//! Metadata filters applied during the search (DESIGN.md D18).
//!
//! A filter is a bitset of allowed rows. The scan skips disallowed slots at admission, so they
//! never take a place in the candidate buffer. What changes with selectivity is where the
//! allowed rows are: at a fixed probe count, a 1% filter leaves about 1% of the scanned slots
//! eligible. So the planner either probes further (until a target count of allowed slots has
//! been scanned, using per-cluster allowed counts computed once per filter) or, for filters
//! small enough, skips the index and scores the allowed rows exactly.

use rayon::prelude::*;

use crate::ivf::{Corpus, Index, PAD, Params, Scratch, Timing, View};
use crate::lut::BLOCK;
use crate::quant::{dot_i8i8, query_i8};
use crate::topk::TopK;
use crate::DIM;

pub struct Filter {
    pub allow: Vec<u64>,
    /// The allowed rows, ascending.
    pub rows: Vec<u32>,
    /// Allowed slots per cluster.
    pub per_cluster: Vec<u32>,
}

impl Filter {
    pub fn new(allow: Vec<u64>, index: &Index) -> Self {
        let rows = allow.iter().enumerate().flat_map(|(w, &b)| (0..64).filter(move |i| b >> i & 1 == 1).map(move |i| (w * 64 + i) as u32)).collect();
        let ids = index.ids.as_slice();
        let per_cluster = (0..index.k)
            .into_par_iter()
            .map(|c| {
                let (b0, b1) = (index.block_off[c] as usize, index.block_off[c + 1] as usize);
                ids[b0 * BLOCK..b1 * BLOCK].iter().filter(|&&id| id != PAD && allow.get(id as usize >> 6).is_some_and(|w| w >> (id & 63) & 1 == 1)).count() as u32
            })
            .collect();
        Self { allow, rows, per_cluster }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Plan {
    /// The unfiltered probes, disallowed rows skipped at admission.
    Post,
    /// Probe clusters in centroid order until this many allowed slots are covered.
    Expand(usize),
    /// Exact int8 scores of every allowed row; no index.
    Brute,
    /// Pick one of the above from the filter's size (thresholds from the M1 sweep, D18).
    Auto,
}

/// At or below this many allowed rows, scoring them all beats any index plan (184k rows: 3.2 ms
/// exact, against 15 ms for an expanding probe at R@10 0.88).
pub const BRUTE_MAX: usize = 250_000;
/// At or above this allowed fraction, the unfiltered probes still cover enough allowed slots.
pub const POST_MIN: f64 = 0.3;
/// Allowed slots an expanding probe covers between the two.
pub const EXPAND_TARGET: usize = 200_000;

impl Plan {
    pub fn resolve(self, f: &Filter, n: usize) -> Plan {
        match self {
            Plan::Auto if f.rows.len() <= BRUTE_MAX => Plan::Brute,
            Plan::Auto if f.rows.len() as f64 >= POST_MIN * n as f64 => Plan::Post,
            Plan::Auto => Plan::Expand(EXPAND_TARGET),
            p => p,
        }
    }
}

/// The top `p.k` allowed rows as (score, row), best first.
#[allow(clippy::too_many_arguments)]
pub fn search(index: &Index, corpus: &Corpus, f: &Filter, plan: Plan, q768: &[f32], q256: &[f32], p: &Params, s: &mut Scratch, t: &mut Timing) -> Vec<(f32, u32)> {
    let v = View { allow: Some(&f.allow), ..Default::default() };
    match plan.resolve(f, corpus.n) {
        Plan::Post => index.search_view(corpus, q768, q256, p, &v, s, t),
        Plan::Expand(target) => {
            let t0 = std::time::Instant::now();
            let all = index.route(q256, &Params { nprobe: index.k, margin: None, budget: None, ..*p }, s);
            let mut covered = 0usize;
            let probe: Vec<(f32, u32)> = all
                .into_iter()
                .filter(|&(_, c)| f.per_cluster[c as usize] > 0)
                .take_while(|&(_, c)| {
                    let go = covered < target;
                    covered += f.per_cluster[c as usize] as usize;
                    go
                })
                .collect();
            t.route += t0.elapsed().as_nanos() as u64;
            let cand = index.candidates(q256, &probe, p, &v, s, t);
            index.rerank(corpus, q768, q256, cand, p, &v, s, t)
        }
        Plan::Brute => {
            let t0 = std::time::Instant::now();
            let w: Vec<f32> = q768.iter().zip(&corpus.quant.scale).map(|(&x, &sc)| x / sc).collect();
            let mut wq = Vec::new();
            let inv = 1.0 / query_i8(&w, &mut wq);
            let mut top = TopK::new(p.k);
            for &r in &f.rows {
                top.push(dot_i8i8(&wq, corpus.rerank_row(r)) as f32 * inv, r);
            }
            t.reranked += f.rows.len() as u64;
            t.rerank += t0.elapsed().as_nanos() as u64;
            top.into_sorted()
        }
        Plan::Auto => unreachable!(),
    }
}

/// Exact top `k` among allowed rows for many queries at once, in one pass over the rows: each
/// row is scored against every query while it is in cache. Scores are the int8 inner products
/// the rerank uses.
pub fn truth(corpus: &Corpus, f: &Filter, queries: &[&[f32]], k: usize) -> Vec<Vec<u32>> {
    let qs: Vec<(Vec<i8>, f32)> = queries
        .iter()
        .map(|q| {
            let w: Vec<f32> = q.iter().zip(&corpus.quant.scale).map(|(&x, &sc)| x / sc).collect();
            let mut wq = Vec::new();
            let inv = 1.0 / query_i8(&w, &mut wq);
            (wq, inv)
        })
        .collect();
    let merged = f
        .rows
        .par_chunks(16384)
        .map(|chunk| {
            let mut tops: Vec<TopK<f32>> = (0..qs.len()).map(|_| TopK::new(k)).collect();
            for &r in chunk {
                let row = corpus.rerank_row(r);
                debug_assert_eq!(row.len(), DIM);
                for (top, (wq, inv)) in tops.iter_mut().zip(&qs) {
                    top.push(dot_i8i8(wq, row) as f32 * inv, r);
                }
            }
            tops
        })
        .reduce(
            || (0..qs.len()).map(|_| TopK::new(k)).collect(),
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    for (s, r) in y.into_sorted() {
                        x.push(s, r);
                    }
                }
                a
            },
        );
    merged.into_iter().map(|t| t.into_sorted().into_iter().map(|x| x.1).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_plan() {
        let f = |rows: usize| Filter { allow: Vec::new(), rows: vec![0; rows], per_cluster: Vec::new() };
        let n = 10_000_000;
        assert!(matches!(Plan::Auto.resolve(&f(BRUTE_MAX), n), Plan::Brute));
        assert!(matches!(Plan::Auto.resolve(&f(1_000_000), n), Plan::Expand(EXPAND_TARGET)));
        assert!(matches!(Plan::Auto.resolve(&f(3_000_000), n), Plan::Post));
        assert!(matches!(Plan::Expand(7).resolve(&f(1), n), Plan::Expand(7)));
    }
}
