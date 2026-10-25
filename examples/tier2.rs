//! Which RAM-resident middle tier can stand in for the int8 256-d prefix? For each query, take
//! the 1000 LUT candidates, rescore them with a cheaper representation (Matryoshka prefix width
//! × bits per dim), keep the best `r`, rerank those with int8 768-d, and measure recall@10.
//! cargo run --release --example tier2 -- data/ingest data/index

use rayon::prelude::*;
use sieve::ivf::{Corpus, Index, Params, Rerank, Scratch, Timing};
use sieve::quant::prefix;
use sieve::store::read_vec;
use sieve::{DIM, PDIM};
use std::path::Path;

/// Cosine between a query prefix and a stored row's prefix of width `d`, with the row's int8
/// values requantised to `bits` (8 = as stored, 4 = 16 levels, 2 = 4 levels, 1 = sign).
fn tier_score(q: &[f32], row: &[i8], scale: &[f32], d: usize, bits: u32) -> f32 {
    let shift = 8 - bits;
    let (mut dot, mut nn) = (0f32, 0f32);
    for i in 0..d {
        let v = if bits == 1 {
            if row[i] >= 0 { 1.0 } else { -1.0 }
        } else {
            let step = (1i32 << shift) as f32;
            ((row[i] as f32 / step).round() * step) / scale[i]
        };
        dot += q[i] * v;
        nn += v * v;
    }
    dot / nn.sqrt().max(1e-12)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (ing, idx) = (Path::new(&args[1]), Path::new(&args[2]));
    let corpus = Corpus::open(ing).unwrap();
    let index = Index::open(idx).unwrap();
    let q768: Vec<f32> = read_vec(&ing.join("queries.f32")).unwrap();
    let nq = q768.len() / DIM;
    let gt: Vec<u32> = read_vec(&ing.join("gt768.ids")).unwrap();
    let gk = gt.len() / nq;
    let p = Params { nprobe: 32, cand: 1000, rerank: Rerank::None, k: 1000, prune: None };
    let w768 = |q: &[f32]| corpus.quant.query_weights(q);

    let cands: Vec<Vec<u32>> = (0..nq)
        .into_par_iter()
        .map_init(Scratch::new, |s, i| {
            let q = &q768[i * DIM..][..DIM];
            let mut q256 = [0f32; PDIM];
            prefix(q, &mut q256);
            index.search(&corpus, q, &q256, &p, s, &mut Timing::default()).into_iter().map(|x| x.1).collect()
        })
        .collect();

    println!("| tier 2 | bytes/vector | GB at 113.5M | R@10 r=50 | R@10 r=100 | R@10 r=200 |");
    println!("|---|---|---|---|---|---|");
    for (d, bits) in [(256, 8), (256, 4), (128, 8), (256, 2), (128, 4), (64, 8), (128, 2), (64, 4), (256, 1)] {
        let mut row = format!("| {d}-d × {bits} bit | {} | {:.1} |", d * bits as usize / 8, 113.5e6 * (d * bits as usize / 8) as f64 / 1e9);
        for r in [50usize, 100, 200] {
            let rec: f64 = (0..nq)
                .into_par_iter()
                .map(|i| {
                    let q = &q768[i * DIM..][..DIM];
                    let mut tier: Vec<(f32, u32)> = cands[i].iter().map(|&id| (tier_score(q, corpus.rerank_row(id), &corpus.quant.scale, d, bits), id)).collect();
                    tier.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let w = w768(q);
                    let mut full: Vec<(f32, u32)> = tier.iter().take(r).map(|&(_, id)| (sieve::quant::dot_i8(&w, corpus.rerank_row(id)), id)).collect();
                    full.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let ids: Vec<u32> = full.iter().take(10).map(|x| x.1).collect();
                    sieve::eval::recall_at(10, &ids, &gt[i * gk..][..gk])
                })
                .sum::<f64>() / nq as f64;
            row += &format!(" {rec:.4} |");
        }
        println!("{row}");
    }
}
