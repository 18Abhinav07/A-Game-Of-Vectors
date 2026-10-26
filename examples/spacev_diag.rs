//! Where SPACEV-1B queries spend their I/O, measured on held-out queries (10,000 onwards).
//!
//! 1. Rerank: for `NQ` queries at nprobe 64, the top `C` candidates by code estimate with their
//!    exact scores and whether each is a true top-10 row, dumped to results/spacev-diag.bin for
//!    offline policy simulation (f32 est, f32 exact, u8 truth per candidate).
//! 2. Leaf skew: probe frequencies learned on queries 10,000.., coverage of probed bytes on
//!    queries 0..10,000 by a cache of the hottest leaves.
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use sieve::bigann::{self, D, Index, Params, Timing};

const NQ: usize = 1000;
const C: usize = 4000;

fn main() -> anyhow::Result<()> {
    let dir = Path::new("data/sv1b-b2");
    let index = Index::open(dir, Path::new("data/spacev"), 0)?;
    let (_, _, qs) = bigann::read_bin::<i8>(Path::new("data/spacev/query.i8bin"))?;
    let (nq, gk, gt_ids, gt_dist) = bigann::read_gt(Path::new("data/spacev/public_query_gt100.bin"))?;
    let qs: Vec<[i8; D]> = qs.chunks_exact(D).map(|q| q.try_into().unwrap()).collect();

    let p = Params { nprobe: 64, cand: C, k: 10, refine: 0, margin: 0.0 };
    let mut out = std::io::BufWriter::new(std::fs::File::create("results/spacev-diag.bin")?);
    for qi in 10_000..10_000 + NQ {
        let q = &qs[qi];
        let mut t = Timing::default();
        let (probe, dots) = index.route(q, &p, &mut t);
        let cand = index.candidates(q, &probe, &dots, &p, &mut t);
        let (ids, dist) = (&gt_ids[qi * gk..][..gk], &gt_dist[qi * gk..][..gk]);
        let truth: HashSet<u32> = ids.iter().zip(dist).take_while(|(_, d)| **d <= dist[9]).map(|(i, _)| *i).collect();
        let rows = index.base.rows(&cand.iter().map(|c| c.0).collect::<Vec<_>>());
        let mut rec = vec![0u8; C * 9];
        for (j, (&(id, est, nx), x)) in cand.iter().zip(rows.chunks_exact(D)).enumerate() {
            let dot: i32 = q.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum();
            let exact = 2.0 * dot as f32 - nx;
            rec[j * 9..][..4].copy_from_slice(&est.to_le_bytes());
            rec[j * 9 + 4..][..4].copy_from_slice(&exact.to_le_bytes());
            rec[j * 9 + 8] = truth.contains(&id) as u8;
        }
        out.write_all(&rec)?;
        if qi % 100 == 0 {
            eprint!("\rrerank dump {}/{NQ}", qi - 10_000 + 1);
        }
    }
    out.flush()?;
    eprintln!();

    // Leaf skew.
    let off: Vec<u64> = sieve::store::read_vec(&dir.join("postings.off"))?;
    let bytes: Vec<u64> = off.windows(2).map(|w| w[1] - w[0]).collect();
    let k = bytes.len();
    let probes = |range: std::ops::Range<usize>| {
        let mut hits = vec![0u64; k];
        for qi in range {
            let (probe, _) = index.route(&qs[qi], &Params { nprobe: 64, cand: 1, k: 10, refine: 0, margin: 0.0 }, &mut Timing::default());
            probe.iter().for_each(|&c| hits[c as usize] += 1);
        }
        hits
    };
    let (learn, test) = (probes(10_000..nq), probes(0..10_000));
    let mut order: Vec<usize> = (0..k).collect();
    // Hottest per byte first: hits per byte is what a byte-budgeted cache should maximise.
    order.sort_by(|&a, &b| (learn[b] as f64 / bytes[b].max(1) as f64).total_cmp(&(learn[a] as f64 / bytes[a].max(1) as f64)));
    let total: f64 = (0..k).map(|c| (test[c] * bytes[c]) as f64).sum();
    let (mut used, mut got) = (0u64, 0f64);
    let mut budgets = vec![2u64 << 30, 4 << 30, 8 << 30, 12 << 30, 16 << 30].into_iter().peekable();
    for &c in &order {
        while budgets.peek().is_some_and(|&b| used + bytes[c] > b) {
            println!("cache {:>2} GiB of hottest leaves: {:.1}% of probed bytes", budgets.next().unwrap() >> 30, 100.0 * got / total);
        }
        used += bytes[c];
        got += (test[c] * bytes[c]) as f64;
    }
    println!("probed bytes per query (test): {:.1} MB", total / 1e6 / 10_000.0);
    Ok(())
}
