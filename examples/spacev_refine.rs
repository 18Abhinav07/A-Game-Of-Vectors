//! Would a multi-bit residual code cut the rerank? For held-out queries at nprobe 64, the top
//! C1 candidates of the 1-bit scan are re-encoded from their rows with B-bit uniform codes
//! (sign plane = today's 1-bit code, plus B-1 extra planes), re-ranked by that estimate, and
//! only the top C2 reranked exactly. Reports R@10 per (B, C2) against the 1-bit baseline.
use std::collections::HashSet;
use std::path::Path;

use sieve::bigann::{self, D, Index, PD, Params, Timing, pad_rotate};

const NQ: usize = 1000;
const C1: usize = 2000;

/// B-bit uniform code of r: levels (u - (2^B-1)/2)·Δ, Δ chosen to maximise cos(r, r̂).
fn quantize(r: &[f32], bits: u32) -> Vec<f32> {
    let h = ((1u32 << bits) - 1) as f32 / 2.0;
    let amax = r.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut best = (f32::MIN, vec![]);
    for s in 1..=40 {
        let delta = amax / h * (0.25 + s as f32 * 0.025);
        let rh: Vec<f32> = r.iter().map(|&v| ((v / delta + h).floor().clamp(0.0, 2.0 * h) - h) * delta).collect();
        let (dot, n2) = r.iter().zip(&rh).fold((0f32, 0f32), |(d, n), (a, b)| (d + a * b, n + b * b));
        let cos = dot / n2.sqrt().max(1e-9);
        if cos > best.0 {
            best = (cos, rh);
        }
    }
    best.1
}

fn main() -> anyhow::Result<()> {
    let index = Index::open(Path::new("data/sv1b-b2"), Path::new("data/spacev"), 0)?;
    let (_, _, qs) = bigann::read_bin::<i8>(Path::new("data/spacev/query.i8bin"))?;
    let (_, gk, gt_ids, gt_dist) = bigann::read_gt(Path::new("data/spacev/public_query_gt100.bin"))?;
    let qs: Vec<[i8; D]> = qs.chunks_exact(D).map(|q| q.try_into().unwrap()).collect();
    let p = Params { nprobe: 64, cand: C1, k: 10, refine: 0, margin: 0.0 };
    let c2s = [20usize, 50, 100, 150, 200, 300, 500];
    let bitss = [1u32, 2, 3, 4];
    let mut rec = vec![vec![0f64; c2s.len()]; bitss.len()];
    let mut err = vec![0f64; bitss.len()];
    for qi in 10_000..10_000 + NQ {
        let q = &qs[qi];
        let mut t = Timing::default();
        let (probe, dots) = index.route(q, &p, &mut t);
        let cand = index.candidates(q, &probe, &dots, &p, &mut t);
        let (ids, dist) = (&gt_ids[qi * gk..][..gk], &gt_dist[qi * gk..][..gk]);
        let truth: HashSet<u32> = ids.iter().zip(dist).take_while(|(_, d)| **d <= dist[9]).map(|(i, _)| *i).collect();
        let rows = index.base.rows(&cand.iter().map(|c| c.0).collect::<Vec<_>>());
        let xf: Vec<f32> = rows.iter().map(|&v| v as f32).collect();
        let leaves = index.model.assign(&xf);
        let qf: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        let qr = pad_rotate(&index.model.rot, &qf);
        let exact: Vec<f32> = cand.iter().zip(rows.chunks_exact(D)).map(|(c, x)| 2.0 * q.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum::<i32>() as f32 - c.2).collect();
        for (bi, &bits) in bitss.iter().enumerate() {
            let est: Vec<f32> = (0..cand.len())
                .map(|j| {
                    if bits == 1 {
                        return cand[j].1;
                    }
                    let xr = pad_rotate(&index.model.rot, &xf[j * D..][..D]);
                    let c = index.model.leaf_rot(leaves[j] as usize);
                    let r: Vec<f32> = (0..PD).map(|i| xr[i] - c[i]).collect();
                    let rh = quantize(&r, bits);
                    let (rr, rrh) = r.iter().zip(&rh).fold((0f32, 0f32), |(a, b), (x, y)| (a + x * x, b + x * y));
                    let qrh: f32 = qr.iter().zip(&rh).map(|(a, b)| a * b).sum();
                    let qc: f32 = qr.iter().zip(c).map(|(a, b)| a * b).sum();
                    2.0 * (qc + qrh * rr / rrh) - cand[j].2
                })
                .collect();
            err[bi] += (est.iter().zip(&exact).map(|(e, x)| ((e - x) as f64).powi(2)).sum::<f64>() / est.len() as f64).sqrt();
            let mut order: Vec<usize> = (0..cand.len()).collect();
            order.sort_by(|&a, &b| est[b].total_cmp(&est[a]));
            for (ci, &c2) in c2s.iter().enumerate() {
                let mut top: Vec<usize> = order[..c2.min(order.len())].to_vec();
                top.sort_by(|&a, &b| exact[b].total_cmp(&exact[a]));
                rec[bi][ci] += top.iter().take(10).filter(|&&j| truth.contains(&cand[j].0)).count() as f64 / 10.0;
            }
        }
    }
    for (bi, bits) in bitss.iter().enumerate() {
        let r: Vec<String> = c2s.iter().zip(&rec[bi]).map(|(c, r)| format!("{c}:{:.4}", r / NQ as f64)).collect();
        println!("{bits}-bit (rms err {:.0}): {}", err[bi] / NQ as f64, r.join("  "));
    }
    Ok(())
}
