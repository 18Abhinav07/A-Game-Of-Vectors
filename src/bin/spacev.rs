//! MS SPACEV-1B (DESIGN.md D19): train, add chunks, pack, search.
//!
//!   spacev train   two-level k-means on a sample of one base chunk
//!   spacev add     encode one chunk (or a prefix of it) into rerank rows and run files
//!   spacev pack    merge the runs into SSD posting records
//!   spacev search  recall@10 against the big-ann ground truth, latency and QPS, warm or cold

use anyhow::Result;
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sieve::bigann::{self, D, Index, Model, Params, Timing};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "data/sv")]
    index: PathBuf,
    /// Results are appended here as JSON lines.
    #[arg(long, default_value = "results/spacev.jsonl")]
    log: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Train {
        #[arg(long)]
        base: PathBuf,
        /// Sample only from the first this many rows of the chunk.
        #[arg(long)]
        rows: Option<usize>,
        #[arg(long, default_value_t = 512)]
        k1: usize,
        #[arg(long, default_value_t = 256)]
        k2: usize,
        #[arg(long, default_value_t = 10_000_000)]
        sample: usize,
        #[arg(long, default_value_t = 10)]
        iters: usize,
        /// Refinement code width stored per slot (D20): 1 = 1-bit codes only.
        #[arg(long, default_value_t = 1)]
        bits: usize,
    },
    Add {
        #[arg(long)]
        base: PathBuf,
        /// Id of the chunk's first row (chunk c of SPACEV starts at c·1e8).
        #[arg(long)]
        first: u32,
        /// Only the first this many rows.
        #[arg(long)]
        rows: Option<usize>,
        #[arg(long, default_value_t = 64)]
        buckets: usize,
    },
    Pack {
        #[arg(long, default_value_t = 64)]
        buckets: usize,
    },
    Search {
        #[arg(long, default_value = "data/spacev/query.i8bin")]
        queries: PathBuf,
        #[arg(long)]
        gt: PathBuf,
        /// nprobe:cand[:refine], comma separated (refine: the 1-bit shortlist re-estimated by a
        /// multi-bit index before the best `cand` are reranked).
        #[arg(long, default_value = "32:500,64:500,128:1000,256:1000")]
        configs: String,
        /// Use the first this many queries.
        #[arg(long, default_value_t = 10_000)]
        nq: usize,
        /// Drop the index from the page cache and bypass it for every read.
        #[arg(long)]
        cold: bool,
        /// Hot-cluster cache, in MiB.
        #[arg(long, default_value_t = 0)]
        cache_mb: usize,
        /// Label for the log.
        #[arg(long, default_value = "")]
        label: String,
        /// The base chunks (base.NN), read for the exact rerank.
        #[arg(long, default_value = "data/spacev")]
        base_dir: PathBuf,
        /// Also report how many true top-10 rows reach the candidates.
        #[arg(long)]
        diag: bool,
        /// Pin the hottest leaves in RAM up to this many GiB. Hotness is learned from the
        /// held-out queries (those after the first `nq`), never from the evaluated ones.
        #[arg(long, default_value_t = 0.0)]
        hot_gb: f64,
    },
}

fn log(path: &Path, v: serde_json::Value) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{v}")?;
    Ok(())
}

/// big-ann recall@10 with ties: a result counts if it is any ground-truth row at most as far as
/// the 10th nearest (at 1B, 31% of queries have a tie at rank 10, mostly exact duplicates).
fn recall10(got: &[(f32, u32)], gt_ids: &[u32], gt_dist: &[f32]) -> f64 {
    let cut = gt_dist[9];
    let ok: HashSet<u32> = gt_ids.iter().zip(gt_dist).take_while(|(_, d)| **d <= cut).map(|(i, _)| *i).collect();
    got.iter().take(10).filter(|(_, id)| ok.contains(id)).count() as f64 / 10.0
}

/// Strict recall@10: only the first ten ground-truth ids count, as DiskANN's benchmark scores it.
fn recall10_strict(got: &[(f32, u32)], gt_ids: &[u32]) -> f64 {
    got.iter().take(10).filter(|(_, id)| gt_ids[..10].contains(id)).count() as f64 / 10.0
}

/// Of the true top-10 (with ties), the fraction inside the candidates, against recall@10.
fn diagnose(index: &Index, qs: &[[i8; D]], gt_ids: &[u32], gt_dist: &[f32], gk: usize, p: &Params) {
    let (mut in_cand, mut exact) = (0.0, 0.0);
    for (i, q) in qs.iter().enumerate() {
        let mut t = Timing::default();
        let (probe, dots) = index.route(q, p, &mut t);
        let cand = index.candidates(q, &probe, &dots, p, &mut t);
        let (ids, dist) = (&gt_ids[i * gk..][..gk], &gt_dist[i * gk..][..gk]);
        let truth: Vec<u32> = ids.iter().zip(dist).take_while(|(_, d)| **d <= dist[9]).map(|(i, _)| *i).collect();
        let c: HashSet<u32> = cand.iter().map(|c| c.0).collect();
        in_cand += (truth.iter().filter(|i| c.contains(i)).count().min(10)) as f64 / 10.0;
        exact += recall10(&index.rerank(q, &cand, p, &mut t), ids, dist);
    }
    let n = qs.len() as f64;
    println!("  diag nprobe {} cand {}: true top-10 in candidates {:.4}, R@10 {:.4}", p.nprobe, p.cand, in_cand / n, exact / n);
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

fn main() -> Result<()> {
    let a = Args::parse();
    match a.cmd {
        Cmd::Train { base, rows, k1, k2, sample, iters, bits } => {
            let m = bigann::map_base(&base)?;
            let b = m.as_slice();
            let b = &b[..rows.map_or(b.len(), |r| r * D)];
            anyhow::ensure!((1..=bigann::MAX_BITS).contains(&bits), "--bits must be 1..={}", bigann::MAX_BITS);
            let st = bigann::train(b, &a.index, k1, k2, sample, iters)?;
            sieve::store::write_vec(&a.index.join("bits.u64"), &[bits as u64])?;
            log(&a.log, serde_json::json!({"kind": "train", "index": a.index, "k1": k1, "k2": k2, "leaves": st.leaves, "bits": bits, "sample": sample, "iters": iters, "seconds": st.seconds}))?;
        }
        Cmd::Add { base, first, rows, buckets } => {
            let model = Model::open(&a.index)?;
            let m = bigann::map_base(&base)?;
            let b = m.as_slice();
            let b = &b[..rows.map_or(b.len(), |r| r * D)];
            let st = bigann::add(&a.index, &model, b, first, buckets)?;
            eprintln!("added {} rows from id {first} in {:.0} s ({:.0} rows/s)", st.rows, st.seconds, st.rows as f64 / st.seconds);
            log(&a.log, serde_json::json!({"kind": "add", "index": a.index, "first": first, "rows": st.rows, "seconds": st.seconds}))?;
        }
        Cmd::Pack { buckets } => {
            let t0 = Instant::now();
            let k = Model::open(&a.index)?.k();
            let st = bigann::pack(&a.index, k, buckets)?;
            let s = t0.elapsed().as_secs_f64();
            eprintln!(
                "packed {} slots into {k} leaves, {:.2} GB, largest leaf {}, empty {}, {s:.0} s",
                st.slots,
                st.bytes as f64 / 1e9,
                st.max_leaf,
                st.empty
            );
            log(&a.log, serde_json::json!({"kind": "pack", "index": a.index, "slots": st.slots, "leaves": k, "bytes": st.bytes, "max_leaf": st.max_leaf, "empty": st.empty, "seconds": s}))?;
        }
        Cmd::Search { queries, gt, configs, nq, cold, cache_mb, label, base_dir, diag, hot_gb } => {
            let (qn, qd, qs) = bigann::read_bin::<i8>(&queries)?;
            anyhow::ensure!(qd == D, "queries are {qd}-dim");
            let (gn, gk, gt_ids, gt_dist) = bigann::read_gt(&gt)?;
            anyhow::ensure!(gn == qn, "ground truth has {gn} queries, query file {qn}");
            let nq = nq.min(qn);
            let all: Vec<[i8; D]> = qs.chunks_exact(D).map(|q| q.try_into().unwrap()).collect();
            let qs = &all[..nq];
            let index = Index::open(&a.index, &base_dir, cache_mb << 20)?;
            let rows: u64 = index.postings.file_bytes();
            eprintln!("{} leaves, postings {:.2} GB, {nq} queries", index.model.k(), rows as f64 / 1e9);
            let mut hot_bytes = 0;
            if hot_gb > 0.0 {
                anyhow::ensure!(nq < qn, "--hot-gb learns from the queries after the first --nq; none are left");
                // Routing of the widest config: its nprobe, and its margin if it has one.
                let (nprobe, margin) = configs
                    .split(',')
                    .map(|c| {
                        let f: Vec<&str> = c.split(':').collect();
                        (f[0].parse::<usize>().unwrap_or(64), f.get(3).and_then(|m| m.trim_start_matches('m').parse::<f32>().ok()).unwrap_or(0.0))
                    })
                    .max_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)))
                    .unwrap_or((64, 0.0));
                let t0 = Instant::now();
                let hot = bigann::hot_leaves(&index, &all[nq..], nprobe, margin, (hot_gb * (1u64 << 30) as f64) as u64);
                hot_bytes = index.postings.pin(&hot);
                eprintln!(
                    "pinned {} hottest leaves ({:.2} GiB, learned from {} held-out queries at nprobe {nprobe}, margin {margin}) in {:.1} s",
                    hot.len(),
                    hot_bytes as f64 / (1u64 << 30) as f64,
                    qn - nq,
                    t0.elapsed().as_secs_f64()
                );
            }
            for cfg in configs.split(',') {
                let mut it = cfg.split(':');
                let nprobe: usize = it.next().unwrap().parse()?;
                let cand: usize = it.next().unwrap_or("500").parse()?;
                let refine: usize = it.next().unwrap_or("0").parse()?;
                let margin: f32 = it.next().map(|m| m.trim_start_matches('m').parse()).transpose()?.unwrap_or(0.0);
                let p = Params { nprobe, cand, k: 10, refine, margin };
                if diag {
                    diagnose(&index, qs, &gt_ids, &gt_dist, gk, &p);
                }
                if cold {
                    index.go_cold(&a.index)?;
                } else {
                    // Warm: one untimed pass first.
                    qs.par_iter().for_each(|q| drop(index.search(q, &p, &mut Timing::default())));
                }
                let b0 = index.postings.stats.bytes_read.load(std::sync::atomic::Ordering::Relaxed);
                let mut lat = Vec::with_capacity(nq);
                let mut tm = Timing::default();
                let (mut rec, mut strict) = (0.0, 0.0);
                for (i, q) in qs.iter().enumerate() {
                    let t0 = Instant::now();
                    let got = index.search(q, &p, &mut tm);
                    lat.push(t0.elapsed().as_secs_f64() * 1e3);
                    rec += recall10(&got, &gt_ids[i * gk..][..gk], &gt_dist[i * gk..][..gk]);
                    strict += recall10_strict(&got, &gt_ids[i * gk..][..gk]);
                }
                let mb = (index.postings.stats.bytes_read.load(std::sync::atomic::Ordering::Relaxed) - b0) as f64 / 1e6 / nq as f64;
                if cold {
                    index.go_cold(&a.index)?;
                }
                let t0 = Instant::now();
                qs.par_iter().for_each(|q| drop(index.search(q, &p, &mut Timing::default())));
                let qps = nq as f64 / t0.elapsed().as_secs_f64();
                let n = nq as f64;
                let ms = |ns: u64| ns as f64 / 1e6 / n;
                let (mean, p50, p99) = (lat.iter().sum::<f64>() / n, pct(&mut lat, 0.5), pct(&mut lat, 0.99));
                println!(
                    "{label} {cfg}{}: R@10 {:.4} (strict {:.4}), p50 {p50:.2} ms, p99 {p99:.2} ms, mean {mean:.2} (route {:.2}, scan {:.2}, io {:.2}, rerank {:.2}), {:.1} probes, {:.0} scanned, {mb:.1} MB/q, {qps:.0} QPS",
                    if cold { " cold" } else { "" },
                    rec / n,
                    strict / n,
                    ms(tm.route),
                    ms(tm.scan),
                    ms(tm.io_wait),
                    ms(tm.rerank),
                    tm.probes as f64 / n,
                    tm.scanned as f64 / n
                );
                log(&a.log, serde_json::json!({
                    "kind": "search", "label": label, "index": a.index, "gt": gt, "nq": nq, "nprobe": nprobe, "cand": cand, "refine": refine, "margin": margin, "bits": index.model.bits,
                    "cold": cold, "cache_mb": cache_mb, "hot_bytes": hot_bytes, "recall10": rec / n, "recall10_strict": strict / n, "p50_ms": p50, "p99_ms": p99, "mean_ms": mean,
                    "route_ms": ms(tm.route), "scan_ms": ms(tm.scan), "io_ms": ms(tm.io_wait), "rerank_ms": ms(tm.rerank),
                    "probes": tm.probes as f64 / n, "scanned": tm.scanned as f64 / n, "mb_per_query": mb, "qps": qps,
                }))?;
            }
        }
    }
    Ok(())
}
