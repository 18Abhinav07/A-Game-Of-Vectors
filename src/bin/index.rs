//! `index`: trains IVF centroids and writes the cluster-major fast-scan index.

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use sieve::ivf::{Corpus, build};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "data/ingest")]
    ingest: PathBuf,
    #[arg(long, default_value = "data/index")]
    out: PathBuf,
    /// Number of clusters. Exa uses 100k at billions of vectors; ~sqrt(n) is the usual start.
    #[arg(long, default_value_t = 4096)]
    k: usize,
    /// Vectors sampled to train the centroids.
    #[arg(long, default_value_t = 400_000)]
    sample: usize,
    #[arg(long, default_value_t = 12)]
    iters: usize,
    /// Spill every vector into a second, SOAR-chosen cluster with this λ (1.0 in the paper).
    #[arg(long)]
    soar: Option<f32>,
    /// Store RaBitQ residual codes (per-cluster, rotated) instead of global sign codes.
    #[arg(long)]
    rabitq: bool,
    /// Only write SSD posting lists (postings.bin, postings.off) for the existing index at --out.
    #[arg(long)]
    pack: bool,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let t0 = Instant::now();
    if a.pack {
        let st = sieve::postings::pack(&a.out)?;
        println!("{} clusters, {:.2} GB of posting lists in {:.1}s", st.clusters, st.bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
        return Ok(());
    }
    let corpus = Corpus::open(&a.ingest)?;
    let stats = build(&corpus, &a.out, a.k, a.sample, a.iters, a.soar, a.rabitq)?;
    let secs = t0.elapsed().as_secs_f64();

    let mut sizes = stats.sizes.clone();
    sizes.sort_unstable();
    let pct = |p: f64| sizes[((sizes.len() - 1) as f64 * p) as usize];
    let padded: usize = sizes.iter().map(|s| s.div_ceil(sieve::lut::BLOCK) * sieve::lut::BLOCK).sum();
    let meta = serde_json::json!({
        "rows": corpus.n, "k": stats.k, "soar_lambda": a.soar, "rabitq": a.rabitq, "sample": a.sample, "iters": a.iters, "seconds": secs,
        "cluster_size": { "min": pct(0.0), "p50": pct(0.5), "p99": pct(0.99), "max": pct(1.0) },
        "padding_overhead": padded as f64 / corpus.n as f64 - 1.0,
        "stored_per_vector": padded as f64 / corpus.n as f64,
    });
    std::fs::write(a.out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    println!("{}", serde_json::to_string_pretty(&meta)?);
    Ok(())
}
