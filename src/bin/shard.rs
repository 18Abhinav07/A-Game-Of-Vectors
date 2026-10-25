//! A shard server (DESIGN.md D16): owns a balanced subset of the clusters and answers
//! scan-and-rerank requests for them from a coordinator (`search --shards ...`).
//!
//!   shard --index data/index-rabitq --shard 0 --of 4 --listen 127.0.0.1:7000

use anyhow::Result;
use clap::Parser;
use std::net::TcpListener;
use std::path::PathBuf;

use sieve::ivf::{Corpus, Index};
use sieve::shard;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "data/ingest")]
    ingest: PathBuf,
    #[arg(long, default_value = "data/index")]
    index: PathBuf,
    #[arg(long)]
    listen: String,
    /// This shard's number, from 0.
    #[arg(long)]
    shard: u16,
    /// Number of shards the clusters are split across.
    #[arg(long)]
    of: usize,
    /// SSD mode (D13): read owned clusters from postings.bin.
    #[arg(long)]
    ssd: bool,
    #[arg(long, default_value_t = 0)]
    cache_mb: usize,
}

fn main() -> Result<()> {
    let a = Args::parse();
    anyhow::ensure!((a.shard as usize) < a.of, "--shard must be below --of");
    let corpus = Corpus::open(&a.ingest)?;
    let mut index = Index::open(&a.index)?;
    if a.ssd {
        index.use_postings(&a.index, a.cache_mb << 20)?;
    }
    let owner = shard::assign(&index, a.of);
    let mine: Vec<u32> = (0..index.k as u32).filter(|&c| owner[c as usize] == a.shard).collect();
    let slots: usize = mine.iter().map(|&c| index.slots(c)).sum();
    let listener = TcpListener::bind(&a.listen)?;
    eprintln!("shard {}/{}: {} clusters, {slots} slots, listening on {}", a.shard, a.of, mine.len(), a.listen);
    shard::serve(listener, &index, &corpus, &owner, a.shard)
}
