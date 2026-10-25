//! Builds the BM25 index and host table from the passage text (DESIGN.md D18).

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use sieve::ivf::Corpus;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "data/ingest")]
    ingest: PathBuf,
    /// Directory of msmarco_v2.1_doc_segmented_NN.json.gz, read in name order.
    #[arg(long, default_value = "data/raw/text")]
    text: PathBuf,
    #[arg(long, default_value = "data/bm25")]
    out: PathBuf,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let corpus = Corpus::open(&a.ingest)?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(&a.text)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.to_string_lossy().ends_with(".json.gz")).collect();
    files.sort();
    let s = sieve::bm25::build(&corpus, &files, &a.out)?;
    let meta = serde_json::json!({
        "rows": s.rows, "skipped_passages": s.skipped, "terms": s.terms, "postings": s.postings,
        "posting_bytes": s.bytes, "seconds": s.seconds, "k1": sieve::bm25::K1, "b": sieve::bm25::B,
    });
    std::fs::write(a.out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    println!("{meta}");
    Ok(())
}
