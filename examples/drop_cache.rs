//! Drops files from the page cache, so the next reads come from the device:
//! `cargo run --release --example drop_cache -- <file>...`. Used before cold runs of other
//! programs (the DiskANN head-to-head, DESIGN.md D22).
use sieve::store::{evict, resident};

fn main() -> anyhow::Result<()> {
    for p in std::env::args().skip(1).map(std::path::PathBuf::from) {
        let before = resident(&p)?;
        evict(&p)?;
        println!("{}: resident {before:.3} -> {:.3}", p.display(), resident(&p)?);
    }
    Ok(())
}
