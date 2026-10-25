//! Checks that `store::evict` really drops a file from the page cache:
//! `cargo run --release --example evict -- <file>`.
use sieve::store::{evict, resident};
use std::io::Read;

/// Reads the whole file through a small buffer and returns the bytes read.
fn read_all(path: &std::path::Path) -> anyhow::Result<u64> {
    let (mut f, mut buf, mut n) = (std::fs::File::open(path)?, vec![0u8; 8 << 20], 0u64);
    loop {
        match f.read(&mut buf)? {
            0 => return Ok(n),
            k => n += k as u64,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from(std::env::args().nth(1).expect("usage: evict <file>"));
    let t = std::time::Instant::now();
    let n = read_all(&path)?;
    println!("read {} MB in {:.2}s, resident {:.3}", n >> 20, t.elapsed().as_secs_f64(), resident(&path)?);
    evict(&path)?;
    println!("after evict: resident {:.3}", resident(&path)?);
    let t = std::time::Instant::now();
    read_all(&path)?;
    println!("re-read in {:.2}s, resident {:.3}", t.elapsed().as_secs_f64(), resident(&path)?);
    Ok(())
}
