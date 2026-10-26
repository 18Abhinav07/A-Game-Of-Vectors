//! Cold rerank-read throughput through `Base::rows`: 12 concurrent "queries" of `n` random ids.
use rayon::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let n: usize = std::env::args().nth(1).map_or(500, |v| v.parse().unwrap());
    let base = sieve::bigann::Base::open(std::path::Path::new("data/spacev"))?;
    base.go_cold()?;
    let mix = |i: u64| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20) % 1_000_000_000;
    let qs = 2000usize;
    let t = Instant::now();
    (0..qs).into_par_iter().for_each(|q| {
        let ids: Vec<u32> = (0..n).map(|j| mix((q * n + j) as u64 + 12345) as u32).collect();
        std::hint::black_box(base.rows(&ids));
    });
    let s = t.elapsed().as_secs_f64();
    println!("{n} rows/query: {:.0} queries/s, {:.0}k rows/s", qs as f64 / s, (qs * n) as f64 / s / 1e3);
    Ok(())
}
