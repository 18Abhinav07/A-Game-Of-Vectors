//! Fast-scan kernel throughput by working-set size (L1, L2, DRAM), single thread.
//! cargo run --release --example kernel

use sieve::lut::{BLOCK, BLOCK_BYTES, Lut, scan};
use std::time::Instant;

fn main() {
    let q: Vec<f32> = (0..sieve::PDIM).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
    let lut = Lut::new(&q);
    for kb in [32usize, 1 << 10, 8 << 10, 1 << 20] {
        let nblocks = kb * 1024 / BLOCK_BYTES;
        let blocks: Vec<u8> = (0..nblocks * BLOCK_BYTES).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let mut out = vec![0u16; nblocks * BLOCK];
        let reps = ((1usize << 30) / (kb * 1024)).max(3);
        scan(&lut, &blocks, &mut out);
        let t = Instant::now();
        for _ in 0..reps {
            scan(&lut, std::hint::black_box(&blocks), &mut out);
            std::hint::black_box(&out);
        }
        let s = t.elapsed().as_secs_f64();
        let codes = (nblocks * BLOCK * reps) as f64;
        println!("{kb:>8} KiB  {:>6.0} M codes/s  {:>5.1} GB/s  {:.1} ns/block", codes / s / 1e6, codes * 32.0 / s / 1e9, s * 1e9 / (nblocks * reps) as f64);
    }
}
