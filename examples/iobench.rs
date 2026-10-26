//! Device ceilings on an uncached file: random reads (100 B unaligned and aligned, 512 B, 4 and 8 KiB) and sequential 256 KiB
//! reads at several thread counts. Usage: iobench <file>
use std::os::unix::fs::FileExt;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("file");
    let p = std::path::Path::new(&path);
    let f = std::fs::File::open(p).unwrap();
    sieve::store::no_cache(&f);
    let len = f.metadata().unwrap().len();
    let mix = |i: u64| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
    for (size, shift, n) in [(100usize, 1000u64, 200_000u64), (100, 0, 200_000), (512, 0, 200_000), (4096, 0, 200_000), (8192, 0, 200_000)] {
        for th in [64usize, 128] {
            sieve::store::evict(p).unwrap();
            let t = Instant::now();
            std::thread::scope(|s| {
                for c in 0..th as u64 {
                    let f = &f;
                    s.spawn(move || {
                        let mut b = vec![0u8; size];
                        for i in (c..n).step_by(th) {
                            let o = (mix(i + 7 * th as u64) % (len / 4096 - 1)) * 4096 + shift;
                            f.read_exact_at(&mut b, o).unwrap();
                        }
                    });
                }
            });
            let s = t.elapsed().as_secs_f64();
            println!("random {size:>4} B +{shift:>4}, {th:>3} threads: {:>6.0}k IOPS, {:.2} GB/s", n as f64 / s / 1e3, (n as usize * size) as f64 / s / 1e9);
        }
    }
    const SEQ: usize = 256 << 10;
    for th in [8usize, 32] {
        sieve::store::evict(p).unwrap();
        let n = 40_000u64;
        let t = Instant::now();
        std::thread::scope(|s| {
            for c in 0..th as u64 {
                let f = &f;
                s.spawn(move || {
                    let mut b = vec![0u8; SEQ];
                    for i in (c..n).step_by(th) {
                        let o = (mix(i) % (len / SEQ as u64 - 1)) * SEQ as u64;
                        f.read_exact_at(&mut b, o).unwrap();
                    }
                });
            }
        });
        let s = t.elapsed().as_secs_f64();
        println!("256 KiB reads, {th:>2} threads: {:.2} GB/s", (n as usize * SEQ) as f64 / s / 1e9);
    }
}
