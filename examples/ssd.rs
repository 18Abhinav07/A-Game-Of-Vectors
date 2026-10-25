//! Random 768-byte reads from the int8 rerank file with the page cache bypassed (F_NOCACHE),
//! at several queue depths (one blocking pread per thread). The physics of the SSD rerank tier.
//! cargo run --release --example ssd -- data/ingest/rerank.i8

use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("path to rerank.i8");
    let f = std::fs::File::open(&path).unwrap();
    // SAFETY: fcntl on an fd we own.
    unsafe { libc_fcntl(f.as_raw_fd(), 48 /* F_NOCACHE */, 1) };
    let rows = f.metadata().unwrap().len() / 768;
    for qd in [1usize, 4, 16, 64] {
        let per = 20_000 / qd.min(16) ;
        let t = Instant::now();
        std::thread::scope(|s| {
            for th in 0..qd {
                let f = &f;
                s.spawn(move || {
                    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ th as u64;
                    let mut buf = [0u8; 768];
                    for _ in 0..per {
                        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                        f.read_exact_at(&mut buf, (x % rows) * 768).unwrap();
                    }
                });
            }
        });
        let s = t.elapsed().as_secs_f64();
        let n = (per * qd) as f64;
        println!("QD {qd:>2}: {:>7.0} reads/s  {:>6.1} us/read latency  -> 1000-candidate rerank {:.2} ms", n / s, s * 1e6 / per as f64, 1000.0 / (n / s) * 1e3);
    }
}

unsafe extern "C" {
    #[link_name = "fcntl"]
    fn libc_fcntl(fd: i32, cmd: i32, ...) -> i32;
}
