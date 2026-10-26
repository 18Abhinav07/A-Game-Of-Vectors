use std::os::unix::fs::FileExt;
use std::time::Instant;
fn main() {
    let p = std::path::Path::new("data/spacev/base.00");
    let f = std::fs::File::open(p).unwrap();
    let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let n = 20_000u64;
    for th in [1usize, 16, 64] {
        sieve::store::evict(p).unwrap();
        let ids: Vec<u64> = (0..n).map(|i| ((i + th as u64 * 77777).wrapping_mul(2654435761) % 100_000_000) * 100).collect();
        let t = Instant::now();
        std::thread::scope(|s| for c in ids.chunks(ids.len() / th) { let m = &m; s.spawn(move || { let mut x = 0u8; for &o in c { x ^= unsafe { std::ptr::read_volatile(&m[o as usize]) }; } x }); });
        println!("mmap faults, {th} threads: {:.2} us/read aggregate", t.elapsed().as_secs_f64() * 1e6 / n as f64);
        sieve::store::evict(p).unwrap();
        let t = Instant::now();
        std::thread::scope(|s| for c in ids.chunks(ids.len() / th) { let f = &f; s.spawn(move || { let mut b = [0u8; 100]; for &o in c { f.read_exact_at(&mut b, o + 3).unwrap(); } }); });
        println!("pread misses, {th} threads: {:.2} us/read aggregate", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
}
