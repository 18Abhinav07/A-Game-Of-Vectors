use std::os::unix::fs::FileExt;
use std::time::Instant;
fn main() {
    let f = std::sync::Arc::new(std::fs::File::open("data/spacev/base.00").unwrap());
    let n = 200_000u64;
    let ids: Vec<u64> = (0..n).map(|i| (i.wrapping_mul(2654435761) % 10_000_000) * 100).collect();
    let mut b = [0u8; 100];
    for &o in &ids { f.read_exact_at(&mut b, o).unwrap(); } // warm
    let t = Instant::now();
    for &o in &ids { f.read_exact_at(&mut b, o).unwrap(); }
    println!("1 thread: {:.2} us/read", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    for th in [4usize, 12, 32] {
        let t = Instant::now();
        std::thread::scope(|s| for c in ids.chunks(ids.len() / th) { let f = f.clone(); s.spawn(move || { let mut b = [0u8; 100]; for &o in c { f.read_exact_at(&mut b, o).unwrap(); } }); });
        println!("{th} threads: {:.3} us/read aggregate", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
    let m = unsafe { memmap2::Mmap::map(&*f).unwrap() };
    let t = Instant::now(); let mut s = 0u64;
    for &o in &ids { s += m[o as usize] as u64; }
    println!("mmap: {:.3} us/read ({s})", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    unsafe extern "C" { fn mincore(addr: *const u8, len: usize, vec: *mut u8) -> i32; }
    let t = Instant::now(); let mut hits = 0;
    for &o in &ids {
        let pg = (o as usize) & !16383; let mut v = 0u8;
        unsafe { mincore(m.as_ptr().add(pg), 16384, &mut v) };
        hits += (v & 1) as usize;
    }
    println!("mincore: {:.3} us/call, {hits} resident", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    for th in [12usize] {
        let t = Instant::now();
        std::thread::scope(|s| for c in ids.chunks(ids.len() / th) { let m = &m; s.spawn(move || { let mut v = 0u8; for &o in c { let pg = (o as usize) & !16383; unsafe { mincore(m.as_ptr().add(pg), 16384, &mut v) }; } }); });
        println!("mincore {th} threads: {:.3} us aggregate", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
}
