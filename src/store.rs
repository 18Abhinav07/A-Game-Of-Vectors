//! Flat binary files of plain-old-data, written by appending and read by mmap.

use anyhow::{Context, Result};
use bytemuck::Pod;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::Path;

pub struct Appender {
    w: BufWriter<File>,
}

impl Appender {
    pub fn create(path: &Path) -> Result<Self> {
        let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
        Ok(Self { w: BufWriter::with_capacity(8 << 20, f) })
    }

    pub fn write<T: Pod>(&mut self, xs: &[T]) -> Result<()> {
        Ok(self.w.write_all(bytemuck::cast_slice(xs))?)
    }

    pub fn finish(mut self) -> Result<()> {
        self.w.flush()?;
        Ok(())
    }
}

pub fn write_vec<T: Pod>(path: &Path, xs: &[T]) -> Result<()> {
    let mut a = Appender::create(path)?;
    a.write(xs)?;
    a.finish()
}

pub fn read_vec<T: Pod>(path: &Path) -> Result<Vec<T>> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let len = f.metadata()?.len() as usize;
    anyhow::ensure!(len.is_multiple_of(size_of::<T>()), "{} is not a whole number of items", path.display());
    let mut v = vec![T::zeroed(); len / size_of::<T>()];
    f.read_exact(bytemuck::cast_slice_mut(&mut v))?;
    Ok(v)
}

/// A read-only typed view of a file. mmap is page-aligned, so any Pod type casts cleanly.
pub struct Mapped<T> {
    map: Option<Mmap>,
    _t: PhantomData<T>,
}

impl<T: Pod> Mapped<T> {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        if f.metadata()?.len() == 0 {
            return Ok(Self { map: None, _t: PhantomData });
        }
        // SAFETY: index files are written once and never modified while mapped.
        let map = unsafe { Mmap::map(&f)? };
        Ok(Self { map: Some(map), _t: PhantomData })
    }

    pub fn as_slice(&self) -> &[T] {
        self.map.as_deref().map_or(&[], bytemuck::cast_slice)
    }

    /// Hint that the whole file will be read soon (pre-faults it into the page cache).
    pub fn will_need(&self) {
        if let Some(m) = &self.map {
            let _ = m.advise(memmap2::Advice::WillNeed);
        }
    }
}

/// F_NOCACHE on macOS: reads through this descriptor no longer add pages to the cache. Pages
/// that are already resident are still served from it; `evict` removes those.
pub fn no_cache(f: &File) {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        unsafe extern "C" {
            fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        }
        // SAFETY: fcntl(F_NOCACHE = 48) on a descriptor the caller owns.
        unsafe {
            fcntl(f.as_raw_fd(), 48, 1);
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = f;
}

unsafe extern "C" {
    fn msync(addr: *mut std::ffi::c_void, len: usize, flags: i32) -> i32;
    fn mincore(addr: *const std::ffi::c_void, len: usize, vec: *mut u8) -> i32;
    fn getpagesize() -> i32;
}

/// Same value on macOS and Linux.
const MS_INVALIDATE: i32 = 2;

/// Drops a file's clean pages from the page cache (what `vmtouch -e` does off Linux), so the
/// next reads come from the device. Lets a laptop measure cold reads on files that fit in RAM.
pub fn evict(path: &Path) -> Result<()> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    if f.metadata()?.len() == 0 {
        return Ok(());
    }
    // SAFETY: a private read-only map of a file nobody writes while it is mapped.
    let map = unsafe { Mmap::map(&f)? };
    // SAFETY: the range is exactly the live mapping.
    let rc = unsafe { msync(map.as_ptr() as *mut _, map.len(), MS_INVALIDATE) };
    anyhow::ensure!(rc == 0, "msync(MS_INVALIDATE) failed for {}", path.display());
    Ok(())
}

/// Fraction of a file's pages that are resident in the page cache.
pub fn resident(path: &Path) -> Result<f64> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    if f.metadata()?.len() == 0 {
        return Ok(0.0);
    }
    // SAFETY: as in `evict`.
    let map = unsafe { Mmap::map(&f)? };
    // SAFETY: no preconditions.
    let page = unsafe { getpagesize() } as usize;
    let mut vec = vec![0u8; map.len().div_ceil(page)];
    // SAFETY: `vec` has one byte per page of the mapping.
    let rc = unsafe { mincore(map.as_ptr() as *const _, map.len(), vec.as_mut_ptr()) };
    anyhow::ensure!(rc == 0, "mincore failed for {}", path.display());
    Ok(vec.iter().filter(|&&b| b & 1 == 1).count() as f64 / vec.len() as f64)
}

/// A pool of threads that only block in positioned reads. macOS has no io_uring, so queue
/// depth on the SSD equals the number of reads blocked at once; the compute pool (one thread
/// per core) would cap it near 12, while the device keeps scaling to QD64 (BENCHMARKS.md,
/// kernels). Sized by SIEVE_IO_THREADS, default 64.
///
/// Callers block on a channel until their jobs finish. A rayon pool would not do: a rayon
/// worker that `install`s into another pool keeps stealing work from its own pool while it
/// waits, so each one starts more queries on top of the pending ones (measured: 13 GB peak RSS
/// and 7× lower QPS).
pub struct IoPool {
    tx: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>,
    threads: usize,
}

impl IoPool {
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs every job on the pool and returns their results in order.
    pub fn run<T: Send + 'static>(&self, jobs: Vec<Box<dyn FnOnce() -> T + Send>>) -> Vec<T> {
        let n = jobs.len();
        let rrx = self.stream(jobs);
        let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
        for _ in 0..n {
            let (i, v) = rrx.recv().expect("io job panicked");
            out[i] = Some(v);
        }
        out.into_iter().map(|v| v.expect("every job reports")).collect()
    }

    /// Submits every job and returns `(job index, result)` in completion order, so the caller
    /// can work on early results while later reads are still in flight.
    pub fn stream<T: Send + 'static>(&self, jobs: Vec<Box<dyn FnOnce() -> T + Send>>) -> std::sync::mpsc::Receiver<(usize, T)> {
        let (rtx, rrx) = std::sync::mpsc::channel();
        for (i, job) in jobs.into_iter().enumerate() {
            let rtx = rtx.clone();
            self.tx.send(Box::new(move || drop(rtx.send((i, job()))))).expect("io pool is alive");
        }
        rrx
    }
}

pub fn io_pool() -> &'static IoPool {
    static POOL: std::sync::OnceLock<IoPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("SIEVE_IO_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        for i in 0..n {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("sieve-io-{i}"))
                .spawn(move || loop {
                    let job = rx.lock().unwrap().recv();
                    match job {
                        Ok(job) => job(),
                        Err(_) => return,
                    }
                })
                .expect("spawn io thread");
        }
        IoPool { tx, threads: n }
    })
}
