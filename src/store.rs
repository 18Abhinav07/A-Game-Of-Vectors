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
    anyhow::ensure!(len % size_of::<T>() == 0, "{} is not a whole number of items", path.display());
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
