//! SSD-resident posting lists (DESIGN.md D13): the scan tier moves out of RAM.
//!
//! In RAM mode the index keeps every code, id and RaBitQ factor resident (about 17 GB at
//! 113.5M with SOAR). Here each cluster becomes one contiguous record on disk, so probing a
//! cluster costs one positioned read, and RAM holds only the centroids, the record offsets and an
//! optional cache of hot clusters. This is the SPANN layout (Chen et al., NeurIPS 2021), applied
//! to 1-bit RaBitQ codes instead of full vectors, so a record is about 44 B per slot.
//!
//! Record c starts at `off[c]` (4 KiB aligned) and holds, for its `nb` blocks of 32 slots:
//!
//!   blocks   nb · BLOCK_BYTES    nibble-transposed codes, as in blocks.bin
//!   factors  nb · BLOCK · 2 f32  RaBitQ [mul, err] per slot
//!   ids      nb · BLOCK u32      corpus row per slot (PAD for padding)

use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::lut::{BLOCK, BLOCK_BYTES};
use crate::store::{Mapped, read_vec, write_vec};

const ALIGN: u64 = 4096;
/// Bytes of one block's slice of a record.
pub const BLOCK_RECORD: usize = BLOCK_BYTES + BLOCK * 2 * 4 + BLOCK * 4;

/// One cluster's record, read from disk or the cache. u32 storage keeps it 4-byte aligned, so
/// the factor and id sections cast in place.
pub struct Record {
    nb: usize,
    words: Vec<u32>,
}

impl Record {
    fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }

    pub fn blocks(&self) -> &[u8] {
        &self.bytes()[..self.nb * BLOCK_BYTES]
    }

    pub fn factors(&self) -> &[f32] {
        let a = self.nb * BLOCK_BYTES / 4;
        bytemuck::cast_slice(&self.words[a..a + self.nb * BLOCK * 2])
    }

    pub fn ids(&self) -> &[u32] {
        let a = (self.nb * BLOCK_BYTES + self.nb * BLOCK * 8) / 4;
        &self.words[a..a + self.nb * BLOCK]
    }

    fn len_bytes(&self) -> usize {
        self.words.len() * 4
    }
}

pub struct PackStats {
    pub clusters: usize,
    pub bytes: u64,
}

/// Rewrites a RaBitQ index's blocks.bin, factors.f32 and ids.u32 as postings.bin plus
/// postings.off. The index's own files are left in place, so either mode can open it.
pub fn pack(dir: &Path) -> Result<PackStats> {
    let block_off: Vec<u64> = read_vec(&dir.join("block_off.u64"))?;
    let blocks: Mapped<u8> = Mapped::open(&dir.join("blocks.bin"))?;
    let factors: Mapped<f32> = Mapped::open(&dir.join("factors.f32")).context("postings need a RaBitQ index")?;
    let ids: Mapped<u32> = Mapped::open(&dir.join("ids.u32"))?;
    let (blocks, factors, ids) = (blocks.as_slice(), factors.as_slice(), ids.as_slice());
    let k = block_off.len() - 1;
    let path = dir.join("postings.bin");
    let mut w = std::io::BufWriter::with_capacity(8 << 20, File::create(&path).with_context(|| format!("create {}", path.display()))?);
    let (mut off, mut pos) = (Vec::with_capacity(k + 1), 0u64);
    for c in 0..k {
        let (b0, b1) = (block_off[c] as usize, block_off[c + 1] as usize);
        off.push(pos);
        w.write_all(&blocks[b0 * BLOCK_BYTES..b1 * BLOCK_BYTES])?;
        w.write_all(bytemuck::cast_slice(&factors[b0 * BLOCK * 2..b1 * BLOCK * 2]))?;
        w.write_all(bytemuck::cast_slice(&ids[b0 * BLOCK..b1 * BLOCK]))?;
        pos += ((b1 - b0) * BLOCK_RECORD) as u64;
        let pad = pos.next_multiple_of(ALIGN) - pos;
        w.write_all(&vec![0u8; pad as usize])?;
        pos += pad;
    }
    off.push(pos);
    w.flush()?;
    write_vec(&dir.join("postings.off"), &off)?;
    Ok(PackStats { clusters: k, bytes: pos })
}

/// Second-chance (CLOCK) cache of whole records under a byte budget, sharded by cluster so
/// concurrent queries rarely contend on one lock.
struct Cache {
    shards: Vec<Mutex<Shard>>,
}

struct Shard {
    map: HashMap<u32, (Arc<Record>, bool)>,
    ring: VecDeque<u32>,
    bytes: usize,
    cap: usize,
}

const SHARDS: usize = 16;

impl Cache {
    fn new(cap: usize) -> Self {
        let shard = || Mutex::new(Shard { map: HashMap::new(), ring: VecDeque::new(), bytes: 0, cap: cap / SHARDS });
        Self { shards: (0..SHARDS).map(|_| shard()).collect() }
    }

    fn get(&self, c: u32) -> Option<Arc<Record>> {
        let mut s = self.shards[c as usize % SHARDS].lock().unwrap();
        s.map.get_mut(&c).map(|(r, referenced)| {
            *referenced = true;
            r.clone()
        })
    }

    fn insert(&self, c: u32, r: Arc<Record>) {
        let mut s = self.shards[c as usize % SHARDS].lock().unwrap();
        let len = r.len_bytes();
        if len > s.cap || s.map.contains_key(&c) {
            return;
        }
        while s.bytes + len > s.cap {
            let Some(v) = s.ring.pop_front() else { break };
            let (rec, referenced) = s.map.get_mut(&v).expect("ring and map agree");
            if *referenced {
                *referenced = false;
                s.ring.push_back(v);
            } else {
                let freed = rec.len_bytes();
                s.map.remove(&v);
                s.bytes -= freed;
            }
        }
        s.bytes += len;
        s.ring.push_back(c);
        s.map.insert(c, (r, false));
    }

    fn clear(&self) {
        for s in &self.shards {
            let mut s = s.lock().unwrap();
            s.map.clear();
            s.ring.clear();
            s.bytes = 0;
        }
    }
}

#[derive(Default)]
pub struct IoStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub bytes_read: AtomicU64,
}

pub struct Postings {
    file: Arc<File>,
    off: Vec<u64>,
    nblocks: Vec<u32>,
    cache: Option<Cache>,
    pub stats: IoStats,
}

impl Postings {
    /// `cache_bytes` = 0 disables the cache: every probe is a read.
    pub fn open(dir: &Path, cache_bytes: usize) -> Result<Self> {
        let off: Vec<u64> = read_vec(&dir.join("postings.off")).context("postings.off: run `index --pack` first")?;
        let block_off: Vec<u64> = read_vec(&dir.join("block_off.u64"))?;
        anyhow::ensure!(off.len() == block_off.len(), "postings.off does not match block_off.u64");
        let nblocks = block_off.windows(2).map(|w| (w[1] - w[0]) as u32).collect();
        let path = dir.join("postings.bin");
        Ok(Self {
            file: Arc::new(File::open(&path).with_context(|| format!("open {}", path.display()))?),
            off,
            nblocks,
            cache: (cache_bytes > 0).then(|| Cache::new(cache_bytes)),
            stats: IoStats::default(),
        })
    }

    /// Total size of the records, the bytes that would otherwise be resident.
    pub fn file_bytes(&self) -> u64 {
        *self.off.last().unwrap_or(&0)
    }

    pub fn bypass_cache(&self) {
        crate::store::no_cache(&self.file);
    }

    pub fn clear_cache(&self) {
        if let Some(c) = &self.cache {
            c.clear();
        }
    }

    /// The records of `clusters`, in order. Misses are read as jobs on the I/O pool, so a query
    /// keeps one request per missed cluster in flight.
    pub fn fetch(&self, clusters: &[u32]) -> Vec<Arc<Record>> {
        let mut out: Vec<Option<Arc<Record>>> = (0..clusters.len()).map(|_| None).collect();
        self.fetch_each(clusters, |i, r| out[i] = Some(r.clone()));
        out.into_iter().map(|r| r.expect("hit or read")).collect()
    }

    /// Calls `f(i, record of clusters[i])` once per cluster: cache hits first, then reads in
    /// completion order, so the caller scans while the remaining reads are in flight. Returns
    /// the nanoseconds spent blocked waiting for reads.
    pub fn fetch_each(&self, clusters: &[u32], mut f: impl FnMut(usize, &Arc<Record>)) -> u64 {
        let hits: Vec<Option<Arc<Record>>> = clusters.iter().map(|&c| self.cache.as_ref().and_then(|cache| cache.get(c))).collect();
        let miss: Vec<usize> = (0..clusters.len()).filter(|&i| hits[i].is_none()).collect();
        self.stats.hits.fetch_add((clusters.len() - miss.len()) as u64, Relaxed);
        self.stats.misses.fetch_add(miss.len() as u64, Relaxed);
        // Submit the reads before touching the hits, so the device starts working at once.
        let jobs = miss
            .iter()
            .map(|&i| {
                let c = clusters[i] as usize;
                let (file, nb, off) = (self.file.clone(), self.nblocks[c] as usize, self.off[c]);
                Box::new(move || {
                    let mut words = vec![0u32; nb * BLOCK_RECORD / 4];
                    file.read_exact_at(bytemuck::cast_slice_mut(&mut words), off).expect("postings.bin read");
                    Record { nb, words }
                }) as Box<dyn FnOnce() -> Record + Send>
            })
            .collect();
        let done = crate::store::io_pool().stream(jobs);
        for (i, r) in hits.iter().enumerate() {
            if let Some(r) = r {
                f(i, r);
            }
        }
        let mut waited = 0u64;
        for _ in 0..miss.len() {
            let t = std::time::Instant::now();
            let (j, r) = done.recv().expect("io job panicked");
            waited += t.elapsed().as_nanos() as u64;
            self.stats.bytes_read.fetch_add(r.len_bytes() as u64, Relaxed);
            let (i, r) = (miss[j], Arc::new(r));
            if let Some(cache) = &self.cache {
                cache.insert(clusters[i], r.clone());
            }
            f(i, &r);
        }
        waited
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(nb: usize) -> Arc<Record> {
        Arc::new(Record { nb, words: vec![0; nb * BLOCK_RECORD / 4] })
    }

    #[test]
    fn record_sections_tile_the_record() {
        let r = rec(3);
        assert_eq!(r.blocks().len(), 3 * BLOCK_BYTES);
        assert_eq!(r.factors().len(), 3 * BLOCK * 2);
        assert_eq!(r.ids().len(), 3 * BLOCK);
        assert_eq!(r.blocks().len() + r.factors().len() * 4 + r.ids().len() * 4, r.len_bytes());
    }

    #[test]
    fn clock_cache_keeps_referenced_records() {
        // One shard's budget fits two one-block records.
        let cache = Cache::new(SHARDS * 2 * BLOCK_RECORD);
        let (a, b, c) = (0, SHARDS as u32, 2 * SHARDS as u32);
        cache.insert(a, rec(1));
        cache.insert(b, rec(1));
        assert!(cache.get(a).is_some());
        // a was referenced, so inserting c evicts b.
        cache.insert(c, rec(1));
        assert!(cache.get(a).is_some() && cache.get(c).is_some());
        assert!(cache.get(b).is_none());
        // Too large for a shard: not cached.
        cache.insert(3 * SHARDS as u32, rec(3));
        assert!(cache.get(3 * SHARDS as u32).is_none());
    }
}
