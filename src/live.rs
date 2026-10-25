//! Inserts, deletes and upserts over an immutable IVF index without a rebuild (DESIGN.md D17).
//!
//! The built index stays read-only. Deletes set a bit in a tombstone bitset that the scan checks
//! at admission. Inserts are encoded exactly as the build encodes a row (nearest centroid plus a
//! SOAR spill, RaBitQ residual codes against each centroid, a 2-bit tier code, an int8 rerank
//! row) and appended to per-cluster delta lists that the query path scans right after the
//! cluster's base list. Every change is written to a write-ahead log and fsynced before it is
//! applied, so a restart replays the log to the same state. Compaction folds the delta lists and
//! tombstones into a fresh index directory, and the log can then be truncated.
//!
//! Row ids: corpus rows keep their ids, and delta rows are numbered from `corpus.n` on. Callers
//! see external ids: a corpus row's external id is its row id, a delta row's is whatever it was
//! inserted under. Inserting an external id that is already live is an upsert: the old row is
//! tombstoned.
//!
//! One writer at a time: `insert` and `delete` take `&mut self`, `search` takes `&self`, so a
//! server wraps a `Live` in an `RwLock`. Compaction reads `&self` and does not block searches.

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use crate::ivf::{Corpus, Index, PAD, Params, Scratch, Timing, View, assign, assign_top, soar_secondary};
use crate::lut::{BLOCK, BLOCK_BYTES, pack_block, unpack_slot};
use crate::quant::normalize;
use crate::rabitq;
use crate::store::{Appender, read_vec, write_vec};
use crate::tier2;
use crate::{CODE_BYTES, DIM, PDIM};

/// SOAR secondary candidates, as in `ivf::build`.
const SOAR_CAND: usize = 16;

/// One cluster's delta list, in the base index's block layout so the same scan reads it.
#[derive(Default)]
struct DeltaList {
    blocks: Vec<u8>,
    factors: Vec<f32>,
    ids: Vec<u32>,
    /// Raw codes of the last (partial) block, repacked on every insert into it.
    tail: Vec<[u8; CODE_BYTES]>,
    len: usize,
}

impl DeltaList {
    fn push(&mut self, code: [u8; CODE_BYTES], f: [f32; 2], id: u32) {
        let k = self.len % BLOCK;
        if k == 0 {
            self.blocks.resize(self.blocks.len() + BLOCK_BYTES, 0);
            self.factors.resize(self.factors.len() + BLOCK * 2, 0.0);
            self.ids.resize(self.ids.len() + BLOCK, PAD);
            self.tail.clear();
        }
        self.tail.push(code);
        let codes: Vec<&[u8]> = self.tail.iter().map(|c| &c[..]).collect();
        let b = self.blocks.len() - BLOCK_BYTES;
        pack_block(&codes, &mut self.blocks[b..]);
        let slot = self.ids.len() - BLOCK + k;
        self.ids[slot] = id;
        self.factors[slot * 2..slot * 2 + 2].copy_from_slice(&f);
        self.len += 1;
    }
}

/// Rows added since the build: their rerank rows, tier codes and external ids, plus the delta
/// lists that point at the ones not yet compacted into the base lists.
#[derive(Default)]
pub struct Delta {
    /// Row id of the first delta row (the corpus size).
    base: usize,
    lists: Vec<DeltaList>,
    rows: Vec<i8>,
    norm256: Vec<f32>,
    t2: Vec<u8>,
    t2_norm: Vec<f32>,
    ext: Vec<u32>,
}

impl Delta {
    #[inline]
    fn local(&self, id: u32) -> Option<usize> {
        (id as usize).checked_sub(self.base).filter(|&i| i < self.ext.len())
    }

    #[inline]
    pub fn row(&self, id: u32) -> Option<&[i8]> {
        self.local(id).map(|i| &self.rows[i * DIM..][..DIM])
    }

    #[inline]
    pub fn norm256(&self, id: u32) -> Option<f32> {
        self.local(id).map(|i| self.norm256[i])
    }

    #[inline]
    pub fn tier2(&self, id: u32) -> Option<(&[u8], f32)> {
        self.local(id).filter(|_| !self.t2_norm.is_empty()).map(|i| (&self.t2[i * tier2::BYTES..][..tier2::BYTES], self.t2_norm[i]))
    }

    /// Cluster `c`'s delta list as (blocks, factors, ids), if it has any rows.
    #[inline]
    pub fn list(&self, c: u32) -> Option<(&[u8], &[f32], &[u32])> {
        self.lists.get(c as usize).filter(|l| l.len > 0).map(|l| (&l.blocks[..], &l.factors[..], &l.ids[..]))
    }

    pub fn rows(&self) -> usize {
        self.ext.len()
    }

    /// Slots in delta lists (rows not yet compacted, counted once per cluster they sit in).
    pub fn slots(&self) -> usize {
        self.lists.iter().map(|l| l.len).sum()
    }
}

fn bit(set: &[u64], id: u32) -> bool {
    set.get(id as usize >> 6).is_some_and(|w| w >> (id & 63) & 1 == 1)
}

fn set_bit(set: &mut Vec<u64>, id: u32) {
    let w = id as usize >> 6;
    if set.len() <= w {
        set.resize(w + 1, 0);
    }
    set[w] |= 1 << (id & 63);
}

const OP_INSERT: u8 = 1;
const OP_DELETE: u8 = 2;
/// op, external id, prefix norm, int8 row.
const INSERT_BYTES: usize = 1 + 4 + 4 + DIM;
const DELETE_BYTES: usize = 1 + 4;

pub struct ReplayStats {
    pub inserts: usize,
    pub deletes: usize,
    /// Bytes of a torn final record, cut off the log.
    pub torn: u64,
    pub seconds: f64,
}

pub struct CompactStats {
    pub slots: u64,
    pub dropped: u64,
    pub seconds: f64,
}

pub struct Live<'a> {
    pub index: &'a Index,
    pub corpus: &'a Corpus,
    cents_rot: Vec<f32>,
    soar: Option<f32>,
    delta: Delta,
    /// Tombstones over row ids, corpus and delta alike.
    deleted: Vec<u64>,
    /// External id → row id, for every live delta row. Corpus rows map to themselves.
    by_ext: HashMap<u32, u32>,
    wal: Option<BufWriter<File>>,
    index_dir: std::path::PathBuf,
}

impl<'a> Live<'a> {
    /// Opens `index` (built in `dir`) for updates. A compacted index carries its row segment
    /// (`live.*`) in `dir`; with `wal`, the log is replayed and then appended to.
    pub fn open(index: &'a Index, corpus: &'a Corpus, dir: &Path, wal: Option<&Path>) -> Result<(Self, Option<ReplayStats>)> {
        let (rot, _) = index.rabitq.as_ref().context("live updates need a RaBitQ index")?;
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).context("index meta.json")?)?;
        let soar = meta["soar_lambda"].as_f64().map(|l| l as f32);
        let cents_rot = index
            .centroids
            .chunks_exact(PDIM)
            .flat_map(|c| {
                let mut c = c.to_vec();
                rot.apply(&mut c);
                c
            })
            .collect();
        let mut delta = Delta { base: corpus.n, lists: (0..index.k).map(|_| DeltaList::default()).collect(), ..Default::default() };
        let (mut deleted, mut by_ext) = (Vec::new(), HashMap::new());
        if dir.join("live.ext.u32").exists() {
            delta.rows = read_vec(&dir.join("live.rows.i8"))?;
            delta.norm256 = read_vec(&dir.join("live.norm256.f32"))?;
            delta.t2 = read_vec(&dir.join("live.t2.bin"))?;
            delta.t2_norm = read_vec(&dir.join("live.t2.norm"))?;
            delta.ext = read_vec(&dir.join("live.ext.u32"))?;
            deleted = read_vec(&dir.join("live.deleted.u64"))?;
            for (i, &e) in delta.ext.iter().enumerate() {
                let id = (corpus.n + i) as u32;
                if !bit(&deleted, id) {
                    by_ext.insert(e, id);
                }
            }
        }
        let mut live = Self { index, corpus, cents_rot, soar, delta, deleted, by_ext, wal: None, index_dir: dir.to_path_buf() };
        let stats = match wal {
            None => None,
            Some(path) => {
                let stats = live.replay(path)?;
                live.wal = Some(BufWriter::with_capacity(8 << 20, OpenOptions::new().create(true).append(true).open(path)?));
                Some(stats)
            }
        };
        Ok((live, stats))
    }

    fn replay(&mut self, path: &Path) -> Result<ReplayStats> {
        let t0 = std::time::Instant::now();
        let mut buf = Vec::new();
        if let Ok(mut f) = File::open(path) {
            f.read_to_end(&mut buf)?;
        }
        let (mut pos, mut stats) = (0usize, ReplayStats { inserts: 0, deletes: 0, torn: 0, seconds: 0.0 });
        // Consecutive inserts are applied as one batch so encoding runs in parallel.
        let (mut ext, mut rows, mut norms) = (Vec::new(), Vec::new(), Vec::new());
        let le32 = |b: &[u8]| u32::from_le_bytes(b[..4].try_into().unwrap());
        while pos < buf.len() {
            let need = match buf[pos] {
                OP_INSERT => INSERT_BYTES,
                OP_DELETE => DELETE_BYTES,
                op => anyhow::bail!("{}: bad op {op} at byte {pos}", path.display()),
            };
            if pos + need > buf.len() {
                break;
            }
            let r = &buf[pos..pos + need];
            if r[0] == OP_INSERT {
                ext.push(le32(&r[1..]));
                norms.push(f32::from_le_bytes(r[5..9].try_into().unwrap()));
                rows.extend_from_slice(bytemuck::cast_slice::<u8, i8>(&r[9..]));
                stats.inserts += 1;
            } else {
                self.apply_insert(&ext, &rows, &norms);
                (ext.clear(), rows.clear(), norms.clear());
                self.apply_delete(&[le32(&r[1..])]);
                stats.deletes += 1;
            }
            pos += need;
        }
        self.apply_insert(&ext, &rows, &norms);
        stats.torn = (buf.len() - pos) as u64;
        if stats.torn > 0 {
            OpenOptions::new().write(true).open(path)?.set_len(pos as u64)?;
        }
        stats.seconds = t0.elapsed().as_secs_f64();
        Ok(stats)
    }

    fn log(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(w) = &mut self.wal {
            w.write_all(bytes)?;
            w.flush()?;
            w.get_ref().sync_data()?;
        }
        Ok(())
    }

    /// The row id an external id currently resolves to.
    pub fn current(&self, ext: u32) -> Option<u32> {
        match self.by_ext.get(&ext) {
            Some(&id) => Some(id),
            None => ((ext as usize) < self.corpus.n && !bit(&self.deleted, ext)).then_some(ext),
        }
    }

    fn ext_of(&self, id: u32) -> u32 {
        match self.delta.local(id) {
            Some(i) => self.delta.ext[i],
            None => id,
        }
    }

    /// Inserts (or replaces) `ext[i]` with the unit 768-d vector `vecs[i]`. Durable on return.
    pub fn insert(&mut self, ext: &[u32], vecs: &[f32]) -> Result<()> {
        ensure!(vecs.len() == ext.len() * DIM, "insert: {} ids, {} floats", ext.len(), vecs.len());
        let mut rows = vec![0i8; vecs.len()];
        let mut norms = vec![0f32; ext.len()];
        rows.par_chunks_mut(DIM).zip(vecs.par_chunks(DIM)).zip(norms.par_iter_mut()).for_each(|((r, v), n)| {
            self.corpus.quant.encode(v, r);
            *n = v[..PDIM].iter().map(|x| x * x).sum::<f32>().sqrt();
        });
        let mut rec = Vec::with_capacity(ext.len() * INSERT_BYTES);
        for ((&e, r), &n) in ext.iter().zip(rows.chunks_exact(DIM)).zip(&norms) {
            rec.push(OP_INSERT);
            rec.extend_from_slice(&e.to_le_bytes());
            rec.extend_from_slice(&n.to_le_bytes());
            rec.extend_from_slice(bytemuck::cast_slice(r));
        }
        self.log(&rec)?;
        self.apply_insert(ext, &rows, &norms);
        Ok(())
    }

    /// Deletes external ids; unknown ones are ignored. Durable on return.
    pub fn delete(&mut self, ext: &[u32]) -> Result<()> {
        let mut rec = Vec::with_capacity(ext.len() * DELETE_BYTES);
        for &e in ext {
            rec.push(OP_DELETE);
            rec.extend_from_slice(&e.to_le_bytes());
        }
        self.log(&rec)?;
        self.apply_delete(ext);
        Ok(())
    }

    fn apply_delete(&mut self, ext: &[u32]) {
        for &e in ext {
            if let Some(id) = self.current(e) {
                set_bit(&mut self.deleted, id);
                self.by_ext.remove(&e);
            }
        }
    }

    /// Encodes rows as `ivf::build` does: the codes are built from the prefix of the int8 row,
    /// so a replayed log reproduces them bit for bit.
    fn apply_insert(&mut self, ext: &[u32], rows: &[i8], norms: &[f32]) {
        if ext.is_empty() {
            return;
        }
        let (index, scale) = (self.index, &self.corpus.quant.scale);
        let (rot, _) = index.rabitq.as_ref().unwrap();
        let mut xs = vec![0f32; ext.len() * PDIM];
        xs.par_chunks_mut(PDIM).zip(rows.par_chunks(DIM)).for_each(|(x, r)| {
            x.iter_mut().zip(&r[..PDIM]).zip(scale).for_each(|((x, &q), &s)| *x = q as f32 / s);
            normalize(x);
        });
        let pairs: Vec<[u32; 2]> = match self.soar {
            None => assign(&xs, &index.centroids, index.k).into_iter().map(|c| [c, PAD]).collect(),
            Some(lambda) => {
                let top = assign_top(&xs, &index.centroids, index.k, SOAR_CAND);
                xs.par_chunks(PDIM).zip(top.par_chunks(SOAR_CAND)).map(|(x, cand)| [cand[0], soar_secondary(x, &index.centroids, cand, lambda)]).collect()
            }
        };
        let cents_rot = &self.cents_rot;
        let enc: Vec<[([u8; CODE_BYTES], [f32; 2]); 2]> = xs
            .par_chunks(PDIM)
            .zip(&pairs)
            .map(|(x, pair)| {
                let mut xr = x.to_vec();
                rot.apply(&mut xr);
                pair.map(|c| {
                    let mut code = [0u8; CODE_BYTES];
                    if c == PAD {
                        return (code, [0.0; 2]);
                    }
                    let f = rabitq::encode(&xr, &cents_rot[c as usize * PDIM..][..PDIM], &mut code);
                    (code, f)
                })
            })
            .collect();
        if let Some((t2, ..)) = &index.tier2 {
            let at = self.delta.t2.len();
            self.delta.t2.resize(at + ext.len() * tier2::BYTES, 0);
            let mut n = vec![0f32; ext.len()];
            self.delta.t2[at..].par_chunks_mut(tier2::BYTES).zip(xs.par_chunks(PDIM)).zip(n.par_iter_mut()).for_each(|((c, x), n)| *n = t2.encode(x, c));
            self.delta.t2_norm.extend_from_slice(&n);
        }
        for (j, &e) in ext.iter().enumerate() {
            if let Some(old) = self.current(e) {
                set_bit(&mut self.deleted, old);
            }
            let id = (self.delta.base + self.delta.ext.len()) as u32;
            self.delta.ext.push(e);
            self.delta.rows.extend_from_slice(&rows[j * DIM..][..DIM]);
            self.delta.norm256.push(norms[j]);
            self.by_ext.insert(e, id);
            for (h, &c) in pairs[j].iter().enumerate().filter(|x| *x.1 != PAD) {
                self.delta.lists[c as usize].push(enc[j][h].0, enc[j][h].1, id);
            }
        }
    }

    pub fn view(&self) -> View<'_> {
        View { deleted: Some(&self.deleted), delta: Some(&self.delta), allow: None }
    }

    /// `Index::search` over the live state, returning external ids.
    pub fn search(&self, q768: &[f32], q256: &[f32], p: &Params, s: &mut Scratch, t: &mut Timing) -> Vec<(f32, u32)> {
        let mut r = self.index.search_view(self.corpus, q768, q256, p, &self.view(), s, t);
        r.iter_mut().for_each(|x| x.1 = self.ext_of(x.1));
        r
    }

    pub fn delta(&self) -> &Delta {
        &self.delta
    }

    pub fn deleted(&self) -> usize {
        self.deleted.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Writes a new index directory with the delta lists merged into the base lists and
    /// tombstoned slots dropped, plus SSD posting lists and the row segment. Centroids,
    /// rotation and the corpus-indexed tier files are shared by symlink.
    pub fn compact(&self, out: &Path) -> Result<CompactStats> {
        let t0 = std::time::Instant::now();
        let src = std::fs::canonicalize(index_dir_of(out, self)?)?;
        std::fs::create_dir_all(out)?;
        let (_, factors) = self.index.rabitq.as_ref().unwrap();
        let (blocks, factors, ids) = (self.index.blocks.as_slice(), factors.as_slice(), self.index.ids.as_slice());
        let mut w_blocks = Appender::create(&out.join("blocks.bin"))?;
        let mut w_factors = Appender::create(&out.join("factors.f32"))?;
        let mut w_ids = Appender::create(&out.join("ids.u32"))?;
        let (mut block_off, mut nblocks) = (Vec::with_capacity(self.index.k + 1), 0u64);
        let (mut slots, mut dropped) = (0u64, 0u64);
        let mut buf = vec![0u8; BLOCK_BYTES];
        // (code, factors, row id) of every live slot of one cluster.
        let mut live: Vec<([u8; CODE_BYTES], [f32; 2], u32)> = Vec::new();
        for c in 0..self.index.k {
            live.clear();
            let (b0, b1) = (self.index.block_off[c] as usize, self.index.block_off[c + 1] as usize);
            let base = (&blocks[b0 * BLOCK_BYTES..b1 * BLOCK_BYTES], &factors[b0 * BLOCK * 2..b1 * BLOCK * 2], &ids[b0 * BLOCK..b1 * BLOCK]);
            for (bl, fs, is) in [Some(base), self.delta.list(c as u32)].into_iter().flatten() {
                for (j, &id) in is.iter().enumerate() {
                    if id == PAD {
                        continue;
                    }
                    if bit(&self.deleted, id) {
                        dropped += 1;
                        continue;
                    }
                    let mut code = [0u8; CODE_BYTES];
                    unpack_slot(&bl[j / BLOCK * BLOCK_BYTES..][..BLOCK_BYTES], j % BLOCK, &mut code);
                    live.push((code, [fs[j * 2], fs[j * 2 + 1]], id));
                }
            }
            block_off.push(nblocks);
            for chunk in live.chunks(BLOCK) {
                let codes: Vec<&[u8]> = chunk.iter().map(|x| &x.0[..]).collect();
                pack_block(&codes, &mut buf);
                w_blocks.write(&buf)?;
                let (mut f, mut slot) = ([[0f32; 2]; BLOCK], [PAD; BLOCK]);
                for (k, x) in chunk.iter().enumerate() {
                    (f[k], slot[k]) = (x.1, x.2);
                }
                w_factors.write(f.as_flattened())?;
                w_ids.write(&slot)?;
                nblocks += 1;
            }
            slots += live.len() as u64;
        }
        block_off.push(nblocks);
        (w_blocks.finish()?, w_factors.finish()?, w_ids.finish()?);
        write_vec(&out.join("block_off.u64"), &block_off)?;
        for f in ["centroids.f32", "rotation.f32", "tier2.sigma", "tier2.bin", "tier2.norm"] {
            let (from, to) = (src.join(f), out.join(f));
            if from.exists() {
                let _ = std::fs::remove_file(&to);
                std::os::unix::fs::symlink(std::fs::canonicalize(&from)?, &to)?;
            }
        }
        let mut meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(src.join("meta.json"))?)?;
        meta["compacted_from"] = src.display().to_string().into();
        meta["live_rows"] = self.delta.rows().into();
        meta["deleted"] = self.deleted().into();
        std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
        write_vec(&out.join("live.rows.i8"), &self.delta.rows)?;
        write_vec(&out.join("live.norm256.f32"), &self.delta.norm256)?;
        write_vec(&out.join("live.t2.bin"), &self.delta.t2)?;
        write_vec(&out.join("live.t2.norm"), &self.delta.t2_norm)?;
        write_vec(&out.join("live.ext.u32"), &self.delta.ext)?;
        write_vec(&out.join("live.deleted.u64"), &self.deleted)?;
        crate::postings::pack(out)?;
        Ok(CompactStats { slots, dropped, seconds: t0.elapsed().as_secs_f64() })
    }

    /// Empties the log once a compaction has made its records redundant. Replaying records onto
    /// the compacted index would also be harmless: every record is an upsert or a delete.
    pub fn truncate_wal(&mut self) -> Result<()> {
        if let Some(w) = &mut self.wal {
            w.flush()?;
            w.get_ref().set_len(0)?;
            w.get_ref().sync_all()?;
        }
        Ok(())
    }
}

/// Where the index being compacted lives: it must differ from `out`.
fn index_dir_of(out: &Path, live: &Live) -> Result<std::path::PathBuf> {
    let dir = live.index_dir.clone();
    ensure!(std::fs::canonicalize(out).ok() != std::fs::canonicalize(&dir).ok(), "compaction must write a new directory");
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_list_matches_build_layout() {
        let mut l = DeltaList::default();
        let codes: Vec<[u8; CODE_BYTES]> = (0..40u8).map(|i| [i.wrapping_mul(37); CODE_BYTES].map(|b| b ^ i)).collect();
        for (i, c) in codes.iter().enumerate() {
            l.push(*c, [i as f32, -(i as f32)], i as u32);
        }
        assert_eq!(l.blocks.len(), 2 * BLOCK_BYTES);
        let mut want = vec![0u8; BLOCK_BYTES];
        let refs: Vec<&[u8]> = codes[32..].iter().map(|c| &c[..]).collect();
        pack_block(&refs, &mut want);
        assert_eq!(&l.blocks[BLOCK_BYTES..], &want[..]);
        assert_eq!(l.ids[39], 39);
        assert_eq!(l.ids[40], PAD);
        assert_eq!(&l.factors[39 * 2..39 * 2 + 2], &[39.0, -39.0]);
        let mut back = [0u8; CODE_BYTES];
        for (i, c) in codes.iter().enumerate() {
            unpack_slot(&l.blocks[i / BLOCK * BLOCK_BYTES..][..BLOCK_BYTES], i % BLOCK, &mut back);
            assert_eq!(&back, c);
        }
    }

    #[test]
    fn bitset() {
        let mut s = Vec::new();
        set_bit(&mut s, 130);
        assert!(bit(&s, 130) && !bit(&s, 129) && !bit(&s, 100_000));
    }
}
