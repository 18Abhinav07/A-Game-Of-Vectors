//! Billion-scale L2 search over int8 vectors (DESIGN.md D19), built for MS SPACEV-1B
//! (big-ann-benchmarks NeurIPS'21: 1B × 100-d int8 Bing web-document vectors, Euclidean).
//!
//! The M1 machinery carries over: IVF over k-means leaves, RaBitQ residual codes scanned by the
//! NEON LUT kernel, SSD records read on the I/O pool. What changes at 1B on a 24 GB laptop:
//!
//! - **L2, not inner product.** A query ranks rows by s(x) = 2⟨q,x⟩ − ‖x‖² (= ‖q‖² − ‖q−x‖²).
//!   ⟨q,x⟩ = ⟨q,c⟩ + ⟨q,r⟩ splits as in rabitq.rs, and ‖x‖² is stored exactly per slot, so the
//!   code still only estimates the residual term.
//! - **Two-level centroids.** 131k leaves trained flat would cost 1B × 131k × 100 × 2 flops to
//!   assign (hours of sgemm). Leaves are trained per coarse cell instead (K1 coarse × K2 fine),
//!   and a row is assigned among the children of its few nearest coarse cells: about 1.5k
//!   centroid dots per row. Queries route over all leaves at once (one sgemv).
//! - **Out-of-core build.** The base arrives in 1e8-row chunks that never coexist with the
//!   index. Each chunk is encoded into run files bucketed by leaf range and sorted; `pack`
//!   then streams the buckets in leaf order into the SSD records and deletes each bucket's runs
//!   as it goes, so the peak is the index plus one bucket.
//! - **Exact rerank from the base chunks themselves.** The downloaded chunks are the rerank
//!   store: candidate rows are read at id·100 from base.NN (positioned reads on the I/O pool,
//!   or the page cache when warm). A 4-bit residual row (54 B) was tried first and dropped:
//!   its score error (std 206 on SPACEV) is several times the median gap between the 10th and
//!   11th neighbour (60 at 10M, 26 at 1B), and it cost 8 points of recall@10 at 10M.
//!
//! Slot factors are [mul, ‖x‖²]: est = 2·(⟨q,c⟩ + mul·(a·lut + b)) − ‖x‖².

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::blas::gemm_abt;
use crate::ivf::Rng;
use crate::lut::{BLOCK, pack_block, scan};
use crate::postings::{ALIGN, Postings, block_record};
use crate::rabitq::{self, Rotation};
use crate::store::{Mapped, read_vec, write_vec};
use crate::topk::TopK;

/// SPACEV dimensions.
pub const D: usize = 100;
/// Padded to a power of two for the rotation and the codes.
pub const PD: usize = 128;
const G: usize = PD / 4;
pub const CODE: usize = PD / 8;
pub const BLOCK_BYTES: usize = G * 16;
/// Rows per base chunk file (base.NN holds ids [NN·1e8, (NN+1)·1e8)).
pub const CHUNK_ROWS: u64 = 100_000_000;
/// Coarse cells a row is assigned among (its leaf is the best child of these).
const TOP_COARSE: usize = 4;
const PAD: u32 = u32::MAX;

/// For every row, the `m` centroids maximising 2⟨x,c⟩ − ‖c‖² (nearest in L2), best first, with
/// that score. sgemm in row blocks so the score matrix stays small.
pub fn top_l2(rows: &[f32], cents: &[f32], cnorm: &[f32], m: usize) -> Vec<(u32, f32)> {
    const B: usize = 4096;
    let (n, k) = (rows.len() / D, cnorm.len());
    let mut out = vec![(0u32, 0f32); n * m];
    let mut scores = vec![0f32; B.min(n) * k];
    for (blk, o) in rows.chunks(B * D).zip(out.chunks_mut(B * m)) {
        let rn = blk.len() / D;
        let s = &mut scores[..rn * k];
        gemm_abt(blk, cents, rn, k, D, s);
        s.par_chunks(k).zip(o.par_chunks_mut(m)).for_each(|(row, o)| {
            let mut t = TopK::new(m);
            row.iter().zip(cnorm).enumerate().for_each(|(j, (&v, &cn))| t.push(2.0 * v - cn, j as u32));
            o.iter_mut().zip(t.into_sorted()).for_each(|(o, (s, j))| *o = (j, s));
        });
    }
    out
}

fn norms(cents: &[f32]) -> Vec<f32> {
    cents.chunks_exact(D).map(|c| c.iter().map(|x| x * x).sum()).collect()
}

/// Lloyd's k-means in L2. Empty clusters are reseeded on a random row.
pub fn kmeans_l2(sample: &[f32], k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = sample.len() / D;
    let mut rng = Rng::new(seed);
    let mut cents: Vec<f32> = (0..k).flat_map(|_| sample[rng.below(n) * D..][..D].to_vec()).collect();
    for _ in 0..iters {
        let a = top_l2(sample, &cents, &norms(&cents), 1);
        let mut sums = vec![0f64; k * D];
        let mut counts = vec![0usize; k];
        for (row, &(c, _)) in sample.chunks_exact(D).zip(&a) {
            counts[c as usize] += 1;
            sums[c as usize * D..][..D].iter_mut().zip(row).for_each(|(s, &x)| *s += x as f64);
        }
        for c in 0..k {
            let dst = &mut cents[c * D..][..D];
            match counts[c] {
                0 => dst.copy_from_slice(&sample[rng.below(n) * D..][..D]),
                m => dst.iter_mut().zip(&sums[c * D..][..D]).for_each(|(d, &s)| *d = (s / m as f64) as f32),
            }
        }
    }
    cents
}

/// Rows `[start, start + n)` of an int8 base as f32.
fn to_f32(base: &[i8], start: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n * D];
    v.par_chunks_mut(D).zip(base[start * D..(start + n) * D].par_chunks(D)).for_each(|(o, x)| o.iter_mut().zip(x).for_each(|(o, &x)| *o = x as f32));
    v
}

pub struct TrainStats {
    pub leaves: usize,
    pub seconds: f64,
}

/// Trains `k1` coarse cells on part of a random sample of `base`, then `k2` leaves inside each
/// cell on the sample rows it owns. Writes coarse.f32, centroids.f32 (the k1·k2 leaves, cell
/// major) and rotation.f32.
pub fn train(base: &[i8], out: &Path, k1: usize, k2: usize, sample_n: usize, iters: usize) -> Result<TrainStats> {
    let t0 = Instant::now();
    std::fs::create_dir_all(out)?;
    let n = base.len() / D;
    let mut rng = Rng::new(42);
    let mut sample = vec![0f32; sample_n * D];
    for row in sample.chunks_exact_mut(D) {
        let i = rng.below(n);
        row.iter_mut().zip(&base[i * D..][..D]).for_each(|(o, &x)| *o = x as f32);
    }
    let coarse_n = sample_n.min(k1 * 512);
    let coarse = kmeans_l2(&sample[..coarse_n * D], k1, iters, 7);
    eprintln!("coarse: {k1} cells on {coarse_n} rows in {:.0} s", t0.elapsed().as_secs_f64());
    let a = top_l2(&sample, &coarse, &norms(&coarse), 1);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); k1];
    for (i, &(c, _)) in a.iter().enumerate() {
        members[c as usize].push(i as u32);
    }
    let leaves: Vec<f32> = members
        .par_iter()
        .enumerate()
        .flat_map_iter(|(c, m)| {
            // A cell with fewer rows than leaves borrows its coarse centroid for every leaf; the
            // duplicates stay empty and cost nothing at query time.
            let rows: Vec<f32> = match m.len() {
                0 => coarse[c * D..][..D].to_vec(),
                _ => m.iter().flat_map(|&i| sample[i as usize * D..][..D].iter().copied()).collect(),
            };
            kmeans_l2(&rows, k2, iters, c as u64)
        })
        .collect();
    let sizes: Vec<usize> = members.iter().map(Vec::len).collect();
    eprintln!(
        "leaves: {} in {:.0} s (sample rows per cell: min {}, max {})",
        k1 * k2,
        t0.elapsed().as_secs_f64(),
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap()
    );
    write_vec(&out.join("coarse.f32"), &coarse)?;
    write_vec(&out.join("centroids.f32"), &leaves)?;
    write_vec(&out.join("rotation.f32"), Rotation::with_dim(11, PD).signs())?;
    write_vec(&out.join("k2.u64"), &[k2 as u64])?;
    Ok(TrainStats { leaves: k1 * k2, seconds: t0.elapsed().as_secs_f64() })
}

/// The trained centroids and rotation.
pub struct Model {
    coarse: Vec<f32>,
    coarse_norm: Vec<f32>,
    pub leaves: Vec<f32>,
    pub leaf_norm: Vec<f32>,
    leaves_rot: Vec<f32>,
    k2: usize,
    pub rot: Rotation,
}

fn pad_rotate(rot: &Rotation, x: &[f32]) -> [f32; PD] {
    let mut v = [0f32; PD];
    v[..D].copy_from_slice(x);
    rot.apply(&mut v);
    v
}

impl Model {
    pub fn open(dir: &Path) -> Result<Self> {
        let coarse: Vec<f32> = read_vec(&dir.join("coarse.f32"))?;
        let leaves: Vec<f32> = read_vec(&dir.join("centroids.f32"))?;
        let rot = Rotation::from_signs(read_vec(&dir.join("rotation.f32"))?);
        anyhow::ensure!(rot.dim() == PD, "rotation.f32 is not {PD}-dim");
        let k2 = read_vec::<u64>(&dir.join("k2.u64"))?[0] as usize;
        let leaves_rot = leaves.par_chunks(D).flat_map_iter(|c| pad_rotate(&rot, c)).collect();
        Ok(Self { coarse_norm: norms(&coarse), coarse, leaf_norm: norms(&leaves), leaves, leaves_rot, k2, rot })
    }

    pub fn k(&self) -> usize {
        self.leaf_norm.len()
    }

    /// Each row's leaf: the best child among its TOP_COARSE nearest coarse cells.
    pub fn assign(&self, rows: &[f32]) -> Vec<u32> {
        let n = rows.len() / D;
        let top = top_l2(rows, &self.coarse, &self.coarse_norm, TOP_COARSE);
        let mut by_cell: Vec<Vec<u32>> = vec![Vec::new(); self.coarse_norm.len()];
        for (i, t) in top.chunks_exact(TOP_COARSE).enumerate() {
            t.iter().for_each(|&(c, _)| by_cell[c as usize].push(i as u32));
        }
        let k2 = self.k2;
        let hits: Vec<Vec<(u32, u32, f32)>> = by_cell
            .par_iter()
            .enumerate()
            .filter(|(_, m)| !m.is_empty())
            .map(|(c, m)| {
                let x: Vec<f32> = m.iter().flat_map(|&i| rows[i as usize * D..][..D].iter().copied()).collect();
                let best = top_l2(&x, &self.leaves[c * k2 * D..][..k2 * D], &self.leaf_norm[c * k2..][..k2], 1);
                m.iter().zip(best).map(|(&i, (j, s))| (i, (c * k2) as u32 + j, s)).collect()
            })
            .collect();
        let mut best = vec![(0u32, f32::MIN); n];
        for (i, leaf, s) in hits.into_iter().flatten() {
            if s > best[i as usize].1 {
                best[i as usize] = (leaf, s);
            }
        }
        best.into_iter().map(|b| b.0).collect()
    }
}

/// One stored row before packing: its leaf, id, RaBitQ code and [mul, ‖x‖²].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Slot {
    pub leaf: u32,
    pub id: u32,
    pub code: [u8; CODE],
    pub mul: f32,
    pub nx: f32,
}

fn run_path(dir: &Path, bucket: usize, first: u32) -> PathBuf {
    dir.join("runs").join(format!("b{bucket:03}.{first:010}"))
}

pub struct AddStats {
    pub rows: usize,
    pub seconds: f64,
}

/// Encodes `base` (int8 rows with ids from `first`) into leaf-sorted run files, one per bucket
/// of `buckets` leaf ranges.
pub fn add(dir: &Path, model: &Model, base: &[i8], first: u32, buckets: usize) -> Result<AddStats> {
    const BATCH: usize = 1 << 20;
    let t0 = Instant::now();
    let n = base.len() / D;
    std::fs::create_dir_all(dir.join("runs"))?;
    let mut slots: Vec<Slot> = Vec::with_capacity(n);
    for start in (0..n).step_by(BATCH) {
        let m = BATCH.min(n - start);
        let x = to_f32(base, start, m);
        let leaves = model.assign(&x);
        let enc: Vec<Slot> = x
            .par_chunks(D)
            .zip(&leaves)
            .enumerate()
            .map(|(j, (x, &leaf))| {
                let xr = pad_rotate(&model.rot, x);
                let mut code = [0u8; CODE];
                let [mul, _] = rabitq::encode(&xr, &model.leaves_rot[leaf as usize * PD..][..PD], &mut code);
                let nx = x.iter().map(|v| v * v).sum();
                Slot { leaf, id: first + (start + j) as u32, code, mul, nx }
            })
            .collect();
        slots.extend_from_slice(&enc);
        eprint!("\rencoded {}/{n} ({:.0} s)", start + m, t0.elapsed().as_secs_f64());
    }
    eprintln!();
    slots.par_sort_unstable_by_key(|s| (s.leaf, s.id));
    let k = model.k();
    let mut i = 0;
    for b in 0..buckets {
        let end = slots[i..].partition_point(|s| (s.leaf as usize * buckets / k) <= b) + i;
        write_vec(&run_path(dir, b, first), &slots[i..end])?;
        i = end;
    }
    Ok(AddStats { rows: n, seconds: t0.elapsed().as_secs_f64() })
}

pub struct PackStats {
    pub slots: u64,
    pub bytes: u64,
    pub max_leaf: usize,
    pub empty: usize,
}

/// Merges the run files bucket by bucket into postings.bin (the postings.rs record layout with
/// 512-byte code blocks), writing block_off.u64 and postings.off. Each bucket's runs are deleted
/// once its records are written.
pub fn pack(dir: &Path, k: usize, buckets: usize) -> Result<PackStats> {
    let runs = dir.join("runs");
    let mut names: Vec<String> = std::fs::read_dir(&runs)?.map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned())).collect::<Result<_, _>>()?;
    names.sort();
    let path = dir.join("postings.bin");
    let mut w = std::io::BufWriter::with_capacity(8 << 20, File::create(&path).with_context(|| format!("create {}", path.display()))?);
    let (mut off, mut block_off) = (Vec::with_capacity(k + 1), Vec::with_capacity(k + 1));
    let (mut pos, mut nblocks) = (0u64, 0u64);
    let mut st = PackStats { slots: 0, bytes: 0, max_leaf: 0, empty: 0 };
    let mut leaf = 0usize;
    for b in 0..buckets {
        let prefix = format!("b{b:03}.");
        let files: Vec<&String> = names.iter().filter(|n| n.starts_with(&prefix)).collect();
        let mut slots: Vec<Slot> = Vec::new();
        for f in &files {
            slots.extend(read_vec::<Slot>(&runs.join(f))?);
        }
        slots.par_sort_unstable_by_key(|s| (s.leaf, s.id));
        let mut i = 0;
        while leaf < k && leaf * buckets / k <= b {
            let end = slots[i..].partition_point(|s| s.leaf as usize == leaf) + i;
            let m = &slots[i..end];
            st.max_leaf = st.max_leaf.max(m.len());
            st.empty += m.is_empty() as usize;
            off.push(pos);
            block_off.push(nblocks);
            let nb = m.len().div_ceil(BLOCK);
            let mut blocks = vec![0u8; nb * BLOCK_BYTES];
            let mut factors = vec![0f32; nb * BLOCK * 2];
            let mut ids = vec![PAD; nb * BLOCK];
            for (bi, chunk) in m.chunks(BLOCK).enumerate() {
                let codes: Vec<&[u8]> = chunk.iter().map(|s| &s.code[..]).collect();
                pack_block(&codes, &mut blocks[bi * BLOCK_BYTES..][..BLOCK_BYTES]);
            }
            for (j, s) in m.iter().enumerate() {
                factors[2 * j] = s.mul;
                factors[2 * j + 1] = s.nx;
                ids[j] = s.id;
            }
            w.write_all(&blocks)?;
            w.write_all(bytemuck::cast_slice(&factors))?;
            w.write_all(bytemuck::cast_slice(&ids))?;
            pos += (nb * block_record(BLOCK_BYTES)) as u64;
            let pad = pos.next_multiple_of(ALIGN) - pos;
            w.write_all(&vec![0u8; pad as usize])?;
            pos += pad;
            nblocks += nb as u64;
            st.slots += m.len() as u64;
            i = end;
            leaf += 1;
        }
        anyhow::ensure!(i == slots.len(), "bucket {b}: {} slots outside its leaf range", slots.len() - i);
        for f in files {
            std::fs::remove_file(runs.join(f))?;
        }
        eprint!("\rpacked bucket {}/{buckets}", b + 1);
    }
    eprintln!();
    off.push(pos);
    block_off.push(nblocks);
    w.flush()?;
    write_vec(&dir.join("postings.off"), &off)?;
    write_vec(&dir.join("block_off.u64"), &block_off)?;
    st.bytes = pos;
    Ok(st)
}

/// The base chunks, read for the exact rerank with positioned reads on the I/O pool. Not
/// mapped: 1B rows (100 GB) do not fit the page cache, and a mapped read that misses is a
/// serial page fault, where the pool keeps 64 reads in flight.
pub struct Base {
    files: Vec<std::sync::Arc<File>>,
    paths: Vec<PathBuf>,
}

impl Base {
    /// base.00, base.01, ... from `dir`, as many as are complete (scripts/spacev.sh writes
    /// base.NN.done after checking a chunk's size).
    pub fn open(dir: &Path) -> Result<Self> {
        let paths: Vec<PathBuf> = (0..)
            .map(|c| dir.join(format!("base.{c:02}")))
            .take_while(|p| p.with_extension(format!("{}.done", p.extension().unwrap().to_str().unwrap())).exists())
            .collect();
        anyhow::ensure!(!paths.is_empty(), "no base.NN chunks in {}", dir.display());
        let files = paths.iter().map(|p| File::open(p).map(std::sync::Arc::new)).collect::<std::io::Result<_>>()?;
        Ok(Self { files, paths })
    }

    /// Rows of `ids`, in order.
    fn rows(&self, ids: &[u32]) -> Vec<i8> {
        // One job per pool thread: enough reads in flight to keep the SSD busy when cold, few
        // enough hand-offs that cached reads are not dominated by them.
        let per_job = ids.len().div_ceil(crate::store::io_pool().threads()).max(1);
        let jobs = ids
            .chunks(per_job)
            .map(|chunk| {
                let chunk = chunk.to_vec();
                let files = self.files.clone();
                Box::new(move || {
                    let mut rows = vec![0i8; chunk.len() * D];
                    for (row, &id) in rows.chunks_exact_mut(D).zip(&chunk) {
                        let (c, r) = (id as u64 / CHUNK_ROWS, id as u64 % CHUNK_ROWS);
                        files[c as usize].read_exact_at(bytemuck::cast_slice_mut(row), r * D as u64).expect("base read");
                    }
                    rows
                }) as Box<dyn FnOnce() -> Vec<i8> + Send>
            })
            .collect();
        crate::store::io_pool().run(jobs).concat()
    }

    /// Drops the chunks from the page cache and bypasses it from now on.
    fn go_cold(&self) -> Result<()> {
        for (f, p) in self.files.iter().zip(&self.paths) {
            crate::store::no_cache(f);
            crate::store::evict(p)?;
        }
        Ok(())
    }
}

pub struct Index {
    pub model: Model,
    pub postings: Postings,
    pub base: Base,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub nprobe: usize,
    /// Candidates kept from the code scan for the exact rerank.
    pub cand: usize,
    pub k: usize,
}

#[derive(Default, Clone, Copy)]
pub struct Timing {
    pub route: u64,
    pub scan: u64,
    pub io_wait: u64,
    pub rerank: u64,
    pub scanned: u64,
}

impl Index {
    pub fn open(dir: &Path, base_dir: &Path, cache_bytes: usize) -> Result<Self> {
        Ok(Self {
            model: Model::open(dir)?,
            postings: Postings::open_with(dir, cache_bytes, BLOCK_BYTES)?,
            base: Base::open(base_dir)?,
        })
    }

    /// Every later read comes from the device: pages already cached are dropped and new reads
    /// bypass the cache.
    pub fn go_cold(&self, dir: &Path) -> Result<()> {
        self.postings.bypass_cache();
        self.postings.clear_cache();
        crate::store::evict(&dir.join("postings.bin"))?;
        self.base.go_cold()
    }

    /// The `p.k` nearest rows to `q` as (score, id), best first; score = ‖q‖² − ‖q − x‖².
    pub fn search(&self, q: &[i8; D], p: &Params, t: &mut Timing) -> Vec<(f32, u32)> {
        let (probe, dots) = self.route(q, p, t);
        let cand = self.candidates(q, &probe, &dots, p, t);
        self.rerank(q, &cand, p, t)
    }

    /// The `p.nprobe` nearest leaves, and ⟨q, c⟩ for every leaf.
    pub fn route(&self, q: &[i8; D], p: &Params, t: &mut Timing) -> (Vec<u32>, Vec<f32>) {
        let t0 = Instant::now();
        let m = &self.model;
        let qf: Vec<f32> = q.iter().map(|&x| x as f32).collect();
        let mut dots = vec![0f32; m.k()];
        gemm_abt(&qf, &m.leaves, 1, m.k(), D, &mut dots);
        let mut top = TopK::new(p.nprobe);
        dots.iter().zip(&m.leaf_norm).enumerate().for_each(|(c, (&d, &n))| top.push(2.0 * d - n, c as u32));
        t.route += t0.elapsed().as_nanos() as u64;
        (top.into_sorted().into_iter().map(|x| x.1).collect(), dots)
    }

    /// The `p.cand` best slots of the probed leaves by code estimate, best first, as
    /// (id, ⟨q,c⟩, ‖x‖²).
    pub fn candidates(&self, q: &[i8; D], probe: &[u32], dots: &[f32], p: &Params, t: &mut Timing) -> Vec<(u32, f32, f32)> {
        let t1 = Instant::now();
        let mut qp = [0f32; PD];
        qp.iter_mut().zip(q).for_each(|(o, &x)| *o = x as f32);
        let rq: rabitq::Query<G> = rabitq::Query::new(&self.model.rot, &qp);
        // (id, ⟨q,c⟩, ‖x‖²) of every slot admitted to the candidate top-k, indexed by its u32.
        let mut info: Vec<(u32, f32, f32)> = Vec::new();
        let mut cand = TopK::new(p.cand);
        let (mut scores, mut scanned) = (Vec::new(), 0u64);
        let waited = self.postings.fetch_each(probe, |i, r| {
            let cent = dots[probe[i] as usize];
            let (blocks, f, ids) = (r.blocks(), r.factors(), r.ids());
            scores.resize(ids.len(), 0);
            scan(&rq.lut, blocks, &mut scores);
            for (j, (&id, &sc)) in ids.iter().zip(&scores).enumerate() {
                if id == PAD {
                    continue;
                }
                let (mul, nx) = (f[2 * j], f[2 * j + 1]);
                let est = 2.0 * (cent + mul * (rq.a * sc as f32 + rq.b)) - nx;
                if cand.threshold().is_none_or(|th| est > th) {
                    cand.push(est, info.len() as u32);
                    info.push((id, cent, nx));
                }
            }
            scanned += ids.len() as u64;
        });
        t.scanned += scanned;
        t.io_wait += waited;
        t.scan += (t1.elapsed().as_nanos() as u64).saturating_sub(waited);
        cand.into_sorted().into_iter().map(|(_, i)| info[i as usize]).collect()
    }

    /// Rescores candidates exactly from their base rows.
    pub fn rerank(&self, q: &[i8; D], cand: &[(u32, f32, f32)], p: &Params, t: &mut Timing) -> Vec<(f32, u32)> {
        let t2 = Instant::now();
        let rows = self.base.rows(&cand.iter().map(|c| c.0).collect::<Vec<_>>());
        let mut out = TopK::new(p.k);
        for (&(id, _, nx), x) in cand.iter().zip(rows.chunks_exact(D)) {
            let dot: i32 = q.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum();
            out.push(2.0 * dot as f32 - nx, id);
        }
        t.rerank += t2.elapsed().as_nanos() as u64;
        out.into_sorted()
    }
}

/// Copies little-endian bytes into an aligned Vec.
fn pod_vec<T: Pod>(b: &[u8]) -> Vec<T> {
    let mut v = vec![T::zeroed(); b.len() / size_of::<T>()];
    bytemuck::cast_slice_mut::<T, u8>(&mut v).copy_from_slice(b);
    v
}

/// big-ann-benchmarks binary files: u32 n, u32 d, then n·d items.
pub fn read_bin<T: Pod>(path: &Path) -> Result<(usize, usize, Vec<T>)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let n = u32::from_le_bytes(bytes[0..4].try_into()?) as usize;
    let d = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let body = &bytes[8..8 + n * d * size_of::<T>()];
    Ok((n, d, pod_vec(body)))
}

/// Ground truth: (nq, k, ids nq·k, distances nq·k).
pub fn read_gt(path: &Path) -> Result<(usize, usize, Vec<u32>, Vec<f32>)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let nq = u32::from_le_bytes(bytes[0..4].try_into()?) as usize;
    let k = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let ids = pod_vec(&bytes[8..8 + nq * k * 4]);
    let dist = pod_vec(&bytes[8 + nq * k * 4..8 + nq * k * 8]);
    Ok((nq, k, ids, dist))
}

/// The base file as int8 rows, mapped.
pub fn map_base(path: &Path) -> Result<Mapped<i8>> {
    let m = Mapped::open(path)?;
    anyhow::ensure!(m.as_slice().len() % D == 0, "{} is not whole {D}-byte rows", path.display());
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_finds_separated_clusters() {
        let mut rng = Rng::new(3);
        let mut rows = Vec::new();
        for i in 0..400 {
            let center = if i % 2 == 0 { 30.0 } else { -30.0 };
            rows.extend((0..D).map(|_| center + (rng.below(5) as f32 - 2.0)));
        }
        let cents = kmeans_l2(&rows, 2, 5, 1);
        let a = top_l2(&rows, &cents, &norms(&cents), 1);
        assert!(a.chunks(2).all(|p| p[0].0 != p[1].0));
    }
}
