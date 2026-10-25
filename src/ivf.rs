//! The corpus files written by ingest, the IVF build, and the query path (DESIGN.md D1–D4).

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::Path;

use crate::blas::gemm_abt;
use crate::lut::{BLOCK, BLOCK_BYTES, Lut, pack_block};
use crate::postings::Postings;
use crate::quant::{Int8Quantizer, dot_f32, dot_i8i8, normalize, query_i8};
use crate::rabitq::{self, Rotation};
use crate::store::{Appender, Mapped, read_vec, write_vec};
use crate::tier2::{self, Tier2};
use crate::topk::TopK;
use crate::{CODE_BYTES, DIM, PDIM};

/// Padding slots in a cluster's last block carry this id and are never returned.
pub const PAD: u32 = u32::MAX;

/// Ingest output: codes (RAM tier), int8 rerank vectors (SSD tier), prefix norms, doc ids.
pub struct Corpus {
    pub n: usize,
    pub codes: Mapped<u8>,
    pub rerank: Mapped<i8>,
    /// Mapped, not read: only the prefix rerank and the flat baselines touch it.
    pub norm256: Mapped<f32>,
    pub quant: Int8Quantizer,
    docid_bytes: Mapped<u8>,
    docid_off: Mapped<u64>,
    /// pread access to rerank.i8 for the SSD tier, optionally bypassing the page cache.
    pub ssd: RowReader,
}

/// Reads int8 rerank rows with positioned reads issued in parallel, so a query keeps many SSD
/// requests in flight (one blocking pread per worker; macOS has no io_uring).
pub struct RowReader {
    file: std::sync::Arc<std::fs::File>,
}

impl RowReader {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self { file: std::sync::Arc::new(std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?) })
    }

    /// F_NOCACHE: reads no longer fill the page cache (see `store::no_cache`).
    pub fn bypass_cache(&self) {
        crate::store::no_cache(&self.file);
    }

    /// Reads rows `ids` into `out`, a few rows per job on the I/O pool, so many requests are in
    /// flight at once.
    pub fn read_rows(&self, ids: &[u32], out: &mut [i8]) {
        use std::os::unix::fs::FileExt;
        const PER_JOB: usize = 4;
        let jobs = ids
            .chunks(PER_JOB)
            .map(|chunk| {
                let (file, chunk) = (self.file.clone(), chunk.to_vec());
                Box::new(move || {
                    let mut rows = vec![0i8; chunk.len() * DIM];
                    for (row, &id) in rows.chunks_exact_mut(DIM).zip(&chunk) {
                        file.read_exact_at(bytemuck::cast_slice_mut(row), id as u64 * DIM as u64).expect("rerank.i8 read");
                    }
                    rows
                }) as Box<dyn FnOnce() -> Vec<i8> + Send>
            })
            .collect();
        for (o, rows) in out.chunks_mut(DIM * PER_JOB).zip(crate::store::io_pool().run(jobs)) {
            o.copy_from_slice(&rows);
        }
    }
}

impl Corpus {
    pub fn open(dir: &Path) -> Result<Self> {
        let norm256: Mapped<f32> = Mapped::open(&dir.join("norm256.f32"))?;
        let c = Self {
            n: norm256.as_slice().len(),
            codes: Mapped::open(&dir.join("codes.bin"))?,
            rerank: Mapped::open(&dir.join("rerank.i8"))?,
            quant: Int8Quantizer { scale: read_vec(&dir.join("quant.f32"))? },
            docid_bytes: Mapped::open(&dir.join("docids.bin"))?,
            docid_off: Mapped::open(&dir.join("docids.off"))?,
            ssd: RowReader::open(&dir.join("rerank.i8"))?,
            norm256,
        };
        anyhow::ensure!(c.codes.as_slice().len() == c.n * CODE_BYTES, "codes.bin size");
        anyhow::ensure!(c.rerank.as_slice().len() == c.n * DIM, "rerank.i8 size");
        Ok(c)
    }

    #[inline]
    pub fn code(&self, i: u32) -> &[u8] {
        let i = i as usize * CODE_BYTES;
        &self.codes.as_slice()[i..i + CODE_BYTES]
    }

    #[inline]
    pub fn rerank_row(&self, i: u32) -> &[i8] {
        let i = i as usize * DIM;
        &self.rerank.as_slice()[i..i + DIM]
    }

    pub fn doc_id(&self, i: u32) -> &str {
        let off = self.docid_off.as_slice();
        let (a, b) = (off[i as usize] as usize, off[i as usize + 1] as usize);
        std::str::from_utf8(&self.docid_bytes.as_slice()[a..b]).unwrap_or("")
    }

    /// The normalised 256-dim prefix reconstructed from the int8 rerank row.
    pub fn prefix_f32(&self, i: u32, out: &mut [f32]) {
        for ((o, &x), &s) in out.iter_mut().zip(&self.rerank_row(i)[..PDIM]).zip(&self.quant.scale) {
            *o = x as f32 / s;
        }
        normalize(out);
    }
}

/// splitmix64, for reproducible sampling without a dependency.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The `m` nearest centroids (max inner product) of every row, best first, via sgemm in blocks.
pub fn assign_top(rows: &[f32], cents: &[f32], k: usize, m: usize) -> Vec<u32> {
    const B: usize = 8192;
    let n = rows.len() / PDIM;
    let mut out = vec![0u32; n * m];
    let mut scores = vec![0f32; B * k];
    for (blk, o) in rows.chunks(B * PDIM).zip(out.chunks_mut(B * m)) {
        let rows_in = blk.len() / PDIM;
        let s = &mut scores[..rows_in * k];
        gemm_abt(blk, cents, rows_in, k, PDIM, s);
        s.par_chunks(k).zip(o.par_chunks_mut(m)).for_each(|(row, a)| {
            if m == 1 {
                a[0] = row.iter().enumerate().fold((0, f32::MIN), |b, (j, &v)| if v > b.1 { (j, v) } else { b }).0 as u32;
                return;
            }
            let mut t = TopK::new(m);
            row.iter().enumerate().for_each(|(j, &v)| t.push(v, j as u32));
            a.iter_mut().zip(t.into_sorted()).for_each(|(a, (_, j))| *a = j);
        });
    }
    out
}

pub fn assign(rows: &[f32], cents: &[f32], k: usize) -> Vec<u32> {
    assign_top(rows, cents, k, 1)
}

/// SOAR secondary assignment (Sun et al., NeurIPS 2023). Among the candidate centroids, pick the
/// one minimising ||x − c||² + λ·⟨r̂, x − c⟩², where r is the residual to the primary centroid.
/// The penalty favours a centroid whose residual is orthogonal to r, so the query that misses
/// the primary cluster (large ⟨q, r⟩) is likely to reach the secondary one.
pub fn soar_secondary(x: &[f32], cents: &[f32], cand: &[u32], lambda: f32) -> u32 {
    let c1 = &cents[cand[0] as usize * PDIM..][..PDIM];
    let x_c1 = dot_f32(x, c1);
    // x and the centroids are unit length: ||r||² = 2 − 2⟨x,c1⟩, ⟨r,x⟩ = 1 − ⟨x,c1⟩.
    let r2 = (2.0 - 2.0 * x_c1).max(1e-12);
    let r_x = 1.0 - x_c1;
    let mut best = (f32::MAX, cand[1]);
    for &c in &cand[1..] {
        let cv = &cents[c as usize * PDIM..][..PDIM];
        let x_c = dot_f32(x, cv);
        let r_c = x_c - dot_f32(c1, cv);
        let along = r_x - r_c;
        let loss = (2.0 - 2.0 * x_c) + lambda * along * along / r2;
        if loss < best.0 {
            best = (loss, c);
        }
    }
    best.1
}

/// Spherical k-means: centroids stay unit length, assignment by inner product.
/// Spherical k-means (Lloyd on unit vectors). With `balance = Some(r)`, every iteration but the
/// last two also splits clusters holding more than r× the mean: the centroid of the smallest
/// cluster moves onto a random member of the oversized one, and the next Lloyd step divides the
/// members between the two. Assignment stays nearest-centroid, so routing is unaffected; only
/// the centroids move. This is the split step of SPANN and SPFresh, run during training.
pub fn kmeans(sample: &[f32], k: usize, iters: usize, seed: u64, balance: Option<f32>) -> Vec<f32> {
    let n = sample.len() / PDIM;
    let mut rng = Rng::new(seed);
    let mut cents: Vec<f32> = (0..k).flat_map(|_| sample[rng.below(n) * PDIM..][..PDIM].to_vec()).collect();
    for it in 0..iters {
        let a = assign(sample, &cents, k);
        let mut sums = vec![0f32; k * PDIM];
        let mut counts = vec![0usize; k];
        for (row, &c) in sample.chunks_exact(PDIM).zip(&a) {
            counts[c as usize] += 1;
            sums[c as usize * PDIM..][..PDIM].iter_mut().zip(row).for_each(|(s, x)| *s += x);
        }
        let mut empty = 0;
        for c in 0..k {
            let dst = &mut cents[c * PDIM..][..PDIM];
            if counts[c] == 0 {
                empty += 1;
                dst.copy_from_slice(&sample[rng.below(n) * PDIM..][..PDIM]);
            } else {
                dst.copy_from_slice(&sums[c * PDIM..][..PDIM]);
                normalize(dst);
            }
        }
        let max = counts.iter().max().unwrap();
        let split = match balance {
            Some(r) if it + 2 < iters => split_large(sample, &a, &counts, &mut cents, r, &mut rng),
            _ => 0,
        };
        eprintln!("kmeans iter {it}: largest cluster {max}, empty {empty}, split {split}");
    }
    cents
}

/// Pairs clusters above `r`× the mean size (largest first) with the smallest clusters, and
/// moves each small centroid onto a random member of its large partner. Returns the number of
/// splits. A small cluster is only given up if it is below half the mean, so the
/// fix cannot create a new imbalance.
fn split_large(sample: &[f32], a: &[u32], counts: &[usize], cents: &mut [f32], r: f32, rng: &mut Rng) -> usize {
    let k = counts.len();
    let mean = a.len() as f32 / k as f32;
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_unstable_by_key(|&c| std::cmp::Reverse(counts[c]));
    let large: Vec<usize> = order.iter().copied().take_while(|&c| counts[c] as f32 > r * mean).collect();
    let small: Vec<usize> = order.iter().rev().copied().take_while(|&c| (counts[c] as f32) < 0.5 * mean).take(large.len()).collect();
    if small.is_empty() {
        return 0;
    }
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
    let is_large: std::collections::HashSet<usize> = large.iter().copied().take(small.len()).collect();
    for (i, &c) in a.iter().enumerate() {
        if is_large.contains(&(c as usize)) {
            members[c as usize].push(i as u32);
        }
    }
    for (&big, &tiny) in large.iter().zip(&small) {
        let m = &members[big];
        let pick = m[rng.below(m.len())] as usize;
        cents[tiny * PDIM..][..PDIM].copy_from_slice(&sample[pick * PDIM..][..PDIM]);
    }
    small.len()
}

pub struct BuildStats {
    pub k: usize,
    pub sizes: Vec<usize>,
}

/// One stored slot's RaBitQ residual code and its `[mul, err]` factors.
type ResSlot = ([u8; CODE_BYTES], [f32; 2]);

/// Trains centroids on a sample, assigns every vector and writes the cluster-major layout.
/// With `soar_lambda` set, every vector is also spilled into a SOAR secondary cluster. With
/// `rabitq`, each slot holds the RaBitQ code of the vector's residual to that slot's cluster
/// (so the two SOAR copies get different codes) instead of the global sign code.
pub fn build(corpus: &Corpus, out: &Path, k: usize, sample_n: usize, iters: usize, balance: Option<f32>, soar_lambda: Option<f32>, rabitq: bool) -> Result<BuildStats> {
    std::fs::create_dir_all(out)?;
    let n = corpus.n;
    // Assign in chunks so the reconstructed prefixes never all sit in RAM at once.
    const CHUNK: usize = 1 << 20;
    let mut rng = Rng::new(42);
    let sample_n = sample_n.min(n);
    let mut sample = vec![0f32; sample_n * PDIM];
    for row in sample.chunks_exact_mut(PDIM) {
        corpus.prefix_f32(rng.below(n) as u32, row);
    }
    let cents = kmeans(&sample, k, iters, 7, balance);
    let t2 = Tier2::fit(&sample);
    write_vec(&out.join("tier2.sigma"), &t2.sigma)?;
    drop(sample);
    let mut t2_codes = Appender::create(&out.join("tier2.bin"))?;
    let mut t2_norms = Appender::create(&out.join("tier2.norm"))?;
    let mut code_buf = vec![0u8; CHUNK * tier2::BYTES];
    let mut norm_buf = vec![0f32; CHUNK];
    write_vec(&out.join("centroids.f32"), &cents)?;
    let rot = rabitq.then(|| Rotation::new(11));
    let cents_rot: Vec<f32> = match &rot {
        Some(r) => cents.chunks_exact(PDIM).flat_map(|c| {
            let mut c = c.to_vec();
            r.apply(&mut c);
            c
        }).collect(),
        None => Vec::new(),
    };

    const SOAR_CAND: usize = 16;
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
    let mut res_members: Vec<Vec<ResSlot>> = vec![Vec::new(); if rabitq { k } else { 0 }];
    let mut rows = vec![0f32; CHUNK * PDIM];
    for start in (0..n).step_by(CHUNK) {
        let m = CHUNK.min(n - start);
        let rows = &mut rows[..m * PDIM];
        rows.par_chunks_mut(PDIM).enumerate().for_each(|(j, r)| corpus.prefix_f32((start + j) as u32, r));
        rows.par_chunks(PDIM)
            .zip(code_buf[..m * tier2::BYTES].par_chunks_mut(tier2::BYTES))
            .zip(norm_buf[..m].par_iter_mut())
            .for_each(|((r, c), nrm)| *nrm = t2.encode(r, c));
        t2_codes.write(&code_buf[..m * tier2::BYTES])?;
        t2_norms.write(&norm_buf[..m])?;
        // Up to two clusters per row; PAD marks "no secondary".
        let pairs: Vec<[u32; 2]> = match soar_lambda {
            None => assign(rows, &cents, k).into_iter().map(|c| [c, PAD]).collect(),
            Some(lambda) => {
                let top = assign_top(rows, &cents, k, SOAR_CAND);
                rows.par_chunks(PDIM)
                    .zip(top.par_chunks(SOAR_CAND))
                    .map(|(x, cand)| [cand[0], soar_secondary(x, &cents, cand, lambda)])
                    .collect()
            }
        };
        let enc: Vec<[ResSlot; 2]> = match &rot {
            Some(r) => rows
                .par_chunks(PDIM)
                .zip(&pairs)
                .map(|(x, pair)| {
                    let mut xr = x.to_vec();
                    r.apply(&mut xr);
                    pair.map(|c| {
                        let mut code = [0u8; CODE_BYTES];
                        if c == PAD {
                            return (code, [0.0; 2]);
                        }
                        let f = rabitq::encode(&xr, &cents_rot[c as usize * PDIM..][..PDIM], &mut code);
                        (code, f)
                    })
                })
                .collect(),
            None => Vec::new(),
        };
        for (j, pair) in pairs.iter().enumerate() {
            for (h, &c) in pair.iter().enumerate().filter(|x| *x.1 != PAD) {
                members[c as usize].push((start + j) as u32);
                if rabitq {
                    res_members[c as usize].push(enc[j][h]);
                }
            }
        }
        eprintln!("assigned {}/{n}", start + m);
    }
    let sizes: Vec<usize> = members.iter().map(Vec::len).collect();

    let mut block_off = Vec::with_capacity(k + 1);
    let mut ids = Appender::create(&out.join("ids.u32"))?;
    let mut blocks = Appender::create(&out.join("blocks.bin"))?;
    let mut factors = match rabitq {
        true => Some(Appender::create(&out.join("factors.f32"))?),
        false => None,
    };
    let mut nblocks = 0u64;
    let mut buf = vec![0u8; BLOCK_BYTES];
    for (c, m) in members.iter().enumerate() {
        block_off.push(nblocks);
        for (bi, chunk) in m.chunks(BLOCK).enumerate() {
            let codes: Vec<&[u8]> = match rabitq {
                true => {
                    let res = &res_members[c][bi * BLOCK..][..chunk.len()];
                    let mut f = [[0f32; 2]; BLOCK];
                    f.iter_mut().zip(res).for_each(|(f, r)| *f = r.1);
                    factors.as_mut().unwrap().write(f.as_flattened())?;
                    res.iter().map(|r| &r.0[..]).collect()
                }
                false => chunk.iter().map(|&i| corpus.code(i)).collect(),
            };
            pack_block(&codes, &mut buf);
            blocks.write(&buf)?;
            let mut slot = [PAD; BLOCK];
            slot[..chunk.len()].copy_from_slice(chunk);
            ids.write(&slot)?;
            nblocks += 1;
        }
    }
    block_off.push(nblocks);
    ids.finish()?;
    blocks.finish()?;
    write_vec(&out.join("block_off.u64"), &block_off)?;
    if let (Some(f), Some(r)) = (factors, &rot) {
        f.finish()?;
        write_vec(&out.join("rotation.f32"), r.signs())?;
    }
    t2_codes.finish()?;
    t2_norms.finish()?;
    Ok(BuildStats { k, sizes })
}

/// How many candidates ahead the rerank loop prefetches.
const PREFETCH_ROWS: usize = 16;

/// Hints every cache line of `row` into L1. A hint only: it never faults and never blocks.
#[inline(always)]
fn prefetch(row: &[i8]) {
    #[cfg(target_arch = "aarch64")]
    for off in (0..row.len()).step_by(64) {
        // SAFETY: PRFM is a hint and has no architectural effect; the address is inside `row`.
        unsafe { std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) row.as_ptr().add(off), options(nostack, preserves_flags, readonly)) };
    }
}

pub struct Index {
    pub k: usize,
    pub centroids: Vec<f32>,
    pub block_off: Vec<u64>,
    pub ids: Mapped<u32>,
    pub blocks: Mapped<u8>,
    /// The RAM middle tier, if this index was built with one.
    pub tier2: Option<(Tier2, Mapped<u8>, Mapped<f32>)>,
    /// RaBitQ residual codes: the rotation and `[mul, err]` per slot, if built with them.
    pub rabitq: Option<(Rotation, Mapped<f32>)>,
    /// SSD mode (D13): when set, probed clusters are read from postings.bin instead of the
    /// mapped blocks, ids and factors, which then stay untouched and use no RAM.
    pub postings: Option<Postings>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rerank {
    /// Order by the LUT score alone.
    None,
    /// Exact-ish cosine on the int8 256-dim prefix.
    Prefix,
    /// Exact-ish cosine on the int8 768-dim vector.
    Full,
    /// Prefix rerank of every candidate, then full rerank of the best `r`: the SSD tier then
    /// serves `r` reads per query instead of `cand`.
    Tiered(usize),
    /// Full rerank of the best `r` by code estimate, read from the SSD with parallel preads.
    /// Needs no RAM tier, so it is the rerank of SSD mode.
    Ssd(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub nprobe: usize,
    pub cand: usize,
    pub rerank: Rerank,
    pub k: usize,
    /// Bound-driven pruning (RaBitQ indexes only), as `(m, w)`: before rerank, drop every
    /// candidate whose upper bound `est + w·err` is below the `m`-th best lower bound
    /// `est - w·err`. `w = 1` is the paper's ε0 = 1.9 bound; smaller `w` trades certainty for
    /// fewer rerank reads.
    pub prune: Option<(usize, f32)>,
    /// Query-aware probing (SPANN): of the top `nprobe` clusters, scan only those whose centroid
    /// score is within this margin of the best centroid's. Easy queries, whose neighbours sit in
    /// one dominant cluster, scan less; `nprobe` stays the cap for hard ones.
    pub margin: Option<f32>,
    /// Per-query scan budget in stored slots (44 B each on SSD): clusters are taken in score
    /// order until the next one would exceed it, so flat-score queries cannot probe up to the
    /// cap. The best cluster is always scanned.
    pub budget: Option<usize>,
}

/// What a query sees beyond the immutable base index (D17): rows to hide (tombstones) and a
/// delta segment of rows inserted since the build. The default view is the base alone.
#[derive(Clone, Copy, Default)]
pub struct View<'a> {
    /// Bitset over row ids; a set bit hides the row.
    pub deleted: Option<&'a [u64]>,
    pub delta: Option<&'a crate::live::Delta>,
}

#[inline]
fn bit(set: &[u64], id: u32) -> bool {
    set.get(id as usize >> 6).is_some_and(|w| w >> (id & 63) & 1 == 1)
}

impl View<'_> {
    #[inline]
    fn hidden(&self, id: u32) -> bool {
        self.deleted.is_some_and(|d| bit(d, id))
    }

    /// The int8 row of `id`, from the delta's row store or the corpus.
    #[inline]
    fn row<'c>(&self, corpus: &'c Corpus, id: u32) -> &'c [i8]
    where
        Self: 'c,
    {
        match self.delta.and_then(|d| d.row(id)) {
            Some(r) => r,
            None => corpus.rerank_row(id),
        }
    }
}

/// Per-thread buffers so the query path does not allocate.
pub struct Scratch {
    cent: Vec<f32>,
    scores: Vec<u16>,
    w: Vec<f32>,
    wq: Vec<i8>,
    rows: Vec<i8>,
    /// (corpus row, error half-width) of every slot admitted to the candidate buffer.
    meta: Vec<(u32, f32)>,
}

impl Scratch {
    pub fn new() -> Self {
        Self { cent: Vec::new(), scores: Vec::new(), w: Vec::new(), wq: Vec::new(), rows: Vec::new(), meta: Vec::new() }
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a query spends its time, in nanoseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub route: u64,
    pub scan: u64,
    pub rerank: u64,
    /// Part of `rerank` spent in the RAM middle tier (tiered rerank only).
    pub tier2: u64,
    pub scanned: u64,
    /// Candidates that reached the rerank stage after dedup and pruning.
    pub reranked: u64,
    /// Clusters scanned.
    pub probed: u64,
    /// Part of `scan` spent waiting for posting lists (SSD mode only).
    pub fetch: u64,
    /// Coordinator only (D16): wall time from fanning a query out to the last shard's reply.
    pub wait: u64,
}

/// The `m`-th largest lower bound `est - w·err` among the candidates (−∞ if there are fewer
/// than `m`). Any candidate whose upper bound is below it is, with high probability, outranked
/// by at least `m` others, so it cannot reach the top `m`.
pub fn prune_threshold(cand: &[(f32, u32, f32)], m: usize, w: f32) -> f32 {
    if m == 0 || cand.len() < m {
        return f32::NEG_INFINITY;
    }
    let mut lo: Vec<f32> = cand.iter().map(|&(est, _, err)| est - w * err).collect();
    let (_, &mut v, _) = lo.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
    v
}

impl Index {
    pub fn open(dir: &Path) -> Result<Self> {
        let centroids: Vec<f32> = read_vec(&dir.join("centroids.f32"))?;
        let block_off: Vec<u64> = read_vec(&dir.join("block_off.u64"))?;
        let k = centroids.len() / PDIM;
        anyhow::ensure!(block_off.len() == k + 1, "block_off.u64 does not match centroids");
        Ok(Self {
            k,
            centroids,
            block_off,
            ids: Mapped::open(&dir.join("ids.u32")).context("ids.u32")?,
            blocks: Mapped::open(&dir.join("blocks.bin")).context("blocks.bin")?,
            tier2: match dir.join("tier2.bin").exists() {
                false => None,
                true => Some((
                    Tier2 { sigma: read_vec(&dir.join("tier2.sigma"))? },
                    Mapped::open(&dir.join("tier2.bin"))?,
                    Mapped::open(&dir.join("tier2.norm"))?,
                )),
            },
            rabitq: match dir.join("rotation.f32").exists() {
                false => None,
                true => Some((Rotation::from_signs(read_vec(&dir.join("rotation.f32"))?), Mapped::open(&dir.join("factors.f32"))?)),
            },
            postings: None,
        })
    }

    /// Switches to SSD mode with a cluster cache of `cache_bytes` (0 for none).
    pub fn use_postings(&mut self, dir: &Path, cache_bytes: usize) -> Result<()> {
        anyhow::ensure!(self.rabitq.is_some(), "SSD mode needs a RaBitQ index");
        self.postings = Some(Postings::open(dir, cache_bytes)?);
        Ok(())
    }

    /// The clusters a query probes, best first, as (centroid score, cluster): the top `nprobe`,
    /// cut by the margin and the scan budget. `q256` must be unit length.
    pub fn route(&self, q256: &[f32], p: &Params, s: &mut Scratch) -> Vec<(f32, u32)> {
        s.cent.resize(self.k, 0.0);
        // One query against all centroids is a matrix-vector product: Accelerate runs it on AMX.
        gemm_abt(q256, &self.centroids, 1, self.k, PDIM, &mut s.cent);
        let mut probe = TopK::new(p.nprobe.min(self.k));
        for (c, &v) in s.cent.iter().enumerate() {
            probe.push(v, c as u32);
        }
        let mut probe = probe.into_sorted();
        if let (Some(m), Some(&(best, _))) = (p.margin, probe.first()) {
            probe.retain(|&(v, _)| v >= best - m);
        }
        if let Some(b) = p.budget {
            let mut used = 0;
            let keep = probe
                .iter()
                .take_while(|&&(_, c)| {
                    used += self.slots(c);
                    used <= b
                })
                .count();
            probe.truncate(keep.max(1));
        }
        probe
    }

    /// Stored slots of cluster `c`, padding included.
    pub fn slots(&self, c: u32) -> usize {
        (self.block_off[c as usize + 1] - self.block_off[c as usize]) as usize * BLOCK
    }

    /// `q768` and `q256` must be unit length. Returns (score, corpus row), best first.
    pub fn search(&self, corpus: &Corpus, q768: &[f32], q256: &[f32], p: &Params, s: &mut Scratch, t: &mut Timing) -> Vec<(f32, u32)> {
        self.search_view(corpus, q768, q256, p, &View::default(), s, t)
    }

    /// `search` over the base plus a live view (tombstones, delta segment).
    #[allow(clippy::too_many_arguments)]
    pub fn search_view(&self, corpus: &Corpus, q768: &[f32], q256: &[f32], p: &Params, v: &View, s: &mut Scratch, t: &mut Timing) -> Vec<(f32, u32)> {
        let t0 = std::time::Instant::now();
        let probe = self.route(q256, p, s);
        t.route += t0.elapsed().as_nanos() as u64;
        let cand = self.candidates(q256, &probe, p, v, s, t);
        self.rerank(corpus, q768, q256, cand, p, v, s, t)
    }

    /// Scans the given (centroid score, cluster) list and reranks: everything after routing.
    /// A shard (D16) runs this on the clusters it owns out of a coordinator's probe list.
    #[allow(clippy::too_many_arguments)]
    pub fn search_probes(&self, corpus: &Corpus, q768: &[f32], q256: &[f32], probe: &[(f32, u32)], p: &Params, s: &mut Scratch, t: &mut Timing) -> Vec<(f32, u32)> {
        let cand = self.candidates(q256, probe, p, &View::default(), s, t);
        self.rerank(corpus, q768, q256, cand, p, &View::default(), s, t)
    }

    /// The scan: the best `p.cand` slots of the probed clusters by code estimate, deduplicated by
    /// corpus row, as (estimate, row, error half-width), in row order.
    /// Rows the view hides are skipped at admission; the view's delta lists are scanned after
    /// each probed cluster's base list.
    pub fn candidates(&self, q256: &[f32], probe: &[(f32, u32)], p: &Params, v: &View, s: &mut Scratch, t: &mut Timing) -> Vec<(f32, u32, f32)> {
        t.probed += probe.len() as u64;
        let t1 = std::time::Instant::now();

        let ids = self.ids.as_slice();
        let blocks = self.blocks.as_slice();
        // (estimate, corpus row, error half-width). Global sign codes carry no bound, so their
        // width is infinite and pruning keeps everything.
        let mut cand: Vec<(f32, u32, f32)> = match &self.rabitq {
            Some((rot, factors)) => {
                let rq = rabitq::Query::new(rot, q256);
                let mut cand = TopK::new(p.cand);
                let (scores, meta) = (&mut s.scores, &mut s.meta);
                meta.clear();
                // Only slots that enter the buffer record their row and error factor, so the
                // hot loop touches the ids and errors of a few thousand slots, not all of them.
                let mut scan = |cent: f32, blocks: &[u8], factors: &[f32], ids: &[u32]| {
                    scores.resize(blocks.len() / BLOCK_BYTES * BLOCK, 0);
                    crate::lut::scan(&rq.lut, blocks, scores);
                    t.scanned += scores.len() as u64;
                    for (j, (&sc, f)) in scores.iter().zip(factors.chunks_exact(2)).enumerate() {
                        let est = rq.estimate(cent, f[0], sc);
                        if cand.threshold().is_some_and(|th| est <= th) || ids[j] == PAD || v.hidden(ids[j]) {
                            continue;
                        }
                        cand.push(est, meta.len() as u32);
                        meta.push((ids[j], f[1]));
                    }
                };
                match &self.postings {
                    None => {
                        let factors = factors.as_slice();
                        for &(cent, c) in probe {
                            let (b0, b1) = (self.block_off[c as usize] as usize, self.block_off[c as usize + 1] as usize);
                            scan(cent, &blocks[b0 * BLOCK_BYTES..b1 * BLOCK_BYTES], &factors[b0 * BLOCK * 2..b1 * BLOCK * 2], &ids[b0 * BLOCK..b1 * BLOCK]);
                            if let Some((b, f, i)) = v.delta.and_then(|d| d.list(c)) {
                                scan(cent, b, f, i);
                            }
                        }
                    }
                    Some(post) => {
                        // Scan each record as its read completes; `fetch` is time spent blocked.
                        let clusters: Vec<u32> = probe.iter().map(|x| x.1).collect();
                        t.fetch += post.fetch_each(&clusters, |i, r| {
                            scan(probe[i].0, r.blocks(), r.factors(), r.ids());
                            if let Some((b, f, ids)) = v.delta.and_then(|d| d.list(probe[i].1)) {
                                scan(probe[i].0, b, f, ids);
                            }
                        });
                    }
                }
                cand.into_sorted().into_iter().map(|(est, k)| (est, meta[k as usize].0, meta[k as usize].1)).collect()
            }
            None => {
                let lut = Lut::new(q256);
                let mut cand = TopK::new(p.cand);
                for &(_, c) in probe {
                    let (b0, b1) = (self.block_off[c as usize] as usize, self.block_off[c as usize + 1] as usize);
                    let nb = b1 - b0;
                    s.scores.resize(nb * BLOCK, 0);
                    crate::lut::scan(&lut, &blocks[b0 * BLOCK_BYTES..b1 * BLOCK_BYTES], &mut s.scores);
                    t.scanned += (nb * BLOCK) as u64;
                    let slot_ids = &ids[b0 * BLOCK..b1 * BLOCK];
                    for (&sc, &id) in s.scores.iter().zip(slot_ids) {
                        if cand.threshold().is_some_and(|th| sc <= th) || id == PAD || v.hidden(id) {
                            continue;
                        }
                        cand.push(sc, id);
                    }
                }
                cand.into_sorted().into_iter().map(|(sc, id)| (lut.to_float(sc), id, f32::INFINITY)).collect()
            }
        };
        // A spilled vector can be reached through both of its clusters: keep its best estimate
        // (with residual codes the two copies score differently). Rerank rows are random reads
        // from a file much larger than cache, and visiting them in id order turns neighbouring
        // candidates into neighbouring pages; prefetching a few rows ahead keeps several misses
        // in flight instead of stalling on each one.
        cand.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(b.0.total_cmp(&a.0)));
        cand.dedup_by_key(|x| x.1);
        t.scan += t1.elapsed().as_nanos() as u64;
        cand
    }

    /// Bound pruning, then the rerank of `p.rerank`, over candidates from `candidates` (or a
    /// coordinator's global cut of several shards' candidates). Returns the top `p.k`.
    #[allow(clippy::too_many_arguments)]
    pub fn rerank(&self, corpus: &Corpus, q768: &[f32], q256: &[f32], mut cand: Vec<(f32, u32, f32)>, p: &Params, v: &View, s: &mut Scratch, t: &mut Timing) -> Vec<(f32, u32)> {
        let t2 = std::time::Instant::now();
        if let Some((m, w)) = p.prune {
            let tau = prune_threshold(&cand, m, w);
            cand.retain(|&(est, _, err)| est + w * err >= tau);
        }
        t.reranked += cand.len() as u64;
        let cand: Vec<(f32, u32)> = cand.into_iter().map(|(est, id, _)| (est, id)).collect();

        let mut out = TopK::new(p.k);
        let mut t_mid = std::time::Instant::now();
        let t2_start = |m: &mut std::time::Instant| {
            let e = m.elapsed().as_nanos() as u64;
            *m = std::time::Instant::now();
            e
        };
        match p.rerank {
            Rerank::None => cand.iter().for_each(|&(sc, id)| out.push(sc, id)),
            Rerank::Full => {
                s.w.clear();
                s.w.extend(q768.iter().zip(&corpus.quant.scale).map(|(&x, &sc)| x / sc));
                let inv = 1.0 / query_i8(&s.w, &mut s.wq);
                for (i, &(_, id)) in cand.iter().enumerate() {
                    if let Some(&(_, ahead)) = cand.get(i + PREFETCH_ROWS) {
                        prefetch(v.row(corpus, ahead));
                    }
                    out.push(dot_i8i8(&s.wq, v.row(corpus, id)) as f32 * inv, id);
                }
            }
            Rerank::Ssd(r) => {
                let mut best = TopK::new(r);
                cand.iter().for_each(|&(est, id)| best.push(est, id));
                let ids: Vec<u32> = best.into_sorted().into_iter().map(|x| x.1).collect();
                read_and_rerank(corpus, v, q768, &ids, s, &mut out);
            }
            Rerank::Tiered(r) if self.tier2.is_some() => {
                let (t2, codes, norms) = self.tier2.as_ref().unwrap();
                let codes = codes.as_slice();
                let tables = t2.tables(q256);
                let mut pre = TopK::new(r);
                for &(_, id) in &cand {
                    let (c, n) = match v.delta.and_then(|d| d.tier2(id)) {
                        Some(x) => x,
                        None => (&codes[id as usize * tier2::BYTES..][..tier2::BYTES], norms.as_slice()[id as usize]),
                    };
                    pre.push(tier2::score(&tables, c, n), id);
                }
                let ids: Vec<u32> = pre.into_sorted().into_iter().map(|x| x.1).collect();
                t.tier2 += t2_start(&mut t_mid);
                read_and_rerank(corpus, v, q768, &ids, s, &mut out);
            }
            Rerank::Prefix | Rerank::Tiered(_) => {
                s.w.clear();
                s.w.extend(q256.iter().zip(&corpus.quant.scale).map(|(&x, &sc)| x / sc));
                let inv = 1.0 / query_i8(&s.w, &mut s.wq);
                let mut pre = TopK::new(match p.rerank {
                    Rerank::Tiered(r) => r,
                    _ => p.k,
                });
                for (i, &(_, id)) in cand.iter().enumerate() {
                    if let Some(&(_, ahead)) = cand.get(i + PREFETCH_ROWS) {
                        prefetch(&v.row(corpus, ahead)[..PDIM]);
                    }
                    let r = &v.row(corpus, id)[..PDIM];
                    let norm = v.delta.and_then(|d| d.norm256(id)).unwrap_or_else(|| corpus.norm256.as_slice()[id as usize]);
                    pre.push(dot_i8i8(&s.wq, r) as f32 * inv / norm, id);
                }
                if p.rerank == Rerank::Prefix {
                    out = pre;
                } else {
                    s.w.clear();
                    s.w.extend(q768.iter().zip(&corpus.quant.scale).map(|(&x, &sc)| x / sc));
                    let inv = 1.0 / query_i8(&s.w, &mut s.wq);
                    for (_, id) in pre.into_sorted() {
                        out.push(dot_i8i8(&s.wq, v.row(corpus, id)) as f32 * inv, id);
                    }
                }
            }
        }
        t.rerank += t2.elapsed().as_nanos() as u64;
        out.into_sorted()
    }
}

/// Exact int8 scores of `ids`: corpus rows with parallel preads, delta rows from RAM.
fn read_and_rerank(corpus: &Corpus, v: &View, q768: &[f32], ids: &[u32], s: &mut Scratch, out: &mut TopK<f32>) {
    let (base, extra): (Vec<u32>, Vec<u32>) = ids.iter().partition(|&&id| (id as usize) < corpus.n);
    s.rows.resize(base.len() * DIM, 0);
    corpus.ssd.read_rows(&base, &mut s.rows);
    s.w.clear();
    s.w.extend(q768.iter().zip(&corpus.quant.scale).map(|(&x, &sc)| x / sc));
    let inv = 1.0 / query_i8(&s.w, &mut s.wq);
    for (row, &id) in s.rows.chunks_exact(DIM).zip(&base) {
        out.push(dot_i8i8(&s.wq, row) as f32 * inv, id);
    }
    for &id in &extra {
        out.push(dot_i8i8(&s.wq, v.row(corpus, id)) as f32 * inv, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_moves_smallest_centroid_into_largest_cluster() {
        // 3 clusters: 0 holds 8 of 10 points, 1 holds 2, 2 holds none.
        let mut sample = vec![0f32; 10 * PDIM];
        for (i, row) in sample.chunks_exact_mut(PDIM).enumerate() {
            row[i] = 1.0;
        }
        let a = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
        let counts = [8, 2, 0];
        let mut cents = vec![0f32; 3 * PDIM];
        let n = split_large(&sample, &a, &counts, &mut cents, 2.0, &mut Rng::new(1));
        assert_eq!(n, 1);
        let moved = &cents[2 * PDIM..3 * PDIM];
        let member = (0..8).any(|i| moved == &sample[i * PDIM..(i + 1) * PDIM]);
        assert!(member, "new centroid must be a member of the oversized cluster");
        assert!(cents[PDIM..2 * PDIM].iter().all(|&x| x == 0.0), "cluster 1 is not below half the mean");
    }

    #[test]
    fn prune_threshold_is_mth_lower_bound() {
        // Lower bounds at w = 1: 0.8, 0.5, 0.65, 0.1.
        let cand = [(0.9, 0, 0.1), (0.7, 1, 0.2), (0.7, 2, 0.05), (0.3, 3, 0.2)];
        assert_eq!(prune_threshold(&cand, 2, 1.0), 0.65);
        assert_eq!(prune_threshold(&cand, 5, 1.0), f32::NEG_INFINITY);
        // The top-m candidates always survive: each one's upper bound is at least τ.
        let tau = prune_threshold(&cand, 2, 1.0);
        let kept: Vec<u32> = cand.iter().filter(|c| c.0 + c.2 >= tau).map(|c| c.1).collect();
        assert_eq!(kept, [0, 1, 2]);
        // Narrower bounds raise τ and prune harder.
        assert!(prune_threshold(&cand, 2, 0.5) > tau);
    }
}
