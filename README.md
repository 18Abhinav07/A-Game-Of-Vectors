# sieve

A Rust vector search engine that reproduces, measures and extends the design Exa published in
[How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db).
It serves **113.5M passages** (the TREC-RAG 2024 MS MARCO v2.1 corpus, Arctic-Embed-M v1.5)
on one 24 GB laptop: R@10 0.909 at 4.8 ms p50 and 460 QPS with the codes in RAM, or R@10 0.902
at 13 ms in **225 MB of RAM** with everything on SSD.

Exa's post describes the design but publishes no recall or latency numbers. This repo
measures it against exact search and NIST relevance labels, finds where it loses recall,
and fixes that.

## What's in it
- **Exa's pipeline, built from scratch**: Matryoshka 256-d prefix → 1-bit codes → asymmetric
  lookup-table scoring with a float query (NEON `vqtbl1q_u8`, 32-doc blocks, about 950M
  codes/s/core) → IVF → rerank.
- **SOAR spilling** (Sun et al., NeurIPS 2023): measurement showed routing, not the 1-bit codes,
  was losing recall. Each vector is also stored in a second, orthogonality-aware cluster.
  R@10 rises 0.815 → 0.883 at 32 probes.
- **RaBitQ residual codes** (Gao & Long, SIGMOD 2024): the same 32 bytes, but each code
  describes the vector's rotated offset from its own cluster centroid, with an unbiased
  estimator and an error bound. The code ceiling rises from 0.949 to 0.981, and 500
  candidates now beat the 1000 the global codes needed.
- **Three-tier storage**: the 1-bit scan and a 2-bit RAM tier pick the 200 candidates worth
  fetching, then int8 rows are read from SSD with parallel `pread`. This is lossless versus
  reranking all 1000, so 87 GB of rerank vectors never have to fit in RAM.
- **SSD mode** (SPANN-style): posting lists packed one 4 KiB-aligned record per cluster and read
  with one `pread` per probe, a CLOCK cluster cache, a blocking I/O pool. Same recall as RAM
  mode at 468 MB peak RSS for 18.4M vectors; RAM grows with the cluster count, not N.
- **Bound-driven rerank and query-aware probing**: the RaBitQ error bound decides which
  candidates are worth reranking (28% fewer rows with no loss, 48% for −0.1 pt), and a
  centroid-score margin decides how many clusters each query probes (R@10 0.939 vs 0.940 for
  fixed 128 probes, with 27% less scan). A per-query scan budget bounds the tail: SSD-mode
  cold p99 41.7 → 17.7 ms at higher recall.
- **Sharding with a two-phase merge**: clusters are partitioned across shard processes, and a
  coordinator routes, fans out over TCP, and merges. Shards return calibrated RaBitQ estimates,
  so the coordinator makes the global candidate and rerank cuts itself. The result is
  single-node recall at single-node work: 4 shards give R@10 0.9194 vs 0.9191, and p50
  2.13 → 1.20 ms from intra-query parallelism.
- **Live updates without a rebuild**: tombstones, per-cluster delta lists in the base block
  layout, a write-ahead log, and compaction into a new index. Delete 10% of M1 and reinsert it:
  R@10 0.9192 before, after reinsertion, after a restart from the log, and after compaction.
  Inserts run at 100k rows/s.
- **Measured against FAISS** (IVF-PQ, fast-scan PQ, IVF-RaBitQ) at the same code budget.
- **Integer rerank** on `sdot` (int8 query × int8 row) and centroid routing on AMX via
  Accelerate.
- **One-pass streaming ingest** at 232k rows/s. Exact ground truth is computed with sgemm while
  the data streams past, and shards are downloaded, consumed and deleted so 350 GB of parquet
  never has to sit on a 512 GB disk.

## Results: full corpus (M2: 113.5M passages, 16384 clusters)
| | R@10 | nDCG@10 dl21 / dl22 / dl23 | p50 (cold) | QPS | RAM |
|---|---|---|---|---|---|
| exact flat 768-d | 1.000 | 0.694 / 0.552 / 0.470 | | | |
| 64 probes, 2-bit tier → 200 rows from SSD | 0.909 | 0.686 / 0.547 / 0.457 | 4.8 ms | 460 | 14 GB |
| 128 probes, 1000 cand | 0.937 | 0.693 / 0.551 / 0.460 | 9.3 ms | 308 | 15 GB |
| SSD mode, 64 probes, top 300 from SSD | 0.902 | 0.684 / 0.546 / 0.454 | 13 ms | 90 | 225 MB |
| SSD mode + 1 GB cluster cache, margin probing | 0.907 | 0.690 / 0.545 / 0.460 | 7.2 ms | 117 | 3.3 GB |

At 128 probes (1.6% of the corpus scanned) nDCG@10 is within 0.011 of exact search on every
query set; at 256 probes, within 0.006.

## Results: ablations (M1: 18.4M passages)
| | R@10 | p50 | QPS (12 threads) |
|---|---|---|---|
| plain IVF, 32 probes | 0.815 | | |
| + SOAR | 0.883 | 0.76 ms | 12.8k |
| + RaBitQ residual codes | 0.894 | 0.78 ms | 13.5k (at 500 cand) |
| + 2-bit tier, 200 rows from SSD | 0.893 | | |
| ceiling of the codes (exhaustive scan + rerank): Exa's sign codes → RaBitQ | 0.949 → 0.981 | | |

**Against FAISS** on the same slice, with the same 32 B budget and the same int8 rerank: for
R@10 ≈ 0.885, FAISS's fastest IVF index (`IVF4096,PQ64x4fs`) needs 128 probes and 1.54 ms just
to scan the codes. sieve needs 32 probes and 0.63 ms end to end, so it is 2.4× faster, mostly
because of SOAR. FAISS's `IVF,RaBitQ` reaches 0.903 at 128 probes; sieve reaches 0.919 at 64.

Full tables, ablations and kernel numbers: [BENCHMARKS.md](BENCHMARKS.md). Every design decision
and the evidence for it: [DESIGN.md](DESIGN.md).

## Scaling to 1B: what the measurements imply
These are projections from measured per-slot sizes, not runs. SSD mode stores 44 B per slot
(32 B code, two f32 factors, u32 id), and SOAR stores each vector twice.

| | M1 (measured) | M2 (measured) | 1B (projected) |
|---|---|---|---|
| clusters | 4,096 | 16,384 | ~131k |
| posting lists on SSD | 1.63 GB | 10.03 GB | ~88 GB |
| int8 rerank rows on SSD | 14 GB | 87 GB | 768 GB |
| RAM: centroids + offsets | ~4 MB | ~17 MB | ~135 MB |
| peak RSS, no cache (incl. queries, truth) | 287 MB | 225 MB | well under 1 GB |
| bytes read per query, 64 probes | 27 MB | 40 MB | ~43 MB |

RAM stops being the limit: it scales with the cluster count, not N. Disk capacity and
**bandwidth** take its place. At ~40 MB per query, one NVMe drive with ~3–6 GB/s sustains roughly
75–150 cold queries/s, and a cluster cache lifts that in proportion to its hit rate. So a 1B index
fits on one machine with a 1 TB SSD, but a production QPS target means sharding across
nodes, which is how Exa describes its own deployment. Recall at 1B for a fixed probe count is
untested. Bigger clusters and more vectors per probe both work against it.

## Production gaps
What a real service would still need, roughly in order:
- **Replication**: shards and the coordinator exist (D16), but each shard is one process with no
  replica or hedged requests, and shards map the full files instead of their own split.
- **Cluster maintenance under drift**: live updates exist (D17), but centroids are fixed at
  build time. Data that drifts away from them needs SPFresh-style split and re-centering.
- **Filters and hybrid search**: metadata predicates during the scan, plus BM25 fusion.
- **I/O**: macOS has no io_uring, so queue depth comes from threads blocked in `pread`. On Linux,
  io_uring with registered buffers would remove the thread pool.
- **Observability**: per-stage histograms, cache and I/O metrics exported, not just bench JSON.
- **Graph baselines**: FAISS IVF-PQ, fast-scan and RaBitQ are measured. HNSW/DiskANN on the
  same slice is not.

## Reproduce
```sh
cargo test --release
./scripts/m1.sh          # 10 shards: download, ingest, index (plain + SOAR), eval, bench
./scripts/m2.sh          # all 60 shards, streamed and deleted as they are ingested
```
Binaries: `ingest` (parquet → codes, int8 rows, ground truth), `index` (k-means, SOAR, tiers),
`search [--ssd --cache-mb N] [--shards host:port,... [--two-phase]] eval | routing | bench [--cold]`,
`search live` (delete, reinsert, replay, compact), `shard --shard i --of n --listen host:port [--ssd]`. Config strings are
`nprobe:cand:rerank[:bM[xW]][:pΔ][:sN[k]]`, with rerank one of `none | prefix | full | tieredN | ssdN`.

## Layout
```
src/quant.rs   normalisation, sign codes, int8 quantiser, sdot kernels
src/lut.rs     asymmetric LUT scan (NEON + scalar reference, property-tested)
src/rabitq.rs  RaBitQ rotation, residual codes, estimator and error bound
src/tier2.rs   2-bit Lloyd-Max middle tier with per-byte score tables
src/ivf.rs     k-means, SOAR assignment, index build, search, SSD row reader
src/postings.rs  SSD posting lists, CLOCK cluster cache
src/live.rs    tombstones, delta lists, write-ahead log, compaction
src/shard.rs   cluster→shard assignment, wire protocol, shard server, coordinator client
src/store.rs   mmap, page eviction/residency, blocking I/O pool
src/blas.rs    Accelerate sgemm (AMX)
src/data.rs    parquet streaming, topics, qrels
src/eval.rs    recall, nDCG@10 (MaxP)
examples/      kernel, SSD and middle-tier experiments
```
