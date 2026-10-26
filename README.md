<p><img src="site/logo.svg" width="56" alt="sieve logo"></p>

# sieve

**[Results site: 18abhinav07.github.io/A-Game-Of-Vectors](https://18abhinav07.github.io/A-Game-Of-Vectors/)**, with the 1B latency budget and the DiskANN head-to-head as interactive charts.

A vector search engine written from scratch in Rust, inspired by the architecture Exa described in
[How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db).
It serves **113.5M passages** (the TREC-RAG 2024 MS MARCO v2.1 corpus, Arctic-Embed-M v1.5)
on one 24 GB laptop: R@10 0.909 at 4.8 ms p50 and 460 QPS with the codes in RAM, or R@10 0.902
at 13 ms in **225 MB of RAM** with everything on SSD. On **1 billion vectors** (MS SPACEV-1B) the
same laptop builds an index in about 22 minutes and serves R@10 0.906 at 420 QPS (2.7 ms p50)
or 0.960 at 127 QPS with 12 GiB of RAM and everything else read cold from SSD, or R@10 0.911 at
7.3 ms in 298 MB of RAM.

Exa's post describes the design but publishes no code and no recall or latency numbers. The
implementation, tuning and every number here are this repo's own: it measures the design
against exact search and NIST relevance labels, finds where it loses recall, and fixes that.

## What's in it
- **The described pipeline, implemented from scratch**: Matryoshka 256-d prefix → 1-bit codes → asymmetric
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
  Inserts run at 70–100k rows/s.
- **Filters and hybrid search**: domain-allowlist filters with a planner (post-filter, an
  expanding probe, or exact brute force over the allowed rows) that holds filtered R@10 at or
  above 0.911 from 100% down to 0.1% selectivity. A BM25 index over all 18.4M passages, and
  weighted reciprocal rank fusion: equal-weight fusion lowers nDCG@10 (DL21 0.566 → 0.536),
  and a BM25 weight of 0.2 to 0.3 gains up to 1.4 pt (DL23).
- **One billion vectors on one laptop**: SPACEV-1B (1B × 100-d int8, L2) streamed in ten
  10 GB chunks, two-level k-means into 131,072 leaves, out-of-core encode and pack (28 GB of
  postings), and an exact rerank read straight from the downloaded chunks. Compressed 4-bit
  rerank rows lost 8 points of recall, so they were dropped. Recall does not fall from 100M to
  1B at a fixed config. Page-aligned reads, a hot set pinned by hits per byte (learned from
  held-out queries) and a 2-bit refinement of the shortlist take it from 147 to 420 QPS at
  R@10 0.906, until the SSD's 3.6 GB/s is the limit.
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

## Results: one billion vectors (SPACEV-1B, 131,072 leaves)
The M2 corpus stops at 113.5M because that is all of it, so the 1B claim is tested on the
big-ann-benchmarks SPACEV-1B set instead, on the same 24 GB laptop with no cloud.

| nprobe:cand | R@10 (100M) | R@10 (1B) | p50 (1B) | p99 (1B) | QPS (1B) |
|---|---|---|---|---|---|
| 64:500 | 0.899 | 0.911 | 7.3 ms | 9.9 ms | 147 |
| 128:1000 | 0.938 | 0.942 | 14.6 ms | 24.2 ms | 69 |
| 256:2000 | 0.965 | 0.964 | 28.2 ms | 37.3 ms | 35 |
| 128:1000, cold | 0.938 | 0.944 | 18.2 ms | 25.4 ms | 55 |

- **Build**: train 326 s, encode 1.8M rows/s (ten 100M chunks at about 58 s each, overlapped
  with the download), pack 38 s. Disk: 28.3 GB of postings and the 100 GB of base chunks,
  which double as the exact rerank store.
- **RAM is not the limit**: search peaks at 298 MB of RSS. It scales with the leaf count, not N.
- **The bottleneck was the exact rerank**: 100 GB of rows cannot be cached in 24 GB, so 55–70%
  of query time went to reading candidate rows, and warm was barely faster than cold.

### Spending 12 GiB of RAM on it (DESIGN.md D20)
| 2-bit, 12 GiB pinned, rest cold | R@10 | p50 | p99 | QPS |
|---|---|---|---|---|
| 128:100:1000:m0.3 | 0.871 | 2.2 ms | 9.1 ms | 599 |
| 128:100:1000:m0.4 | 0.906 | 2.7 ms | 11.2 ms | 420 |
| 96:100:1000:m0.45 | 0.911 | 3.0 ms | 10.3 ms | 413 |
| 96:100:1000:m0.6 | 0.927 | 3.8 ms | 11.2 ms | 314 |
| 192:200:2000:m1.0 | 0.960 | 8.7 ms | 21.6 ms | 127 |
| fixed 40:100:1000 (no margin) | 0.906 | 3.5 ms | 7.2 ms | 372 |

- **Reads the device likes**: sub-page random reads cap at 138k/s on this SSD, 4 KiB pages at
  223k/s, so the rerank reads whole aligned pages.
- **A pinned hot set, not the page cache**: rerank traffic flushes the OS cache, so the leaves
  with the most probe hits per byte, learned from held-out queries, are read once into RAM.
  12 GiB covers 75% of probed bytes.
- **2-bit refinement**: the extended-RaBitQ low plane plus one scale per slot (48 B instead of
  28 B) re-estimates the 1,000-slot shortlist, so 100 exact reads do the work of 1,000. At equal
  RAM it beats 1-bit and 3-bit at 100M.
- **Query-aware probing** (D21): each query probes the leaves within a distance margin of its
  nearest, so easy queries scan less. 12–14% more QPS than a fixed nprobe at equal recall.
- **Against the NeurIPS'21 SSD track**: DiskANN reached R@10 0.901 at the 1,500 QPS bar on an
  8-vCPU, 64 GB machine. This laptop gets 0.906 at 420 QPS with 24 GB, about 28% of the
  throughput on 37% of the RAM, and the SSD's 3.6 GB/s is now the limit.
- **Against DiskANN on this laptop** (D22, 10M SPACEV, same queries and ground truth): in RAM,
  DiskANN is 45–50× faster at equal recall (0.903 at 146k QPS against sieve's 0.907 at 3.1k).
  A flat IVF scores 120× more candidates than a graph walk. On SSD with ~0.5 GB of RAM each it
  is close, and closer as recall rises. DiskANN's peak-QPS lead is 1.3× at 0.93 and ~1.1× at
  0.945–0.97, while sieve answers in 1.6–2.8 ms against DiskANN's 10–26 ms at those points.

## Production gaps
What a real service would still need, roughly in order:
- **Replication**: shards and the coordinator exist (D16), but each shard is one process with no
  replica or hedged requests, and shards map the full files instead of their own split.
- **Cluster maintenance under drift**: live updates exist (D17), but centroids are fixed at
  build time. Data that drifts away from them needs SPFresh-style split and re-centering.
- **Filters and lexical search**: host allowlists and BM25 fusion exist (D18), but filters on
  other fields, a filter-aware index for the 1 to 10% band, and WAND/MaxScore pruning (BM25 is
  24 ms p50 by exhaustive TAAT) do not.
- **I/O**: macOS has no io_uring, so queue depth comes from threads blocked in `pread`. On Linux,
  io_uring with registered buffers would remove the thread pool.
- **Observability**: per-stage histograms, cache and I/O metrics exported, not just bench JSON.
- **Leaf balance at 1B**: the largest SPACEV leaf is 17× the mean, which sets the scan tail.
- **A graph router**: DiskANN, measured on the same laptop (D22), beats flat IVF by 45–50× in
  RAM. Closing that needs SPANN's structure: many small posting lists behind a graph over the
  centroids.

## Reproduce
```sh
cargo test --release
./scripts/m1.sh          # 10 shards: download, ingest, index (plain + SOAR), eval, bench
./scripts/m2.sh          # all 60 shards, streamed and deleted as they are ingested
./scripts/ac-bench.sh    # M1 latency re-runs on AC power (refuses on battery or Low Power Mode)
./scripts/spacev.sh      # SPACEV-1B: download the ten 10 GB base chunks, queries, ground truth
./scripts/spacev-1b.sh   # run alongside: train, add each chunk as it lands, pack, search
```
Binaries: `ingest` (parquet → codes, int8 rows, ground truth), `index` (k-means, SOAR, tiers),
`search [--ssd --cache-mb N] [--shards host:port,... [--two-phase]] eval | routing | bench [--cold]`,
`search live` (delete, reinsert, replay, compact), `bm25` (inverted index from the passage text),
`search hybrid | filter`, `shard --shard i --of n --listen host:port [--ssd]`, `spacev train [--bits 2] | add | pack | search [--cold] [--hot-gb N]`
(SPACEV-1B). Config strings are
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
src/bm25.rs    tokenizer, inverted index build, TAAT BM25, weighted RRF
src/filter.rs  allowlist filters: post, expanding probe, brute force, planner
src/shard.rs   cluster→shard assignment, wire protocol, shard server, coordinator client
src/store.rs   mmap, page eviction/residency, blocking I/O pool
src/blas.rs    Accelerate sgemm (AMX)
src/data.rs    parquet streaming, topics, qrels
src/eval.rs    recall, nDCG@10 (MaxP)
src/bigann.rs  SPACEV-1B: L2 RaBitQ, two-level leaves, out-of-core add/pack, chunk rerank
src/bin/spacev.rs  train | add | pack | search for the 1B track
examples/      kernel, SSD, middle-tier and pread/mmap experiments
```
