# sieve: design

A Rust vector search engine that reproduces, measures and extends the design Exa published in
[How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db)
(Dec 2024): Matryoshka truncation, 1-bit document codes, asymmetric lookup-table scoring,
IVF clustering, rescoring with uncompressed vectors, and inverted-index filters.

The post gives the design but no recall, latency or filter numbers. This project measures them
on a public 113.5M-passage corpus with human relevance labels, on one laptop.

## Data

[Snowflake/msmarco-v2.1-snowflake-arctic-embed-m-v1.5](https://huggingface.co/datasets/Snowflake/msmarco-v2.1-snowflake-arctic-embed-m-v1.5):
the TREC-RAG 2024 corpus (MS MARCO v2.1 segmented, 113,520,750 passages) embedded with Arctic
Embed M v1.5, which is Matryoshka-trained (768 dims, intended truncation 256).

- Corpus: 60 parquet shards, `doc_id: string`, `embedding: list<f32>` (768), not normalized.
- Queries: TREC DL21/22/23 (NIST graded labels) and MS MARCO v2 doc dev/dev2, with embeddings.
- Published baselines (flat, full corpus, nDCG@10): DL21 0.694 at 768d, 0.694 at 256d; BM25 0.578.
- Passage text and URLs (for BM25 and domain filters) come later from
  `CohereLabs/msmarco-v2.1-embed-english-v3/passages_jsonl`, joined on `doc_id`.

## Pipeline

```
ingest (one pass over parquet, streaming)
  embedding ─ normalize(768) ─┬─ prefix 256 → renormalize → sign bits      codes.bin   32 B/vec (RAM)
                              ├─ int8 per-dim scaled 768                    rerank.i8  768 B/vec (SSD)
                              └─ exact top-k for every eval query (256d and 768d) → gt/*.bin
  doc_id                      → docids.bin + docids.off

build
  spherical k-means on a sample (256d) → K centroids
  assign every vector (Accelerate sgemm on macOS, portable fallback elsewhere)
  cluster-major, fast-scan layout: per cluster, blocks of 32 docs, nibble-transposed

query
  ① route:   q256 · centroids → top nprobe clusters
  ② scan:    LUT kernel over the probed clusters' blocks → top C candidates (u16 scores)
  ③ rerank:  C int8 vectors from SSD → exact 768d (or 256d) scores → top 10
```

## Decisions

### D1. Clustering (IVF), not a graph index
Exa's stated reasons: filters restrict the scan set directly, and clusters shard without a
global graph. Graphs (HNSW, DiskANN) are the comparison baseline (`usearch`), not the design.
The filtered-search comparison measures that claim instead of repeating it.

### D2. 1-bit codes of the 256-dim Matryoshka prefix, scored asymmetrically
Documents are sign bits; the query stays float. The 256 dims split into 64 groups of 4; each
group gets a 16-entry table of partial dot products, so one table lookup scores 4 dims. The
symmetric Hamming (XOR + popcount) variant is kept as an ablation: it is what most binary
quantization tutorials do and what Exa says it moved away from.

### D3. u8 tables, NEON `vqtbl1q_u8`, u16 accumulators (fast-scan)
A 16-entry u8 table is exactly one NEON register, and `TBL` does 16 lookups per instruction.
Codes are stored so one 16-byte load holds group g's nibble for 32 documents. Tables are
quantized per query (shared scale, per-group offset); the rerank stage absorbs the error. An
AVX2 `pshufb` path covers x86. A scalar implementation is the reference for property tests.
Exa's post notes their kernel needed C++; this records what Rust with `std::arch` reaches,
against the machine's memory bandwidth (roofline).

### D4. Rerank from int8 768-d vectors on SSD
Full f16 768-d is 174 GB for the corpus; int8 is 87 GB and Arctic is trained to tolerate it.
The first 256 int8 dims double as a cheaper rerank tier. Latency is reported cold
(`F_NOCACHE` reads) and warm.

### D5. Ground truth computed during ingest
Exact top-k for every eval query in both 256d and 768d space is accumulated while the parquet
streams through, so recall is measured against the true neighbours of whatever slice is
indexed, and no second pass over 350 GB is needed.

### D6. One crate, library plus binaries
`src/lib.rs` holds the engine; `src/bin/` holds `ingest`, `index` and `search` (eval, bench, routing).
Split into crates only if a boundary earns it (the shard server, later).

### D7. SOAR spilling: each vector in two clusters
Measured on shard 00 (1.76M vectors, 1024 clusters): the end-to-end recall@10 sits within 1–2
points of the routing ceiling (fraction of the exact top-10 inside the probed clusters) at every
nprobe, so routing, not the 1-bit codes, is where recall is lost. SOAR (Sun et al., NeurIPS 2023)
stores each vector again in the cluster minimising ||x − c||² + λ⟨r̂, x − c⟩², where r is the
residual to its primary centroid; queries that miss the primary cluster tend to hit the secondary.
At equal vectors scanned it raises recall@10 by 1.4–2.9 points (0.763 → 0.792 at ~29k scanned,
0.885 → 0.899 at ~117k). Cost: codes are stored twice (+32 B/vector, 7.2 GB at 113.5M) and
candidates are deduplicated before rerank (copies score identically, so they sort adjacently).
Exa's post does not describe spilling; plain IVF stays available as the ablation (`--soar` off).

### D8. Integer rerank: int8 query × int8 row on `sdot`
The query weights q_i/scale_i are quantised to int8 under one scale, so each candidate is one
integer dot product (`vdotq_s32`, 16 MACs per instruction) instead of i8→f32 conversions.
Rerank went from 0.17 ms to 0.068 ms per query at 1000 candidates; recall@10 moved by ≤ 0.001.

### D9. Three tiers so 113.5M vectors fit in 24 GB: 1-bit scan, 2-bit RAM rerank, int8 on SSD
At full scale the int8 rerank rows are 87 GB, which won't fit in RAM, and a full rerank of 1000
candidates from SSD costs ≥ 4.6 ms of device time (measured, `examples/ssd.rs`: 18.7k random
768 B reads/s at QD1, 215k/s at QD64), capping the machine near 215 QPS. So rerank is split:
1. **RAM middle tier**: the 256-d prefix in 2-bit Lloyd-Max codes (per-dimension σ;
   thresholds 0, ±0.98σ; levels ±0.45σ, ±1.51σ), 64 B/vector (7.3 GB at 113.5M). It is scored
   with 64 per-query 256-entry byte tables (one per 4 dimensions), so a candidate costs 64 loads
   plus adds, and it keeps the top r of the 1000 candidates.
2. **SSD final tier**: those r int8 rows are read with positioned reads issued in parallel
   (`RowReader`, about 8 rows per task, so many requests are in flight), then scored with `sdot` (D8).
Middle-tier choice came from simulating seven variants on M1 (`examples/tier2.rs`): 256×2-bit
beats 128×4-bit and 64×8-bit at the same bytes, because the prefix's leading dimensions carry
most of the score and dropping them costs more than coarse quantisation. Measured with the real
tier on M1 (32 probes, SOAR): recall@10 0.8641 / 0.8799 / 0.8828 at r = 50 / 100 / 200, against
0.8831 for a full int8 rerank of all 1000, so r = 200 is lossless while cutting SSD reads 5×.
The RAM budget at 113.5M is codes ×2 for SOAR (7.3 GB) + ids (0.9 GB) + tier2 (7.3 GB)
≈ 15.5 GB. `search bench --cold` sets `F_NOCACHE` on the reader. That only stops new pages being
cached, so it measures cold reads only when the file is larger than RAM, as it is at full scale.

### D10. RaBitQ residual codes: same 32 bytes, code ceiling 0.949 → 0.981
Exa's codes are the signs of the vector itself. With exhaustive scan plus rerank, that code
ceiling is R@10 0.949, so even perfect routing would lose 5 points to the codes. RaBitQ
(Gao & Long, SIGMOD 2024) instead codes the residual to the vector's own cluster centroid, after
a random rotation (3 rounds of sign flips and a Walsh-Hadamard transform, O(D log D)):
⟨x,q⟩ ≈ ⟨c,q⟩ + ‖r‖·⟨x̄,Pq⟩/⟨x̄,o'⟩. The centroid term is the routing score the query already
has. The code only describes the residual, which is much smaller than x, and the rotation makes
every bit equally informative. The estimator is unbiased with a stated error bound
(`rabitq::encode` returns it; a test checks coverage).
- The LUT kernel is unchanged: Σ sign·q = 2·Σ bit·q − Σ q, so each slot costs one FMA on top of
  the table lookups. The scan takes 6% longer (0.554 vs 0.521 ms at 32 probes).
- Each slot stores `[mul, err]`, 8 B more per slot (16 B per vector with SOAR). `err` drives
  rerank pruning (D11).
- The two SOAR copies of a vector now carry different codes and scores, so dedup keeps the
  better copy.
- M1, 32 probes: R@10 0.883 → 0.894 at 1000 candidates, 0.820 → 0.867 at 200. 500 RaBitQ
  candidates (0.888) beat 1000 global-code candidates (0.883), halving rerank work. At 64 probes:
  0.910 → 0.926.

### D11. Bound-driven rerank: rerank depth set per query by the RaBitQ error bound
A fixed candidate count spends the same rerank budget on every query, though an easy query's
top 10 stand far above the rest and a hard query's do not. Each RaBitQ estimate comes with a
half-width `err`, so after dedup the engine computes τ, the m-th best lower bound `est − w·err`,
and drops every candidate whose upper bound `est + w·err` is below τ: with high probability it
is outranked by m others. The rest go to rerank (`search ... 32:1000:full:b10x0.8`).
- At `w = 1` (the paper's ε0 = 1.9) it is lossless (0.8936 at 32 probes) but prunes only 850 →
  612 rows. The bound is conservative, and it describes the 256-d prefix score while the rerank
  ranks by 768-d.
- Narrowed bounds are the useful setting. At equal rerank work they beat a fixed cutoff:
  0.890 vs 0.885 at about 340 rows (w = 0.7), and 0.882 vs 0.874 at about 230 (w = 0.6).
  w = 0.8 reranks 446 rows for 0.8926, against 850 rows for 0.8936.
- Combined with the 2-bit tier (D9) at w = 1, recall is unchanged and the tier scores 28% fewer
  rows. The tier's own top-200 cut remains the cheaper way to limit SSD reads at fixed recall;
  pruning removes the rows neither tier needs to see.
- Global sign codes carry no bound (width ∞), so pruning is a no-op on those indexes.

### D12. Query-aware probing: nprobe set per query by centroid scores
A fixed nprobe scans 32 clusters for a query whose neighbours sit in one dominant cluster just
as it does for one spread over many. SPANN (Chen et al., NeurIPS 2021) instead probes a cluster
only if it is nearly as close as the closest one. The inner-product version: of the top-nprobe
clusters, scan those with ⟨q,c⟩ ≥ ⟨q,c₁⟩ − Δ, with nprobe as the cap (`256:500:full:p0.16`).
It costs nothing at query time, since the centroid scores already exist.
- On M1 (4096 clusters, RaBitQ, 500 candidates, full rerank) it beats fixed nprobe at every
  scan budget from 260k vectors up: +0.2 points at 260k, +0.8 at 376k, +0.5 at 700k–925k.
- Δ = 0.16 reaches 0.9387 with 925k vectors scanned, against 0.9404 for 128 fixed probes at
  1.26M (27% fewer). Δ = 0.18 beats 128 probes outright (0.9448) with 7% fewer vectors.
- Small Δ collapses (0.62 at Δ = 0.02), because many queries then scan one or two clusters.
  A margin is useful only well above the routing noise.
- Per-query work now varies, so p99 matters more than mean; the M2 bench reports both. The
  margin depends on how tightly the centroids pack, so it is re-tuned at 16384 clusters.

### D13. SSD mode: posting lists on disk, 113.5M vectors in well under 2 GB of RAM
RAM mode keeps every code, id and RaBitQ factor resident: about 17 GB at 113.5M with SOAR and
the 2-bit tier, and at 1B that would be roughly 150 GB. SSD mode follows SPANN (Chen et al.,
NeurIPS 2021): `index --pack` rewrites each cluster as one contiguous record, 4 KiB aligned
(codes, then `[mul, err]`, then ids: 44 B per slot), so probing a cluster costs one positioned
read. RAM holds only the centroids, the record offsets and an optional hot-cluster cache.
- **Same results.** SSD mode reads the same bytes, so recall is identical to RAM mode (checked:
  0.8876 at 32:500 in both modes).
- **No RAM tier**, so the final tier picks rows by code estimate alone (`ssdN`). That needs more
  SSD reads for the same recall: on M1 at 64 probes, 200 rows give 0.901 and 300 give 0.912
  (RAM mode with the 2-bit tier: 0.919 at 200). 400 rows give 0.917.
- **Measured peak RSS: 468 MB** for the M1 index (1.63 GB of posting lists), including query
  data and ground truth. The resident set is centroids plus offsets, so it grows with K, not N.
- **I/O pool.** macOS has no io_uring, so queue depth is the number of threads blocked in
  `pread`. Reads run as owned jobs on a dedicated pool of 64 blocking threads. A rayon pool
  failed here: a rayon worker that `install`s into another pool keeps stealing from its own
  while it waits, so it started more queries on top of the pending ones (13 GB RSS, 7× lower QPS).
- **Cold measurement.** `store::evict` drops a file's pages with `msync(MS_INVALIDATE)` (checked
  with `mincore`: residency 1.0 → 0.0), and `F_NOCACHE` stops reads refilling the cache. Only
  this combination makes `--cold` honest on a file that fits in RAM.
- **Hot-cluster cache.** A sharded second-chance (CLOCK) cache of whole records under a byte
  budget. On M1 at 512 MB (31% of the lists) the hit rate is 0.55. That cuts bytes read from
  27.4 to 11.4 MB per query and raises QPS 85 → 139. Eval queries are replayed, so the
  throughput loop's hit rate is optimistic; the latency pass starts from an empty cache.
- **The cost is bandwidth.** A query at 64 probes reads 27 MB on M1 and about 39 MB on M2
  (610 KB clusters). Cold, single-threaded, with the M2 ingest competing for the SSD
  (preliminary): p50 13 ms, of which 8 ms is fetch and 4.4 ms is the 300-row rerank, at 85 QPS.
  Query-aware probing (D12) cuts the fetch with the probe count: p50 7.4 ms at 21.6 MB per query.
  Idle-machine numbers come with M2.
- **Scan while reading.** `fetch_each` submits every miss, scans the cache hits, then scans
  each record as its read completes, so the scan hides inside the I/O wait instead of following it.
  A/B against the batch version, two rounds each, cold, 64:500:ssd300: p50 12.9/13.3 → 12.1/11.8 ms,
  time blocked on reads 8.1/8.8 → 7.1/6.8 ms, and peak RSS 494/423 → 287/303 MB, because each
  record is dropped after its scan instead of all 64 being held at once. Recall is unchanged
  (0.912). With margin probing the difference is within noise (`results/m1-stream-ab.txt`).

### D14. Balanced k-means: tried, no gain at equal scan (kept behind `--balance`, off)
Cluster sizes are skewed (M1 stored slots: p50 8.5k, max 45k; M2: p50 13k, max 85k), and a
query that probes the big clusters pays for them in scan and SSD bytes. `index --balance r`
adds a split step to training: after each centroid update, every cluster larger than r × the
mean gives a random member to one of the clusters smaller than half the mean, which restarts
there. Assignment stays nearest-centroid, so the estimator and SOAR are unchanged.
- **It balances the sample, not the corpus.** At r = 1.5 the largest sample cluster falls from
  21× to 6× the mean (2609 → 724 of 122). But about 800 centroids move every iteration
  without settling, and on the full 18.4M rows the largest cluster only falls 45k → 42k, with p99
  20.9k → 21.5k. The tail comes from dense regions of the data, which a 500k sample does not
  pin down.
- **Recall at equal scan is unchanged.** At 16 probes, R@10 goes 0.846 → 0.853, but the vectors
  scanned per query go 151k → 170k, because queries land in the dense regions. The baseline's
  own curve gives 0.851 at 170k. At 64 probes (tiered200): 0.9191 vs 0.9197 at 623k vs 683k
  scanned. r = 2.0 is the same, within 0.006.
- **What would work instead** is bounding the work per probe rather than the cluster size: split
  oversized clusters after assignment, on the real counts, into sub-clusters with their own
  centroids (SPANN's hierarchical balanced clustering), or cap the bytes read per query (the D12
  tail item). Both change the layout, so they are left for the sharded build.
  (`results/m1-balance.jsonl`)

### D15. Per-query scan budget: bounds margin probing's tail
Margin probing (D12) gives easy queries few clusters, but queries with flat centroid scores take
every cluster up to the cap. On M1 at Δ = 0.12 the mean scan is 491k slots, but p99 is 3.0M,
3.4× the p99 of fixed 64 probes. That tail is the long p99 latency. `:sN` sets a budget in stored
slots (44 B each on SSD): clusters are taken in score order until the next would exceed it, and
the best cluster is always scanned. `eval` now reports the scan p99 and max per query.
- **Better recall, and a hard worst case.** Δ = 0.16 with a 1M budget gives R@10 0.9244 at a mean
  scan of 536k and a max of 1.0M. Fixed 64 probes gives 0.9191 at 623k mean (max 975k), and
  uncapped Δ = 0.12 gives 0.9099 with a max of 3.3M. Δ = 0.20 with 1.2M gives 0.9358 against
  fixed 96's 0.9332, at a lower mean (836k vs 942k) and max (1.2M vs 1.4M).
- **Latency (RAM, warm, same session, Low Power Mode):** p99 8.2 → 3.3 ms against uncapped
  Δ = 0.12. Against fixed 64 probes, p50 2.10 → 1.70 ms and QPS 1980 → 2247; p99 is 2.8 vs 3.3 ms.
- **SSD mode, cold:** Δ = 0.16 with 800k matches fixed 64's recall (0.9122 vs 0.9120) at
  p50 10.6 vs 13.1 ms and p99 17.7 vs 31.5 ms. Against uncapped Δ = 0.12, p99 drops 41.7 → 17.7 ms.
  Single cold runs vary by several ms at p99.
- A budget in slots is a budget in bytes read, which is what an SSD-bound shard has to provision.
  The coordinator can set it per query from a latency target. (`results/m1-budget.jsonl`)

### D16. Sharding: cluster-partitioned shards, a routing coordinator, a two-phase merge
Clusters, not vectors, are the unit of partition. Each shard process (`shard --shard i --of n`)
owns a set of clusters: their posting lists, and the int8 rows of the vectors in them. A
coordinator (`search --shards a,b,...`) keeps only the centroids (4 MB at M1, 17 MB at M2),
routes each query with them (nprobe, margin and budget are all applied here), and sends each
shard the probed clusters it owns. Clusters are assigned by longest-processing-time greedy on
stored slots, a pure function of `block_off`, so coordinator and shards agree with no
coordination; at M1 four shards hold 9,206,016 / 9,206,016 / 9,205,952 / 9,205,984 slots.
Partitioning by cluster rather than by vector means a query touches only the shards owning its
probed clusters. Random vector sharding would make every shard scan nprobe clusters of its own,
multiplying the scan by the fan-out.

The wire protocol is length-prefixed binary over TCP with `TCP_NODELAY`. Each coordinator thread
keeps one persistent connection per shard, and each shard runs a thread per connection. Two merge
protocols:
- **One-shot**: each shard scans and reranks its clusters with the full `cand` and rerank
  budgets and returns exact top-k. One round trip, but the union of per-shard top-`cand` lists
  is up to n× bigger than the global one, so rerank work grows with the fan-out. Recall is ≥
  single-node: M1, 4 shards, 32:500:full gives 0.8959 vs 0.8876, and 64:500:tiered200 gives
  0.9282 vs 0.9191, at 4.6× the reranked rows (1954 vs 423). Dividing the budgets by n to
  equalise work loses about 0.9 pt (0.8789, 0.9101): a per-shard cut is not the global cut,
  because the shard with most of the good candidates gets the same quota as the others.
- **Two-phase** (`--two-phase`): phase 1 returns each shard's top-`cand` (estimate, row, error);
  the coordinator makes the single-node cuts (global top-`cand` slots, dedup, bound pruning,
  and for `ssdN` the top-N rows by estimate). Phase 2 has each row reranked by the shard that
  returned it. The global cut is valid because RaBitQ estimates are calibrated inner products,
  not per-cluster scores, so they compare across shards. The tier-2 cut of `tieredN` is by a
  score only the shard computes, so each shard gets its proportional share of N. The result
  reproduces single-node recall at single-node work. M1, 4 shards: 0.8879 / 0.9194 / 0.9120 /
  0.9118 / 0.8932 / 0.4536 against single-node 0.8876 / 0.9191 / 0.9120 / 0.9115 / 0.8932 /
  0.4536 for full, tiered200, ssd300, margin+budget, pruned ssd300, and no rerank.
- **Latency (4 shard processes and the coordinator on one 12-core machine, RAM mode, warm,
  Low Power Mode, 64:500:tiered200):** single process p50 2.13 ms, 1677 QPS. Two-phase p50
  1.20 ms (the four shards scan one query in parallel), 1380 QPS (82%, the protocol's cost on
  shared cores). One-shot p50 1.90 ms, 735 QPS, paying for 4.6× the rerank. Two-phase is the
  default choice: the extra round trip costs tens of µs on a LAN; the extra rerank work costs
  half the throughput.
- Not built: replication (each shard is one process; a replica set per cluster group and
  hedged requests to the faster replica are the standard fix), and physically splitting the
  files. Shards map the full files and touch only their own clusters' pages, so in SSD mode a
  shard's reads already cover only its records. (`results/m1-shard.jsonl`)

### D17. Live updates: tombstones, delta lists, a write-ahead log, compaction
The built index stays immutable; updates sit beside it and the query path reads both through a
`View` (`src/live.rs`, `Index::search_view`).
- **Deletes** set a bit in a tombstone bitset over row ids. The scan checks it at admission,
  after the threshold test, so a dead slot costs a load only when it would otherwise have
  entered the candidate buffer, and never occupies a place in it.
- **Inserts** are encoded exactly as `ivf::build` encodes a row: the prefix is rebuilt from the
  int8 row, assigned to its nearest centroid plus a SOAR secondary, RaBitQ-encoded against both
  centroids, given a 2-bit tier code. The two slots are appended to per-cluster delta lists in
  the base block layout, with the last partial block repacked on each insert, so the unchanged
  LUT scan reads them right after the cluster's base list, in RAM and SSD mode alike. Delta rows
  get row ids from `corpus.n` up, and their int8 rows, tier codes and prefix norms live in RAM;
  every rerank path asks the view for a row before the corpus. The SSD rerank splits its reads:
  corpus rows by `pread`, delta rows from RAM.
- **External ids**: corpus rows keep their row ids as external ids, and a delta row carries the id
  it was inserted under. Inserting a live id is an upsert that tombstones the old row; search
  maps rows back to external ids.
- **Durability**: every batch is written to a log (op, id, prefix norm, int8 row: 777 bytes per
  insert, 5 per delete) and fsynced before it is applied. The log stores the int8 row, not the
  float vector, because the build derives every code from the int8 row, so a replay rebuilds
  the delta bit for bit. A torn final record is cut off on open.
- **Compaction** writes a new index directory: per cluster, the live base slots (codes
  recovered with `lut::unpack_slot`, the inverse of `pack_block`) followed by the live delta
  slots, then SSD posting lists. Centroids, rotation and the corpus-indexed tier files are
  symlinked; the delta rows become a row segment (`live.*`) that the reopened index serves
  rows from. Compaction reads the live state through `&self`, so searches continue while it
  runs; records replayed onto a compacted index are harmless because each is an upsert or a
  delete.
- **Result** (M1, delete a random 10% = 1,838,056 rows, reinsert all of them under their own
  ids; R@10 against the exact top-10 of the rows live at each step): baseline 0.9192 / 0.9191 /
  0.9120 for 64:500 full / tiered200 / ssd300. After the deletes, 0.9183 / 0.9182 / 0.9111; the
  live truth now reaches further down the exact ranking, into clusters not always probed.
  After reinsertion, and again after a restart from the log alone, and again after compaction,
  0.9192 / 0.9191 / 0.9120: the delta path loses nothing. SSD mode gives 0.9120 at every step
  too. Inserts run at 70–100k rows/s on 12 threads (69k in the AC re-run), fsync per 10k rows. The log for 1.84M rows is
  1.44 GB and replays in 12 s; compaction takes 8 s plus the posting-list pack.
- **Cost of an uncompacted delta**: with 10% of the corpus in delta lists and 10% dead slots in
  the base, scanned slots rise 10% (622,762 → 685,851 per query) and p50 rises about 25% (1.67 →
  2.08 ms for full rerank, Low Power Mode, relative only). Compaction restores both (622,762;
  1.80 ms). So compaction is a latency task, not a correctness one, and can run on a
  dead-slot or delta-size threshold.
- Not built: splitting or re-centering clusters as the data drifts (SPFresh's LIRE). With a
  fixed set of centroids, inserts far from all of them land in clusters that probe badly, and
  recall would decay; the reinsert test cannot show this because its rows come from the
  distribution the centroids were trained on. Concurrency is one writer at a time behind an
  `RwLock`; there is no MVCC snapshot for long scans. (`results/m1-live.jsonl`)

### D18. Filters and hybrid search: an allowlist planner, BM25 with weighted fusion
**Filters** (`src/filter.rs`). A filter is a bitset of allowed rows, passed to the scan through
the same `View` as the D17 tombstones, so a disallowed slot is skipped at admission and never
takes a candidate place. The test filter is a domain allowlist (Exa's `includeDomains`): hosts
are parsed from each passage's URL (385,712 hosts in M1), put in a fixed pseudo-random order,
and taken in that order until they hold the target fraction of rows. Whole domains are allowed,
not random rows, so allowed rows cluster by topic the way real ones do. Recall is against the
exact filtered top-10 under the same int8 scores the rerank uses (`filter::truth`: one batched
pass over the allowed rows per filter).
- **Post-filter** (the unfiltered probes, disallowed slots skipped) holds at high selectivity
  (R@10 0.922 at 100%, 0.911 at 50%) and degrades below it: 0.887 at 10%, 0.778 at 1%. At a
  fixed 64 probes, a 1% filter leaves about 1% of the scanned slots eligible.
- **Expanding probe**: clusters are taken in centroid order until a target count of *allowed*
  slots is covered, from per-cluster allowed counts computed once per filter. At 10% it beats
  post (0.937 at 200k allowed slots, 6.2 ms). At 1% and below it has to walk most of the index
  (1,588 to 3,322 probes, 46 to 93 ms), because the allowed rows sit in few clusters that are
  often far from the query.
- **Brute force**: below a few hundred thousand allowed rows, scoring every allowed int8 row
  is exact and cheaper than any index plan: 3.2 ms for 184k rows (1%), 0.34 ms for 18k (0.1%).
- **Planner** (`Plan::Auto`): brute force at or below 250k allowed rows, post-filter at or above
  30% of the corpus, an expanding probe to 200k allowed slots between. Across 100% to 0.1%, its
  filtered R@10 never drops below 0.911 and p50 never exceeds 6.2 ms (single-thread, Low Power
  Mode). The thresholds are M1's; the brute-force bound scales with row cost, not corpus size,
  while the post threshold depends on how many allowed slots the default probes reach.
- Not built: filters on fields other than host, a per-cluster filter index (bitmaps or ACORN-style
  graph links) for the 1 to 10% band, and selectivity estimation for predicates whose row count is
  not known up front. Here the allowed set is materialised once per filter, which is right
  for a fixed allowlist and wrong for per-query predicates.

**BM25** (`src/bm25.rs`, `bin/bm25`). An inverted index over the passage segment text (no title
or headings), built in one pass over the 10 gzipped JSONL files, which list passages in corpus
row order; every one of the 18,380,565 rows matched. Terms are hash-sharded across threads, each
appending (row delta, tf) varints to its own posting lists without locks. The term table is
sorted and looked up by binary search, so nothing is rebuilt on open. Tokenizer: lower-cased
alphanumeric runs, Lucene's stop set, Harman's S-stemmer. k1 0.9, b 0.4 (Anserini's MS MARCO
defaults). Build: 137 s, 7.5 GB peak, 6.32M terms, 1.54B postings, 3.3 GB on disk. Scoring is
term-at-a-time into a dense accumulator: 6.6M postings decoded per query, p50 24.5 ms, p99 124
ms. That is ten times the vector path; MaxScore or WAND would cut it and are not built.

**Fusion**. Reciprocal rank fusion (k 60, top 100 of each list), with a weight on the BM25
list's terms. nDCG@10 on the M1 slice, exact dense / BM25 / fused:
- BM25 alone is far behind the dense model: 0.422 vs 0.566 on DL21, 0.193 vs 0.314 on DL22.
- **Equal-weight RRF hurts**: DL21 0.566 → 0.536, dev 0.242 → 0.224. Fusing a much weaker
  list at equal weight pulls its errors into the top 10.
- **Down-weighted BM25 helps a little**: at weight 0.2 to 0.3, DL23 rises 0.255 → 0.269
  (+1.4 pt) and DL22 0.314 → 0.317, while DL21 and dev stay level. With the IVF list in place
  of exact search, the same pattern holds (DL23 0.251 → 0.265 at weight 0.3).
- So on this corpus, with a strong embedding model, lexical fusion is a small gain on some query
  sets, not a default; its weight has to be tuned, and its latency is the larger cost. Exa's
  own product exposes keyword and neural search as separate modes, which fits this result.
  (`results/m1-hybrid.jsonl`, `results/m1-filter.jsonl`)

### D19. One billion vectors: MS SPACEV-1B on one laptop
M1 and M2 stop at 113.5M passages because that is the whole embedded corpus. To test the 1B
claim directly, `src/bigann.rs` and `bin/spacev` index **MS SPACEV-1B** (big-ann-benchmarks
NeurIPS'21: 1B × 100-d int8 Bing web-document vectors, L2, 29,316 queries, ground truth for the
10M, 100M and 1B prefixes) on the same M4 Pro, 24 GB RAM, with no cloud.
- **L2 through the same kernel.** Rows rank by 2⟨q,x⟩ − ‖x‖². ⟨q,x⟩ = ⟨q,c⟩ + ⟨q,r⟩ splits as
  in D12, and ‖x‖² is stored exactly per slot (the vectors are int8, so it is an exact integer),
  so the RaBitQ code still estimates only the residual term. 100 dims pad to 128: 16-byte codes,
  32 LUT groups. `Lut`, `Query` and the record layout became generic over the group count;
  the M1/M2 paths are unchanged (G = 64 by default).
- **Two-level centroids.** 131,072 flat leaves would cost 1B × 131k × 100 × 2 flops to assign:
  hours of sgemm. Leaves are trained per coarse cell instead (K1 512 coarse × K2 256 fine, each
  cell's k-means in parallel), and a row picks the best child among its 4 nearest coarse
  cells: about 1.5k centroid dots per row. Queries still route over all leaves with one sgemv.
- **Out-of-core build.** The base arrives as ten 10 GB chunks (`scripts/spacev.sh`: 16 parallel
  range requests written in place, ~5 minutes per chunk). `add` encodes one chunk into run files
  bucketed by leaf range and sorted. `pack` streams the buckets in leaf order into the D13 SSD
  records and deletes each bucket's runs once written. Encoding runs at 2M rows/s: 50 s per
  100M rows.
- **Ties.** At 1B, 30.8% of queries have a tie at rank 10 (mostly exact duplicates at distance
  0); at 10M, 0.75%. Recall follows big-ann's rule: a result counts if it is any ground-truth
  row at most as far as the 10th.
- **Exact rerank from the chunks themselves, not compressed rows.** The first design kept a
  4-bit residual row per vector (54 B: f16 offset and step, a nibble per dim) and deleted the
  chunks. It cost 8 points: R@10 0.850 against 0.928 for an exact rerank of the same 500
  candidates at 10M. Measured on 20k rows, its score error has std 206 while the median gap
  between the 10th and 11th neighbour is 60 at 10M and 26 at 1B. 6 bits gives std 49 and
  8 bits std 12, and 8 bits costs as much disk as the raw int8 rows. So the downloaded chunks
  are the rerank store: a candidate's row is read at (id mod 1e8)·100 in base.(id / 1e8). Disk
  at 1B: 100 GB of chunks, ~28 GB of postings, ~32 GB of transient runs.
- **Rerank reads are pread on the I/O pool, not mmap.** On macOS, cached preads on one file
  serialise: 0.44 µs per read on one thread, 1.1 µs aggregate at 12 to 32 threads
  (`examples/preadbench.rs`). mmap hits cost 0.15 µs and scale, and `mincore` to sort hits from
  misses costs 0.46 µs and serialises too. But misses decide it: on an evicted file, mmap faults
  cost 107 µs aggregate at 64 threads against 9.3 µs for preads (`examples/faultbench.rs`).
  100 GB of rows cannot stay in 24 GB of RAM, so at 1B nearly every rerank read is a miss.
  Ten chunk files also spread the per-file lock.
- **Result.** The 1B index builds in about 15 minutes of compute once the chunks are on disk:
  train 326 s (the coarse step's random-row gather was 311 s of it; the sample ids are now
  sorted first, which this run predates), ten adds at 54–60 s each (1.8M rows/s), and pack in
  38 s for 28.3 GB of postings. Search holds 298 MB of RSS. At 64 probes and 500 candidates,
  R@10 is 0.911 at 7.3 ms p50 and 147 QPS; 128:1000 gives 0.942 at 14.6 ms; 256:2000 gives 0.964
  at 28 ms. Recall at a fixed nprobe does not fall from 100M to 1B (64:500: 0.899 → 0.911),
  because the leaf count grew with N (14,336 → 131,072) and the routing sgemv stays under
  0.5 ms. The rerank reads are 55–70% of query time, and cold runs are only 2–4 ms slower than
  warm, because warm is already missing on 100 GB of chunks. That cost, not the scan, is what
  the next step would attack: fewer candidates via a bound-driven cut (D11) or a 6–8-bit
  middle tier in RAM (D9) to pick the rows worth reading.
- **Leaf imbalance.** The largest 1B leaf holds 129,388 slots, 17× the mean of 7,629, and the
  top-4 coarse assignment does nothing to cap it. Balanced k-means or splitting oversized
  leaves would cut the scan tail.

### D20. SPACEV-1B at the device limit: page reads, a pinned hot set, 2-bit refinement
D19 left 1B at R@10 0.911 and 147 QPS, with the exact rerank the bottleneck. The published
reference point is the NeurIPS'21 big-ann T2 (SSD) track (Simhadri et al., PMLR 176, 2022):
DiskANN reaches R@10 0.901 at the 1,500 QPS bar on msspacev-1B, on an 8-vCPU Azure L8s_v2
with **64 GB of RAM** and a 1 TB NVMe. This laptop has 24 GB. Three measured changes, each
attacking whatever the device was spending its time on:
- **Device ceilings first** (`examples/iobench.rs`, on the uncached postings file). Sequential
  256 KiB reads: 3.6 GB/s. Random 4 KiB aligned reads: 223k/s at 64 threads, 241k at 128. But
  random reads of 100 B or 512 B with the page cache bypassed top out at **138k/s**: a sub-page
  read costs the device a page and completes slower. D19's rerank read 100-byte rows.
- **Rerank reads whole aligned pages.** `Base::rows` now reads the 4 KiB page (two for the 2.4%
  of rows that straddle a boundary) into a page-aligned buffer. Isolated, the cold rerank path
  goes to 221k rows/s (`examples/rerank_io.rs`).
- **A pinned hot set instead of the page cache.** Probe traffic is skewed: the hottest leaves
  by hits per byte cover 22% of probed bytes at 2 GiB, 59% at 8 GiB, 75% at 12 GiB, 87% at
  16 GiB (`examples/spacev_diag.rs`). The OS cache could not exploit that, because 100 GB of
  rerank reads flush it (D19: warm ≈ cold). `Postings::pin` reads a static set once;
  `spacev search --hot-gb` chooses it greedily by hits per byte at the run's nprobe, **learned
  from the held-out queries 10,000–29,315**, never from the 10,000 evaluated. Everything else
  is read with F_NOCACHE. With 12 GiB pinned, 1-bit 64:500 goes from 130 to 266 QPS cold
  (posting traffic 14.8 → 3.7 MB/query), and the rerank is again the bottleneck.
- **2-bit refinement to shrink the rerank.** Re-ranking real 1B candidates offline
  (`examples/spacev_refine.rs`, 1,000 held-out queries, nprobe 64) shows the 1-bit estimate's
  error (rms 1,899) is what forces 500–1,000 reads. A uniform B-bit code of the residual, with
  a per-row step that maximises cos(r, r̂) (the extended-RaBitQ idea, Gao et al. SIGMOD 2025),
  gives R@10 at 50 / 100 / 500 candidates of 0.771 / 0.841 / 0.917 (1-bit), 0.889 / 0.917 /
  0.932 (2-bit, rms 869), 0.928 / 0.931 / 0.932 (3-bit, rms 478). The top plane of that code is
  the sign, which is the 1-bit code already in the block, so a `bits`-bit index stores only the
  low planes and one scale f per slot (`refine_encode`; 2-bit: +20 B, 48 B per slot). The scan
  is unchanged; each slot that enters the 1-bit shortlist (`refine`, 1,000) is re-estimated
  from all planes with 16 table lookups per plane (`Refiner`: per-query sums of q over each
  byte value), and only the best `cand` are reranked. Config syntax: `nprobe:cand:refine`.
- **2-bit, not 3.** Bigger slots cost posting bandwidth and hot-set coverage. At 100M with a
  1.2 GiB hot set (the 1B budget scaled), at matched recall: 1-bit 64:1000 R@10 0.910 at 136
  QPS; 2-bit 64:100:1000 0.914 at 268 QPS; 3-bit 64:50:1000 0.917 at 194 QPS.
- **Result at 1B**, 2-bit, 12 GiB pinned, everything else cold, 10,000 queries: 40:100:1000
  R@10 0.906 at 372 QPS (p50 3.5 ms, p99 7.2 ms); 48:100:1000 0.914 at 345 QPS; 64:100:1000
  0.925 at 275 QPS; 128:200:2000 0.951 at 137 QPS (D19: 0.942 at 69 warm). Rebuild: ten adds
  at 83–88 s (1.15M rows/s, the step search adds 35%), pack 101 s, 48.3 GB of postings.
- **Where it stops.** At 48:100 the SSD streams 8.9 MB × 345 = 3.1 GB/s of postings, against a
  3.6 GB/s sequential ceiling, plus 35k page reads/s: the device, not the CPU (12 cores, about
  2.5 ms of CPU per query), is the limit. DiskANN's 1,500 QPS at 0.90 comes with 64 GB of RAM
  for its PQ codes and graph cache; at 12 GiB of RAM this design gets about a quarter of that
  throughput. The next gains are fewer bytes per probe (f16 factors save 4 of 48 B; ids implied
  by a base reordered into posting order save 4 more) or more RAM for the hot set, which pays
  back at 75%→87% coverage between 12 and 16 GiB.

### D21. Query-aware probing at 1B
D12 showed on M1 that a per-query margin on the centroid scores beats a fixed nprobe. At 1B it
is an L2 rule: of the `nprobe` nearest leaves, keep those with ‖q − c‖² ≤ (1 + m)·‖q − c₁‖²
(config `nprobe:cand:refine:mM`). The hot set is learned with the same rule.
- **The useful margins are large.** In 100 dimensions with 131,072 leaves the nearest leaf
  distances bunch up: m 0.04 keeps 1.6 leaves and gets R@10 0.597; m 0.22 keeps 14.5 (0.821).
  The range that matters is m 0.3–1.0 (27–147 leaves).
- **Result, 2-bit, 12 GiB pinned, cold, 10,000 queries.** At matched recall it gives 12–14% more
  QPS, and lower p50, than a fixed nprobe:
  0.906: fixed 40 probes 372 QPS, m 0.4 (43.8 probes on average) 420 QPS;
  0.911–0.914: fixed 48 345 QPS, 96:m0.45 413 QPS;
  0.925–0.927: fixed 64 275 QPS, 96:m0.6 314 QPS;
  0.95–0.96: fixed 128:200:2000 0.951 at 137 QPS, 192:200:2000:m1.0 0.960 at 127 QPS.
- **The cost is the tail.** Hard queries probe up to the cap, so p99 rises (fixed 40: 7.2 ms;
  m 0.4 capped at 128: 11.2 ms; capped at 96: 10.3 ms). The cap is the tail knob; a scan budget
  (D15) would bound it more tightly.

### D22. Head-to-head with DiskANN on the same laptop
The big-ann comparison in D20 set numbers from other hardware side by side. This one runs
Microsoft's DiskANN (the Rust rewrite, upstream `8a9ff2d`) on the same M4 Pro, on the same
data (the first 10M rows of SPACEV), the same 10,000 queries and the same ground truth.
- **Making it a fair fight.** DiskANN's disk index has no macOS reader: its fallback copies the
  whole index file into RAM, so an SSD run would really be an in-RAM run. A small reader
  (`bench/diskann/macos-reader.patch`, one new file) does `F_NOCACHE` + `pread`, the macOS analogue of its
  Linux `O_DIRECT` + io_uring path. Index files are evicted before each cold run
  (`examples/drop_cache.rs`). DiskANN's benchmark scores strict recall (only the first ten
  ground-truth ids count); sieve now logs that too. At 10M the two agree to 0.001.
- **In RAM, the graph wins by 45–50×.** DiskANN (R 64, L 100) reaches strict R@10 0.903 at
  146k QPS (68 µs mean at 12 threads); sieve gets 0.907 at 3.1k QPS. At ~0.945 it is 88k QPS against
  1.8k. Per query on one thread, sieve scores 222k codes at ~3 ns each, while DiskANN scores
  1,840 full vectors in 39 hops at ~28 ns each: about 9× cheaper per candidate, but 120× more candidates. Flat IVF must
  scan whole clusters, and on SPACEV the true top-10 is spread over ~32 of 1,536 of them.
  This is why my earlier projection (within ~2× of DiskANN) was wrong: I assumed cheap codes
  would offset the extra candidates, and they cannot.
- **On SSD at matched RAM (~0.5 GB), it is close.** DiskANN keeps 48-byte PQ codes in RAM
  (0.48 GB, the same as sieve's pinned postings) and reads 4 KiB graph nodes. sieve reads only
  its rerank rows. Both are I/O-bound at a similar number of reads per query (30–60 for
  DiskANN, 50 for sieve):
  - sieve has **~3× lower latency** (6–7× against DiskANN's peak-throughput setting): its 50 reads go out at once, while DiskANN's 30–60 are a
    chain of dependent hops (~85 µs each). At 0.93: sieve 1.6 ms mean, DiskANN 4.4–5.2 ms
    at 12 threads and 10–12 ms at the 48 threads it needs for peak throughput.
  - DiskANN has **~1.3× more peak throughput** at 0.93: 4,606–4,662 QPS at 0.923–0.930
    against sieve's 3,583. At low recall the gap is ~2× (7,781 QPS at 0.875 against 3,574
    at 0.839). sieve tops out at ~178k reads/s whatever its thread counts; DiskANN
    reaches ~228k.
  - At the same 12 threads, sieve is ahead on both: 0.930 at 3,313 QPS against DiskANN's
    0.923–0.930 at 2,265–2,727 (beam 1 and 4).
  - **The gap closes as recall rises.** Tuning sieve's rerank count (50, 70, 100) the way
    DiskANN's L was swept: at 0.945 sieve does 3,016 QPS against 3,355; at 0.955, 2,528
    against 2,629; at 0.97, 1,631 against 1,821. DiskANN's dependent chain grows with L
    (69–129 reads, 14–26 ms at 48 threads), while sieve's batch stays at 50–100 reads
    (1.9–2.8 ms). Above 0.94 the two are level on throughput and sieve is 7–10× faster per
    query.
  - With its whole disk index cached (~4.2 GB of RAM), DiskANN does 21k QPS at 0.923 and
    8.2k at 0.968, ~8× sieve fully in RAM: give a graph the RAM and the in-RAM result returns.
- **Build**: sieve 7 s, DiskANN 169 s (in RAM) and 211 s (disk), on 12 threads.
- **What it means.** Flat IVF is the wrong structure when the index fits in RAM. Its case is
  the SSD (latency, cheap builds, updates), and there it is level with the best graph index
  rather than ahead (behind at low recall, level above 0.94, with far lower latency). The way to close the in-RAM gap is SPANN's: many more, smaller posting
  lists, routed by a graph over their centroids.
- **Caveats.** DiskANN warns that its AArch64 kernels are unoptimised; that matters for the
  in-RAM result (DiskANN's lead would grow), not the I/O-bound SSD one. macOS has no io_uring,
  so a DiskANN beam's reads are issued one after another; beam width 1 was its best setting
  here anyway. 100M was not run. DiskANN builds it in ~6 overlapping shards (each vector in 2),
  and the merge needs the base, shard data, shard graphs and merged graph on disk at once:
  ~96–106 GB at peak, with a 52 GB final index. This disk has 59 GB free next to the 1B data.

## Milestones
- M1: 10 shards (17.6M vectors) end to end; recall and nDCG against exact flat search.
- M2: all 60 shards (113.5M); QPS, p50/p99, roofline, flamegraphs.
- M3: compressed BM25 (Exa's frequency bucketing + delta doc IDs), domain/keyword filters,
  hybrid fusion; filtered search vs HNSW across selectivities.
- M4: shard processes + coordinator; live inserts with compaction (stretch); write-up.
