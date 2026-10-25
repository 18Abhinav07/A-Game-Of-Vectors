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
- Each slot stores `[mul, err]`, 8 B more per slot (16 B per vector with SOAR). `err` is not used
  yet; it is stored for bound-driven rerank.
- The two SOAR copies of a vector now carry different codes and scores, so dedup keeps the
  better copy.
- M1, 32 probes: R@10 0.883 → 0.894 at 1000 candidates, 0.820 → 0.867 at 200. 500 RaBitQ
  candidates (0.888) beat 1000 global-code candidates (0.883), halving rerank work. At 64 probes:
  0.910 → 0.926.

## Milestones
- M1: 10 shards (17.6M vectors) end to end; recall and nDCG against exact flat search.
- M2: all 60 shards (113.5M); QPS, p50/p99, roofline, flamegraphs.
- M3: compressed BM25 (Exa's frequency bucketing + delta doc IDs), domain/keyword filters,
  hybrid fusion; filtered search vs HNSW across selectivities.
- M4: shard processes + coordinator; live inserts with compaction (stretch); write-up.
