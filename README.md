# sieve

A Rust vector search engine that reproduces, measures and extends the design Exa published in
[How we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db).
It serves **113.5M passages** (the TREC-RAG 2024 MS MARCO v2.1 corpus, Arctic-Embed-M v1.5)
on one 24 GB laptop.

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
- **Integer rerank** on `sdot` (int8 query × int8 row) and centroid routing on AMX via
  Accelerate.
- **One-pass streaming ingest** at 232k rows/s. Exact ground truth is computed with sgemm while
  the data streams past, and shards are downloaded, consumed and deleted so 350 GB of parquet
  never has to sit on a 512 GB disk.

## Results (M1: 18.4M passages; full 113.5M in progress)
| | R@10 | p50 | QPS (12 threads) |
|---|---|---|---|
| plain IVF, 32 probes | 0.815 | | |
| + SOAR | 0.883 | 0.76 ms | 12.8k |
| + RaBitQ residual codes | 0.894 | 0.78 ms | 13.5k (at 500 cand) |
| + 2-bit tier, 200 rows from SSD | 0.893 | | |
| ceiling of the codes (exhaustive scan + rerank): Exa's sign codes → RaBitQ | 0.949 → 0.981 | | |

Full tables, ablations and kernel numbers: [BENCHMARKS.md](BENCHMARKS.md). Every design decision
and the evidence for it: [DESIGN.md](DESIGN.md).

## Reproduce
```sh
cargo test --release
./scripts/m1.sh          # 10 shards: download, ingest, index (plain + SOAR), eval, bench
./scripts/m2.sh          # all 60 shards, streamed and deleted as they are ingested
```
Binaries: `ingest` (parquet → codes, int8 rows, ground truth), `index` (k-means, SOAR, tiers),
`search eval | routing | bench [--cold]`.

## Layout
```
src/quant.rs   normalisation, sign codes, int8 quantiser, sdot kernels
src/lut.rs     asymmetric LUT scan (NEON + scalar reference, property-tested)
src/rabitq.rs  RaBitQ rotation, residual codes, estimator and error bound
src/tier2.rs   2-bit Lloyd-Max middle tier with per-byte score tables
src/ivf.rs     k-means, SOAR assignment, index build, search, SSD row reader
src/blas.rs    Accelerate sgemm (AMX)
src/data.rs    parquet streaming, topics, qrels
src/eval.rs    recall, nDCG@10 (MaxP)
examples/      kernel, SSD and middle-tier experiments
```
