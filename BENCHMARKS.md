# Benchmarks

Machine: Apple M4 Pro (12 cores), 24 GB RAM, internal NVMe. Rust 2024, `target-cpu=native`,
fat LTO. Every number below comes from `results/*.jsonl`, written by the binaries, and can be
reproduced with `scripts/m1.sh` and `scripts/m2.sh`.

Queries: all 1,677 TREC DL21–23 topics plus the first 1,000 MS MARCO v2.1 doc dev queries
(2,677 in total; R@10 and latency use all of them). nDCG@10 uses the topics with NIST graded
labels: DL21 (53), DL22 (76), DL23 (82). *R@10* is overlap with the exact 768-d top-10 over the indexed
slice (computed during ingest, D5). *nDCG@10* is document-level MaxP against the qrels. On a
slice of the corpus it is a relative signal only, since relevant documents outside the slice
count as misses.

## M1: 10 shards, 18,380,565 passages, 4096 clusters

### Ingest
One streaming pass over 58 GB of parquet takes 79 s (232k rows/s). Of that, 43 s is exact
ground truth for 2,677 queries at 768-d and 256-d, computed with Accelerate sgemm while the
data streams past.

### Routing is the recall bottleneck, and SOAR fixes most of it
Fraction of the exact top-10 that lies in the probed clusters:

| nprobe | plain IVF | SOAR (λ=1) |
|---|---|---|
| 8 | 0.702 | 0.810 |
| 16 | 0.770 | 0.865 |
| 32 | 0.830 | 0.909 |
| 64 | 0.878 | 0.942 |
| 128 | 0.920 | 0.966 |

### End to end: 1-bit LUT scan → 1000 candidates → rerank
| config | R@10 | nDCG@10 dl21 | dl22 | dl23 | dev | vectors scanned |
|---|---|---|---|---|---|---|
| exact flat 768-d | 1.000 | 0.566 | 0.314 | 0.255 | 0.242 | 18.4M |
| exact flat 256-d prefix | 0.823 | 0.566 | 0.310 | 0.247 | 0.243 | 18.4M |
| plain IVF, 32 probes, full rerank | 0.815 | 0.536 | 0.309 | 0.227 | 0.219 | 147k |
| plain IVF, 64 probes, full rerank | 0.859 | 0.537 | 0.313 | 0.236 | 0.226 | 301k |
| **SOAR, 32 probes, full rerank** | **0.883** | 0.556 | 0.309 | 0.251 | 0.235 | 305k |
| SOAR, 64 probes, full rerank | 0.910 | 0.556 | 0.314 | 0.253 | 0.238 | 623k |
| SOAR, 128 probes, full rerank | 0.927 | 0.555 | 0.317 | 0.251 | 0.241 | 1.26M |
| SOAR, 32 probes, **2-bit tier → top 200 → int8** | **0.883** | 0.554 | 0.310 | 0.251 | 0.235 | 305k |
| SOAR, 32 probes, 2-bit tier → top 100 → int8 | 0.880 | 0.554 | 0.309 | 0.250 | 0.235 | 305k |
| SOAR, 32 probes, 2-bit tier → top 50 → int8 | 0.864 | 0.550 | 0.306 | 0.249 | 0.234 | 305k |
| SOAR, 32 probes, int8 256-d prefix rerank | 0.761 | 0.557 | 0.305 | 0.245 | 0.235 | 305k |
| SOAR, 32 probes, no rerank | 0.404 | 0.481 | 0.250 | 0.196 | 0.210 | 305k |

### RaBitQ residual codes (DESIGN.md D10): same 32 bytes per code
All rows SOAR. The left column is Exa's global sign codes, the right column RaBitQ codes of
the residual to the slot's centroid.

| config | global sign codes | RaBitQ residual |
|---|---|---|
| 8 probes, 1000 cand, full rerank | 0.795 | 0.800 |
| 16 probes, 1000 cand, full rerank | 0.845 | 0.852 |
| 32 probes, 1000 cand, full rerank | 0.883 | **0.894** |
| 64 probes, 1000 cand, full rerank | 0.910 | **0.926** |
| 32 probes, 200 cand, full rerank | 0.820 | **0.867** |
| 32 probes, 500 cand, full rerank | — | 0.888 |
| 32 probes, 1000 cand, 2-bit tier → 200 → int8 | 0.883 | 0.893 |
| 64 probes, 500 cand, 2-bit tier → 200 → int8 | — | 0.919 |
| 32 probes, no rerank | 0.404 | 0.441 |
| all probes, 1000 cand (code ceiling) | 0.949 | **0.981** |

At 32 probes, recall is now within 1.6 points of the routing ceiling (0.909). The remaining
loss is mostly routing, which SOAR and more probes address. Scan time rises 6%
(0.554 vs 0.521 ms).

### Bound-driven rerank (DESIGN.md D11)
RaBitQ index, 32 probes, 1000 candidates unless noted. *reranked* is rows per query after SOAR
dedup and pruning.

| config | R@10 | reranked |
|---|---|---|
| full rerank, no pruning | 0.8936 | 850 |
| pruned at rank 10, w = 1.0 (ε0 = 1.9) | 0.8936 | 612 |
| pruned at rank 10, w = 0.8 | 0.8926 | 446 |
| pruned at rank 10, w = 0.7 | 0.8901 | 346 |
| fixed 400 candidates | 0.8848 | 338 |
| pruned at rank 10, w = 0.6 | 0.8822 | 243 |
| fixed 250 candidates | 0.8737 | 211 |
| pruned at rank 10, w = 0.4 | 0.8277 | 90 |
| 2-bit tier → 200, no pruning | 0.8929 | 850 |
| 2-bit tier → 200, pruned at rank 10, w = 1.0 | 0.8929 | 612 |
| 64 probes, 500 cand, 2-bit tier → 200, pruned w = 1.0 | 0.9191 | 362 |

### Query-aware probing (DESIGN.md D12)
RaBitQ index, 500 candidates, full rerank. Fixed nprobe against a centroid-score margin Δ, with
the nprobe cap high enough not to bind.

| config | R@10 | vectors scanned |
|---|---|---|
| fixed 24 probes | 0.8712 | 227k |
| Δ = 0.10 (cap 128) | 0.8801 | 260k |
| fixed 32 probes | 0.8876 | 305k |
| Δ = 0.12 (cap 128) | 0.9049 | 376k |
| fixed 48 probes | 0.9082 | 463k |
| fixed 64 probes | 0.9192 | 623k |
| Δ = 0.14 (cap 256) | 0.9278 | 696k |
| Δ = 0.16 (cap 256) | 0.9387 | 925k |
| fixed 96 probes | 0.9334 | 942k |
| Δ = 0.18 (cap 256) | 0.9448 | 1.17M |
| fixed 128 probes | 0.9404 | 1.26M |
| fixed 192 probes | 0.9483 | 1.89M |
| Δ = 0.02 / 0.06 / 0.08 | 0.624 / 0.784 / 0.841 | 18k / 79k / 155k |

Ablations that show where recall comes from:
- **LUT vs Hamming**: scoring all 18.4M codes, then reranking 1000, gives R@10 0.949 with the
  asymmetric LUT and 0.884 with Hamming distance. The float query is worth 6.5 points.
- **Rerank depth**: at 32 probes, 200 candidates give 0.820 and 1000 give 0.883.
- **SOAR at equal scan cost**: at about 300k vectors scanned, SOAR reaches 0.883 and plain IVF
  0.859. At about 150k it is 0.845 against 0.815.
- **Middle tier**: 2-bit codes of the 256-d prefix (64 B/vector, in RAM) choose the 200 rows
  worth fetching from SSD with no recall loss (0.8828 vs 0.8831). See DESIGN.md D9.

### Latency and throughput (warm, index resident)
Single-thread latency is measured over all 2,677 queries. QPS is 12 threads for 10 s.

| config | p50 | p99 | route | scan | rerank | QPS |
|---|---|---|---|---|---|---|
| SOAR 16 probes, full rerank | 0.76 ms | 6.29 ms | 0.064 | 0.29 | 0.91 | 17.8k |
| SOAR 32 probes, full rerank | 0.76 ms | 1.49 ms | 0.065 | 0.51 | 0.24 | 12.8k |
| SOAR 64 probes, full rerank | 1.24 ms | 2.47 ms | 0.068 | 1.00 | 0.26 | 7.6k |

These rows predate two later changes: routing on AMX (sgemm, 65 → 28 µs) and the rerank
prefetch. The 16-probe p99 includes page faults on rerank rows. They will be re-run on an idle
machine after the M2 ingest.

### SSD mode (DESIGN.md D13): posting lists on disk
`index --pack` turns the RaBitQ index into 1.63 GB of 4 KiB-aligned cluster records. RAM holds
centroids, offsets and an optional cluster cache. Recall equals RAM mode for the same scan; the
difference is the last tier, which picks SSD rows by code estimate (`ssdN`) because there is no
2-bit RAM tier.

| config | R@10 |
|---|---|
| 32 probes, 500 cand, 2-bit RAM tier → 200 (RAM mode) | 0.893 |
| 32 probes, 500 cand, top 200 by estimate | 0.870 |
| 32 probes, 500 cand, top 300 / 400 | 0.881 / 0.886 |
| 64 probes, 500 cand, top 300 / 400 | 0.912 / 0.917 |
| 64 probes, 1000 cand, top 500 | 0.921 |
| Δ = 0.12 (cap 128), 500 cand, top 300 | 0.898 |

Cold, every byte from SSD (`bench --cold` evicts both files; latency is single-threaded, QPS
is 12 threads). **Preliminary**: the M2 ingest was writing to the same SSD, which then served
115k random reads/s at QD64 against 215k idle. Re-run pending on an idle machine.

| config | cache | p50 | p99 | fetch | rerank | MB read/query | QPS | peak RSS |
|---|---|---|---|---|---|---|---|---|
| 64:500:ssd300 | none | 13.2 ms | 22.3 ms | 8.1 ms | 4.4 ms | 27.4 | 85 | 468 MB |
| 64:500:ssd300 | 512 MB (hit 0.55) | 8.8 ms | | | | 11.4 | 139 | 1.34 GB |
| 256:500:ssd300:p0.12 | none | 7.4 ms | 50 ms | | | 21.6 | 84 | |

### Kernels (`cargo run --release --example kernel`, `--example ssd`)
- 1-bit LUT scan (`vqtbl1q_u8`, 64 tables × 16 entries, u16 accumulators): ~950M codes/s per
  core, flat from a 1 MiB to a 1 GiB working set, so it is compute-bound, not bandwidth-bound.
- int8 × int8 rerank on `sdot`: 0.068 ms for 1000 × 768-d rows (was 0.17 ms with i8→f32).
- SSD, random 768 B `pread` with `F_NOCACHE`: 18.7k/s at QD1 (53 µs), 71.7k/s at QD4,
  176k/s at QD16, 215k/s at QD64.

## M2: 60 shards, 113,520,750 passages, 16384 clusters

Pending: ingest is in progress, streaming each shard off the network and deleting it once read.
