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
- **Balanced k-means** (D14, a negative result): splitting large clusters during training barely
  changes the corpus skew (max stored cluster 45k → 42k at r = 1.5) and gives no recall at equal
  scan.

  | index | 16:500:full R@10 / scanned | 32:500:full | 64:500:tiered200 |
  |---|---|---|---|
  | baseline | 0.8464 / 151k | 0.8876 / 305k | 0.9191 / 623k |
  | `--balance 2.0` | 0.8521 / 154k | 0.8885 / 309k | 0.9188 / 619k |
  | `--balance 1.5` | 0.8525 / 170k | 0.8905 / 341k | 0.9197 / 683k |

### Latency and throughput (warm, index resident)
Single-thread latency is measured over all 2,677 queries. QPS is 12 threads for 10 s.

| config | p50 | p99 | route | scan | rerank | QPS |
|---|---|---|---|---|---|---|
| SOAR 16 probes, full rerank | 0.76 ms | 6.29 ms | 0.064 | 0.29 | 0.91 | 17.8k |
| SOAR 32 probes, full rerank | 0.76 ms | 1.49 ms | 0.065 | 0.51 | 0.24 | 12.8k |
| SOAR 64 probes, full rerank | 1.24 ms | 2.47 ms | 0.068 | 1.00 | 0.26 | 7.6k |

These rows predate two later changes: routing on AMX (sgemm, 65 → 28 µs) and the rerank
prefetch. The 16-probe p99 includes page faults on rerank rows. A re-run after the M2 ingest
(`results/m1-lowpower.jsonl`) turned out to be on battery in Low Power Mode: the scan stage at
64 probes took 1.64 ms against 1.00 ms here, and 12-thread QPS fell 3×. So those numbers are kept
only as a same-session reference for the FAISS comparison below. The on-AC re-run is the next
section.

### On AC power (`scripts/ac-bench.sh`, `results/m1*-ac.jsonl`)
The re-run of every M1 latency marked "Low Power Mode" above and below, on AC power with Low
Power Mode off and the machine otherwise idle. Recall is identical to the earlier runs in
every row; only times change. Latency is single-thread over 2,677 queries, QPS is 12 threads.

| config | | p50 | p99 | scan | fetch | rerank | QPS | peak RSS |
|---|---|---|---|---|---|---|---|---|
| 16:500:full | warm | 0.37 ms | 1.61 ms | 0.32 | | 0.19 | 23,687 | 8.1 GB |
| 32:500:full | warm | 0.62 ms | 0.90 ms | 0.58 | | 0.04 | 12,527 | 8.1 GB |
| 64:500:full | warm | 1.16 ms | 1.62 ms | 1.12 | | 0.04 | 5,962 | 8.0 GB |
| 32:1000:tiered200 | warm | 1.03 ms | 1.37 ms | 0.65 | | 0.37 | 3,677 | 2.8 GB |
| 64:500:tiered200 | warm | 1.44 ms | 1.90 ms | 1.13 | | 0.31 | 3,273 | 2.8 GB |
| 256:500:tiered200:p0.12 | warm | 0.69 ms | 5.47 ms | 0.91 | | 0.31 | 3,660 | 2.8 GB |
| 64:500:tiered200 | cold | 4.06 ms | 5.52 ms | 1.10 | | 2.98 | 471 | 2.8 GB |
| 256:500:tiered200:p0.12 | cold | 3.48 ms | 8.22 ms | 0.88 | | 2.98 | 473 | 2.8 GB |
| 64:500:ssd300, SSD mode | cold | 12.2 ms | 25.8 ms | 8.49 | 7.08 | 4.06 | 87 | 155 MB |
| same, 512 MB cluster cache | cold | 7.52 ms | 19.4 ms | 3.73 | 2.23 | 4.05 | 137 | 1.19 GB |
| 256:500:ssd300:p0.12, SSD mode | cold | 6.73 ms | 41.1 ms | 6.42 | 5.34 | 4.10 | 94 | 156 MB |
| same, 512 MB cluster cache | cold | 5.32 ms | 19.1 ms | 2.57 | 1.43 | 3.88 | 177 | 1.18 GB |
| 64:500:ssd300:s1000k, SSD mode | cold | 12.1 ms | 17.0 ms | 8.34 | 6.92 | 3.97 | 78 | 159 MB |

Full rerank cold (the 15 GB of int8 rows evicted) is 15–16 ms p50, almost all of it faulting
rows back in; the tiered rerank reads 200 rows instead of 500 and stays at 4 ms.

Sharding (D16), 64:500:tiered200, warm, all processes on this machine over loopback:

| | p50 | p99 | QPS | shard wait per query |
|---|---|---|---|---|
| single process | 1.45 ms | 1.93 ms | 3,344 | |
| 4 shards, one-shot | 2.53 ms | 5.24 ms | 599 | 2.57 ms |
| 4 shards, two-phase | **0.91 ms** | **1.74 ms** | 2,869 | 0.93 ms |

Live updates (D17), 64:500:full p50: baseline 1.28 ms, 10% deleted 1.27, reinserted 1.84,
replayed from the log 1.43, compacted 1.28. Delete 1.84M rows 0.76 s; insert 26.5 s (69k rows/s,
10k-row batches with fsync); replay 8.4 s; compaction 3.6 s.

Filters (D18), auto plan p50 at 100% / 50% / 10% / 1% / 0.1% allowed: 1.15 / 1.19 / 4.19 / 2.24 /
0.22 ms. BM25 (exhaustive TAAT): 16.8 ms p50, 80 ms p99.

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
| 64:500:ssd300, scan while reading | none | 11.8 ms | 18.4 ms | 6.8 ms blocked | 4.1 ms | 27.4 | 86 | 303 MB |
| 256:500:ssd300:p0.12, scan while reading | none | 7.2 ms | 51 ms | 5.9 ms blocked | 4.8 ms | 21.6 | 101 | 296 MB |

### Scan budget (DESIGN.md D15): bounding margin probing's tail
Vectors scanned per query, mean / p99 / max, and R@10 (tiered200 in RAM mode, ssd300 in SSD
mode). Latency is from one session on battery in Low Power Mode; compare rows, not absolute values.

| config | R@10 | scanned mean / p99 / max | warm p50 / p99 |
|---|---|---|---|
| fixed 64 probes | 0.9191 | 623k / 880k / 975k | 2.10 / 2.81 ms |
| fixed 96 probes | 0.9332 | 942k / 1.28M / 1.41M | 2.87 / 3.78 ms |
| Δ = 0.12, cap 256 | 0.9099 | 491k / 3.01M / 3.30M | 0.97 / 8.20 ms |
| Δ = 0.16, cap 256 | 0.9385 | 925k / 3.15M / 3.30M | |
| Δ = 0.12, budget 1M | 0.9016 | 328k / 999k / 1.00M | |
| **Δ = 0.16, budget 1M** | **0.9244** | 536k / 1.00M / 1.00M | 1.70 / 3.33 ms |
| Δ = 0.16, budget 1.2M | 0.9275 | 601k / 1.20M / 1.20M | |
| **Δ = 0.20, budget 1.2M** | **0.9358** | 836k / 1.20M / 1.20M | 3.43 / 3.88 ms |

SSD mode, cold, no cache:

| config | R@10 | p50 | p99 | QPS |
|---|---|---|---|---|
| 64:500:ssd300 | 0.9120 | 13.1 ms | 31.5 ms | 90 |
| Δ = 0.12 | 0.9024 | 6.7 ms | 41.7 ms | 99 |
| Δ = 0.16, budget 1M | 0.9173 | 11.5 ms | 32.5 ms | 97 |
| **Δ = 0.16, budget 800k** | **0.9122** | 10.6 ms | **17.7 ms** | 109 |

Δ = 0.20 with 1.2M dominates fixed 96 probes on recall, mean scan and worst-case scan, but its
warm p50 is higher (3.43 vs 2.87 ms): many of its queries sit at the budget, and 256 small
probes cost more per slot than 96 large ones.

### Sharding (DESIGN.md D16): 4 shard processes and a coordinator
Clusters are split across 4 shard processes by stored slots (9.21M slots each). The coordinator
routes and merges. R@10 vs exact 768-d search; "reranked" counts rows entering the rerank, per
query, summed over shards.

| config | single node | 4 shards, one-shot | one-shot, budgets ÷ 4 | 4 shards, two-phase |
|---|---|---|---|---|
| 32:500:full | 0.8876 (423 rows) | 0.8959 (1954) | 0.8789 (490, 32:125) | **0.8879** (440) |
| 64:500:tiered200 | 0.9191 (415) | 0.9282 (1945) | 0.9101 (488, 64:125:tiered50) | **0.9194** (433) |
| 64:500:ssd300 | 0.9120 (300 read) | 0.9251 (≤1200 read) | 0.8973 (300 read, ssd75) | **0.9120** (300 read) |
| 64:500:full:p0.16:s1000k | 0.9115 | 0.9202 | | **0.9118** |
| 64:500:ssd300:b10x0.5 | 0.8932 | | | **0.8932** |
| 64:500:none | 0.4536 | | | **0.4536** |

Latency, RAM mode, warm, 64:500:tiered200, everything on one 12-core machine over loopback (Low
Power Mode, same session, so compare rows):

| | p50 | p99 | QPS | shard wait per query |
|---|---|---|---|---|
| single process | 2.13 ms | 2.95 ms | 1677 | |
| 4 shards, one-shot | 1.90 ms | 2.61 ms | 735 | 1.87 ms |
| 4 shards, two-phase | **1.20 ms** | **1.93 ms** | 1380 | 1.19 ms |

### Live updates (DESIGN.md D17): delete 10%, reinsert through the delta path, restart, compact
1,838,056 random rows (10%) are deleted, then reinserted under their own ids through the delta
path. R@10 is against the exact top-10 of the rows live at each step. Latency is single-thread,
warm, in Low Power Mode, so compare rows.

| step | full | tiered200 | ssd300 | ssd300, SSD mode | scanned per query | p50 (full) |
|---|---|---|---|---|---|---|
| baseline | 0.9192 | 0.9191 | 0.9120 | 0.9120 | 622,762 | 1.67 ms |
| 10% deleted (tombstones) | 0.9183 | 0.9182 | 0.9111 | 0.9111 | 622,762 | 1.70 ms |
| reinserted (1.84M rows in delta lists) | **0.9192** | **0.9191** | **0.9120** | **0.9120** | 685,851 | 2.08 ms |
| restarted from the log | 0.9192 | 0.9191 | 0.9120 | 0.9120 | 685,851 | 2.29 ms |
| compacted | 0.9192 | 0.9191 | 0.9120 | 0.9120 | 622,762 | 1.80 ms |

| operation | |
|---|---|
| delete 1.84M (184 batches, fsync each) | 0.75 s |
| insert 1.84M (10k per batch, fsync each) | 18.3 s, 100k rows/s |
| log size | 1.44 GB (777 B per insert) |
| replay on open | 12.0 s |
| compaction (merge 36.8M slots, drop 3.68M dead) | 8.2 s, plus the posting-list pack |

### Filters (DESIGN.md D18): domain allowlists from 100% to 0.1% of rows
Whole hosts are allowed until they hold the target fraction of rows. R@10 is against the exact
filtered top-10 (int8 scores), over every 4th query (670). Config 64:500:full. Latency is
single-thread, warm, Low Power Mode, so compare rows.

| allowed rows (hosts) | post | expand 20k | expand 60k | expand 200k | brute | **auto** |
|---|---|---|---|---|---|---|
| 100%: 18,380,565 (385,713) | 0.9222, 1.66 ms | 0.6622, 0.26 ms | 0.7737, 0.38 ms | 0.8615, 0.74 ms | - | **post: 0.9222, 1.68 ms** |
| 50%: 9,190,342 (205,897) | 0.9110, 1.78 ms | 0.7167, 0.35 ms | 0.8169, 0.59 ms | 0.8918, 1.35 ms | - | **post: 0.9110, 1.77 ms** |
| 10%: 2,073,361 (35,652) | 0.8867, 2.03 ms | 0.8037, 1.03 ms | 0.8760, 2.31 ms | 0.9366, 6.18 ms | - | **expand 200k: 0.9366, 6.19 ms** |
| 1%: 183,811 (1,389) | 0.7776, 2.97 ms | 0.8775, 15.3 ms | 0.9213, 23.7 ms | 0.9499, 46.7 ms | 0.9999, 3.21 ms | **brute: 0.9999, 3.21 ms** |
| 0.1%: 18,390 (659) | 0.7712, 1.99 ms | 0.9879, 56.6 ms | 0.9888, 93.5 ms | 0.9888, 93.4 ms | 0.9987, 0.33 ms | **brute: 0.9987, 0.34 ms** |

Probes for the expanding plan at 200k allowed slots: 22, 45, 208, 1,588, 3,322 (down the rows).
Exact filtered truth for 670 queries: 33 s at 100%, 0.2 s at 0.1%.

### Hybrid (DESIGN.md D18): BM25, dense, and weighted reciprocal rank fusion
nDCG@10 (MaxP), all 2,677 queries. RRF k 60 over the top 100 of each list; "x w" is the weight
on the BM25 list.

| run | R@10 vs 768 | DL21 | DL22 | DL23 | dev |
|---|---|---|---|---|---|
| BM25 (k1 0.9, b 0.4) | 0.1896 | 0.4222 | 0.1928 | 0.1706 | 0.1653 |
| exact 768-d | 1.0000 | 0.5657 | 0.3142 | 0.2551 | 0.2420 |
| rrf(BM25 x1, exact) | 0.4552 | 0.5361 | 0.2982 | 0.2547 | 0.2239 |
| rrf(BM25 x0.5, exact) | 0.6431 | 0.5598 | 0.3166 | 0.2665 | 0.2371 |
| rrf(BM25 x0.3, exact) | 0.7583 | 0.5635 | 0.3162 | **0.2694** | 0.2408 |
| rrf(BM25 x0.2, exact) | 0.8318 | **0.5666** | **0.3166** | 0.2665 | **0.2431** |
| rrf(BM25 x0.1, exact) | 0.9154 | 0.5679 | 0.3117 | 0.2587 | 0.2431 |
| ivf 64:500:tiered200 | 0.9191 | 0.5564 | 0.3130 | 0.2512 | 0.2378 |
| rrf(BM25 x1, ivf) | 0.4474 | 0.5346 | 0.2973 | 0.2539 | 0.2216 |
| rrf(BM25 x0.3, ivf) | 0.7164 | 0.5560 | 0.3158 | 0.2647 | 0.2376 |
| rrf(BM25 x0.2, ivf) | 0.7825 | 0.5567 | 0.3148 | 0.2614 | 0.2395 |

| BM25 index | |
|---|---|
| build (10 files, 12 threads) | 137 s, 7.5 GB peak |
| terms / postings / on disk | 6.32M / 1.54B / 3.3 GB |
| query (TAAT, single-thread) | p50 24.5 ms, p99 124 ms, 6.6M postings decoded |

### Against FAISS (same slice, same code budget, same rerank)
`scripts/faiss_baseline.py` (faiss-cpu 1.15.1) indexes the same 256-d prefixes with 4096 IVF
lists and 32 B codes, then reranks the top 500 against the same 768-d int8 rows, so only the
index and the code differ. R@10 is against exact 768-d search. Latency is single-threaded, one
query at a time, and covers FAISS's **code scan only**. sieve's numbers include routing and the
rerank. Both were measured in the same session, on battery in Low Power Mode, so compare the
ratios, not the absolute values.

| index | probes | R@10 | p50 | 12-thread QPS |
|---|---|---|---|---|
| FAISS IVF4096,PQ32 | 64 / 128 | 0.861 / 0.893 | 2.96 / 5.79 ms | 2.3k / 1.1k (scan) |
| FAISS IVF4096,PQ64x4fs (fast-scan) | 64 / 128 | 0.852 / 0.883 | 0.88 / 1.54 ms | 11.7k / 6.9k (scan) |
| FAISS IVF4096,RaBitQ (40 B) | 64 / 128 | 0.868 / 0.903 | 4.36 / 8.66 ms | 1.4k / 0.7k (scan) |
| **sieve** SOAR + RaBitQ, 500 cand, full rerank | 16 / 32 | 0.846 / **0.888** | 0.34 / **0.63 ms** | 22k / 6.1k (end to end) |
| **sieve** + 2-bit tier → 200 rows | 64 | **0.919** | 2.10 ms | 1.8k |

- **At equal recall, sieve is 2.4× faster than FAISS's fastest IVF code.** For 0.883–0.888,
  FAISS fast-scan needs 128 probes and 1.54 ms just to scan, against 0.63 ms end to end for sieve
  at 32 probes. FAISS's own `IVF,RaBitQ` needs about 100 probes and ~7 ms. sieve evaluates the same
  estimator through the 32-document `vqtbl1q_u8` LUT kernel.
- **The gain is mostly SOAR.** sieve stores each vector twice, so 32 of its probes scan about as
  many slots as 64 FAISS probes. At that roughly equal scan, R@10 is 0.888 against 0.868 for
  FAISS RaBitQ and 0.852 for fast-scan PQ. The price is 2× the code memory: 1.18 GB against
  588 MB of PQ codes.
- **No FAISS IVF index here reaches sieve's 0.919** below 128 probes. FAISS RaBitQ reaches 0.903 at
  128 probes and 8.7 ms of scan.
- Build: FAISS trains and adds in 2.5–3.8 min (12 threads). sieve's `index` takes 1.7 min,
  including SOAR and the 2-bit tier (`results/faiss-m1.jsonl`).

### Kernels (`cargo run --release --example kernel`, `--example ssd`)
- 1-bit LUT scan (`vqtbl1q_u8`, 64 tables × 16 entries, u16 accumulators): ~950M codes/s per
  core, flat from a 1 MiB to a 1 GiB working set, so it is compute-bound, not bandwidth-bound.
- int8 × int8 rerank on `sdot`: 0.068 ms for 1000 × 768-d rows (was 0.17 ms with i8→f32).
- SSD, random 768 B `pread` with `F_NOCACHE`: 18.7k/s at QD1 (53 µs), 71.7k/s at QD4,
  176k/s at QD16, 215k/s at QD64.

## M2: 60 shards, 113,520,750 passages, 16384 clusters (the full corpus)

Index: RaBitQ residual codes with SOAR (λ = 1, 2.002 stored per vector) and the 2-bit tier.
On the full corpus, nDCG@10 is an absolute number, no longer a relative signal.

### Build
| step | time |
|---|---|
| download + ingest, 60 shards (network-bound, shards deleted as read) | 3 h 19 min |
| of which exact ground truth, 2,677 queries × 113.5M at 768-d and 256-d (sgemm) | 356 s |
| index: k-means (1M sample, 12 iters), SOAR, RaBitQ, 2-bit tier | 13 min |
| `index --pack`: 10.03 GB of posting lists | 9 s |

Cluster sizes: p50 13,244, p99 35,635, max 85,003, min 14. The 6× spread between p50 and max
drives the p99 tail of the scan, and is the case for balanced k-means.

### Routing: fraction of the exact top-10 inside the probed clusters
| nprobe | % of clusters | reachable |
|---|---|---|
| 8 | 0.05% | 0.818 |
| 16 | 0.10% | 0.865 |
| 32 | 0.20% | 0.904 |
| 64 | 0.39% | 0.935 |
| 128 | 0.78% | 0.957 |
| 256 | 1.56% | 0.974 |
| 512 | 3.12% | 0.984 |

With 4× the clusters for 6.2× the data, routing at a given nprobe is about where M1 was
(0.909 → 0.904 at 32 probes), while 32 probes now cover 0.20% of the clusters instead of 0.78%.

### End to end
| config | R@10 | nDCG@10 dl21 | dl22 | dl23 | dev | scanned | reranked |
|---|---|---|---|---|---|---|---|
| exact flat 768-d | 1.000 | 0.694 | 0.552 | 0.470 | 0.334 | 113.5M | |
| exact flat 256-d prefix | 0.808 | 0.693 | 0.556 | 0.452 | 0.331 | 113.5M | |
| 16 probes, 1000 cand, 2-bit tier → 200 | 0.848 | 0.672 | 0.534 | 0.432 | 0.303 | 231k | 870 |
| 32 probes, 1000 cand, 2-bit tier → 200 | 0.886 | 0.681 | 0.542 | 0.445 | 0.314 | 457k | 851 |
| 32 probes, 500 cand, 2-bit tier → 200 | 0.879 | 0.680 | 0.541 | 0.442 | 0.314 | 457k | 426 |
| **64 probes, 500 cand, 2-bit tier → 200** | **0.909** | 0.686 | 0.547 | 0.457 | 0.323 | 913k | 418 |
| 64 probes, 1000 cand, 2-bit tier → 200 | 0.916 | 0.688 | 0.548 | 0.457 | 0.324 | 913k | 834 |
| 64 probes, 1000 cand, 2-bit tier → 100 | 0.909 | 0.688 | 0.548 | 0.456 | 0.324 | 913k | 834 |
| 128 probes, 1000 cand, 2-bit tier → 200 | 0.937 | 0.693 | 0.551 | 0.460 | 0.328 | 1.84M | 821 |
| 256 probes, 1000 cand, 2-bit tier → 200 | 0.952 | 0.692 | 0.551 | 0.465 | 0.332 | 3.71M | 811 |

At 128 probes (1.6% of the corpus scanned) nDCG@10 is within 0.011 of exact search on every
query set; at 256 probes, within 0.006.

### Bound-driven rerank and query-aware probing at 113.5M
| config | R@10 | scanned | reranked |
|---|---|---|---|
| 64 probes, 500 cand, 2-bit tier → 200 | 0.9090 | 913k | 418 |
| same, bound-pruned w = 1.0 | 0.9090 | 913k | 380 |
| 64 probes, 1000 cand, full rerank, bound-pruned w = 0.8 | 0.9159 | 913k | 487 |
| Δ = 0.08 (cap 256), 500 cand, 2-bit tier → 200 | 0.8499 | 268k | 455 |
| Δ = 0.10 | 0.8866 | 487k | 443 |
| Δ = 0.12 | 0.9100 | 769k | 433 |
| Δ = 0.14 | 0.9235 | 1.09M | 425 |
| Δ = 0.16 | 0.9340 | 1.45M | 419 |

Pruning stays lossless at w = 1 (9% fewer rows). At w = 0.8, a full-precision rerank of the
487 surviving rows matches the 1000-candidate tiered run (0.9159 vs 0.9160). Margin probing still beats fixed nprobe at equal scan: Δ = 0.12 reaches
the 64-probe recall (0.910) with 16% less scan, and Δ = 0.16 lands between 64 and 128 probes
at 79% of the 128-probe scan.

### SSD mode at 113.5M
| config | R@10 |
|---|---|
| 64 probes, 500 cand, top 300 by estimate | 0.9021 |
| 64 probes, 500 cand, top 400 | 0.9078 |
| 128 probes, 1000 cand, top 400 | 0.9282 |
| Δ = 0.12 (cap 256), 500 cand, top 400 | 0.9074 |

### Latency and throughput (cold: rerank rows and posting lists come from the SSD)
Single-thread latency over 2,677 queries; QPS is 12 threads for 10 s. The SSD was idle
(ingest finished). rerank.i8 is 87 GB, so RAM mode reads the final tier from the SSD regardless.

| config | p50 | p99 | route | scan | 2-bit | rerank | QPS | peak RSS |
|---|---|---|---|---|---|---|---|---|
| RAM, 32:500:tiered200 | 4.07 ms | 8.67 ms | 0.09 | 0.91 | 0.34 | 3.24 | 465 | 13.9 GB |
| RAM, **64:500:tiered200** | 4.82 ms | 7.39 ms | 0.12 | 1.75 | 0.14 | 3.09 | 460 | 14.2 GB |
| RAM, 64:500:tiered200:b10 | 4.71 ms | 6.68 ms | 0.07 | 1.60 | 0.23 | 3.08 | 488 | 14.0 GB |
| RAM, 64:1000:tiered200 | 5.79 ms | 17.2 ms | 0.12 | 2.06 | 1.44 | 4.37 | 420 | 15.2 GB |
| RAM, 128:1000:tiered200 | 9.28 ms | 26.1 ms | 0.14 | 4.48 | 2.97 | 5.91 | 308 | 15.5 GB |
| RAM, 256:500:tiered200:p0.12 | 3.89 ms | 10.8 ms | 0.11 | 1.37 | 0.15 | 3.08 | 474 | 14.1 GB |

| SSD mode | cache | p50 | p99 | blocked on reads | rerank | MB read/query | QPS | peak RSS |
|---|---|---|---|---|---|---|---|---|
| 64:500:ssd300 | none | 13.2 ms | 17.2 ms | 6.7 ms | 3.9 ms | 40.2 | 90 | 225 MB\* |
| 64:500:ssd300 | 1 GB (hit 0.26) | 11.1 ms | 15.9 ms | 4.2 ms | 3.9 ms | 29.1 | 108 | 3.3 GB |
| 256:500:ssd400:p0.12 | none | 7.8 ms | 40.7 ms | 5.3 ms | 4.8 ms | 33.8 | 93 | 1.1 GB\* |
| 256:500:ssd400:p0.12 | 1 GB (hit 0.34) | 7.2 ms | 28.8 ms | 2.7 ms | 4.7 ms | 21.8 | 117 | 3.3 GB |

\* The pipeline run measured 1.1–1.2 GB with no cache. 908 MB of that was two norm arrays that
SSD mode never reads, loaded eagerly; they are now memory-mapped and the measured peak is
225 MB (`results/m2-lazy.jsonl`, A/B against the old binary at the same moment: 1,091 → 225 MB).
Later runs of 64:500:ssd300 measured p50 15.0–16.0 ms with both binaries while a Time Machine
backup was running, so treat SSD-mode latency as ±2 ms.

What this says:
- **RAM mode serves the full corpus at R@10 0.909 in 4.8 ms p50 and 460 QPS**, holding 14 GB of
  codes, ids, factors and the 2-bit tier in RAM and reading 200 int8 rows per query from SSD.
  Rerank (SSD reads) is 64% of the latency.
- **SSD mode serves the same corpus at R@10 0.902 in 225 MB** of RAM, 63× less, at 13 ms p50
  and about 90–117 QPS. At 40 MB per query and ~3.6 GB/s, it is bound by SSD bandwidth.
- The 1 GB cluster cache (10% of the lists) hits 26% at fixed nprobe and 34% with margin
  probing, whose probe set concentrates on fewer, hotter clusters.
- Margin probing gives the best median in both modes (3.89 ms RAM, 7.2 ms SSD) but the worst
  tail: flat-score queries probe up to the 256 cap, into large clusters.

## SPACEV-1B: one billion vectors (DESIGN.md D19)

MS SPACEV-1B from big-ann-benchmarks (NeurIPS'21): 1B × 100-d int8, L2. Queries are the public
query set, with each prefix's own ground truth (10M and 100M: first 5,000 queries; 1B: 10,000
warm, 2,000 cold). R@10 is tie-aware as in big-ann: a result counts if it is any ground-truth
row at most as far as the 10th (30.8% of 1B queries have a tie at rank 10). Index: RaBitQ
128-bit residual codes, two-level k-means leaves, top-500/1000/2000 candidates reranked exactly
from the int8 base chunks with pooled `pread`. Logs: `results/spacev.jsonl`,
`results/spacev-dev.jsonl`; reproduce with `scripts/spacev.sh` (download) and
`scripts/spacev-1b.sh` (build and search).

### Build
| | 10M | 100M | 1B |
|---|---|---|---|
| leaves (K1 × K2) | 64 × 24 = 1,536 | 128 × 112 = 14,336 | 512 × 256 = 131,072 |
| train | 4.6 s | 64 s | 326 s |
| add (encode + run files) | 3.4 s | 50 s | 10 × 54–60 s (1.8M rows/s) |
| pack | 0.2 s | 2 s | 38 s |
| postings on SSD | 0.28 GB | 2.83 GB | 28.32 GB |
| largest leaf (mean) | 99,023 (6,510) | 51,396 (6,975) | 129,388 (7,629) |
| rerank store | first 1 GB of chunk 0 | 10 GB of chunks | 100 GB of chunks |

Download: ten 10 GB chunks at 303–325 s each (16 parallel range requests), overlapped with the
adds. At 1B, search peak RSS is 298 MB (`/usr/bin/time -l`, 64:500 cold).

### Search
| nprobe:cand | 100M R@10 | p50 | p99 | QPS | 1B R@10 | p50 | p99 | QPS |
|---|---|---|---|---|---|---|---|---|
| 32:500 | 0.864 | 1.27 ms | 3.66 ms | 1,353 | | | | |
| 64:500 | 0.899 | 2.15 ms | 3.59 ms | 842 | 0.911 | 7.25 ms | 9.87 ms | 147 |
| 64:1000 | 0.911 | 3.46 ms | 6.72 ms | 414 | 0.922 | 12.08 ms | 15.94 ms | 86 |
| 128:1000 | 0.938 | 4.95 ms | 11.9 ms | 374 | 0.942 | 14.56 ms | 24.2 ms | 69 |
| 256:2000 | 0.965 | 7.42 ms | 10.6 ms | 230 | 0.964 | 28.19 ms | 37.3 ms | 35 |
| 64:1000 cold | 0.910 | 14.2 ms | 18.5 ms | 56 | 0.926 | 14.15 ms | 21.1 ms | 73 |
| 128:1000 cold | 0.938 | 20.3 ms | 42.6 ms | 51 | 0.944 | 18.20 ms | 25.4 ms | 55 |

1B stage means (ms) for the warm runs: route 0.37–0.47, scan 0.97–3.60, posting I/O 1.67–7.67,
rerank 4.28–16.60. Cold uses a different 2,000-query sample, so its recall is not comparable
to warm to the third decimal. 100M warm ran right after the 1B runs, with the page cache full
of 1B data.

At 10M (1,536 leaves, all rows cached): 64:500 R@10 0.927 at 0.74 ms p50 and 3,241 QPS;
64:1000 0.940; 256:2000 0.985 at 2.82 ms. Cold 64:1000: 13.7 ms p50, 9.7 ms of it rerank.

What this says:
- **One laptop serves 1B vectors at R@10 0.91–0.96 in 7–28 ms, in 298 MB of RAM.** Recall at a
  fixed config does not degrade from 100M to 1B, because the leaf count scaled with N.
- **The exact rerank is the bottleneck at 1B.** 100 GB of chunks cannot be cached in 24 GB, so
  rerank cost is linear in candidates (4.3 ms per 500) and warm ≈ cold.
- **Compressed rerank rows were not good enough.** 4-bit residual rows (54 B per vector) gave
  R@10 0.850 at 10M 64:500 against 0.927 exact. Their score error std (206) is far above the
  10th-to-11th neighbour gap (median 60 at 10M, 26 at 1B).

### At the device limit (DESIGN.md D20): page reads, a 12 GiB pinned hot set, 2-bit refinement
Device ceilings on the uncached postings file (`examples/iobench.rs`): random 100 B or 512 B
reads 138k/s, random 4 KiB 223k/s (64 threads) and 241k/s (128), random 8 KiB 185k/s,
sequential 256 KiB 3.6 GB/s. The hot set is learned from held-out queries 10,000–29,315 by
hits per byte at each run's nprobe; queries 0–9,999 are evaluated. Everything not pinned is
read with F_NOCACHE, including the rerank rows. Config: `nprobe:cand[:refine]`.

Offline, 1B, 1,000 held-out queries, nprobe 64, R@10 after reranking the top-k by each estimate
(`examples/spacev_refine.rs`):

| estimate (rms error) | 20 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|
| 1-bit (1,899) | 0.618 | 0.771 | 0.841 | 0.887 | 0.917 |
| 2-bit (869) | 0.797 | 0.889 | 0.917 | 0.928 | 0.932 |
| 3-bit (478) | 0.898 | 0.928 | 0.931 | 0.932 | 0.932 |
| 4-bit (264) | 0.923 | 0.932 | 0.932 | 0.932 | 0.932 |

100M, bit width at equal RAM (1.2 GiB pinned, cold otherwise):

| bits (slot) | config | R@10 | p50 | p99 | QPS | MB/q |
|---|---|---|---|---|---|---|
| 1 (28 B) | 64:500 | 0.898 | 4.70 ms | 63.4 ms | 264 | 3.6 |
| 1 | 64:1000 | 0.910 | 6.77 ms | 16.5 ms | 136 | 3.6 |
| 2 (48 B) | 48:100:1000 | 0.899 | 3.41 ms | 9.99 ms | 333 | 8.1 |
| 2 | 64:100:1000 | 0.914 | 4.08 ms | 10.6 ms | 268 | 10.7 |
| 3 (64 B) | 48:50:1000 | 0.902 | 4.15 ms | 9.86 ms | 266 | 12.8 |
| 3 | 64:50:1000 | 0.917 | 5.37 ms | 13.5 ms | 194 | 16.9 |

1B, 10,000 queries, cold except the pinned set:

| index | hot set | config | R@10 | p50 | p99 | QPS | MB/q |
|---|---|---|---|---|---|---|---|
| 1-bit, D19 (warm) | page cache | 64:500 | 0.911 | 7.25 ms | 9.87 ms | 147 | |
| 1-bit, page reads | none | 64:500 | 0.911 | 8.03 ms | | 130 | 14.8 |
| 1-bit, page reads | 12 GiB | 64:500 | 0.911 | 4.46 ms | | 266 | 3.7 |
| 1-bit, page reads | 12 GiB | 64:1000 | 0.922 | 6.83 ms | | 164 | 3.7 |
| 2-bit | 12 GiB | 32:100:1000 | 0.896 | 3.13 ms | 6.47 ms | 458 | 6.1 |
| 2-bit | 12 GiB | 40:100:1000 | 0.906 | 3.48 ms | 7.24 ms | 372 | 7.5 |
| 2-bit | 12 GiB | 48:100:1000 | 0.914 | 3.88 ms | 8.31 ms | 345 | 8.9 |
| 2-bit | 12 GiB | 64:100:1000 | 0.925 | 4.61 ms | 10.5 ms | 275 | 11.6 |
| 2-bit | 12 GiB | 64:150:2000 | 0.929 | 4.83 ms | 10.3 ms | 254 | 11.6 |
| 2-bit | 12 GiB | 128:200:2000 | 0.951 | 8.18 ms | 17.5 ms | 137 | 22.6 |

2-bit 1B build: train reused, ten adds at 83–88 s (1.15M rows/s), pack 101 s, 965 s in all;
48.33 GB of postings. 2-bit stage means at 48:100:1000 (ms): route 0.35, scan 1.53, posting
I/O 1.41, rerank 0.81. 47.5k–50k leaves fit in 12 GiB depending on nprobe.

Query-aware probing (DESIGN.md D21), same setup; `mM` keeps leaves with ‖q − c‖² within
(1 + M) of the nearest, up to the nprobe cap:

| config | probes | R@10 | p50 | p99 | QPS | MB/q |
|---|---|---|---|---|---|---|
| 16:100:1000 | 16 | 0.859 | 2.32 ms | 4.37 ms | 703 | 3.2 |
| 24:100:1000 | 24 | 0.882 | 2.69 ms | 5.09 ms | 552 | 4.6 |
| 128:100:1000:m0.22 | 14.5 | 0.821 | 1.92 ms | 6.93 ms | 845 | 2.4 |
| 128:100:1000:m0.3 | 27.1 | 0.871 | 2.16 ms | 9.11 ms | 599 | 4.2 |
| 128:100:1000:m0.4 | 43.8 | 0.906 | 2.66 ms | 11.2 ms | 420 | 6.8 |
| 96:100:1000:m0.45 | 42.9 | 0.911 | 2.96 ms | 10.3 ms | 413 | 6.9 |
| 128:100:1000:m0.55 | 65.4 | 0.930 | 3.83 ms | 12.8 ms | 295 | 10.5 |
| 96:100:1000:m0.6 | 57.4 | 0.927 | 3.78 ms | 11.2 ms | 314 | 9.6 |
| 128:100:1000:m0.75 | 86.6 | 0.941 | 5.29 ms | 15.2 ms | 222 | 14.5 |
| 192:200:2000:m1.0 | 147.1 | 0.960 | 8.72 ms | 21.6 ms | 127 | 24.8 |

At matched recall that is 12–14% more QPS than a fixed nprobe, at a higher p99.

Against the published NeurIPS'21 big-ann results on msspacev-1B (Simhadri et al., PMLR 176,
2022; [track README](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips21/t1_t2/README.md)):

| system | hardware | R@10 | QPS |
|---|---|---|---|
| T2 DiskANN baseline | 8 vCPU, 64 GB RAM, 1 TB NVMe | 0.901 | 1,500 (bar) |
| T2 kota-t2 | same | 0.904 | 1,500 (bar) |
| T2 BBANN | same | 0.760 | 1,500 (bar) |
| T1 FAISS baseline (in RAM) | 32 vCPU, 64 GB | 0.729 | 10,000 (bar) |
| T1 best (kst_ann_t1) | same | 0.765 | 10,000 (bar) |
| sieve 2-bit, 128:100:1000:m0.4 | M4 Pro laptop, 24 GB RAM (12 GiB pinned) | 0.906 | 420 |
| sieve 2-bit, 96:100:1000:m0.6 | same | 0.927 | 314 |
| sieve 2-bit, 192:200:2000:m1.0 | same | 0.960 | 127 |

What this says:
- **2.5× the QPS of D19 at equal or better recall, fully cold apart from the pinned set.** The
  hot set halves posting traffic, page-sized reads raise rerank IOPS, and 2-bit refinement cuts
  the rerank from 500–1,000 rows to 100 at the same recall.
- **2-bit beats 3-bit end to end** even though 3-bit ranks better offline: larger slots cost
  scan bandwidth and fit fewer leaves in the same hot set.
- **The SSD is now the limit.** At 48:100:1000, posting reads run at 3.1 GB/s against a 3.6
  GB/s ceiling. DiskANN on the T2 machine does about 4× the QPS at 0.90 with 64 GB of RAM;
  this laptop has 24 GB and pins 12.

### macOS read path (`examples/preadbench.rs`, `examples/faultbench.rs`)
| | per read |
|---|---|
| cached `pread`, one file, 1 thread | 0.44 µs |
| cached `pread`, one file, 12–32 threads (aggregate) | ~1.1 µs (serialised) |
| mmap hit | 0.15 µs |
| `mincore` residency check | 0.46 µs (serialised) |
| evicted file, 64 threads: mmap fault | 107 µs aggregate |
| evicted file, 64 threads: `pread` | 9.3 µs aggregate (~100k IOPS) |
