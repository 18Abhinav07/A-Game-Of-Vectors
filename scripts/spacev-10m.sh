#!/usr/bin/env bash
# sieve's side of the DiskANN head-to-head (DESIGN.md D22): the first 10M rows of SPACEV, the
# first 10,000 queries, the official 10M ground truth. Needs chunk 0 and the query files from
# `scripts/spacev.sh 0` (10 GB; only its first 1 GB is read). Builds in about 10 s.
# INDEX (default data/sv10m-b2) is where the index goes; results go to results/diskann-h2h.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${INDEX:-data/sv10m-b2}
S=data/spacev
cargo build --release
B="target/release/spacev --index $D --log results/diskann-h2h.jsonl"
[ -f $D/k2.u64 ] || $B train --base $S/base.00 --rows 10000000 --k1 64 --k2 24 --sample 2000000 --bits 2
[ -f $D/added.0 ] || { $B add --base $S/base.00 --first 0 --rows 10000000; touch $D/added.0; }
[ -f $D/postings.bin ] || $B pack
G=$S/msspacev-gt-10M
# In RAM (the in-memory table), then cold with 0.5 GB of postings pinned (the SSD table).
$B search --gt $G --configs 16:100:1000,32:100:1000,64:100:1000,128:200:2000 --label h2h-10m-sieve-b2
$B search --gt $G --configs 32:50:1000,64:50:1000,96:50:1000,96:70:1000,128:100:2000,128:200:2000 \
  --cold --hot-gb 0.5 --label h2h-10m-sieve-b2-cold
