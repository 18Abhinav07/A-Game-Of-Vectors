#!/usr/bin/env bash
# M1: shards 00-09 (17.6M passages) end to end. Logs to results/m1.log, metrics to results/m1.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release
mkdir -p results
B=target/release
LOG=results/m1.jsonl
$B/ingest data/raw/corpus/0{0..9}.parquet --out data/ingest 2>&1 | grep -v "rows/s  (" || true
cat data/ingest/meta.json
$B/index --ingest data/ingest --out data/index-plain --k 4096 --sample 500000
$B/index --ingest data/ingest --out data/index --k 4096 --sample 500000 --soar 1.0
for idx in data/index-plain data/index; do
  $B/search --index $idx --log $LOG routing
  $B/search --index $idx --log $LOG eval --configs "8:1000:full,16:1000:full,32:1000:full,64:1000:full,128:1000:full,32:200:full,32:1000:prefix,32:1000:none"
done
$B/search --index data/index --log $LOG eval --hamming --configs "all:1000:full"
for c in 16:1000:full 32:1000:full 64:1000:full; do
  $B/search --index data/index --log $LOG bench --config $c --seconds 10
done
