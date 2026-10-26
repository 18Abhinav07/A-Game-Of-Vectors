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
# RaBitQ residual codes and the 2-bit tier (D9, D10): the index every later M1 run uses.
$B/index --ingest data/ingest --out data/index-rabitq --k 4096 --sample 500000 --soar 1.0 --rabitq
for idx in data/index-plain data/index; do
  $B/search --index $idx --log $LOG routing
  $B/search --index $idx --log $LOG eval --configs "8:1000:full,16:1000:full,32:1000:full,64:1000:full,128:1000:full,32:200:full,32:1000:prefix,32:1000:none"
done
$B/search --index data/index --log $LOG eval --hamming --configs "all:1000:full"
for c in 16:1000:full 32:1000:full 64:1000:full; do
  $B/search --index data/index --log $LOG bench --config $c --seconds 10
done
# D18: BM25 over the passage text (data/raw/text/msmarco_v2.1_doc_segmented_0{0..9}.json.gz),
# hybrid fusion and filters. Needs the RaBitQ index in data/index-rabitq.
if ls data/raw/text/*.json.gz >/dev/null 2>&1; then
  $B/bm25 --ingest data/ingest --text data/raw/text --out data/bm25
  $B/search --index data/index-rabitq --log results/m1-hybrid.jsonl hybrid --weights 1,0.5,0.3,0.2,0.1
  $B/search --index data/index-rabitq --log results/m1-filter.jsonl filter
fi
