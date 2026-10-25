#!/usr/bin/env bash
# M2: the full corpus (60 shards, 113.5M passages). Shards 00-09 are kept from M1; 10-59 are
# downloaded while ingest runs, and each is deleted once read (ingest --wait --consume-from 10).
# Ingest is not resumable: if it dies, the consumed shards must be downloaded again.
set -euo pipefail
cd "$(dirname "$0")/.."
B=target/release
LOG=results/m2.jsonl
./scripts/download.sh 10 59 > data/download-m2.log 2>&1 &
DL=$!
STREAM=()
for i in $(seq -f "%02g" 10 59); do STREAM+=("data/raw/corpus/$i.parquet"); done
$B/ingest data/raw/corpus/0{0..9}.parquet "${STREAM[@]}" --out data/full/ingest --wait --consume-from 10 2>&1 \
  | awk 'NR % 50 == 0 || !/rows\/s  \(/'
wait $DL || true  # the ingest is the source of truth; a restarted downloader may report a signal
cat data/full/ingest/meta.json
[[ "${1:-}" == "ingest" ]] && exit 0
$B/index --ingest data/full/ingest --out data/full/index --k 16384 --sample 1000000 --soar 1.0 --rabitq
$B/search --ingest data/full/ingest --index data/full/index --log $LOG routing
$B/search --ingest data/full/ingest --index data/full/index --log $LOG eval --configs "16:1000:tiered200,32:1000:tiered200,64:1000:tiered200,128:1000:tiered200,256:1000:tiered200,64:1000:tiered100"
# rerank.i8 (87 GB) exceeds RAM, so the final tier is read from the SSD whether or not --cold is set.
for c in 32:1000:tiered200 64:1000:tiered200 128:1000:tiered200 64:1000:tiered100; do
  $B/search --ingest data/full/ingest --index data/full/index --log $LOG bench --config $c --seconds 10 --cold
done
