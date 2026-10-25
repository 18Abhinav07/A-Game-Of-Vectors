#!/usr/bin/env bash
# M2 after ingest: index, routing, eval and bench over data/full (see m2.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
B=target/release
LOG=results/m2.jsonl
$B/index --ingest data/full/ingest --out data/full/index --k 16384 --sample 1000000 --soar 1.0 --rabitq
$B/search --ingest data/full/ingest --index data/full/index --log $LOG routing
$B/search --ingest data/full/ingest --index data/full/index --log $LOG eval --configs "16:1000:tiered200,32:1000:tiered200,64:1000:tiered200,128:1000:tiered200,256:1000:tiered200,32:500:tiered200,64:500:tiered200,64:1000:tiered100"
# rerank.i8 (87 GB) exceeds RAM, so the final tier is read from the SSD whether or not --cold is set.
for c in 32:500:tiered200 64:500:tiered200 64:1000:tiered200 128:1000:tiered200; do
  $B/search --ingest data/full/ingest --index data/full/index --log $LOG bench --config $c --seconds 10 --cold
done
