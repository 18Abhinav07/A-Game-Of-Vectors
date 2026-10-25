#!/usr/bin/env bash
# M2 after ingest: index, routing, eval and bench over data/full (see m2.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
B=target/release
LOG=results/m2.jsonl
$B/index --ingest data/full/ingest --out data/full/index --k 16384 --sample 1000000 --soar 1.0 --rabitq
$B/search --ingest data/full/ingest --index data/full/index --log $LOG routing
$B/search --ingest data/full/ingest --index data/full/index --log $LOG eval --configs "16:1000:tiered200,32:1000:tiered200,64:1000:tiered200,128:1000:tiered200,256:1000:tiered200,32:500:tiered200,64:500:tiered200,64:1000:tiered100"
# Bound pruning (D11) and query-aware probing (D12). Centroid scores are tighter at 16384
# clusters than at 4096, so the margin is swept again.
$B/search --ingest data/full/ingest --index data/full/index --log $LOG eval --configs "64:500:tiered200:b10,64:1000:full:b10x0.8,256:500:tiered200:p0.08,256:500:tiered200:p0.10,256:500:tiered200:p0.12,256:500:tiered200:p0.14,256:500:tiered200:p0.16"
# rerank.i8 (87 GB) exceeds RAM, so the final tier is read from the SSD whether or not --cold is set.
for c in 32:500:tiered200 64:500:tiered200 64:1000:tiered200 128:1000:tiered200 64:500:tiered200:b10 256:500:tiered200:p0.12; do
  $B/search --ingest data/full/ingest --index data/full/index --log $LOG bench --config $c --seconds 10 --cold
done
# SSD mode (D13): posting lists on disk, RAM holds centroids, offsets and an optional cache.
$B/index --out data/full/index --pack
$B/search --ssd --ingest data/full/ingest --index data/full/index --log $LOG eval --configs "64:500:ssd300,64:500:ssd400,128:1000:ssd400,256:500:ssd400:p0.12"
for c in 64:500:ssd300 256:500:ssd400:p0.12; do
  for mb in 0 1024; do
    $B/search --ssd --cache-mb $mb --ingest data/full/ingest --index data/full/index --log $LOG bench --config $c --seconds 10 --cold
  done
done
