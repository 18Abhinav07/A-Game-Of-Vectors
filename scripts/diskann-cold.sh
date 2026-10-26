#!/usr/bin/env bash
# Runs a DiskANN disk-index job cold: the index files are dropped from the page cache first,
# and the macOS reader (F_NOCACHE pread) keeps them out. Usage: scripts/diskann-cold.sh <job.json> <out-prefix> <index-prefix>
set -euo pipefail
cd "$(dirname "$0")/.."
DISKANN=${DISKANN:-../DiskANN/target/release/diskann-benchmark}
target/release/examples/drop_cache "$3"_disk.index
/usr/bin/time -l $DISKANN --quiet run --input-file "$1" --output-file "$2".json > "$2".log 2>&1
