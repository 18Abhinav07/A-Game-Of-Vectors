#!/usr/bin/env bash
# MS SPACEV-1B end to end (DESIGN.md D19): train on chunk 0, add every chunk as soon as
# scripts/spacev.sh has finished it (run that alongside), pack, then search warm and cold.
# Disk: the 10 chunks (100 GB, kept: they are the rerank store), postings ~28 GB, and up to
# ~32 GB of run files that pack deletes as it goes.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release
B="target/release/spacev --index data/sv1b --log results/spacev.jsonl"
S=data/spacev
[ -f data/sv1b/k2.u64 ] || $B train --base $S/base.00 --k1 512 --k2 256 --sample 20000000
for c in 0 1 2 3 4 5 6 7 8 9; do
  [ -f data/sv1b/added.$c ] && continue
  until [ -f $S/base.0$c.done ]; do sleep 30; done
  $B add --base $S/base.0$c --first $((c * 100000000))
  touch data/sv1b/added.$c
done
[ -f data/sv1b/postings.bin ] || $B pack
G=$S/public_query_gt100.bin
$B search --gt $G --configs 64:500,64:1000,128:1000,256:2000 --label 1B
$B search --gt $G --configs 64:1000,128:1000 --nq 2000 --cold --label 1B
