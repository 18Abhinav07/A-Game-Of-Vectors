#!/usr/bin/env bash
# MS SPACEV-1B end to end (DESIGN.md D19, D20): train on chunk 0, add every chunk as soon as
# scripts/spacev.sh has finished it (run that alongside), pack, then search.
# BITS=2 (default) stores the 2-bit refinement plane (48 B slots, ~48 GB of postings) and
# searches cold with a HOT_GB (default 12) pinned hot set; BITS=1 is the D19 index (~28 GB).
# Disk: the 10 chunks (100 GB, kept: they are the rerank store), the postings, and up to
# ~55 GB of run files that pack deletes as it goes.
set -euo pipefail
cd "$(dirname "$0")/.."
BITS=${BITS:-2}
HOT_GB=${HOT_GB:-12}
D=data/sv1b; [ "$BITS" = 1 ] || D=data/sv1b-b$BITS
cargo build --release
B="target/release/spacev --index $D --log results/spacev.jsonl"
S=data/spacev
[ -f $D/k2.u64 ] || $B train --base $S/base.00 --k1 512 --k2 256 --sample 20000000 --bits $BITS
for c in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $D/added.$c ] && continue
  until [ -f $S/base.0$c.done ]; do sleep 30; done
  $B add --base $S/base.0$c --first $((c * 100000000))
  touch $D/added.$c
done
[ -f $D/postings.bin ] || $B pack
G=$S/public_query_gt100.bin
if [ "$BITS" = 1 ]; then
  $B search --gt $G --configs 64:500,64:1000,128:1000,256:2000 --label 1B
  $B search --gt $G --configs 64:1000,128:1000 --nq 2000 --cold --label 1B
else
  $B search --gt $G --configs 128:100:1000:m0.4 --nq 10000 \
    --cold --hot-gb $HOT_GB --label 1B-b$BITS-hot$HOT_GB
  for c in 96:100:1000:m0.6 192:200:2000:m1.0; do
    $B search --gt $G --configs $c --nq 10000 --cold --hot-gb $HOT_GB --label 1B-b$BITS-hot$HOT_GB
  done
fi
