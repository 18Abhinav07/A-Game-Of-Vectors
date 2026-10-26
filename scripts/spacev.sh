#!/usr/bin/env bash
# MS SPACEV-1B (big-ann-benchmarks, NeurIPS'21): 1B x 100-d int8 Bing web-document vectors,
# 29,316 queries, top-100 ground truth for the 10M, 100M and 1B prefixes.
# Usage: scripts/spacev.sh [chunk ...]   (chunk c = vectors [c*100M, (c+1)*100M), 10 GB each)
# The base file is fetched as 1e8-row chunks with 16 parallel range requests, each written in
# place, so a chunk never needs a second copy. Chunks are raw int8 rows with no header.
set -euo pipefail
cd "$(dirname "$0")/.."
U=https://comp21storage.z5.web.core.windows.net/comp21/spacev1b
D=data/spacev
mkdir -p $D
for f in query.i8bin public_query_gt100.bin msspacev-gt-100M msspacev-gt-10M; do
  [ -s $D/$f ] || curl -sf --retry 5 -o $D/$f $U/$f
done
CHUNK=10000000000          # 1e8 rows x 100 bytes
PS=$((160 * 1048576))      # part size, a multiple of dd's 1 MiB block
for c in "$@"; do
  out=$D/base.$(printf %02d $c)
  [ -f $out.done ] && continue
  parts=$(( (CHUNK + PS - 1) / PS ))
  t0=$(date +%s)
  seq 0 $((parts - 1)) | xargs -P 16 -n 1 bash -c '
    set -o pipefail
    p=$1; a=$(( 8 + '$c' * '$CHUNK' + p * '$PS' )); n='$PS'
    rem=$(( '$CHUNK' - p * '$PS' )); [ $rem -lt $n ] && n=$rem
    for try in 1 2 3 4 5; do
      curl -sf -r $a-$((a + n - 1)) '$U'/spacev1b_base.i8bin | dd of='$out' bs=1048576 seek=$((p * 160)) conv=notrunc 2>/dev/null && exit 0
      sleep 2
    done
    echo "part $p failed" >&2; exit 1' _
  size=$(stat -f %z $out)
  [ "$size" -eq "$CHUNK" ] || { echo "$out: $size bytes, want $CHUNK" >&2; exit 1; }
  touch $out.done
  echo "chunk $c: 10 GB in $(( $(date +%s) - t0 )) s"
done
