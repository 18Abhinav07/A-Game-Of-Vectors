#!/usr/bin/env bash
# Download the Snowflake Arctic-M v1.5 MS MARCO v2.1 embeddings from Hugging Face.
# Usage: scripts/download.sh [first_shard] [last_shard]   (shards are 00..59)
# Resumable: re-running continues partial files. Two shards download in parallel.
# Shards that `ingest --consume` already read and deleted (marked .consumed) are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://huggingface.co/datasets/Snowflake/msmarco-v2.1-snowflake-arctic-embed-m-v1.5/resolve/main
RAW=data/raw
FIRST=${1:-0}; LAST=${2:-9}

fetch() { # url dest
  local url=$1 dest=$2
  [[ -f "$dest" ]] && { echo "have $dest"; return; }
  [[ -f "$dest.consumed" ]] && { echo "ingested and deleted $dest"; return; }
  # A network change can leave curl on a dead connection with no error, so abort below
  # 500 KB/s for 60 s and resume from the partial file until the transfer completes.
  until curl -sSL --fail --speed-limit 500000 --speed-time 60 -C - -o "$dest.part" "$url"; do
    echo "retry $dest"; sleep 5
  done
  mv "$dest.part" "$dest"; echo "done $dest"
}

for f in dl21 dl22 dl23 msmarco-v2-doc.dev msmarco-v2-doc.dev2; do
  fetch "$BASE/topics/snowflake-arctic-embed-m-v1.5-topics.$f.parquet" "$RAW/topics/$f.parquet"
done
for f in dl21-doc-msmarco-v2.1 dl22-doc-msmarco-v2.1 dl23-doc-msmarco-v2.1 msmarco-v2.1-doc.dev msmarco-v2.1-doc.dev2; do
  fetch "$BASE/qrels/$f.json" "$RAW/qrels/$f.json"
done

for i in $(seq -f "%02g" "$FIRST" "$LAST"); do
  fetch "$BASE/corpus/$i.parquet" "$RAW/corpus/$i.parquet" &
  while (( $(jobs -rp | wc -l) >= 2 )); do wait -n || true; done
done
wait
echo "all shards $FIRST..$LAST downloaded"
