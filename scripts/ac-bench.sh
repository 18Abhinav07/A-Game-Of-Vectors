#!/usr/bin/env bash
# M1 latency re-runs on AC power with Low Power Mode off. Every latency in BENCHMARKS.md marked
# "Low Power Mode, relative only" has a row here. Recall does not change; only times do.
# Logs go to results/*-ac.jsonl. Takes about 25 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
pmset -g batt | grep -q "AC Power" || { echo "not on AC power" >&2; exit 1; }
pmset -g | grep -Eq "lowpowermode +1" && { echo "Low Power Mode is on" >&2; exit 1; }
cargo build --release
B=target/release
S="$B/search --ingest data/ingest --index data/index-rabitq"

# Core latency tables (warm and cold), then SSD mode with and without a cluster cache.
for c in 16:500:full 32:500:full 64:500:full 32:1000:tiered200 64:500:tiered200 64:500:tiered200:b10 256:500:tiered200:p0.12; do
  $S --log results/m1-ac.jsonl bench --config $c --seconds 10 > /dev/null
  $S --log results/m1-ac.jsonl bench --config $c --seconds 10 --cold > /dev/null
done
for c in 64:500:ssd300 256:500:ssd300:p0.12 64:500:ssd300:s1000k; do
  for mb in 0 512; do $S --log results/m1-ac.jsonl --ssd --cache-mb $mb bench --config $c --seconds 10 --cold > /dev/null; done
done

# D16: single process against 4 shard processes over loopback, one-shot and two-phase.
pids=()
for i in 0 1 2 3; do
  $B/shard --index data/index-rabitq --shard $i --of 4 --listen 127.0.0.1:$((7100 + i)) & pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
for i in 0 1 2 3; do until nc -z 127.0.0.1 $((7100 + i)); do sleep 1; done; done
SH=127.0.0.1:7100,127.0.0.1:7101,127.0.0.1:7102,127.0.0.1:7103
$S --log results/m1-shard-ac.jsonl bench --config 64:500:tiered200 --seconds 10 > /dev/null
$S --log results/m1-shard-ac.jsonl --shards $SH bench --config 64:500:tiered200 --seconds 10 > /dev/null
$S --log results/m1-shard-ac.jsonl --shards $SH --two-phase bench --config 64:500:tiered200 --seconds 10 > /dev/null
kill "${pids[@]}"; trap - EXIT

# D17 and D18 latencies come with their recall runs.
$S --log results/m1-live-ac.jsonl live --configs 64:500:full,64:500:tiered200,64:500:ssd300
$S --log results/m1-filter-ac.jsonl filter
$S --log results/m1-hybrid-ac.jsonl hybrid --weights 1,0.3,0.2
echo done
