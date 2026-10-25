#!/usr/bin/env python3
"""One line per `search bench` JSON object read from stdin (pretty-printed, possibly several)."""
import json, sys

dec, text, i = json.JSONDecoder(), sys.stdin.read(), 0
while (i := text.find("{", i)) >= 0:
    d, i = dec.raw_decode(text, i)
    io = d.get("postings_io") or {}
    lp, qp = io.get("latency_pass") or {}, io.get("qps_pass") or {}
    st = " ".join(f"{k} {v:.2f}" for k, v in d["stage_ms"].items() if v)
    print(f"{d['run']} | {'ssd' if d.get('ssd_mode') else 'ram'} cache {d.get('cache_mb', 0)} MB"
          f" | p50 {d['latency_ms']['p50']:.2f} p99 {d['latency_ms']['p99']:.2f} ms | {st}"
          f" | {d['qps']:.0f} QPS | hit {lp.get('hit_rate', 0):.2f}/{qp.get('hit_rate', 0):.2f}"
          f" | {lp.get('mb_read_per_query', 0):.1f} MB/q | rss {d.get('peak_rss_mb', 0):.0f} MB")
