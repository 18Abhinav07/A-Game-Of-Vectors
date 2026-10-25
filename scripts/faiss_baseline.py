"""FAISS baselines on the M1 slice, at the same 32-byte code budget and the same int8 rerank.

The vectors are the 256-d Matryoshka prefixes reconstructed from data/ingest/rerank.i8 (the same
prefix sieve indexes), so the only differences are the index and the code. Recall@10 is against
exact 768-d search (gt768), after reranking the top `cand` candidates with the 768-d int8 rows,
exactly like sieve's `full` rerank. Latency is single-thread, one query at a time; QPS is a
batch search on all threads, mirroring `search bench`.

  .venv-bench/bin/python -I scripts/faiss_baseline.py --ingest data/ingest --out results/faiss-m1.jsonl
"""

import argparse
import json
import os
import time

import faiss
import numpy as np

DIM, PDIM = 768, 256


def load(ingest):
    rows = np.memmap(os.path.join(ingest, "rerank.i8"), dtype=np.int8, mode="r").reshape(-1, DIM)
    scale = np.fromfile(os.path.join(ingest, "quant.f32"), dtype=np.float32)
    q = np.fromfile(os.path.join(ingest, "queries.f32"), dtype=np.float32).reshape(-1, DIM)
    gt = np.fromfile(os.path.join(ingest, "gt768.ids"), dtype=np.uint32).reshape(len(q), -1)
    return rows, scale, q, gt


def prefix(rows, scale, lo, hi):
    x = rows[lo:hi, :PDIM].astype(np.float32) / scale[:PDIM]
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def build(spec, rows, scale, sample_n, chunk=1 << 20):
    n = len(rows)
    rng = np.random.default_rng(7)
    sample = np.sort(rng.choice(n, sample_n, replace=False))
    xt = unit(rows[sample, :PDIM].astype(np.float32) / scale[:PDIM])
    index = faiss.index_factory(PDIM, spec, faiss.METRIC_INNER_PRODUCT)
    t = time.time()
    index.train(xt)
    train_s = time.time() - t
    t = time.time()
    for lo in range(0, n, chunk):
        index.add(prefix(rows, scale, lo, min(n, lo + chunk)))
    return index, train_s, time.time() - t


def rerank(rows, scale, q768, cand, k=10):
    """Exact 768-d scores against the int8 rows, as sieve's `full` rerank."""
    out = np.empty((len(q768), k), dtype=np.int64)
    w = q768 / scale
    for i, c in enumerate(cand):
        c = c[c >= 0]
        order = np.argsort(c)  # sorted reads, like sieve's pread batches
        s = rows[c[order]].astype(np.float32) @ w[i]
        out[i] = c[order[np.argsort(-s)[:k]]]
    return out


def recall(found, gt, k=10):
    return float(np.mean([len(set(f[:k]) & set(g[:k])) / k for f, g in zip(found, gt)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", default="data/ingest")
    ap.add_argument("--out", default="results/faiss-m1.jsonl")
    ap.add_argument("--sample", type=int, default=500_000)
    ap.add_argument("--specs", default="IVF4096,PQ32;IVF4096,PQ64x4fs;IVF4096,RaBitQ")
    ap.add_argument("--nprobe", default="16,32,64,128")
    ap.add_argument("--cand", type=int, default=500)
    ap.add_argument("--lat-queries", type=int, default=500)
    a = ap.parse_args()

    rows, scale, q768, gt = load(a.ingest)
    q256 = unit(q768[:, :PDIM].copy())
    threads = faiss.omp_get_max_threads()
    print(f"{len(rows)} rows, {len(q768)} queries, {threads} threads")
    for spec in a.specs.split(";"):
        index, train_s, add_s = build(spec, rows, scale, a.sample)
        print(f"{spec}: train {train_s:.0f}s, add {add_s:.0f}s")
        ivf = faiss.extract_index_ivf(index)
        for nprobe in map(int, a.nprobe.split(",")):
            ivf.nprobe = nprobe
            faiss.omp_set_num_threads(threads)
            t = time.time()
            _, cand = index.search(q256, a.cand)
            qps = len(q256) / (time.time() - t)
            r_codes = recall(cand, gt)
            found = rerank(rows, scale, q768, cand)
            r = recall(found, gt)
            # Single-thread latency of the code scan alone; the rerank is the same int8 pread
            # sieve does and is timed by its own bench.
            faiss.omp_set_num_threads(1)
            lat = []
            for i in range(min(a.lat_queries, len(q256))):
                t = time.perf_counter()
                index.search(q256[i : i + 1], a.cand)
                lat.append((time.perf_counter() - t) * 1e3)
            lat = np.array(lat)
            rec = {
                "spec": spec, "nprobe": nprobe, "cand": a.cand, "recall10": r,
                "recall10_codes_only": r_codes, "scan_qps": qps,
                "scan_latency_ms": {"p50": float(np.percentile(lat, 50)), "p99": float(np.percentile(lat, 99))},
                "train_s": train_s, "add_s": add_s, "code_bytes_per_vector": ivf.code_size,
                "faiss": faiss.__version__,
            }
            print(json.dumps(rec))
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        del index


if __name__ == "__main__":
    main()
