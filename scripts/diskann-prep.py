"""Write DiskANN-format inputs for the head-to-head: a 10k-query subset (queries 0..10k, the
set sieve evaluates), its ground truth against 10M and 100M, and headed base files."""
import struct, sys
S, D = "data/spacev", "data/dann"
NQ, DIM, K = 10000, 100, 100

def head(f, n, d): f.write(struct.pack("<II", n, d))

with open(f"{S}/query.i8bin", "rb") as f, open(f"{D}/query10k.i8bin", "wb") as o:
    f.seek(8); head(o, NQ, DIM); o.write(f.read(NQ * DIM))

for n in ("10M", "100M"):
    with open(f"{S}/msspacev-gt-{n}", "rb") as f, open(f"{D}/gt{n.lower()}-10k.bin", "wb") as o:
        tot, k = struct.unpack("<II", f.read(8)); assert k == K
        ids = f.read(NQ * K * 4); f.seek(8 + tot * K * 4)
        head(o, NQ, K); o.write(ids); o.write(f.read(NQ * K * 4))

for rows in map(int, sys.argv[1:]):
    with open(f"{S}/base.00", "rb") as f, open(f"{D}/base{rows // 1000000}m.i8bin", "wb") as o:
        head(o, rows, DIM); left = rows * DIM
        while left:
            b = f.read(min(left, 1 << 28)); o.write(b); left -= len(b)
