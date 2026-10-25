//! Streams corpus shards once and writes every on-disk tier: 1-bit codes, int8 rerank vectors,
//! prefix norms and passage ids. Exact top-k for every evaluation query is computed in the same
//! pass with sgemm, so ground truth costs no second read of the corpus (DESIGN.md D5).

use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use sieve::blas::gemm_abt;
use sieve::data::{EVAL_SETS, read_topics, stream_corpus};
use sieve::quant::{Int8Quantizer, normalize, prefix, sign_code};
use sieve::store::{Appender, write_vec};
use sieve::topk::TopK;
use sieve::{CODE_BYTES, DIM, PDIM};

#[derive(Parser)]
struct Args {
    /// Corpus parquet shards, ingested in the order given.
    #[arg(required = true)]
    shards: Vec<PathBuf>,
    #[arg(long, default_value = "data/ingest")]
    out: PathBuf,
    #[arg(long, default_value = "data/raw/topics")]
    topics: PathBuf,
    /// Depth of the exact ground truth kept per query.
    #[arg(long, default_value_t = 100)]
    gt_k: usize,
    #[arg(long, default_value_t = 65536)]
    batch: usize,
    /// Wait for shards that are still downloading.
    #[arg(long)]
    wait: bool,
    /// Delete each shard from this position on after reading it (stream-and-delete for corpora
    /// larger than the disk). Earlier shards are kept.
    #[arg(long, default_value_t = usize::MAX)]
    consume_from: usize,
}

/// Exact top-k for every query over the rows seen so far.
struct Truth {
    q: Vec<f32>,
    dim: usize,
    heaps: Vec<TopK<f32>>,
    scores: Vec<f32>,
}

impl Truth {
    fn new(q: Vec<f32>, dim: usize, k: usize) -> Self {
        let nq = q.len() / dim;
        Self { q, dim, heaps: (0..nq).map(|_| TopK::new(k)).collect(), scores: Vec::new() }
    }

    fn add(&mut self, rows: &[f32], first_id: u32) {
        let (nq, m) = (self.heaps.len(), rows.len() / self.dim);
        self.scores.resize(nq * m, 0.0);
        gemm_abt(&self.q, rows, nq, m, self.dim, &mut self.scores);
        self.heaps.par_iter_mut().zip(self.scores.par_chunks(m)).for_each(|(h, s)| {
            for (j, &v) in s.iter().enumerate() {
                h.push(v, first_id + j as u32);
            }
        });
    }

    fn write(self, out: &std::path::Path, name: &str, k: usize) -> Result<()> {
        let (mut ids, mut scores) = (Vec::new(), Vec::new());
        for h in self.heaps {
            let mut v = h.into_sorted();
            v.resize(k, (f32::NEG_INFINITY, u32::MAX));
            ids.extend(v.iter().map(|x| x.1));
            scores.extend(v.iter().map(|x| x.0));
        }
        write_vec(&out.join(format!("{name}.ids")), &ids)?;
        write_vec(&out.join(format!("{name}.scores")), &scores)
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    std::fs::create_dir_all(&a.out)?;
    let t0 = Instant::now();

    let mut q768 = Vec::new();
    let mut q256 = Vec::new();
    let mut tsv = std::fs::File::create(a.out.join("queries.tsv"))?;
    for (set, file, _, limit) in EVAL_SETS {
        for mut q in read_topics(&a.topics.join(file), set, *limit)? {
            normalize(&mut q.emb);
            let mut p = [0f32; PDIM];
            prefix(&q.emb, &mut p);
            q768.extend_from_slice(&q.emb);
            q256.extend_from_slice(&p);
            writeln!(tsv, "{}\t{}\t{}", q.set, q.id, q.text.replace(['\t', '\n'], " "))?;
        }
    }
    let nq = q768.len() / DIM;
    write_vec(&a.out.join("queries.f32"), &q768)?;
    eprintln!("{nq} queries");
    let mut gt768 = Truth::new(q768, DIM, a.gt_k);
    let mut gt256 = Truth::new(q256, PDIM, a.gt_k);

    let mut codes = Appender::create(&a.out.join("codes.bin"))?;
    let mut rerank = Appender::create(&a.out.join("rerank.i8"))?;
    let mut norms = Appender::create(&a.out.join("norm256.f32"))?;
    let mut ids = Appender::create(&a.out.join("docids.bin"))?;
    let mut offs: Vec<u64> = vec![0];
    let mut quant: Option<Int8Quantizer> = None;
    let mut n = 0usize;
    let (mut t_gt, mut t_enc) = (0f64, 0f64);

    let mut pre = Vec::new();
    let mut code_buf = Vec::new();
    let mut i8_buf = Vec::new();
    let mut norm_buf = Vec::new();
    for batch in stream_corpus(a.shards.clone(), a.batch, a.wait, a.consume_from) {
        let mut b = batch?;
        let m = b.len();
        let te = Instant::now();
        pre.resize(m * PDIM, 0.0);
        code_buf.resize(m * CODE_BYTES, 0u8);
        i8_buf.resize(m * DIM, 0i8);
        norm_buf.resize(m, 0.0);
        b.emb.par_chunks_mut(DIM).zip(pre.par_chunks_mut(PDIM)).zip(norm_buf.par_iter_mut()).for_each(|((v, p), nrm)| {
            normalize(v);
            *nrm = prefix(v, p);
        });
        // Scales come from the first batch; later outliers clamp to ±127 (a few per million).
        let qz = quant.get_or_insert_with(|| Int8Quantizer::fit(&b.emb));
        pre.par_chunks(PDIM).zip(code_buf.par_chunks_mut(CODE_BYTES)).for_each(|(p, c)| sign_code(p, c));
        b.emb.par_chunks(DIM).zip(i8_buf.par_chunks_mut(DIM)).for_each(|(v, o)| qz.encode(v, o));
        codes.write(&code_buf)?;
        rerank.write(&i8_buf)?;
        norms.write(&norm_buf)?;
        for id in &b.ids {
            ids.write(id.as_bytes())?;
            offs.push(offs.last().unwrap() + id.len() as u64);
        }
        t_enc += te.elapsed().as_secs_f64();

        let tg = Instant::now();
        gt768.add(&b.emb, n as u32);
        gt256.add(&pre, n as u32);
        t_gt += tg.elapsed().as_secs_f64();

        n += m;
        let s = t0.elapsed().as_secs_f64();
        eprintln!("{n} rows  {:.0} rows/s  (encode {t_enc:.1}s, truth {t_gt:.1}s, wall {s:.1}s)", n as f64 / s);
    }
    codes.finish()?;
    rerank.finish()?;
    norms.finish()?;
    ids.finish()?;
    write_vec(&a.out.join("docids.off"), &offs)?;
    write_vec(&a.out.join("quant.f32"), &quant.expect("empty corpus").scale)?;
    gt768.write(&a.out, "gt768", a.gt_k)?;
    gt256.write(&a.out, "gt256", a.gt_k)?;

    let secs = t0.elapsed().as_secs_f64();
    let meta = serde_json::json!({
        "rows": n, "queries": nq, "gt_k": a.gt_k,
        "shards": a.shards.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "seconds": secs, "encode_seconds": t_enc, "truth_seconds": t_gt,
    });
    std::fs::write(a.out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    eprintln!("done: {n} rows in {secs:.1}s ({:.0} rows/s)", n as f64 / secs);
    Ok(())
}
