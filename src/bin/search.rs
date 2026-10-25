//! Quality, latency and throughput for every rung of the ablation chain (DESIGN.md, "Evaluation").
//!
//!   search eval   recall@10 against exact search and nDCG@10 against TREC judgements
//!   search bench  single-thread latency percentiles with a stage breakdown, then multi-thread QPS

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sieve::data::{EVAL_SETS, Qrels, read_qrels};
use sieve::eval::{mean, ndcg10, recall_at};
use sieve::ivf::{Corpus, Index, Params, Rerank, Scratch, Timing};
use sieve::lut::hamming;
use sieve::quant::{dot_i8, sign_code};
use sieve::store::read_vec;
use sieve::topk::TopK;
use sieve::{CODE_BYTES, DIM, PDIM};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "data/ingest")]
    ingest: PathBuf,
    #[arg(long, default_value = "data/index")]
    index: PathBuf,
    #[arg(long, default_value = "data/raw/qrels")]
    qrels: PathBuf,
    /// Results are appended here as JSON lines.
    #[arg(long, default_value = "results/runs.jsonl")]
    log: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Eval {
        /// nprobe:cand:rerank, comma separated; nprobe "all" scans every cluster (flat LUT).
        #[arg(long, default_value = "8:1000:full,16:1000:full,32:1000:full,64:1000:full,128:1000:full,32:200:full,32:1000:prefix,32:1000:none,all:1000:full")]
        configs: String,
        /// Also run the flat Hamming + rerank rung (a full scan per query).
        #[arg(long)]
        hamming: bool,
    },
    /// Fraction of the exact top-10 that lies in the probed clusters: the recall ceiling of routing.
    Routing,
    Bench {
        #[arg(long, default_value = "32:1000:full")]
        config: String,
        /// Seconds of multi-threaded load for the throughput figure.
        #[arg(long, default_value_t = 10.0)]
        seconds: f64,
        /// Tiered rerank reads its final rows with F_NOCACHE, so every one comes from the SSD.
        #[arg(long)]
        cold: bool,
    },
}

struct Queries {
    set: Vec<String>,
    id: Vec<String>,
    q768: Vec<f32>,
    q256: Vec<f32>,
}

impl Queries {
    fn load(dir: &Path) -> Result<Self> {
        let q768: Vec<f32> = read_vec(&dir.join("queries.f32"))?;
        let (mut set, mut id) = (Vec::new(), Vec::new());
        for line in std::fs::read_to_string(dir.join("queries.tsv"))?.lines() {
            let mut f = line.splitn(3, '\t');
            set.push(f.next().context("queries.tsv: set")?.to_owned());
            id.push(f.next().context("queries.tsv: id")?.to_owned());
        }
        anyhow::ensure!(set.len() * DIM == q768.len(), "queries.tsv and queries.f32 disagree");
        let mut q256 = vec![0f32; set.len() * PDIM];
        for (q, p) in q768.chunks_exact(DIM).zip(q256.chunks_exact_mut(PDIM)) {
            sieve::quant::prefix(q, p);
        }
        Ok(Self { set, id, q768, q256 })
    }

    fn len(&self) -> usize {
        self.set.len()
    }

    fn q768(&self, i: usize) -> &[f32] {
        &self.q768[i * DIM..][..DIM]
    }

    fn q256(&self, i: usize) -> &[f32] {
        &self.q256[i * PDIM..][..PDIM]
    }
}

struct Truth {
    k: usize,
    ids: Vec<u32>,
}

impl Truth {
    fn load(dir: &Path, name: &str, nq: usize) -> Result<Self> {
        let ids: Vec<u32> = read_vec(&dir.join(format!("{name}.ids")))?;
        Ok(Self { k: ids.len() / nq, ids })
    }

    fn get(&self, q: usize) -> &[u32] {
        &self.ids[q * self.k..][..self.k]
    }
}

fn parse_config(s: &str, nclusters: usize) -> Result<(String, Params)> {
    let f: Vec<&str> = s.split(':').collect();
    let (np, cand, rr, prune) = match f[..] {
        [np, cand, rr] => (np, cand, rr, None),
        [np, cand, rr, b] if b.starts_with('b') => {
            let (m, w) = b[1..].split_once('x').unwrap_or((&b[1..], "1"));
            (np, cand, rr, Some((m.parse().context("bM")?, w.parse().context("bMxW")?)))
        }
        _ => bail!("config {s}: expected nprobe:cand:rerank[:bM[xW]]"),
    };
    let nprobe = if np == "all" { nclusters } else { np.parse()? };
    let rerank = match rr {
        "none" => Rerank::None,
        "prefix" => Rerank::Prefix,
        "full" => Rerank::Full,
        t if t.starts_with("tiered") => Rerank::Tiered(t["tiered".len()..].parse().context("tieredN")?),
        _ => bail!("rerank must be none, prefix, full or tieredN"),
    };
    let mut label = format!("ivf {np} probes, {cand} cand, rerank {rr}");
    if let Some((m, w)) = prune {
        label += &format!(", bound-pruned at rank {m}, width {w}");
    }
    Ok((label, Params { nprobe, cand: cand.parse()?, rerank, k: 100, prune }))
}

/// One ranked list of corpus rows per query.
type Run = Vec<Vec<u32>>;

struct Scored {
    label: String,
    recall768: f64,
    recall256: f64,
    ndcg: Vec<(String, f64)>,
    ms_per_query: Option<f64>,
    scanned: Option<f64>,
    reranked: Option<f64>,
}

fn score(label: &str, run: &Run, qs: &Queries, gt768: &Truth, gt256: &Truth, corpus: &Corpus, qrels: &HashMap<String, Qrels>) -> Scored {
    let recall768 = mean((0..qs.len()).map(|i| recall_at(10, &run[i], gt768.get(i))));
    let recall256 = mean((0..qs.len()).map(|i| recall_at(10, &run[i], gt256.get(i))));
    let ndcg = EVAL_SETS
        .iter()
        .map(|(set, ..)| {
            let per_query = (0..qs.len()).filter(|&i| qs.set[i] == *set).filter_map(|i| {
                let judged = qrels.get(*set).and_then(|q| q.get(&qs.id[i]));
                ndcg10(run[i].iter().map(|&r| corpus.doc_id(r)), judged)
            });
            (set.to_string(), mean(per_query))
        })
        .collect();
    Scored { label: label.into(), recall768, recall256, ndcg, ms_per_query: None, scanned: None, reranked: None }
}

fn flat_hamming_rerank(corpus: &Corpus, qs: &Queries, cand: usize) -> Run {
    let codes = corpus.codes.as_slice();
    (0..qs.len())
        .into_par_iter()
        .map(|i| {
            let mut qc = [0u8; CODE_BYTES];
            sign_code(qs.q256(i), &mut qc);
            let mut top = TopK::new(cand);
            for (id, c) in codes.chunks_exact(CODE_BYTES).enumerate() {
                top.push((PDIM as u32 - hamming(&qc, c)) as u16, id as u32);
            }
            let w = corpus.quant.query_weights(qs.q768(i));
            let mut out = TopK::new(100);
            for (_, id) in top.into_sorted() {
                out.push(dot_i8(&w, corpus.rerank_row(id)), id);
            }
            out.into_sorted().into_iter().map(|x| x.1).collect()
        })
        .collect()
}

/// Returns the run, inverse throughput (ms) and mean vectors scanned and reranked per query.
fn ivf_run(index: &Index, corpus: &Corpus, qs: &Queries, p: &Params) -> (Run, f64, f64, f64) {
    let t0 = Instant::now();
    let res: Vec<(Vec<u32>, u64, u64)> = (0..qs.len())
        .into_par_iter()
        .map_init(Scratch::new, |s, i| {
            let mut t = Timing::default();
            let r = index.search(corpus, qs.q768(i), qs.q256(i), p, s, &mut t);
            (r.into_iter().map(|x| x.1).collect(), t.scanned, t.reranked)
        })
        .collect();
    let wall = t0.elapsed().as_secs_f64();
    let scanned = mean(res.iter().map(|r| r.1 as f64));
    let reranked = mean(res.iter().map(|r| r.2 as f64));
    // Wall time across threads, so this is inverse throughput, not latency.
    let ms = wall * 1e3 / qs.len() as f64;
    (res.into_iter().map(|r| r.0).collect(), ms, scanned, reranked)
}

fn print_table(rows: &[Scored]) {
    let sets: Vec<&str> = EVAL_SETS.iter().map(|s| s.0).collect();
    print!("| run | R@10 vs 768 | R@10 vs 256 |");
    sets.iter().for_each(|s| print!(" nDCG@10 {s} |"));
    println!(" vectors scanned | reranked |");
    println!("|---|---|---|{}---|---|", "---|".repeat(sets.len()));
    for r in rows {
        print!("| {} | {:.4} | {:.4} |", r.label, r.recall768, r.recall256);
        r.ndcg.iter().for_each(|(_, v)| print!(" {v:.4} |"));
        let f = |v: Option<f64>| v.map_or("-".into(), |s| format!("{s:.0}"));
        println!(" {} | {} |", f(r.scanned), f(r.reranked));
    }
}

fn log(path: &Path, v: serde_json::Value) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{v}")?;
    Ok(())
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<()> {
    let a = Args::parse();
    let corpus = Corpus::open(&a.ingest)?;
    let index = Index::open(&a.index)?;
    let qs = Queries::load(&a.ingest)?;
    eprintln!("{} rows, {} clusters, {} queries", corpus.n, index.k, qs.len());

    match a.cmd {
        Cmd::Eval { configs, hamming } => {
            let gt768 = Truth::load(&a.ingest, "gt768", qs.len())?;
            let gt256 = Truth::load(&a.ingest, "gt256", qs.len())?;
            let mut qrels = HashMap::new();
            for (set, _, file, _) in EVAL_SETS {
                qrels.insert(set.to_string(), read_qrels(&a.qrels.join(file))?);
            }
            let as_run = |t: &Truth| -> Run { (0..qs.len()).map(|i| t.get(i).to_vec()).collect() };

            let mut rows = vec![
                score("flat f32 768-d (exact)", &as_run(&gt768), &qs, &gt768, &gt256, &corpus, &qrels),
                score("flat f32 256-d prefix (exact)", &as_run(&gt256), &qs, &gt768, &gt256, &corpus, &qrels),
            ];
            if hamming {
                let t = Instant::now();
                let run = flat_hamming_rerank(&corpus, &qs, 1000);
                let mut s = score("flat hamming, 1000 cand, rerank full", &run, &qs, &gt768, &gt256, &corpus, &qrels);
                s.ms_per_query = Some(t.elapsed().as_secs_f64() * 1e3 / qs.len() as f64);
                s.scanned = Some(corpus.n as f64);
                rows.push(s);
            }
            for c in configs.split(',') {
                let (label, p) = parse_config(c, index.k)?;
                let (run, ms, scanned, reranked) = ivf_run(&index, &corpus, &qs, &p);
                let mut s = score(&label, &run, &qs, &gt768, &gt256, &corpus, &qrels);
                s.ms_per_query = Some(ms);
                s.scanned = Some(scanned);
                s.reranked = Some(reranked);
                eprintln!("{label}: R@10 {:.4}, {reranked:.0} reranked", s.recall768);
                rows.push(s);
            }
            print_table(&rows);
            for r in &rows {
                log(&a.log, serde_json::json!({
                    "kind": "eval", "rows": corpus.n, "clusters": index.k, "run": r.label,
                    "recall10_768": r.recall768, "recall10_256": r.recall256,
                    "ndcg10": r.ndcg.iter().cloned().collect::<HashMap<_, _>>(),
                    "inverse_qps_ms": r.ms_per_query, "scanned": r.scanned, "reranked": r.reranked,
                }))?;
            }
        }
        Cmd::Routing => {
            let gt768 = Truth::load(&a.ingest, "gt768", qs.len())?;
            let ids = index.ids.as_slice();
            // Up to two clusters per row once SOAR spilling is on.
            let mut cluster_of = vec![[u32::MAX; 2]; corpus.n];
            for c in 0..index.k {
                let (b0, b1) = (index.block_off[c] as usize, index.block_off[c + 1] as usize);
                for &id in &ids[b0 * sieve::lut::BLOCK..b1 * sieve::lut::BLOCK] {
                    if id != sieve::ivf::PAD {
                        let slot = &mut cluster_of[id as usize];
                        slot[(slot[0] != u32::MAX) as usize] = c as u32;
                    }
                }
            }
            // Rank (by centroid score) of the cluster holding each of the exact top-10.
            let ranks: Vec<Vec<usize>> = (0..qs.len())
                .into_par_iter()
                .map(|i| {
                    let s: Vec<f32> = index.centroids.chunks_exact(PDIM).map(|c| sieve::quant::dot_f32(c, qs.q256(i))).collect();
                    let mut order: Vec<u32> = (0..index.k as u32).collect();
                    order.sort_unstable_by(|&a, &b| s[b as usize].partial_cmp(&s[a as usize]).unwrap());
                    let mut rank = vec![0usize; index.k];
                    order.iter().enumerate().for_each(|(r, &c)| rank[c as usize] = r);
                    gt768.get(i)[..10]
                        .iter()
                        .map(|&id| cluster_of[id as usize].iter().filter(|&&c| c != u32::MAX).map(|&c| rank[c as usize]).min().unwrap())
                        .collect()
                })
                .collect();
            println!("| nprobe | % of clusters | top-10 reachable |");
            println!("|---|---|---|");
            let mut np = 1;
            while np <= index.k {
                let reach = mean(ranks.iter().map(|r| r.iter().filter(|&&x| x < np).count() as f64 / 10.0));
                println!("| {np} | {:.2}% | {reach:.4} |", 100.0 * np as f64 / index.k as f64);
                log(&a.log, serde_json::json!({"kind": "routing", "rows": corpus.n, "clusters": index.k, "nprobe": np, "reachable10": reach}))?;
                np *= 2;
            }
        }
        Cmd::Bench { config, seconds, cold } => {
            let (label, p) = parse_config(&config, index.k)?;
            if cold {
                corpus.ssd.bypass_cache();
            }
            index.blocks.will_need();
            // Warm-up pass so the numbers describe a resident index, as a serving node would be.
            ivf_run(&index, &corpus, &qs, &p);

            let mut s = Scratch::new();
            let mut lat = Vec::with_capacity(qs.len());
            let mut stages = Timing::default();
            for i in 0..qs.len() {
                let t = Instant::now();
                std::hint::black_box(index.search(&corpus, qs.q768(i), qs.q256(i), &p, &mut s, &mut stages));
                lat.push(t.elapsed().as_secs_f64() * 1e3);
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let per = |ns: u64| ns as f64 / 1e6 / qs.len() as f64;

            let t0 = Instant::now();
            let mut done = 0usize;
            while t0.elapsed().as_secs_f64() < seconds {
                ivf_run(&index, &corpus, &qs, &p);
                done += qs.len();
            }
            let qps = done as f64 / t0.elapsed().as_secs_f64();

            let v = serde_json::json!({
                "kind": "bench", "rows": corpus.n, "clusters": index.k, "run": label, "cold": cold,
                "threads": rayon::current_num_threads(),
                "latency_ms": { "mean": mean(lat.iter().copied()), "p50": pct(&lat, 0.5), "p99": pct(&lat, 0.99), "max": pct(&lat, 1.0) },
                "stage_ms": { "route": per(stages.route), "scan": per(stages.scan), "rerank": per(stages.rerank), "tier2": per(stages.tier2) },
                "scanned_per_query": stages.scanned as f64 / qs.len() as f64,
                "reranked_per_query": stages.reranked as f64 / qs.len() as f64,
                "qps": qps,
            });
            println!("{}", serde_json::to_string_pretty(&v)?);
            log(&a.log, v)?;
        }
    }
    Ok(())
}
