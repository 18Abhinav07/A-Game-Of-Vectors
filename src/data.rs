//! Readers for the Snowflake Arctic MS MARCO v2.1 release: corpus shards, topics and qrels.

use anyhow::{Context, Result, bail};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, sync_channel};

use crate::DIM;

pub struct Batch {
    pub ids: Vec<String>,
    /// rows × DIM, as stored (not normalised).
    pub emb: Vec<f32>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

fn list_f32(batch: &RecordBatch, col: &str) -> Result<Vec<f32>> {
    let list = batch.column_by_name(col).with_context(|| format!("no column {col}"))?.as_list::<i32>();
    let offs = list.value_offsets();
    for w in offs.windows(2) {
        if (w[1] - w[0]) as usize != DIM {
            bail!("{col}: row has {} values, expected {DIM}", w[1] - w[0]);
        }
    }
    let vals = list.values().as_primitive::<Float32Type>().values();
    Ok(vals[offs[0] as usize..offs[offs.len() - 1] as usize].to_vec())
}

fn strings(batch: &RecordBatch, col: &str) -> Result<Vec<String>> {
    let a = batch.column_by_name(col).with_context(|| format!("no column {col}"))?;
    Ok(a.as_string::<i32>().iter().map(|s| s.unwrap_or_default().to_owned()).collect())
}

/// Decodes the shards in order on a background thread so parquet decoding overlaps compute.
/// With `wait`, a shard that is not on disk yet is waited for (it is still downloading).
/// Shards at position `consume_from` and later are deleted once read, with a `.consumed` marker
/// left in their place, so the full corpus never has to fit on disk at once.
pub fn stream_corpus(paths: Vec<PathBuf>, batch_rows: usize, wait: bool, consume_from: usize) -> Receiver<Result<Batch>> {
    let (tx, rx) = sync_channel(2);
    std::thread::spawn(move || {
        for (i, p) in paths.into_iter().enumerate() {
            let consume = i >= consume_from;
            while wait && !p.exists() {
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            let r = (|| -> Result<()> {
                let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&p)?)?
                    .with_batch_size(batch_rows)
                    .build()?;
                for rb in reader {
                    let rb = rb?;
                    let b = Batch { ids: strings(&rb, "doc_id")?, emb: list_f32(&rb, "embedding")? };
                    if tx.send(Ok(b)).is_err() {
                        return Ok(());
                    }
                }
                if consume {
                    let mut marker = p.clone().into_os_string();
                    marker.push(".consumed");
                    std::fs::write(&marker, b"")?;
                    std::fs::remove_file(&p)?;
                }
                Ok(())
            })();
            if let Err(e) = r {
                let _ = tx.send(Err(e.context(format!("reading {}", p.display()))));
                return;
            }
        }
    });
    rx
}

pub struct Query {
    pub set: String,
    pub id: String,
    pub text: String,
    pub emb: Vec<f32>,
}

pub fn read_topics(path: &Path, set: &str, limit: Option<usize>) -> Result<Vec<Query>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
    let mut out = Vec::new();
    for rb in reader {
        let rb = rb?;
        let ids = strings(&rb, "id")?;
        let texts = strings(&rb, "text")?;
        let emb = list_f32(&rb, "embedding")?;
        for (i, (id, text)) in ids.into_iter().zip(texts).enumerate() {
            out.push(Query { set: set.into(), id, text, emb: emb[i * DIM..(i + 1) * DIM].to_vec() });
        }
    }
    if let Some(l) = limit {
        out.truncate(l);
    }
    Ok(out)
}

/// query id → (document id → graded relevance).
pub type Qrels = HashMap<String, HashMap<String, u32>>;

pub fn read_qrels(path: &Path) -> Result<Qrels> {
    let v: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
    let mut out = Qrels::new();
    for (q, docs) in v.as_object().context("qrels: expected object")? {
        let m = out.entry(q.clone()).or_default();
        for (d, rel) in docs.as_object().context("qrels: expected object per query")? {
            let r = match rel {
                serde_json::Value::String(s) => s.parse()?,
                serde_json::Value::Number(n) => n.as_u64().context("qrels: bad number")? as u32,
                _ => bail!("qrels: bad relevance {rel}"),
            };
            m.insert(d.clone(), r);
        }
    }
    Ok(out)
}

/// The evaluation sets: (set name, topics file, qrels file, query limit).
pub const EVAL_SETS: &[(&str, &str, &str, Option<usize>)] = &[
    ("dl21", "dl21.parquet", "dl21-doc-msmarco-v2.1.json", None),
    ("dl22", "dl22.parquet", "dl22-doc-msmarco-v2.1.json", None),
    ("dl23", "dl23.parquet", "dl23-doc-msmarco-v2.1.json", None),
    ("dev", "msmarco-v2-doc.dev.parquet", "msmarco-v2.1-doc.dev.json", Some(1000)),
];

/// Passage ids look like `msmarco_v2.1_doc_00_0#3_1234`; qrels are judged per document.
pub fn doc_of(passage_id: &str) -> &str {
    passage_id.split('#').next().unwrap_or(passage_id)
}
