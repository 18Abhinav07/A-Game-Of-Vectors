//! BM25 over the passage text, and rank fusion with the vector results (DESIGN.md D18).
//!
//! The inverted index is built in one pass over the gzipped JSONL passage files, which list
//! passages in the same order as the corpus rows. Terms are sharded by hash across threads, each
//! thread appending (row delta, tf) varints to its own terms' posting lists, so the build runs in
//! parallel without locks. The output is a sorted term table (looked up by binary search, so no
//! hash map is rebuilt on open), one posting blob, and per-row lengths and host ids.
//!
//! Scoring is term-at-a-time into a dense accumulator: simple and exact, and its cost is the
//! total posting length of the query terms. Dynamic pruning (MaxScore/WAND) is not implemented.

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use crate::ivf::Corpus;
use crate::store::{Appender, Mapped, read_vec, write_vec};
use crate::topk::TopK;

/// Lucene's English stop set, as Anserini's default analyzer uses.
const STOP: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their",
    "then", "there", "these", "they", "this", "to", "was", "will", "with",
];
const MAX_TOKEN: usize = 32;

/// Harman's S-stemmer: plural endings only. A lighter stemmer than Anserini's Porter.
fn stem(t: &mut String) {
    let n = t.len();
    if n > 4 && t.ends_with("ies") && !t.ends_with("eies") && !t.ends_with("aies") {
        t.truncate(n - 3);
        t.push('y');
    } else if n > 3 && t.ends_with("es") && !(t.ends_with("aes") || t.ends_with("ees") || t.ends_with("oes")) {
        t.truncate(n - 1);
    } else if n > 2 && t.ends_with('s') && !(t.ends_with("us") || t.ends_with("ss")) {
        t.truncate(n - 1);
    }
}

/// Lower-cased alphanumeric runs, stop words dropped, plurals stemmed.
pub fn tokenize(text: &str, mut f: impl FnMut(&str)) {
    let mut cur = String::new();
    let mut flush = |cur: &mut String| {
        if !cur.is_empty() && cur.len() <= MAX_TOKEN && !STOP.contains(&cur.as_str()) {
            stem(cur);
            f(cur);
        }
        cur.clear();
    };
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else {
            flush(&mut cur);
        }
    }
    flush(&mut cur);
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
fn get_varint(b: &[u8], pos: &mut usize) -> u32 {
    let mut v = 0u32;
    let mut shift = 0;
    loop {
        let x = b[*pos];
        *pos += 1;
        v |= ((x & 0x7f) as u32) << shift;
        if x < 0x80 {
            return v;
        }
        shift += 7;
    }
}

/// One thread's share of the vocabulary during the build.
#[derive(Default)]
struct Part {
    dict: HashMap<Box<str>, u32>,
    post: Vec<Vec<u8>>,
    last: Vec<u32>,
    df: Vec<u32>,
}

/// A parsed passage: its row, length in tokens, host, and (hash, term, tf) per distinct term.
struct Doc {
    row: u32,
    len: u32,
    host: String,
    terms: Vec<(u64, String, u32)>,
}

pub struct BuildStats {
    pub rows: usize,
    pub skipped: usize,
    pub terms: usize,
    pub postings: u64,
    pub bytes: u64,
    pub seconds: f64,
}

/// The host of a URL, without a leading "www.".
pub fn host_of(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |x| x.1);
    let host = rest.split(['/', '?', '#', ':']).next().unwrap_or("");
    host.strip_prefix("www.").unwrap_or(host)
}

/// Builds the index in `out` from the passage files `text` (in corpus order).
pub fn build(corpus: &Corpus, text: &[PathBuf], out: &Path) -> Result<BuildStats> {
    let t0 = std::time::Instant::now();
    std::fs::create_dir_all(out)?;
    const PARTS: usize = 64;
    const BATCH: usize = 200_000;
    let mut parts: Vec<Part> = (0..PARTS).map(|_| Part::default()).collect();
    let mut lens = vec![0u16; corpus.n];
    let mut host_of_row = vec![u32::MAX; corpus.n];
    let mut hosts: HashMap<String, u32> = HashMap::new();
    let (mut next, mut skipped, mut postings) = (0usize, 0usize, 0u64);
    'files: for path in text {
        // Decompress on its own thread so parsing overlaps it.
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<String>>(4);
        let p = path.clone();
        let reader = std::thread::spawn(move || -> Result<()> {
            let f = std::fs::File::open(&p).with_context(|| format!("open {}", p.display()))?;
            let r = std::io::BufReader::with_capacity(4 << 20, flate2::read::MultiGzDecoder::new(std::io::BufReader::with_capacity(4 << 20, f)));
            let mut batch = Vec::with_capacity(BATCH);
            for line in r.lines() {
                batch.push(line?);
                if batch.len() == BATCH && tx.send(std::mem::replace(&mut batch, Vec::with_capacity(BATCH))).is_err() {
                    return Ok(());
                }
            }
            let _ = tx.send(batch);
            Ok(())
        });
        let mut done = false;
        for lines in rx.iter() {
            // Match passages to rows in order; a passage missing from the corpus is skipped.
            let parsed: Vec<Option<(String, String, String)>> = lines
                .par_iter()
                .map(|l| {
                    let v: serde_json::Value = serde_json::from_str(l).ok()?;
                    Some((v["docid"].as_str()?.to_owned(), v["url"].as_str().unwrap_or("").to_owned(), v["segment"].as_str()?.to_owned()))
                })
                .collect();
            let mut keep = Vec::with_capacity(parsed.len());
            for p in parsed.into_iter().flatten() {
                if next == corpus.n {
                    break;
                }
                if p.0 == corpus.doc_id(next as u32) {
                    keep.push((next as u32, p.1, p.2));
                    next += 1;
                } else {
                    skipped += 1;
                }
            }
            let docs: Vec<Doc> = keep
                .into_par_iter()
                .map(|(row, url, seg)| {
                    let mut tf: HashMap<String, u32> = HashMap::new();
                    let mut len = 0;
                    tokenize(&seg, |t| {
                        len += 1;
                        *tf.entry(t.to_owned()).or_insert(0) += 1;
                    });
                    Doc { row, len, host: host_of(&url).to_owned(), terms: tf.into_iter().map(|(t, n)| (fnv(&t), t, n)).collect() }
                })
                .collect();
            for d in &docs {
                lens[d.row as usize] = d.len.min(u16::MAX as u32) as u16;
                let n = hosts.len() as u32;
                host_of_row[d.row as usize] = *hosts.entry(d.host.clone()).or_insert(n);
                postings += d.terms.len() as u64;
            }
            parts.par_iter_mut().enumerate().for_each(|(pi, part)| {
                for d in &docs {
                    for (h, t, n) in &d.terms {
                        if *h as usize % PARTS != pi {
                            continue;
                        }
                        let id = match part.dict.get(t.as_str()) {
                            Some(&id) => id,
                            None => {
                                let id = part.post.len() as u32;
                                part.dict.insert(t.as_str().into(), id);
                                part.post.push(Vec::new());
                                part.last.push(0);
                                part.df.push(0);
                                id
                            }
                        } as usize;
                        // The first posting stores row + 1 so the delta of row 0 is not 0.
                        let delta = d.row + 1 - part.last[id];
                        put_varint(&mut part.post[id], delta);
                        put_varint(&mut part.post[id], *n);
                        part.last[id] = d.row + 1;
                        part.df[id] += 1;
                    }
                }
            });
            if next == corpus.n {
                done = true;
                break;
            }
        }
        // Dropping the receiver stops the reader at its next send.
        drop(rx);
        reader.join().unwrap()?;
        eprintln!("{}: matched {next}/{} rows, {skipped} skipped, {:.0} s", path.display(), corpus.n, t0.elapsed().as_secs_f64());
        if done {
            break 'files;
        }
    }
    ensure!(next == corpus.n, "text covers {next} of {} rows", corpus.n);

    // Terms in sorted order across all parts.
    let mut order: Vec<(&str, u16, u32)> = parts.iter().enumerate().flat_map(|(pi, p)| p.dict.iter().map(move |(t, &id)| (&**t, pi as u16, id))).collect();
    order.par_sort_unstable_by(|a, b| a.0.cmp(b.0));
    let mut w_terms = Appender::create(&out.join("terms.bin"))?;
    let mut w_post = Appender::create(&out.join("post.bin"))?;
    let (mut term_off, mut post_off, mut df) = (vec![0u64], vec![0u64], Vec::with_capacity(order.len()));
    for &(t, pi, id) in &order {
        let p = &parts[pi as usize];
        w_terms.write(t.as_bytes())?;
        term_off.push(term_off.last().unwrap() + t.len() as u64);
        w_post.write(&p.post[id as usize])?;
        post_off.push(post_off.last().unwrap() + p.post[id as usize].len() as u64);
        df.push(p.df[id as usize]);
    }
    (w_terms.finish()?, w_post.finish()?);
    write_vec(&out.join("terms.off"), &term_off)?;
    write_vec(&out.join("post.off"), &post_off)?;
    write_vec(&out.join("df.u32"), &df)?;
    write_vec(&out.join("doclen.u16"), &lens)?;
    write_vec(&out.join("host.u32"), &host_of_row)?;
    let mut names = vec![""; hosts.len()];
    hosts.iter().for_each(|(h, &i)| names[i as usize] = h);
    std::fs::write(out.join("hosts.txt"), names.join("\n"))?;
    Ok(BuildStats { rows: next, skipped, terms: order.len(), postings, bytes: *post_off.last().unwrap(), seconds: t0.elapsed().as_secs_f64() })
}

/// Anserini's MS MARCO passage defaults.
pub const K1: f32 = 0.9;
pub const B: f32 = 0.4;

pub struct Bm25 {
    terms: Mapped<u8>,
    term_off: Mapped<u64>,
    post: Mapped<u8>,
    post_off: Mapped<u64>,
    df: Mapped<u32>,
    /// k1·(1 − b + b·len/avglen) per row.
    norm: Vec<f32>,
    pub n: usize,
}

/// Per-thread accumulator for term-at-a-time scoring.
#[derive(Default)]
pub struct Acc {
    score: Vec<f32>,
    touched: Vec<u32>,
}

impl Bm25 {
    pub fn open(dir: &Path) -> Result<Self> {
        let lens: Vec<u16> = read_vec(&dir.join("doclen.u16"))?;
        let avg = lens.iter().map(|&l| l as f64).sum::<f64>() / lens.len().max(1) as f64;
        let norm = lens.iter().map(|&l| K1 * (1.0 - B + B * l as f32 / avg as f32)).collect();
        Ok(Self {
            terms: Mapped::open(&dir.join("terms.bin"))?,
            term_off: Mapped::open(&dir.join("terms.off"))?,
            post: Mapped::open(&dir.join("post.bin"))?,
            post_off: Mapped::open(&dir.join("post.off"))?,
            df: Mapped::open(&dir.join("df.u32"))?,
            n: lens.len(),
            norm,
        })
    }

    fn term(&self, i: usize) -> &[u8] {
        let off = self.term_off.as_slice();
        &self.terms.as_slice()[off[i] as usize..off[i + 1] as usize]
    }

    fn lookup(&self, t: &str) -> Option<usize> {
        let n = self.df.as_slice().len();
        let (mut lo, mut hi) = (0, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.term(mid).cmp(t.as_bytes()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Top `k` rows for `query` as (score, row). Also returns the postings decoded.
    pub fn search(&self, query: &str, k: usize, acc: &mut Acc) -> (Vec<(f32, u32)>, u64) {
        if acc.score.len() != self.n {
            acc.score = vec![0.0; self.n];
        }
        let mut qtf: HashMap<String, u32> = HashMap::new();
        tokenize(query, |t| *qtf.entry(t.to_owned()).or_insert(0) += 1);
        let (post, off, df) = (self.post.as_slice(), self.post_off.as_slice(), self.df.as_slice());
        let mut decoded = 0u64;
        for (t, &qn) in &qtf {
            let Some(i) = self.lookup(t) else { continue };
            let idf = (1.0 + (self.n as f32 - df[i] as f32 + 0.5) / (df[i] as f32 + 0.5)).ln() * qn as f32;
            let b = &post[off[i] as usize..off[i + 1] as usize];
            let (mut pos, mut row) = (0usize, 0u32);
            while pos < b.len() {
                row += get_varint(b, &mut pos);
                let tf = get_varint(b, &mut pos) as f32;
                let r = row - 1;
                let s = &mut acc.score[r as usize];
                if *s == 0.0 {
                    acc.touched.push(r);
                }
                *s += idf * tf * (K1 + 1.0) / (tf + self.norm[r as usize]);
            }
            decoded += df[i] as u64;
        }
        let mut top = TopK::new(k);
        for &r in &acc.touched {
            top.push(std::mem::take(&mut acc.score[r as usize]), r);
        }
        acc.touched.clear();
        (top.into_sorted(), decoded)
    }
}

/// Reciprocal rank fusion (Cormack et al. 2009): Σ 1/(k + rank) over the lists a row is in.
pub fn rrf(lists: &[(&[u32], f32)], k: f32, depth: usize) -> Vec<u32> {
    let mut s: HashMap<u32, f32> = HashMap::new();
    for &(l, w) in lists {
        for (r, &id) in l.iter().enumerate() {
            *s.entry(id).or_insert(0.0) += w / (k + r as f32 + 1.0);
        }
    }
    let mut v: Vec<(f32, u32)> = s.into_iter().map(|(id, x)| (x, id)).collect();
    v.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    v.into_iter().take(depth).map(|x| x.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens() {
        let mut v = Vec::new();
        tokenize("The Cats' 0-60 times, Ponies and BUSES!", |t| v.push(t.to_owned()));
        assert_eq!(v, ["cat", "0", "60", "time", "pony", "buse"]);
    }

    #[test]
    fn varints() {
        let mut b = Vec::new();
        for v in [0, 1, 127, 128, 300, u32::MAX] {
            put_varint(&mut b, v);
        }
        let mut pos = 0;
        let got: Vec<u32> = (0..6).map(|_| get_varint(&b, &mut pos)).collect();
        assert_eq!(got, [0, 1, 127, 128, 300, u32::MAX]);
    }

    #[test]
    fn hosts() {
        assert_eq!(host_of("http://www.example.com:80/a?b"), "example.com");
        assert_eq!(host_of("https://0-60.reviews/0-60-times/"), "0-60.reviews");
    }

    #[test]
    fn fusion() {
        assert_eq!(rrf(&[(&[1, 2, 3], 1.0), (&[3, 1], 1.0)], 60.0, 3), [1, 3, 2]);
        assert_eq!(rrf(&[(&[1, 2], 1.0), (&[2, 1], 0.5)], 60.0, 2), [1, 2]);
    }
}
