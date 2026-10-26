//! Sharding (DESIGN.md D16): clusters are partitioned across shard processes, a coordinator
//! routes each query against the global centroids and fans the probed clusters out to the
//! shards that own them, and each shard scans and reranks its own clusters and returns exact
//! top-k scores, which the coordinator merges.
//!
//! Two protocols. One-shot: each shard scans and reranks its clusters with the full candidate
//! and rerank budgets and returns exact top-k; one round trip, but rerank work grows with the
//! fan-out. Two-phase: shards return their candidates' code estimates, the coordinator makes
//! the global candidate cut, prune and rerank cut that a single node would make (RaBitQ
//! estimates are calibrated inner products, so they compare across shards), then asks each
//! shard to rerank only its share; two round trips, single-node work.
//!
//! The wire format is length-prefixed little-endian binary over TCP: a request is a fixed
//! header, the 768-d query, then (centroid score, cluster) pairs to scan, or in phase 2
//! (estimate, row) pairs to rerank; a reply is the shard's timing counters, then (score, row)
//! pairs best first, or in phase 1 (estimate, row, error) triples.

use anyhow::{Context, Result, bail, ensure};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};

use crate::DIM;
use crate::ivf::{Corpus, Index, Params, Rerank, Scratch, Timing, View, prune_threshold};
use crate::topk::TopK;

/// Owner shard of every cluster: longest-processing-time greedy on stored slots, so each shard
/// holds about the same number of vectors. Deterministic, so the coordinator and every shard
/// compute the same map from `block_off` alone.
pub fn assign(index: &Index, shards: usize) -> Vec<u16> {
    balance(&(0..index.k as u32).map(|c| index.slots(c)).collect::<Vec<_>>(), shards)
}

fn balance(sizes: &[usize], shards: usize) -> Vec<u16> {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&c| (std::cmp::Reverse(sizes[c]), c));
    let mut load = vec![0usize; shards];
    let mut owner = vec![0u16; sizes.len()];
    for c in order {
        let s = (0..shards).min_by_key(|&s| (load[s], s)).unwrap();
        load[s] += sizes[c];
        owner[c] = s as u16;
    }
    owner
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Header {
    cand: u32,
    /// 0 none, 1 prefix, 2 full, 3 tiered, 4 ssd.
    rerank: u32,
    rerank_r: u32,
    k: u32,
    /// 0 when pruning is off.
    prune_m: u32,
    prune_w: f32,
    nprobe: u32,
    /// 0 one-shot, 1 candidates only, 2 rerank the given rows.
    phase: u32,
}

impl Header {
    fn new(p: &Params, nprobe: usize, phase: u32) -> Self {
        let (rerank, rerank_r) = match p.rerank {
            Rerank::None => (0, 0),
            Rerank::Prefix => (1, 0),
            Rerank::Full => (2, 0),
            Rerank::Tiered(r) => (3, r as u32),
            Rerank::Ssd(r) => (4, r as u32),
        };
        let (prune_m, prune_w) = p.prune.map_or((0, 0.0), |(m, w)| (m as u32, w));
        Self { cand: p.cand as u32, rerank, rerank_r, k: p.k as u32, prune_m, prune_w, nprobe: nprobe as u32, phase }
    }

    fn params(&self) -> Result<Params> {
        let r = self.rerank_r as usize;
        let rerank = match self.rerank {
            0 => Rerank::None,
            1 => Rerank::Prefix,
            2 => Rerank::Full,
            3 => Rerank::Tiered(r),
            4 => Rerank::Ssd(r),
            x => bail!("unknown rerank tag {x}"),
        };
        let prune = (self.prune_m > 0).then_some((self.prune_m as usize, self.prune_w));
        // Routing already happened at the coordinator, so margin and budget are not needed here.
        Ok(Params { nprobe: self.nprobe as usize, cand: self.cand as usize, rerank, k: self.k as usize, prune, margin: None, budget: None })
    }
}

/// Timing counters a shard sends back, in `Timing` field order (wait excluded).
const COUNTERS: usize = 8;

fn counters(t: &Timing) -> [u64; COUNTERS] {
    [t.route, t.scan, t.rerank, t.tier2, t.scanned, t.reranked, t.probed, t.fetch]
}

fn add_counters(t: &mut Timing, c: &[u64; COUNTERS]) {
    t.route += c[0];
    t.scan += c[1];
    t.rerank += c[2];
    t.tier2 += c[3];
    t.scanned += c[4];
    t.reranked += c[5];
    t.probed += c[6];
    t.fetch += c[7];
}

fn write_frame(w: &mut impl Write, parts: &[&[u8]]) -> std::io::Result<()> {
    let len: usize = parts.iter().map(|p| p.len()).sum();
    w.write_all(&(len as u32).to_le_bytes())?;
    parts.iter().try_for_each(|p| w.write_all(p))?;
    w.flush()
}

/// Reads one frame into `buf`; `Ok(false)` on a clean end of stream.
fn read_frame(r: &mut impl Read, buf: &mut Vec<u8>) -> std::io::Result<bool> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len) {
        Err(e) if matches!(e.kind(), std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::ConnectionReset) => return Ok(false),
        x => x?,
    }
    buf.resize(u32::from_le_bytes(len) as usize, 0);
    r.read_exact(buf)?;
    Ok(true)
}

/// Pairs travel as two u32 words: f32 bits, then the id.
fn pairs_to_words(pairs: &[(f32, u32)]) -> Vec<u32> {
    pairs.iter().flat_map(|&(v, id)| [v.to_bits(), id]).collect()
}

fn words_to_pairs(bytes: &[u8]) -> Vec<(f32, u32)> {
    bytes.chunks_exact(8).map(|w| (f32_at(w, 0), u32_at(w, 4))).collect()
}

fn triples_to_words(t: &[(f32, u32, f32)]) -> Vec<u32> {
    t.iter().flat_map(|&(v, id, e)| [v.to_bits(), id, e.to_bits()]).collect()
}

fn words_to_triples(bytes: &[u8]) -> Vec<(f32, u32, f32)> {
    bytes.chunks_exact(12).map(|w| (f32_at(w, 0), u32_at(w, 4), f32_at(w, 8))).collect()
}

fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

/// Serves queries for the clusters `owner` assigns to shard `me`, one thread per connection.
pub fn serve(listener: TcpListener, index: &Index, corpus: &Corpus, owner: &[u16], me: u16) -> Result<()> {
    std::thread::scope(|scope| {
        for conn in listener.incoming() {
            let conn = conn?;
            conn.set_nodelay(true)?;
            scope.spawn(move || {
                if let Err(e) = handle(conn, index, corpus, owner, me) {
                    eprintln!("shard {me}: connection closed: {e:#}");
                }
            });
        }
        Ok(())
    })
}

fn handle(conn: TcpStream, index: &Index, corpus: &Corpus, owner: &[u16], me: u16) -> Result<()> {
    let mut r = BufReader::new(conn.try_clone()?);
    let mut w = BufWriter::new(conn);
    let (mut buf, mut s) = (Vec::new(), Scratch::new());
    let mut q256 = vec![0f32; crate::PDIM];
    let hlen = std::mem::size_of::<Header>();
    while read_frame(&mut r, &mut buf)? {
        ensure!(buf.len() >= hlen + DIM * 4, "short request");
        let h: Header = bytemuck::pod_read_unaligned(&buf[..hlen]);
        let p = h.params()?;
        let q768: Vec<f32> = buf[hlen..hlen + DIM * 4].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let pairs = words_to_pairs(&buf[hlen + DIM * 4..]);
        crate::quant::prefix(&q768, &mut q256);
        let mut t = Timing::default();
        if h.phase != 2
            && let Some(&(_, c)) = pairs.iter().find(|&&(_, c)| owner.get(c as usize) != Some(&me)) {
                bail!("cluster {c} is not owned by shard {me}");
            }
        let words = match h.phase {
            0 => pairs_to_words(&index.search_probes(corpus, &q768, &q256, &pairs, &p, &mut s, &mut t)),
            1 => triples_to_words(&index.candidates(&q256, &pairs, &p, &View::default(), &mut s, &mut t)),
            2 => {
                let cand = pairs.iter().map(|&(est, id)| (est, id, 0.0)).collect();
                pairs_to_words(&index.rerank(corpus, &q768, &q256, cand, &p, &View::default(), &mut s, &mut t))
            }
            x => bail!("unknown phase {x}"),
        };
        write_frame(&mut w, &[bytemuck::cast_slice(&counters(&t)), bytemuck::cast_slice(&words)])?;
    }
    Ok(())
}

/// One coordinator thread's connections, one per shard.
pub struct Client {
    conns: Vec<(BufReader<TcpStream>, BufWriter<TcpStream>)>,
    owner: Vec<u16>,
    buf: Vec<u8>,
    two_phase: bool,
}

impl Client {
    pub fn connect(addrs: &[String], owner: Vec<u16>, two_phase: bool) -> Result<Self> {
        let conns = addrs
            .iter()
            .map(|a| {
                let c = TcpStream::connect(a).with_context(|| format!("connect to shard {a}"))?;
                // Without it, Nagle holds the small reply behind the delayed ACK (~40 ms).
                c.set_nodelay(true)?;
                Ok((BufReader::new(c.try_clone()?), BufWriter::new(c)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { conns, owner, buf: Vec::new(), two_phase })
    }

    /// Fans `probe` out to the owning shards and merges their results. Every request of a round
    /// is written before any reply is read, so the shards work in parallel. `t.wait` is the
    /// wall time from the first send to the last reply; the other counters are summed over
    /// shards.
    pub fn search(&mut self, q768: &[f32], probe: &[(f32, u32)], p: &Params, t: &mut Timing) -> Result<Vec<(f32, u32)>> {
        let t0 = std::time::Instant::now();
        let mut per: Vec<Vec<(f32, u32)>> = vec![Vec::new(); self.conns.len()];
        probe.iter().for_each(|&x| per[self.owner[x.1 as usize] as usize].push(x));
        let out = match self.two_phase {
            false => self.round(q768, &Header::new(p, probe.len(), 0), &per, t)?,
            true => self.two_phase(q768, probe.len(), &per, p, t)?,
        };
        t.wait += t0.elapsed().as_nanos() as u64;
        // A SOAR copy can live on two shards; both return the same exact score.
        let mut best: HashMap<u32, f32> = HashMap::new();
        for (v, id) in out.into_iter().flat_map(|(_, words)| words_to_pairs(&words)) {
            let e = best.entry(id).or_insert(v);
            *e = e.max(v);
        }
        let mut top = TopK::new(p.k);
        best.into_iter().for_each(|(id, v)| top.push(v, id));
        Ok(top.into_sorted())
    }

    /// Phase 1 gathers every shard's candidates; the coordinator then cuts them exactly as
    /// `Index::candidates` and `Index::rerank` would on one node, and phase 2 has each row
    /// reranked by the shard that returned it.
    fn two_phase(&mut self, q768: &[f32], nprobe: usize, per: &[Vec<(f32, u32)>], p: &Params, t: &mut Timing) -> Result<Vec<(usize, Vec<u8>)>> {
        let mut all: Vec<(f32, u32, f32, usize)> = Vec::new();
        for (shard, bytes) in self.round(q768, &Header::new(p, nprobe, 1), per, t)? {
            all.extend(words_to_triples(&bytes).into_iter().map(|(est, id, err)| (est, id, err, shard)));
        }
        // Each shard sent its best `cand` slots, so the union holds the global best `cand`.
        all.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        all.truncate(p.cand);
        all.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(b.0.total_cmp(&a.0)));
        all.dedup_by_key(|x| x.1);
        if let Some((m, w)) = p.prune {
            let tri: Vec<(f32, u32, f32)> = all.iter().map(|x| (x.0, x.1, x.2)).collect();
            let tau = prune_threshold(&tri, m, w);
            all.retain(|x| x.0 + w * x.2 >= tau);
        }
        let n = all.len();
        let rerank = match p.rerank {
            Rerank::None => {
                // The estimates are the answer: no second round.
                let words = pairs_to_words(&all.iter().map(|x| (x.0, x.1)).collect::<Vec<_>>());
                return Ok(vec![(0, bytemuck::cast_slice(&words).to_vec())]);
            }
            Rerank::Ssd(r) => {
                // The rows worth reading are chosen by estimate: cut globally here.
                all.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                all.truncate(r);
                None
            }
            other => Some(other),
        };
        let mut rows: Vec<Vec<(f32, u32)>> = vec![Vec::new(); self.conns.len()];
        all.iter().for_each(|x| rows[x.3].push((x.0, x.1)));
        // The second round sends rows, not clusters, so headers differ per shard.
        let headers: Vec<Header> = rows
            .iter()
            .map(|list| {
                let rr = match rerank {
                    None => Rerank::Ssd(list.len()),
                    // The tier-2 cut is by tier-2 score, which only the shard computes: give each
                    // shard its share of `r`.
                    Some(Rerank::Tiered(r)) => Rerank::Tiered((r * list.len()).div_ceil(n.max(1))),
                    Some(x) => x,
                };
                Header::new(&Params { rerank: rr, prune: None, ..*p }, 0, 2)
            })
            .collect();
        self.round_each(q768, &headers, &rows, t)
    }

    fn round(&mut self, q768: &[f32], h: &Header, per: &[Vec<(f32, u32)>], t: &mut Timing) -> Result<Vec<(usize, Vec<u8>)>> {
        let headers = vec![*h; per.len()];
        self.round_each(q768, &headers, per, t)
    }

    /// One request to every shard with a non-empty list, then all replies (payload after the
    /// counters) tagged by shard.
    fn round_each(&mut self, q768: &[f32], headers: &[Header], per: &[Vec<(f32, u32)>], t: &mut Timing) -> Result<Vec<(usize, Vec<u8>)>> {
        for (shard, list) in per.iter().enumerate().filter(|x| !x.1.is_empty()) {
            write_frame(&mut self.conns[shard].1, &[bytemuck::bytes_of(&headers[shard]), bytemuck::cast_slice(q768), bytemuck::cast_slice(&pairs_to_words(list))])?;
        }
        let mut out = Vec::new();
        for (shard, _) in per.iter().enumerate().filter(|x| !x.1.is_empty()) {
            ensure!(read_frame(&mut self.conns[shard].0, &mut self.buf)?, "shard {shard} hung up");
            let c: [u64; COUNTERS] = bytemuck::pod_read_unaligned(&self.buf[..COUNTERS * 8]);
            add_counters(t, &c);
            out.push((shard, self.buf[COUNTERS * 8..].to_vec()));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips_params() {
        let p = Params { nprobe: 64, cand: 500, rerank: Rerank::Ssd(300), k: 100, prune: Some((10, 0.5)), margin: Some(0.1), budget: Some(1000) };
        let q = Header::new(&p, 40, 0).params().unwrap();
        assert_eq!((q.cand, q.rerank, q.k, q.prune, q.nprobe), (500, Rerank::Ssd(300), 100, Some((10, 0.5)), 40));
        assert!(Header::new(&Params { prune: None, ..p }, 1, 0).params().unwrap().prune.is_none());
    }

    #[test]
    fn balance_evens_out_load() {
        let owner = balance(&[9, 7, 5, 4, 3, 2], 2);
        let load = |s| [9, 7, 5, 4, 3, 2].iter().zip(&owner).filter(|x| *x.1 == s).map(|x| x.0).sum::<usize>();
        assert_eq!((load(0), load(1)), (15, 15));
    }

    #[test]
    fn pairs_survive_the_wire() {
        let pairs = [(0.25f32, 7u32), (-1.5, u32::MAX - 1)];
        assert_eq!(words_to_pairs(bytemuck::cast_slice(&pairs_to_words(&pairs))), pairs);
        let tri = [(0.5f32, 3u32, 0.01f32)];
        assert_eq!(words_to_triples(bytemuck::cast_slice(&triples_to_words(&tri))), tri);
    }
}
