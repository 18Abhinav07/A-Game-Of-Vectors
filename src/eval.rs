//! Recall against exact neighbours, and TREC-style nDCG@10 against NIST judgements.

use crate::data::{Qrels, doc_of};
use std::collections::{HashMap, HashSet};

/// |got[..k] ∩ truth[..k]| / k
pub fn recall_at(k: usize, got: &[u32], truth: &[u32]) -> f64 {
    let t: HashSet<u32> = truth.iter().take(k).copied().collect();
    got.iter().take(k).filter(|id| t.contains(id)).count() as f64 / k.min(t.len()).max(1) as f64
}

/// Document-level nDCG@10 from a passage ranking: a document takes the rank of its best passage
/// (MaxP, as in the Snowflake baselines). Linear gain, log2 discount, as trec_eval's ndcg_cut.
/// Returns None for queries without judgements.
pub fn ndcg10<'a>(ranked_passages: impl IntoIterator<Item = &'a str>, judged: Option<&HashMap<String, u32>>) -> Option<f64> {
    let judged = judged.filter(|j| !j.is_empty())?;
    let mut seen = HashSet::new();
    let mut dcg = 0.0;
    for p in ranked_passages {
        let d = doc_of(p);
        if !seen.insert(d) {
            continue;
        }
        let rank = seen.len() - 1;
        dcg += *judged.get(d).unwrap_or(&0) as f64 / (rank as f64 + 2.0).log2();
        if seen.len() == 10 {
            break;
        }
    }
    let mut ideal: Vec<u32> = judged.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal.iter().take(10).enumerate().map(|(i, &r)| r as f64 / (i as f64 + 2.0).log2()).sum();
    (idcg > 0.0).then(|| dcg / idcg)
}

pub fn mean(xs: impl IntoIterator<Item = f64>) -> f64 {
    let (s, n) = xs.into_iter().fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 { f64::NAN } else { s / n as f64 }
}

pub fn qrels_for<'a>(qrels: &'a Qrels, qid: &str) -> Option<&'a HashMap<String, u32>> {
    qrels.get(qid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_perfect_and_maxp() {
        let mut j = HashMap::new();
        j.insert("a".to_string(), 3);
        j.insert("b".to_string(), 1);
        let perfect = ndcg10(["a#0", "a#1", "b#0"], Some(&j)).unwrap();
        assert!((perfect - 1.0).abs() < 1e-12, "second passage of a must not take a rank");
        let swapped = ndcg10(["b#0", "a#0"], Some(&j)).unwrap();
        assert!(swapped < 1.0);
        assert!(ndcg10(["a#0"], None).is_none());
    }

    #[test]
    fn recall_counts_overlap() {
        assert_eq!(recall_at(2, &[1, 2, 3], &[2, 9, 1]), 0.5);
    }
}
