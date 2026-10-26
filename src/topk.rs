//! Bounded top-k with amortised selection: buffer up to 2k items, then keep the best k
//! and raise the admission threshold. Cheaper than a heap when most pushes are rejected.

pub struct TopK<S> {
    k: usize,
    items: Vec<(S, u32)>,
    threshold: Option<S>,
}

impl<S: Copy + PartialOrd> TopK<S> {
    pub fn new(k: usize) -> Self {
        Self { k, items: Vec::with_capacity(2 * k.max(1)), threshold: None }
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.threshold = None;
    }

    /// Scores at or below this can be skipped without calling `push`.
    #[inline]
    pub fn threshold(&self) -> Option<S> {
        self.threshold
    }

    #[inline]
    pub fn push(&mut self, score: S, id: u32) {
        if let Some(t) = self.threshold
            && score <= t {
                return;
            }
        self.items.push((score, id));
        if self.items.len() >= 2 * self.k {
            self.compact();
        }
    }

    fn compact(&mut self) {
        if self.items.len() > self.k {
            self.items.select_nth_unstable_by(self.k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
            self.items.truncate(self.k);
            self.threshold = Some(self.items[self.k - 1].0);
        }
    }

    /// Best first.
    pub fn into_sorted(mut self) -> Vec<(S, u32)> {
        self.sorted_in_place();
        self.items
    }

    pub fn sorted(&mut self) -> &[(S, u32)] {
        self.sorted_in_place();
        &self.items
    }

    fn sorted_in_place(&mut self) {
        self.compact();
        self.items.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        self.items.truncate(self.k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn matches_full_sort(scores in proptest::collection::vec(0u16..500, 0..2000), k in 1usize..64) {
            let mut t = TopK::new(k);
            for (i, &s) in scores.iter().enumerate() {
                t.push(s, i as u32);
            }
            let got: Vec<u16> = t.into_sorted().iter().map(|x| x.0).collect();
            let mut want = scores.clone();
            want.sort_unstable_by(|a, b| b.cmp(a));
            want.truncate(k);
            prop_assert_eq!(got, want);
        }
    }
}
