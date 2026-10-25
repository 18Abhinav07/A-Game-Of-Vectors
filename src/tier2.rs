//! The RAM-resident middle tier (DESIGN.md D9): the 256-dim prefix at 2 bits per dimension,
//! Lloyd-Max levels for a Gaussian, scored through one 256-entry table per byte (4 dims).
//! It narrows the LUT candidates to the few that are read from the SSD tier.

use crate::PDIM;

pub const BYTES: usize = PDIM / 4;

/// Lloyd-Max quantiser for N(0, σ²) at 4 levels: thresholds 0, ±0.9816σ; levels ±0.4528σ, ±1.510σ.
const T: f32 = 0.9816;
const LEVELS: [f32; 4] = [-1.510, -0.4528, 0.4528, 1.510];

#[derive(Clone, Debug)]
pub struct Tier2 {
    pub sigma: Vec<f32>,
}

impl Tier2 {
    /// Fits per-dimension σ on normalised prefixes (n × PDIM).
    pub fn fit(rows: &[f32]) -> Self {
        let n = (rows.len() / PDIM).max(1) as f32;
        let mut ss = vec![0f64; PDIM];
        for r in rows.chunks_exact(PDIM) {
            ss.iter_mut().zip(r).for_each(|(s, &x)| *s += (x * x) as f64);
        }
        Self { sigma: ss.iter().map(|&s| ((s / n as f64).sqrt() as f32).max(1e-6)).collect() }
    }

    #[inline]
    fn level(x: f32, s: f32) -> u8 {
        let t = T * s;
        if x < -t { 0 } else if x < 0.0 { 1 } else if x < t { 2 } else { 3 }
    }

    /// Dim i goes to byte i/4, bits 2(i%4)..2(i%4)+2. Returns the reconstruction's norm.
    pub fn encode(&self, prefix: &[f32], out: &mut [u8]) -> f32 {
        out.fill(0);
        let mut nn = 0f32;
        for (i, (&x, &s)) in prefix.iter().zip(&self.sigma).enumerate() {
            let l = Self::level(x, s);
            out[i >> 2] |= l << ((i & 3) * 2);
            let v = LEVELS[l as usize] * s;
            nn += v * v;
        }
        nn.sqrt()
    }

    /// Per-query tables: entry [g][b] is q · reconstruction for the 4 dims packed in byte value b.
    pub fn tables(&self, q: &[f32]) -> Vec<[f32; 256]> {
        (0..BYTES)
            .map(|g| {
                let mut half = [[0f32; 16]; 2];
                for (h, half) in half.iter_mut().enumerate() {
                    for (b, e) in half.iter_mut().enumerate() {
                        *e = (0..2).map(|j| {
                            let i = 4 * g + 2 * h + j;
                            q[i] * LEVELS[(b >> (2 * j)) & 3] * self.sigma[i]
                        }).sum();
                    }
                }
                std::array::from_fn(|b| half[0][b & 15] + half[1][b >> 4])
            })
            .collect()
    }
}

/// Approximate cosine: Σ_g table[g][code[g]] / ||reconstruction||.
#[inline]
pub fn score(tables: &[[f32; 256]], code: &[u8], norm: f32) -> f32 {
    let mut acc = [0f32; 4];
    for (g, (t, &c)) in tables.iter().zip(code).enumerate() {
        acc[g & 3] += t[c as usize];
    }
    (acc[0] + acc[1] + acc[2] + acc[3]) / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_score_matches_reconstruction() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x >> 11) as f32 / (1u64 << 53) as f32 - 0.5 };
        let rows: Vec<f32> = (0..PDIM * 64).map(|_| rnd()).collect();
        let t2 = Tier2::fit(&rows);
        let q: Vec<f32> = (0..PDIM).map(|_| rnd()).collect();
        let tables = t2.tables(&q);
        for r in rows.chunks_exact(PDIM).take(8) {
            let mut code = [0u8; BYTES];
            let norm = t2.encode(r, &mut code);
            let recon: Vec<f32> = (0..PDIM).map(|i| LEVELS[((code[i >> 2] >> ((i & 3) * 2)) & 3) as usize] * t2.sigma[i]).collect();
            let want: f32 = q.iter().zip(&recon).map(|(a, b)| a * b).sum::<f32>() / norm;
            assert!((score(&tables, &code, norm) - want).abs() < 1e-4);
        }
    }
}
