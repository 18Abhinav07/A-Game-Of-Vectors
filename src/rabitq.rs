//! RaBitQ residual codes (Gao & Long, SIGMOD 2024), for inner product inside IVF clusters.
//!
//! A vector x stored in cluster c is coded by the signs of its residual r = x − c after a random
//! orthogonal rotation P, instead of by the signs of x itself. With o = r/‖r‖, o' = Po and the
//! code direction x̄ = sign(o')/√D, the estimator
//!
//!   ⟨x, q⟩ = ⟨c, q⟩ + ‖r‖·⟨o, q⟩ ≈ ⟨c, q⟩ + ‖r‖ · ⟨x̄, Pq⟩ / ⟨x̄, o'⟩
//!
//! is unbiased, and its error is bounded by ε0·‖r‖·√((1 − ⟨x̄,o'⟩²) / (⟨x̄,o'⟩²(D − 1))) with high
//! probability. The rotation spreads the residual's energy evenly over the dimensions, which is
//! what makes one bit per dimension informative. ⟨c, q⟩ is the routing score the query already
//! computed, so the code only has to describe the small residual.
//!
//! Per stored slot this keeps two floats: `mul` = ‖r‖ / (√D·⟨x̄,o'⟩) = ‖r‖² / ‖r'‖₁ and the error
//! half-width `err`. ⟨x̄, Pq⟩·√D = Σ sign_i·(Pq)_i comes from the existing LUT kernel:
//! Σ sign_i·q_i = 2·Σ bit_i·q_i − Σ q_i.

use crate::PDIM;
use crate::ivf::Rng;
use crate::lut::{GROUPS, Lut};

/// Rounds of random sign flips followed by a Walsh-Hadamard transform. Three rounds make the
/// result behave like a uniformly random rotation for this purpose (as in FALCONN and RaBitQ's
/// fast variant) at O(D log D) per vector instead of a D×D matrix product.
const ROUNDS: usize = 3;

/// Confidence parameter of the error bound (the paper's default).
pub const EPS0: f32 = 1.9;

pub struct Rotation {
    signs: Vec<f32>,
}

impl Rotation {
    pub fn new(seed: u64) -> Self {
        Self::with_dim(seed, PDIM)
    }

    /// A rotation of `d`-dim vectors; `d` must be a power of two.
    pub fn with_dim(seed: u64, d: usize) -> Self {
        assert!(d.is_power_of_two());
        let mut rng = Rng::new(seed);
        Self { signs: (0..ROUNDS * d).map(|_| if rng.next() & 1 == 1 { 1.0 } else { -1.0 }).collect() }
    }

    pub fn from_signs(signs: Vec<f32>) -> Self {
        assert!(signs.len().is_multiple_of(ROUNDS) && (signs.len() / ROUNDS).is_power_of_two(), "rotation file");
        Self { signs }
    }

    pub fn dim(&self) -> usize {
        self.signs.len() / ROUNDS
    }

    pub fn signs(&self) -> &[f32] {
        &self.signs
    }

    /// In place, orthogonal: preserves norms and inner products.
    pub fn apply(&self, v: &mut [f32]) {
        let d = self.dim();
        debug_assert_eq!(v.len(), d);
        for s in self.signs.chunks_exact(d) {
            v.iter_mut().zip(s).for_each(|(x, s)| *x *= s);
            fwht(v);
        }
        let norm = (d as f32).powf(-1.5);
        v.iter_mut().for_each(|x| *x *= norm);
    }
}

/// Unnormalised fast Walsh-Hadamard transform; `v.len()` must be a power of two.
fn fwht(v: &mut [f32]) {
    let mut h = 1;
    while h < v.len() {
        for blk in v.chunks_exact_mut(2 * h) {
            let (a, b) = blk.split_at_mut(h);
            for (x, y) in a.iter_mut().zip(b) {
                (*x, *y) = (*x + *y, *x - *y);
            }
        }
        h *= 2;
    }
}

/// Codes the residual of `x_rot` against `c_rot` (both already rotated) and returns
/// `[mul, err]`. The code uses the bit layout of `quant::sign_code`.
pub fn encode(x_rot: &[f32], c_rot: &[f32], code: &mut [u8]) -> [f32; 2] {
    code.fill(0);
    let (mut nr2, mut l1) = (0f32, 0f32);
    for (i, (&x, &c)) in x_rot.iter().zip(c_rot).enumerate() {
        let r = x - c;
        nr2 += r * r;
        l1 += r.abs();
        if r > 0.0 {
            code[i >> 3] |= 1 << (i & 7);
        }
    }
    if l1 < 1e-12 {
        // x sits on its centroid: ⟨c, q⟩ is already exact.
        return [0.0, 0.0];
    }
    let d = x_rot.len() as f32;
    let nr = nr2.sqrt();
    let od = l1 / (nr * d.sqrt());
    let err = nr * EPS0 * ((1.0 - od * od) / (od * od * (d - 1.0))).max(0.0).sqrt();
    [nr2 / l1, err]
}

/// Per-query state: the LUT over the rotated query and the affine map from a LUT score to
/// Σ sign_i·(Pq)_i, so a slot's estimate is `cent + mul·(a·lut_score + b)`.
pub struct Query<const G: usize = GROUPS> {
    pub lut: Lut<G>,
    pub a: f32,
    pub b: f32,
}

impl<const G: usize> Query<G> {
    pub fn new(rot: &Rotation, q: &[f32]) -> Self {
        let mut qr = q.to_vec();
        rot.apply(&mut qr);
        let lut = Lut::new(&qr);
        let sum: f32 = qr.iter().sum();
        Self { a: 2.0 / lut.scale, b: 2.0 * lut.bias - sum, lut }
    }

    #[inline]
    pub fn estimate(&self, cent: f32, mul: f32, lut_score: u16) -> f32 {
        cent + mul * (self.a * lut_score as f32 + self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dot_f32, normalize};

    fn unit(rng: &mut Rng) -> Vec<f32> {
        // Sum of uniforms: roughly Gaussian, enough for an isotropic direction.
        let mut v: Vec<f32> = (0..PDIM)
            .map(|_| (0..4).map(|_| (rng.next() >> 11) as f32 / (1u64 << 53) as f32 - 0.5).sum())
            .collect();
        normalize(&mut v);
        v
    }

    #[test]
    fn rotation_is_orthogonal() {
        let mut rng = Rng::new(1);
        let rot = Rotation::new(3);
        let (a, b) = (unit(&mut rng), unit(&mut rng));
        let (mut ra, mut rb) = (a.clone(), b.clone());
        rot.apply(&mut ra);
        rot.apply(&mut rb);
        assert!((dot_f32(&ra, &ra) - 1.0).abs() < 1e-4);
        assert!((dot_f32(&ra, &rb) - dot_f32(&a, &b)).abs() < 1e-4);
    }

    #[test]
    fn estimate_is_within_bound() {
        let mut rng = Rng::new(2);
        let rot = Rotation::new(5);
        let (mut inside, trials) = (0, 2000);
        let mut sq_err = 0f64;
        for _ in 0..trials {
            // A centroid near x, as in a real cluster.
            let x = unit(&mut rng);
            let mut c: Vec<f32> = x.iter().zip(unit(&mut rng)).map(|(a, b)| a + 0.8 * b).collect();
            normalize(&mut c);
            let q = unit(&mut rng);
            let (mut xr, mut cr) = (x.clone(), c.clone());
            rot.apply(&mut xr);
            rot.apply(&mut cr);
            let mut code = [0u8; crate::CODE_BYTES];
            let [mul, err] = encode(&xr, &cr, &mut code);
            let qq: Query = Query::new(&rot, &q);
            let est = qq.estimate(dot_f32(&c, &q), mul, qq.lut.score_code(&code));
            let truth = dot_f32(&x, &q);
            sq_err += ((est - truth) as f64).powi(2);
            // The LUT's u8 rounding adds a little on top of the sampling bound.
            if (est - truth).abs() <= err * 1.1 {
                inside += 1;
            }
        }
        assert!(inside as f64 / trials as f64 > 0.95, "{inside}/{trials} inside the bound");
        assert!((sq_err / trials as f64).sqrt() < 0.05);
    }
}
