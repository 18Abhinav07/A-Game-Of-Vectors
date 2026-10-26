//! Vector normalisation, 1-bit sign codes and the int8 rerank quantizer.

use crate::{CODE_BYTES, DIM, PDIM};

/// Scales `v` to unit length in place and returns its original norm.
pub fn normalize(v: &mut [f32]) -> f32 {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        v.iter_mut().for_each(|x| *x *= inv);
    }
    n
}

/// Writes the normalised Matryoshka prefix of a (normalised) full vector into `out`
/// and returns the prefix's norm before renormalising.
pub fn prefix(full: &[f32], out: &mut [f32]) -> f32 {
    out.copy_from_slice(&full[..PDIM]);
    normalize(out)
}

/// Bit i of the code is set when dim i of the prefix is positive.
/// Bit i lives in byte i/8 at position i%8, so group g (dims 4g..4g+3) is one nibble.
pub fn sign_code(prefix: &[f32], out: &mut [u8]) {
    debug_assert_eq!(out.len(), CODE_BYTES);
    out.fill(0);
    for (i, &x) in prefix.iter().enumerate() {
        if x > 0.0 {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
}

/// Symmetric per-dimension int8 quantizer: x_i ≈ q_i / scale_i.
#[derive(Clone, Debug)]
pub struct Int8Quantizer {
    pub scale: Vec<f32>,
}

impl Int8Quantizer {
    /// Fits per-dimension scales on normalised rows (n × DIM) so the largest |x_i| maps to 127.
    pub fn fit(rows: &[f32]) -> Self {
        let mut maxabs = vec![1e-6f32; DIM];
        for row in rows.chunks_exact(DIM) {
            for (m, &x) in maxabs.iter_mut().zip(row) {
                *m = m.max(x.abs());
            }
        }
        Self { scale: maxabs.iter().map(|m| 127.0 / m).collect() }
    }

    pub fn encode(&self, v: &[f32], out: &mut [i8]) {
        for ((o, &x), &s) in out.iter_mut().zip(v).zip(&self.scale) {
            *o = (x * s).round().clamp(-127.0, 127.0) as i8;
        }
    }

    pub fn decode(&self, v: &[i8], out: &mut [f32]) {
        for ((o, &x), &s) in out.iter_mut().zip(v).zip(&self.scale) {
            *o = x as f32 / s;
        }
    }

    /// Per-query weights w_i = q_i / scale_i, so that q · x ≈ Σ w_i · code_i.
    pub fn query_weights(&self, q: &[f32]) -> Vec<f32> {
        q.iter().zip(&self.scale).map(|(&x, &s)| x / s).collect()
    }
}

/// Σ w_i · x_i over int8 codes. Written as fixed-width chunks so it vectorises.
#[inline]
pub fn dot_i8(w: &[f32], x: &[i8]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let mut acc = [0f32; 16];
    let (wc, wr) = w.as_chunks::<16>();
    let (xc, xr) = x.as_chunks::<16>();
    for (wv, xv) in wc.iter().zip(xc) {
        for i in 0..16 {
            acc[i] += wv[i] * xv[i] as f32;
        }
    }
    let tail: f32 = wr.iter().zip(xr).map(|(&a, &b)| a * b as f32).sum();
    acc.iter().sum::<f32>() + tail
}

/// Σ a_i · b_i with independent lanes, so it vectorises despite float non-associativity.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 16];
    let (ac, ar) = a.as_chunks::<16>();
    let (bc, br) = b.as_chunks::<16>();
    for (av, bv) in ac.iter().zip(bc) {
        for i in 0..16 {
            acc[i] += av[i] * bv[i];
        }
    }
    let tail: f32 = ar.iter().zip(br).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

/// Rerank weights w_i = q_i / scale_i quantised to int8 under one scale, so scoring a stored
/// int8 row is an integer dot product (NEON `sdot`: 16 MACs per instruction).
pub fn query_i8(w: &[f32], out: &mut Vec<i8>) -> f32 {
    let m = w.iter().fold(1e-12f32, |m, x| m.max(x.abs()));
    let s = 127.0 / m;
    out.clear();
    out.extend(w.iter().map(|&x| (x * s).round() as i8));
    s
}

#[inline]
pub fn dot_i8i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") && a.len().is_multiple_of(64) {
        // SAFETY: dotprod was detected and the length is a multiple of the 64-byte step.
        return unsafe { dot_i8i8_neon(a, b) };
    }
    dot_i8i8_scalar(a, b)
}

pub fn dot_i8i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dot_i8i8_neon(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [vdupq_n_s32(0); 4];
    for i in (0..a.len()).step_by(64) {
        for (j, acc) in acc.iter_mut().enumerate() {
            let o = i + 16 * j;
            // SAFETY: o + 16 <= len, since len is a multiple of 64 and both slices have it.
            *acc = unsafe { vdotq_s32(*acc, vld1q_s8(pa.add(o)), vld1q_s8(pb.add(o))) };
        }
    }
    vaddvq_s32(vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3])))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_code_bits() {
        let mut p = vec![-1.0f32; PDIM];
        p[0] = 0.5;
        p[9] = 0.1;
        p[255] = 2.0;
        let mut c = [0u8; CODE_BYTES];
        sign_code(&p, &mut c);
        assert_eq!(c[0], 1);
        assert_eq!(c[1], 1 << 1);
        assert_eq!(c[31], 1 << 7);
        assert_eq!(c.iter().map(|b| b.count_ones()).sum::<u32>(), 3);
    }

    #[test]
    fn int8_roundtrip_and_dot() {
        let mut v: Vec<f32> = (0..DIM).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        normalize(&mut v);
        let q = Int8Quantizer::fit(&v);
        let mut c = vec![0i8; DIM];
        q.encode(&v, &mut c);
        let w = q.query_weights(&v);
        let approx = dot_i8(&w, &c);
        assert!((approx - 1.0).abs() < 0.01, "self dot {approx}");
    }

    proptest::proptest! {
        #[test]
        fn sdot_matches_scalar(a in proptest::collection::vec(proptest::num::i8::ANY, DIM), b in proptest::collection::vec(proptest::num::i8::ANY, DIM)) {
            proptest::prop_assert_eq!(dot_i8i8(&a, &b), dot_i8i8_scalar(&a, &b));
            proptest::prop_assert_eq!(dot_i8i8(&a[..PDIM], &b[..PDIM]), dot_i8i8_scalar(&a[..PDIM], &b[..PDIM]));
        }
    }
}
