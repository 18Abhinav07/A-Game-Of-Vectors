//! Asymmetric scoring of 1-bit document codes against a float query (DESIGN.md D2, D3).
//!
//! The 256-dim prefix splits into 64 groups of 4 dims. For group g the query gets a 16-entry
//! table T_g[c] = Σ_{j<4} q[4g+j]·bit_j(c), so Σ_g T_g[nibble_g(doc)] = Σ_i q_i·bit_i(doc).
//! That ranks documents exactly as q·sign(doc) does (the two differ by a per-query constant).
//!
//! Tables are quantized to u8 with one scale and a per-group offset. Codes are stored in blocks
//! of 32 documents, nibble-transposed: byte k of group g holds doc k's nibble in the low half and
//! doc k+16's in the high half. One 16-byte load then feeds two NEON `TBL` lookups (32 docs).
//!
//! The group count is a const parameter so the same kernel serves other code widths: 64 groups
//! for the 256-dim prefix here, 32 for the 128-dim padded SPACEV vectors (D19).

use crate::PDIM;

pub const GROUPS: usize = PDIM / 4;
pub const BLOCK: usize = 32;
pub const BLOCK_BYTES: usize = GROUPS * 16;

pub struct Lut<const G: usize = GROUPS> {
    pub tables: [[u8; 16]; G],
    /// approx(Σ q_i·bit_i) = score / scale + bias
    pub scale: f32,
    pub bias: f32,
}

#[inline]
pub fn nibble(code: &[u8], g: usize) -> u8 {
    (code[g >> 1] >> ((g & 1) * 4)) & 0x0F
}

impl<const G: usize> Lut<G> {
    pub fn new(q: &[f32]) -> Self {
        debug_assert_eq!(q.len(), 4 * G);
        let mut f = [[0f32; 16]; G];
        for (g, t) in f.iter_mut().enumerate() {
            for (c, v) in t.iter_mut().enumerate() {
                *v = (0..4).filter(|j| c >> j & 1 == 1).map(|j| q[4 * g + j]).sum();
            }
        }
        let mins: Vec<f32> = f.iter().map(|t| t.iter().copied().fold(f32::MAX, f32::min)).collect();
        let max_range = f
            .iter()
            .zip(&mins)
            .map(|(t, &m)| t.iter().copied().fold(f32::MIN, f32::max) - m)
            .fold(0f32, f32::max);
        let scale = if max_range > 0.0 { 255.0 / max_range } else { 1.0 };
        let mut tables = [[0u8; 16]; G];
        for g in 0..G {
            for c in 0..16 {
                tables[g][c] = ((f[g][c] - mins[g]) * scale).round().clamp(0.0, 255.0) as u8;
            }
        }
        Self { tables, scale, bias: mins.iter().sum() }
    }

    /// Reference score of one raw (untransposed) code; the kernels must match it exactly.
    pub fn score_code(&self, code: &[u8]) -> u16 {
        (0..G).map(|g| self.tables[g][nibble(code, g) as usize] as u16).sum()
    }

    pub fn to_float(&self, s: u16) -> f32 {
        s as f32 / self.scale + self.bias
    }
}

/// Exact Σ q_i·bit_i, for tests and the quantization-error ablation.
pub fn float_score(q: &[f32], code: &[u8]) -> f32 {
    (0..q.len()).filter(|&i| code[i >> 3] >> (i & 7) & 1 == 1).map(|i| q[i]).sum()
}

/// Packs up to 32 raw codes into one transposed block of `out.len() / 16` groups; missing docs
/// are zero codes.
pub fn pack_block(codes: &[&[u8]], out: &mut [u8]) {
    debug_assert!(codes.len() <= BLOCK && out.len() % 16 == 0);
    let get = |k: usize, g: usize| codes.get(k).map_or(0, |c| nibble(c, g));
    for g in 0..out.len() / 16 {
        for k in 0..16 {
            out[g * 16 + k] = get(k, g) | get(k + 16, g) << 4;
        }
    }
}

/// The raw code of slot `k` in a packed block: the inverse of `pack_block`.
pub fn unpack_slot(block: &[u8], k: usize, code: &mut [u8]) {
    debug_assert!(k < BLOCK && block.len() % 16 == 0);
    code.fill(0);
    for g in 0..block.len() / 16 {
        let b = block[g * 16 + (k & 15)];
        let nib = if k < 16 { b & 0x0F } else { b >> 4 };
        code[g >> 1] |= nib << ((g & 1) * 4);
    }
}

/// Scores every block in `blocks` (a multiple of G · 16 bytes), writing 32 scores per block.
pub fn scan<const G: usize>(lut: &Lut<G>, blocks: &[u8], out: &mut [u16]) {
    debug_assert_eq!(blocks.len() % (G * 16), 0);
    debug_assert_eq!(out.len(), blocks.len() / (G * 16) * BLOCK);
    for (b, o) in blocks.chunks_exact(G * 16).zip(out.chunks_exact_mut(BLOCK)) {
        scan_block(lut, b, o.try_into().unwrap());
    }
}

#[inline]
pub fn scan_block<const G: usize>(lut: &Lut<G>, block: &[u8], out: &mut [u16; BLOCK]) {
    debug_assert_eq!(block.len(), G * 16);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is part of the aarch64 baseline; `block` holds G * 16 bytes.
    unsafe {
        scan_block_neon(lut, block.as_ptr(), out)
    }
    #[cfg(not(target_arch = "aarch64"))]
    scan_block_scalar(lut, block, out)
}

pub fn scan_block_scalar<const G: usize>(lut: &Lut<G>, block: &[u8], out: &mut [u16; BLOCK]) {
    out.fill(0);
    for g in 0..G {
        let t = &lut.tables[g];
        for k in 0..16 {
            let b = block[g * 16 + k];
            out[k] += t[(b & 0x0F) as usize] as u16;
            out[k + 16] += t[(b >> 4) as usize] as u16;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn scan_block_neon<const G: usize>(lut: &Lut<G>, block: *const u8, out: &mut [u16; BLOCK]) {
    use core::arch::aarch64::*;
    unsafe {
        let mask = vdupq_n_u8(0x0F);
        // Two independent accumulator sets (even and odd groups) halve the dependent add chain;
        // each set sums at most 32 · 255, far inside u16. G is even for every code width used.
        let mut a = [vdupq_n_u16(0); 8];
        for g in (0..G).step_by(2) {
            for (h, acc) in a.chunks_exact_mut(4).enumerate() {
                let t = vld1q_u8(lut.tables[g + h].as_ptr());
                let c = vld1q_u8(block.add((g + h) * 16));
                let lo = vqtbl1q_u8(t, vandq_u8(c, mask));
                let hi = vqtbl1q_u8(t, vshrq_n_u8::<4>(c));
                acc[0] = vaddw_u8(acc[0], vget_low_u8(lo));
                acc[1] = vaddw_high_u8(acc[1], lo);
                acc[2] = vaddw_u8(acc[2], vget_low_u8(hi));
                acc[3] = vaddw_high_u8(acc[3], hi);
            }
        }
        let p = out.as_mut_ptr();
        for i in 0..4 {
            vst1q_u16(p.add(8 * i), vaddq_u16(a[i], a[i + 4]));
        }
    }
}

/// Symmetric baseline: Hamming distance between sign codes (lower is closer).
#[inline]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    let (ac, _) = a.as_chunks::<8>();
    let (bc, _) = b.as_chunks::<8>();
    ac.iter()
        .zip(bc)
        .map(|(x, y)| (u64::from_le_bytes(*x) ^ u64::from_le_bytes(*y)).count_ones())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CODE_BYTES;
    use proptest::prelude::*;

    fn codes_strategy(n: usize) -> impl Strategy<Value = Vec<[u8; CODE_BYTES]>> {
        proptest::collection::vec(any::<[u8; CODE_BYTES]>(), n)
    }

    proptest! {
        #[test]
        fn kernel_matches_reference(
            q in proptest::collection::vec(-1.0f32..1.0, PDIM),
            codes in codes_strategy(BLOCK),
        ) {
            let lut: Lut = Lut::new(&q);
            let refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();
            let mut block = vec![0u8; BLOCK_BYTES];
            pack_block(&refs, &mut block);
            let mut got = [0u16; BLOCK];
            scan_block(&lut, &block, &mut got);
            let mut scalar = [0u16; BLOCK];
            scan_block_scalar(&lut, &block, &mut scalar);
            for k in 0..BLOCK {
                prop_assert_eq!(got[k], lut.score_code(&codes[k]));
                prop_assert_eq!(scalar[k], got[k]);
            }
        }

        #[test]
        fn quantized_score_tracks_float(
            q in proptest::collection::vec(-0.2f32..0.2, PDIM),
            code in any::<[u8; CODE_BYTES]>(),
        ) {
            let lut: Lut = Lut::new(&q);
            let exact = float_score(&q, &code);
            let approx = lut.to_float(lut.score_code(&code));
            // 64 groups, each rounded by at most half a quantization step.
            let bound = 64.0 * 0.5 / lut.scale + 1e-4;
            prop_assert!((exact - approx).abs() <= bound, "{exact} vs {approx} (bound {bound})");
        }
    }

    #[test]
    fn narrow_codes_use_the_same_kernel() {
        // 32 groups: the 128-dim SPACEV codes.
        let q: Vec<f32> = (0..128).map(|i| (i as f32 * 0.91).cos()).collect();
        let lut = Lut::<32>::new(&q);
        let codes: Vec<[u8; 16]> = (0..BLOCK).map(|k| core::array::from_fn(|i| (k * 37 + i * 11) as u8)).collect();
        let refs: Vec<&[u8]> = codes.iter().map(|c| c.as_slice()).collect();
        let mut block = vec![0u8; 32 * 16];
        pack_block(&refs, &mut block);
        let (mut got, mut back) = ([0u16; BLOCK], [0u8; 16]);
        scan_block(&lut, &block, &mut got);
        for k in 0..BLOCK {
            assert_eq!(got[k], lut.score_code(&codes[k]));
            unpack_slot(&block, k, &mut back);
            assert_eq!(back, codes[k]);
        }
    }

    #[test]
    fn partial_block_pads_with_zero_codes() {
        let q: Vec<f32> = (0..PDIM).map(|i| (i as f32 * 0.37).sin()).collect();
        let lut: Lut = Lut::new(&q);
        let c = [0xA5u8; CODE_BYTES];
        let mut block = vec![0u8; BLOCK_BYTES];
        pack_block(&[&c], &mut block);
        let mut got = [0u16; BLOCK];
        scan_block(&lut, &block, &mut got);
        assert_eq!(got[0], lut.score_code(&c));
        assert_eq!(got[1], lut.score_code(&[0u8; CODE_BYTES]));
    }

    #[test]
    fn hamming_counts_bits() {
        let a = [0xFFu8; CODE_BYTES];
        let b = [0x0Fu8; CODE_BYTES];
        assert_eq!(hamming(&a, &b), 4 * CODE_BYTES as u32);
    }
}
