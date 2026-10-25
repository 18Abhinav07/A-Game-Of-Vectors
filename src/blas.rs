//! C = A · Bᵀ for row-major f32 matrices. On macOS this is Accelerate's sgemm, which runs on
//! the AMX units; elsewhere a rayon fallback keeps the crate portable.

/// `a` is m × k, `b` is n × k, `c` is m × n.
pub fn gemm_abt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);
    imp::gemm_abt(a, b, m, n, k, c)
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::c_int;

    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int, trans_a: c_int, trans_b: c_int,
            m: c_int, n: c_int, k: c_int,
            alpha: f32, a: *const f32, lda: c_int,
            b: *const f32, ldb: c_int,
            beta: f32, c: *mut f32, ldc: c_int,
        );
    }

    pub fn gemm_abt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
        let (m, n, k) = (m as c_int, n as c_int, k as c_int);
        // SAFETY: the caller checked every slice length against m, n and k.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR, NO_TRANS, TRANS, m, n, k, 1.0,
                a.as_ptr(), k, b.as_ptr(), k, 0.0, c.as_mut_ptr(), n,
            )
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use rayon::prelude::*;

    pub fn gemm_abt(a: &[f32], b: &[f32], _m: usize, n: usize, k: usize, c: &mut [f32]) {
        c.par_chunks_mut(n).zip(a.par_chunks(k)).for_each(|(crow, arow)| {
            for (cv, brow) in crow.iter_mut().zip(b.chunks_exact(k)) {
                *cv = arow.iter().zip(brow).map(|(x, y)| x * y).sum();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn small_product() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 × 3
        let b = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // 2 × 3
        let mut c = [0.0; 4];
        super::gemm_abt(&a, &b, 2, 2, 3, &mut c);
        assert_eq!(c, [4.0, 2.0, 10.0, 5.0]);
    }
}
