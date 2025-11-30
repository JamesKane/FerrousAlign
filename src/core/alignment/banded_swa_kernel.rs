//! Generic kernel surface for banded Smith–Waterman.
//!
//! This module defines the minimal trait (`SwSimd`) and the parameter carrier
//! (`KernelParams`) that a shared DP kernel will use. In this stage, the kernel
//! function is a stub to allow incremental adoption by per‑ISA wrappers without
//! changing behavior yet.

use crate::alignment::banded_swa::OutScore;
use crate::compute::simd_abstraction as simd;
use crate::compute::simd_abstraction::SimdEngine; // bring trait into scope for method resolution


/// Minimal SIMD engine contract for the shared SW kernel.
///
/// Note: We intentionally keep this trait very small initially. As the shared
/// kernel is implemented, only the actually required ops will be added here,
/// and concrete engines (SSE/NEON/AVX2/AVX‑512/portable) will implement it.
pub trait SwSimd: Copy {
    /// Concrete vector type for i8 lanes used in this kernel.
    /// Must be `Copy` so we can pass by value to intrinsic-like helpers without
    /// worrying about move semantics in the kernel code.
    type V8: Copy;

    /// Number of 8‑bit lanes processed in parallel.
    const LANES: usize;

    // Core ops (int8)
    unsafe fn setzero_epi8() -> Self::V8;
    unsafe fn set1_epi8(x: i8) -> Self::V8;
    unsafe fn loadu_epi8(ptr: *const i8) -> Self::V8;
    unsafe fn storeu_epi8(ptr: *mut i8, v: Self::V8);
    unsafe fn adds_epi8(a: Self::V8, b: Self::V8) -> Self::V8; // saturating add
    unsafe fn subs_epi8(a: Self::V8, b: Self::V8) -> Self::V8; // saturating sub
    unsafe fn max_epi8(a: Self::V8, b: Self::V8) -> Self::V8;
    unsafe fn min_epi8(a: Self::V8, b: Self::V8) -> Self::V8;
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8;
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8;
    /// Select bytes from b where mask MSB is set, else from a (like blendv)
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8;
    unsafe fn and_si128(a: Self::V8, b: Self::V8) -> Self::V8;
    unsafe fn or_si128(a: Self::V8, b: Self::V8) -> Self::V8;
    unsafe fn cmpgt_epi8(a: Self::V8, b: Self::V8) -> Self::V8;
    unsafe fn min_epu8(a: Self::V8, b: Self::V8) -> Self::V8;
}

/// Input parameters for the shared SW kernel.
///
/// The kernel operates on Structure‑of‑Arrays (SoA) buffers for query and target
/// sequences. Callers are expected to ensure the buffers are sized as
/// `max_len * E::LANES` and padded appropriately.
#[derive(Debug)]
pub struct KernelParams<'a> {
    /// Batch of alignments (AoS metadata). Tuples are
    /// (qlen, query, tlen, target, band_w, h0).
    pub batch: &'a [(i32, &'a [u8], i32, &'a [u8], i32, i32)],

    /// Query sequences in SoA layout: `query_soa[pos * LANES + lane]`.
    pub query_soa: &'a [u8],

    /// Target sequences in SoA layout: `target_soa[pos * LANES + lane]`.
    pub target_soa: &'a [u8],

    /// Per‑lane query lengths (clamped), in 8‑bit lanes.
    pub qlen: &'a [i8],
    /// Per‑lane target lengths (clamped), in 8‑bit lanes.
    pub tlen: &'a [i8],
    /// Per‑lane initial score h0, in 8‑bit lanes.
    pub h0: &'a [i8],
    /// Per‑lane band width, in 8‑bit lanes.
    pub w: &'a [i8],

    /// Maximum clamped query length across lanes.
    pub max_qlen: i32,
    /// Maximum clamped target length across lanes.
    pub max_tlen: i32,

    /// Gap penalties and z‑drop.
    pub o_del: i32,
    pub e_del: i32,
    pub o_ins: i32,
    pub e_ins: i32,
    pub zdrop: i32,

    /// Scoring matrix (5x5: A,C,G,T,N) and its dimension (typically 5).
    pub mat: &'a [i8; 25],
    pub m: i32,
}

/// Shared banded SW kernel (int8 lanes).
///
/// Notes:
/// - This initial implementation is correctness-focused and mirrors the AVX2
///   compare-and-blend scoring approach using the minimal `SwSimd` ops. It
///   maintains two rows (previous/current) for H and a row for E. F is carried
///   across the row. Banding and z-drop are stubbed to full range for now; they
///   will be refined as wrappers migrate and tests are added around them.
/// - The kernel operates on SoA buffers sized to `max_len * W` where `W == E::LANES`.
///
/// Safety: The caller must ensure that `query_soa` and `target_soa` are laid
/// out as SoA with length at least `max_len * E::LANES`, padded to avoid
/// out‑of‑bounds loads, and that `qlen/tlen/h0/w` slices have length `W`.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn sw_kernel<const W: usize, E: SwSimd>(params: &KernelParams<'_>) -> Vec<OutScore> where <E as SwSimd>::V8: std::fmt::Debug {
    debug_assert_eq!(W, E::LANES, "W generic must match engine lanes");

    let lanes = W;
    let qmax = params.max_qlen.max(0) as usize;
    let tmax = params.max_tlen.max(0) as usize;

    if qmax == 0 || tmax == 0 || lanes == 0 {
        return Vec::new();
    }

    // Score constants (match/mismatch) from matrix similar to MAIN_CODE8 pattern
    let match_score = params.mat[0];
    let mismatch_score = params.mat[1];

    let zero = E::setzero_epi8();
    let match_vec = E::set1_epi8(match_score);
    let mismatch_vec = E::set1_epi8(mismatch_score);

    // Gap penalties (saturating 8-bit arithmetic)
    let oe_del = (params.o_del + params.e_del) as i8;
    let oe_ins = (params.o_ins + params.e_ins) as i8;
    let e_del = params.e_del as i8;
    let e_ins = params.e_ins as i8;
    let oe_del_vec = E::set1_epi8(oe_del);
    let oe_ins_vec = E::set1_epi8(oe_ins);
    let e_del_vec = E::set1_epi8(e_del);
    let e_ins_vec = E::set1_epi8(e_ins);

    // DP buffers
    let mut h_matrix = vec![0i8; qmax * lanes];
    let mut e_matrix = vec![0i8; qmax * lanes];
    
    // Initialize first row
    let h0_vec = E::loadu_epi8(params.h0.as_ptr());
    E::storeu_epi8(h_matrix.as_mut_ptr(), h0_vec);
    
    let h1_vec = E::subs_epi8(h0_vec, oe_ins_vec);
    let h1_vec = E::max_epi8(h1_vec, zero);
    E::storeu_epi8(h_matrix.as_mut_ptr().add(lanes), h1_vec);

    let mut h_prev = h1_vec;
    for j in 2..qmax {
        let h_curr = E::subs_epi8(h_prev, e_ins_vec);
        let h_curr = E::max_epi8(h_curr, zero);
        E::storeu_epi8(h_matrix.as_mut_ptr().add(j * lanes), h_curr);
        h_prev = h_curr;
    }
    
    // Track maxima per lane
    let mut max_scores_vec = E::loadu_epi8(params.h0.as_ptr());
    let mut max_i_vec = zero;
    let mut max_j_vec = zero;
    
    let mut beg = [0i8; W];
    let mut end = [0i8; W];
    let mut terminated = [false; W];
    let mut terminated_count = 0;
    for lane in 0..W {
        end[lane] = params.qlen[lane];
    }

    let qlen_vec = E::loadu_epi8(params.qlen.as_ptr());

    // Main DP loop
    let mut final_row = tmax;
    for i in 0..tmax {
        if terminated_count > params.batch.len() / 2 {
            final_row = i;
            break;
        }

        let mut f_vec = zero;
        let mut h_diag = E::loadu_epi8(h_matrix.as_ptr());

        let s1 = E::loadu_epi8(params.target_soa.as_ptr().add(i * lanes) as *const i8);
        
        let i_vec = E::set1_epi8(i as i8);
        let w_vec = E::loadu_epi8(params.w.as_ptr());
        let beg_vec = E::loadu_epi8(beg.as_ptr());
        let end_vec = E::loadu_epi8(end.as_ptr());

        let i_minus_w = E::subs_epi8(i_vec, w_vec);
        let current_beg_vec = E::max_epi8(beg_vec, i_minus_w);

        let one_vec = E::set1_epi8(1);
        let i_plus_w = E::adds_epi8(i_vec, w_vec);
        let i_plus_w_plus_1 = E::adds_epi8(i_plus_w, one_vec);
        let mut current_end_vec = E::min_epu8(end_vec, i_plus_w_plus_1);
        current_end_vec = E::min_epu8(current_end_vec, qlen_vec);

        let mut current_beg = [0i8; W];
        let mut current_end = [0i8; W];
        E::storeu_epi8(current_beg.as_mut_ptr(), current_beg_vec);
        E::storeu_epi8(current_end.as_mut_ptr(), current_end_vec);

        let mut term_mask_vals = [0i8; W];
        for lane in 0..W {
            if !terminated[lane] && i < params.tlen[lane] as usize {
                term_mask_vals[lane] = -1i8;
            }
        }
        let term_mask = E::loadu_epi8(term_mask_vals.as_ptr());

        for j in 0..qmax {
            let h_top = E::loadu_epi8(h_matrix.as_ptr().add(j * lanes));
            let e_prev = E::loadu_epi8(e_matrix.as_ptr().add(j * lanes));
            
            let h_diag_curr = h_diag;
            h_diag = h_top;

            let s2 = E::loadu_epi8(params.query_soa.as_ptr().add(j * lanes) as *const i8);
            let eq_mask = E::cmpeq_epi8(s1, s2);
            let score_vec = E::blendv_epi8(mismatch_vec, match_vec, eq_mask);
            
            let or_bases = E::or_si128(s1, s2);
            let mut m_vec = E::adds_epi8(h_diag_curr, score_vec);
            m_vec = E::blendv_epi8(m_vec, zero, or_bases);
            m_vec = E::max_epi8(m_vec, zero);

            let e_open = E::subs_epi8(m_vec, oe_del_vec);
            let e_open = E::max_epi8(e_open, zero);
            let e_extend = E::subs_epi8(e_prev, e_del_vec);
            let e_val = E::max_epi8(e_open, e_extend);
            
            let f_open = E::subs_epi8(m_vec, oe_ins_vec);
            let f_open = E::max_epi8(f_open, zero);
            let f_extend = E::subs_epi8(f_vec, e_ins_vec);
            f_vec = E::max_epi8(f_open, f_extend);
            
            let mut h_val = E::max_epi8(m_vec, e_val);
            h_val = E::max_epi8(h_val, f_vec);

            let j_vec = E::set1_epi8(j as i8);
            let in_band_left = E::cmpgt_epi8(j_vec, E::subs_epi8(current_beg_vec, one_vec));
            let in_band_right = E::cmpgt_epi8(current_end_vec, j_vec);
            let in_band_mask = E::and_si128(in_band_left, in_band_right);
            
            let combined_mask = E::and_si128(in_band_mask, term_mask);
            h_val = E::and_si128(h_val, combined_mask);
            let e_val_masked = E::and_si128(e_val, combined_mask);

            E::storeu_epi8(h_matrix.as_mut_ptr().add(j * lanes), h_val);
            E::storeu_epi8(e_matrix.as_mut_ptr().add(j * lanes), e_val_masked);
            
            let is_greater = E::cmpgt_epi8(h_val, max_scores_vec);
            max_scores_vec = E::max_epi8(h_val, max_scores_vec);

            max_i_vec = E::blendv_epi8(max_i_vec, i_vec, is_greater);
            max_j_vec = E::blendv_epi8(max_j_vec, j_vec, is_greater);
        }

        let mut max_score_vals = [0i8; W];
        E::storeu_epi8(max_score_vals.as_mut_ptr(), max_scores_vec);

        if params.zdrop > 0 {
            for lane in 0..W {
                if !terminated[lane] && i > 0 && i < params.tlen[lane] as usize {
                    let mut row_max = 0i8;
                    let row_beg = current_beg[lane] as usize;
                    let row_end = current_end[lane] as usize;

                    for j in row_beg..row_end {
                        let h_val = h_matrix[j * lanes + lane];
                        row_max = row_max.max(h_val);
                    }

                    if row_max == 0 {
                        terminated[lane] = true;
                        terminated_count += 1;
                        continue;
                    }

                    let global_max = max_score_vals[lane];
                    let score_drop = (global_max as i32) - (row_max as i32);

                    if score_drop > params.zdrop {
                        terminated[lane] = true;
                        terminated_count += 1;
                    }
                }
            }
        }
    }

    let mut out_scores = vec![];
    let mut max_scores = [0i8; W];
    let mut max_i = [0i8; W];
    let mut max_j = [0i8; W];

    E::storeu_epi8(max_scores.as_mut_ptr(), max_scores_vec);
    E::storeu_epi8(max_i.as_mut_ptr(), max_i_vec);
    E::storeu_epi8(max_j.as_mut_ptr(), max_j_vec);

    for i in 0..lanes {
        out_scores.push(OutScore {
            score: max_scores[i] as i32,
            target_end_pos: max_i[i] as i32,
            query_end_pos: max_j[i] as i32,
            gtarget_end_pos: 0,
            global_score: 0,
            max_offset: 0,
        });
    }
    
    out_scores.truncate(params.batch.len());
    out_scores
}

// -------------------------------------------------------------------------------------------------
// Adapters: map existing SimdEngine* to SwSimd for int8 ops
// -------------------------------------------------------------------------------------------------

/// SSE/NEON 128-bit engine adapter (16 lanes of i8)
#[derive(Copy, Clone)]
pub struct SwEngine128;

impl SwSimd for SwEngine128 {
    type V8 = <simd::SimdEngine128 as simd::SimdEngine>::Vec8;
    const LANES: usize = 16;

    #[inline(always)]
    unsafe fn setzero_epi8() -> Self::V8 { simd::SimdEngine128::setzero_epi8() }
    #[inline(always)]
    unsafe fn set1_epi8(x: i8) -> Self::V8 { simd::SimdEngine128::set1_epi8(x) }
    #[inline(always)]
    unsafe fn loadu_epi8(ptr: *const i8) -> Self::V8 {
        simd::SimdEngine128::loadu_si128(ptr as *const _)
    }
    #[inline(always)]
    unsafe fn storeu_epi8(ptr: *mut i8, v: Self::V8) {
        simd::SimdEngine128::storeu_si128(ptr as *mut _, v)
    }
    #[inline(always)]
    unsafe fn adds_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::adds_epi8(a, b) }
    #[inline(always)]
    unsafe fn subs_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::subs_epi8(a, b) }
    #[inline(always)]
    unsafe fn max_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::max_epi8(a, b) }
    #[inline(always)]
    unsafe fn min_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::min_epi8(a, b) }
    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::cmpeq_epi8(a, b) }
    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::cmpgt_epi8(b, a) }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        simd::SimdEngine128::blendv_epi8(a, b, mask)
    }
    #[inline(always)]
    unsafe fn and_si128(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::and_si128(a,b) }
    #[inline(always)]
    unsafe fn or_si128(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::or_si128(a,b) }
    #[inline(always)]
    unsafe fn cmpgt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::cmpgt_epi8(a, b) }
    #[inline(always)]
    unsafe fn min_epu8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::min_epu8(a, b) }
}

// -------------------------------------------------------------------------------------------------
// Tests: AVX2-first parity on tiny cases (banding disabled; high zdrop)
// -------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::banded_swa_shared::{pad_batch, soa_transform};

    // Only meaningful on x86_64 where AVX2 path exists
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_sw_vs_avx2_basic_parity() {
        // 2-bit encoded A,C,G,T = 0,1,2,3
        let q: [u8; 8] = [0,1,2,3,0,1,2,3];
        let t: [u8; 8] = [0,1,2,3,0,1,2,3];
        let batch = vec![(8, &q[..], 8, &t[..], 10, 0)];

        // Simple scoring: match=1, mismatch=0
        let mut mat = [0i8; 25];
        for i in 0..4 { mat[i*5 + i] = 1; }

        // Call AVX2 reference path (existing implementation)
        let avx2_out = unsafe {
            crate::alignment::banded_swa_avx2::simd_banded_swa_batch32(
                &batch, 6, 1, 6, 1, 100, &mat, 5,
            )
        };

        // Prepare shared-kernel params (W=32, SwEngine256)
        const W: usize = 32;
        const MAX: usize = 128;
        let (qlen, tlen, h0, w_arr, max_q, max_t, padded) = pad_batch::<W>(&batch);
        let (query_soa, target_soa) = soa_transform::<W, MAX>(&padded);
        let params = KernelParams {
            batch: &batch,
            query_soa: &query_soa,
            target_soa: &target_soa,
            qlen: &qlen,
            tlen: &tlen,
            h0: &h0,
            w: &w_arr,
            max_qlen: max_q,
            max_tlen: max_t,
            o_del: 6,
            e_del: 1,
            o_ins: 6,
            e_ins: 1,
            zdrop: 100,
            mat: &mat,
            m: 5,
        };

        let shared_out = unsafe { sw_kernel::<W, SwEngine256>(&params) };

        assert_eq!(shared_out.len(), avx2_out.len());
        for (a, b) in shared_out.iter().zip(avx2_out.iter()) {
            assert_eq!(a.score, b.score, "score mismatch: shared={} avx2={}", a.score, b.score);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn kernel_sw_vs_avx2_mismatch_case() {
        // Deliberate mismatches
        let q: [u8; 8] = [0,0,0,0,1,1,1,1];
        let t: [u8; 8] = [3,3,3,3,2,2,2,2];
        let batch = vec![(8, &q[..], 8, &t[..], 10, 0)];

        // Simple scoring: match=1, mismatch=0
        let mut mat = [0i8; 25];
        for i in 0..4 { mat[i*5 + i] = 1; }

        let avx2_out = unsafe {
            crate::alignment::banded_swa_avx2::simd_banded_swa_batch32(
                &batch, 6, 1, 6, 1, 100, &mat, 5,
            )
        };

        const W: usize = 32;
        const MAX: usize = 128;
        let (qlen, tlen, h0, w_arr, max_q, max_t, padded) = pad_batch::<W>(&batch);
        let (query_soa, target_soa) = soa_transform::<W, MAX>(&padded);
        let params = KernelParams {
            batch: &batch,
            query_soa: &query_soa,
            target_soa: &target_soa,
            qlen: &qlen,
            tlen: &tlen,
            h0: &h0,
            w: &w_arr,
            max_qlen: max_q,
            max_tlen: max_t,
            o_del: 6,
            e_del: 1,
            o_ins: 6,
            e_ins: 1,
            zdrop: 100,
            mat: &mat,
            m: 5,
        };
        let shared_out = unsafe { sw_kernel::<W, SwEngine256>(&params) };

        assert_eq!(shared_out.len(), avx2_out.len());
        for (a, b) in shared_out.iter().zip(avx2_out.iter()) {
            assert_eq!(a.score, b.score, "score mismatch: shared={} avx2={}", a.score, b.score);
        }
    }
}

/// AVX2 256-bit engine adapter (32 lanes of i8)
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
pub struct SwEngine256;

#[cfg(target_arch = "x86_64")]
impl SwSimd for SwEngine256 {
    type V8 = <simd::SimdEngine256 as simd::SimdEngine>::Vec8;
    const LANES: usize = 32;

    #[inline(always)]
    unsafe fn setzero_epi8() -> Self::V8 { simd::SimdEngine256::setzero_epi8() }
    #[inline(always)]
    unsafe fn set1_epi8(x: i8) -> Self::V8 { simd::SimdEngine256::set1_epi8(x) }
    #[inline(always)]
    unsafe fn loadu_epi8(ptr: *const i8) -> Self::V8 {
        simd::SimdEngine256::loadu_si128(ptr as *const _)
    }
    #[inline(always)]
    unsafe fn storeu_epi8(ptr: *mut i8, v: Self::V8) {
        simd::SimdEngine256::storeu_si128(ptr as *mut _, v)
    }
    #[inline(always)]
    unsafe fn adds_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::adds_epi8(a, b) }
    #[inline(always)]
    unsafe fn subs_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::subs_epi8(a, b) }
    #[inline(always)]
    unsafe fn max_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::max_epi8(a, b) }
    #[inline(always)]
    unsafe fn min_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::min_epi8(a, b) }
    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::cmpeq_epi8(a, b) }
    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::cmpgt_epi8(b, a) }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        simd::SimdEngine256::blendv_epi8(a, b, mask)
    }
    #[inline(always)]
    unsafe fn and_si128(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::and_si128(a,b) }
    #[inline(always)]
    unsafe fn or_si128(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::or_si128(a,b) }
    #[inline(always)]
    unsafe fn cmpgt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::cmpgt_epi8(a, b) }
    #[inline(always)]
    unsafe fn min_epu8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::min_epu8(a, b) }
}

/// AVX-512 512-bit engine adapter (64 lanes of i8)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Copy, Clone)]
pub struct SwEngine512;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl SwSimd for SwEngine512 {
    type V8 = <simd::SimdEngine512 as simd::SimdEngine>::Vec8;
    const LANES: usize = 64;

    #[inline(always)]
    unsafe fn setzero_epi8() -> Self::V8 { simd::SimdEngine512::setzero_epi8() }
    #[inline(always)]
    unsafe fn set1_epi8(x: i8) -> Self::V8 { simd::SimdEngine512::set1_epi8(x) }
    #[inline(always)]
    unsafe fn loadu_epi8(ptr: *const i8) -> Self::V8 {
        simd::SimdEngine512::loadu_si128(ptr as *const _)
    }
    #[inline(always)]
    unsafe fn storeu_epi8(ptr: *mut i8, v: Self::V8) {
        simd::SimdEngine512::storeu_si128(ptr as *mut _, v)
    }
    #[inline(always)]
    unsafe fn adds_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::adds_epi8(a, b) }
    #[inline(always)]
    unsafe fn subs_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::subs_epi8(a, b) }
    #[inline(always)]
    unsafe fn max_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::max_epi8(a, b) }
    #[inline(always)]
    unsafe fn min_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        simd::SimdEngine512::min_epi8(a, b)
    }
    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::cmpeq_epi8(a, b) }
    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::cmpgt_epi8(b, a) }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        simd::SimdEngine512::blendv_epi8(a, b, mask)
    }
    #[inline(always)]
    unsafe fn and_si128(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::and_si128(a,b) }
    #[inline(always)]
    unsafe fn or_si128(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::or_si128(a,b) }
    #[inline(always)]
    unsafe fn cmpgt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::cmpgt_epi8(a, b) }
    #[inline(always)]
    unsafe fn min_epu8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::min_epu8(a, b) }
}