//! Generic kernel surface for banded Smith–Waterman.
//!
//! This module defines the minimal trait (`SwSimd`) and the parameter carrier
//! (`KernelParams`) that a shared DP kernel will use. In this stage, the kernel
//! function is a stub to allow incremental adoption by per‑ISA wrappers without
//! changing behavior yet.

use crate::alignment::banded_swa::OutScore;
use crate::compute::simd_abstraction as simd;
use crate::compute::simd_abstraction::SimdEngine; // bring trait into scope for method resolution
#[cfg(feature = "swk-debug")]
macro_rules! swk_dbg {
    ($($arg:tt)*) => {{
        eprintln!($($arg)*);
    }};
}
#[cfg(not(feature = "swk-debug"))]
macro_rules! swk_dbg {
    ($($arg:tt)*) => {{}};
}

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
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8;
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8;
    /// Select bytes from b where mask MSB is set, else from a (like blendv)
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8;
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
pub unsafe fn sw_kernel<const W: usize, E: SwSimd>(params: &KernelParams<'_>) -> Vec<OutScore> {
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
    let four_vec = E::set1_epi8(4); // base < 4 are valid A/C/G/T

    // Gap penalties (saturating 8-bit arithmetic)
    let oe_del = (params.o_del + params.e_del) as i8;
    let oe_ins = (params.o_ins + params.e_ins) as i8;
    let e_del = params.e_del as i8;
    let e_ins = params.e_ins as i8;
    let oe_del_vec = E::set1_epi8(oe_del);
    let oe_ins_vec = E::set1_epi8(oe_ins);
    let e_del_vec = E::set1_epi8(e_del);
    let e_ins_vec = E::set1_epi8(e_ins);

    // DP buffers (SoA, per position vectors stored in byte arrays)
    let row_bytes = lanes; // bytes per vector row chunk
    let mut h_prev: Vec<i8> = vec![0; qmax * row_bytes];
    let mut h_curr: Vec<i8> = vec![0; qmax * row_bytes];
    let mut e_row: Vec<i8> = vec![0; qmax * row_bytes];

    // Initialize first row from h0 where j==0; propagate insertion penalties along row
    // H[0][0] = h0; H[0][j>0] = max(0, H[0][j-1] - e_ins) with initial open at j==1.
    // Build vector for j==0 from per-lane h0
    // Load h0 into a stack buffer then store to h_prev[0]
    {
        // Build a temporary lane array and store via E::storeu_epi8
        let mut h0_tmp = [0i8; W];
        for lane in 0..lanes { h0_tmp[lane] = params.h0[lane]; }
        let v = E::loadu_epi8(h0_tmp.as_ptr());
        E::storeu_epi8(h_prev.as_mut_ptr() as *mut i8, v);
    }
    // j==1: h = max(0, h0 - (o_ins+e_ins)) then extend by e_ins
    if qmax > 1 {
        let base_ptr = h_prev.as_mut_ptr();
        let h0_vec = E::loadu_epi8(base_ptr);
        let mut h_vec = E::subs_epi8(h0_vec, oe_ins_vec);
        h_vec = E::max_epi8(h_vec, zero);
        E::storeu_epi8(base_ptr.add(row_bytes), h_vec);
        let mut j = 2;
        while j < qmax {
            h_vec = E::subs_epi8(h_vec, e_ins_vec);
            h_vec = E::max_epi8(h_vec, zero);
            E::storeu_epi8(base_ptr.add(j * row_bytes), h_vec);
            j += 1;
        }
    }
    // E row initially zeros

    // Track maxima per lane
    let mut max_scores = [i8::MIN; W];
    let mut max_i = [0i8; W];
    let mut max_j = [0i8; W];

    // Temporary lane buffer for extracting values from vectors
    let mut lane_buf = [0i8; W];
    let mut row_max: [i8; W] = [0; W];
    let mut terminated: [bool; W] = [false; W];
    let ones = E::set1_epi8(-1);

    // Optional debug filters (compile-time; read environment at runtime)
    #[cfg(feature = "swk-debug")]
    let dbg_lane: Option<usize> = std::env::var("SWK_DEBUG_LANE").ok().and_then(|s| s.parse().ok());
    #[cfg(feature = "swk-debug")]
    let dbg_rows: Option<usize> = std::env::var("SWK_DEBUG_ROWS").ok().and_then(|s| s.parse().ok());

    // Main DP sweep over target positions
    for i in 0..tmax {
        #[cfg(feature = "swk-debug")]
        if dbg_rows.map(|lim| i < lim).unwrap_or(true) {
            swk_dbg!("[swk] row i={} (i_row={})", i, i + 1);
        }
        // Reset per-row maxima
        for lane in 0..lanes { row_max[lane] = 0; }
        // F values start at 0 for each row
        let mut f_vec = zero;
        let mut h_left = zero; // H(i, j-1)
        let mut h_diag = zero; // H(i-1, j-1)

        // Pointer to beginning of previous and current rows
        let prev_ptr = h_prev.as_ptr();
        let curr_ptr = h_curr.as_mut_ptr();
        let e_ptr = e_row.as_mut_ptr();

        // Load target bases for this row
        let t_ptr = params.target_soa.as_ptr().add(i * row_bytes);
        let t_vec = E::loadu_epi8(t_ptr as *const i8);

        // Build per-lane constants for this row
        let mut beg_arr = [0i8; W];
        let mut end_arr = [0i8; W];
        let mut tlen_arr = [0i8; W];
        let mut qlen_arr = [0i8; W];
        for lane in 0..lanes {
            let w_lane = params.w[lane] as i32;
            let ql = params.qlen[lane] as i32;
            let tl = params.tlen[lane] as i32;
            // Use 1-based DP row index for band math to mirror legacy SSE
            let ii = (i + 1) as i32;
            let beg_i = (ii - w_lane).max(0);
            // Match SSE band end semantics: end is exclusive upper bound j < end,
            // with end = min(qlen, i_row + w + 1)
            let end_i = (ii + w_lane + 1).min(ql);
            beg_arr[lane] = beg_i.clamp(0, i8::MAX as i32) as i8;
            end_arr[lane] = end_i.clamp(0, i8::MAX as i32) as i8;
            tlen_arr[lane] = tl.clamp(0, i8::MAX as i32) as i8;
            qlen_arr[lane] = ql.clamp(0, i8::MAX as i32) as i8;
            #[cfg(feature = "swk-debug")]
            if dbg_rows.map(|lim| i < lim).unwrap_or(true)
                && dbg_lane.map(|l| l == lane).unwrap_or(true)
            {
                swk_dbg!(
                    "[swk] lane {}: w={}, qlen={}, tlen={}, beg={}, end={}",
                    lane, w_lane, ql, tl, beg_arr[lane], end_arr[lane]
                );
            }
        }
        let beg_vec = E::loadu_epi8(beg_arr.as_ptr());
        let end_vec = E::loadu_epi8(end_arr.as_ptr());
        let tlen_vec = E::loadu_epi8(tlen_arr.as_ptr());
        // Build tlen+1 for i_row <= tlen check
        let mut tlen_p1_arr = [0i8; W];
        for lane in 0..lanes { tlen_p1_arr[lane] = tlen_arr[lane].saturating_add(1); }
        let tlen_p1_vec = E::loadu_epi8(tlen_p1_arr.as_ptr());
        let qlen_vec = E::loadu_epi8(qlen_arr.as_ptr());
        // DP row index is 1-based relative to the initialized H[0,*] row
        let i_row = i + 1; // 1-based for band bounds
        let i_vec = E::set1_epi8((i_row as i32).clamp(0, i8::MAX as i32) as i8);
        let i0_vec = E::set1_epi8((i as i32).clamp(0, i8::MAX as i32) as i8); // 0-based for row activity

        // Build previous row band bounds and masks (i_row - 1)
        let mut prev_beg_arr = [0i8; W];
        let mut prev_end_arr = [0i8; W];
        let i_row_prev = i_row.saturating_sub(1) as i32;
        for lane in 0..lanes {
            let w_lane = params.w[lane] as i32;
            let ql = params.qlen[lane] as i32;
            let beg_prev = (i_row_prev - w_lane).max(0);
            let end_prev = (i_row_prev + w_lane + 1).min(ql);
            prev_beg_arr[lane] = beg_prev.clamp(0, i8::MAX as i32) as i8;
            prev_end_arr[lane] = end_prev.clamp(0, i8::MAX as i32) as i8;
        }
        let prev_beg_vec = E::loadu_epi8(prev_beg_arr.as_ptr());
        let prev_end_vec = E::loadu_epi8(prev_end_arr.as_ptr());
        let i_prev0_vec = if i == 0 {
            // negative value so i_prev < tlen is always true for properly masked H0 usage
            E::set1_epi8(0)
        } else {
            E::set1_epi8(((i - 1) as i32).clamp(0, i8::MAX as i32) as i8)
        };

        for j in 0..qmax {
            // Load query base vector at position j
            let q_ptr = params.query_soa.as_ptr().add(j * row_bytes);
            let q_vec = E::loadu_epi8(q_ptr as *const i8);

            // Build per-lane masks first (band/length/termination), using j
            let j_vec = E::set1_epi8((j as i32).clamp(0, i8::MAX as i32) as i8);
            let j_plus1_vec = E::set1_epi8(((j + 1) as i32).clamp(0, i8::MAX as i32) as i8);
            let m_ge_beg = E::cmplt_epi8(beg_vec, j_plus1_vec);
            let m_lt_end = E::cmplt_epi8(j_vec, end_vec);
            // Row active if 1-based i_row <= tlen  <=> i_row < tlen+1
            let m_row_active = E::cmplt_epi8(i_vec, tlen_p1_vec);
            let m_col_active = E::cmplt_epi8(j_vec, qlen_vec);
            let m_band_start = E::cmpeq_epi8(j_vec, beg_vec);
            let mut active_mask = E::blendv_epi8(zero, ones, m_ge_beg);
            active_mask = E::blendv_epi8(zero, active_mask, m_lt_end);
            active_mask = E::blendv_epi8(zero, active_mask, m_row_active);
            active_mask = E::blendv_epi8(zero, active_mask, m_col_active);
            let mut term_arr = [0i8; W];
            for lane in 0..lanes { term_arr[lane] = if terminated[lane] { 0 } else { -1 }; }
            let term_vec = E::loadu_epi8(term_arr.as_ptr());
            active_mask = E::blendv_epi8(zero, active_mask, term_vec);

            // Load prior row/state
            let mut h_up = E::loadu_epi8(prev_ptr.add(j * row_bytes) as *const i8);
            let mut e_prev = E::loadu_epi8(e_ptr.add(j * row_bytes));
            // Do not mask prior row contributions here; manual AVX2 masks only the
            // outputs for the current row. Prior row values were already stored
            // with masking, so double-masking would undercount.
            // Clamp left/diag at band start before use
            h_left = E::blendv_epi8(h_left, zero, m_band_start);
            h_diag = E::blendv_epi8(h_diag, zero, m_band_start);
            // Note: Do not clear F at band start; the manual AVX2 kernel carries F across columns.

            // Scoring via compare-and-blend; zero-out where bases are not in {0..3}
            // and also when padding sentinels are present (negative values, e.g., 0xFF)
            let eq_mask = E::cmpeq_epi8(q_vec, t_vec);
            let mut score_vec = E::blendv_epi8(mismatch_vec, match_vec, eq_mask);
            // Build invalid mask: (q<0) || (t<0) || (q>=4) || (t>=4)
            let zero_i8 = E::set1_epi8(0);
            let q_neg = E::cmplt_epi8(q_vec, zero_i8);
            let t_neg = E::cmplt_epi8(t_vec, zero_i8);
            let q_lt4 = E::cmplt_epi8(q_vec, four_vec);
            let t_lt4 = E::cmplt_epi8(t_vec, four_vec);
            let q_eq4 = E::cmpeq_epi8(q_vec, four_vec);
            let t_eq4 = E::cmpeq_epi8(t_vec, four_vec);
            let q_gt4 = E::cmplt_epi8(four_vec, q_vec);
            let t_gt4 = E::cmplt_epi8(four_vec, t_vec);
            // Accumulate OR of conditions using blend with ones
            let mut invalid = E::blendv_epi8(zero, ones, q_neg);
            invalid = E::blendv_epi8(invalid, ones, t_neg);
            let q_ge4 = E::blendv_epi8(zero, ones, q_eq4);
            let q_ge4 = E::blendv_epi8(q_ge4, ones, q_gt4);
            let t_ge4 = E::blendv_epi8(zero, ones, t_eq4);
            let t_ge4 = E::blendv_epi8(t_ge4, ones, t_gt4);
            invalid = E::blendv_epi8(invalid, ones, q_ge4);
            invalid = E::blendv_epi8(invalid, ones, t_ge4);
            // Zero score where invalid
            score_vec = E::blendv_epi8(score_vec, zero, invalid);

            // E(i,j) = max(H(i-1,j) - (o_del+e_del), E(i-1,j) - e_del)
            let e_from_open = E::subs_epi8(h_up, oe_del_vec);
            let e_from_ext = E::subs_epi8(e_prev, e_del_vec);
            let mut e_val = E::max_epi8(e_from_open, e_from_ext);

            // F(i,j) = max(H(i,j-1) - (o_ins+e_ins), F(i,j-1) - e_ins)
            let f_from_open = E::subs_epi8(h_left, oe_ins_vec);
            let f_from_ext = E::subs_epi8(f_vec, e_ins_vec);
            f_vec = E::max_epi8(f_from_open, f_from_ext);

            // H(i,j) from diagonal + score
            let h_from_diag = E::adds_epi8(h_diag, score_vec);
            let mut h_val = E::max_epi8(h_from_diag, e_val);
            h_val = E::max_epi8(h_val, f_vec);
            h_val = E::max_epi8(h_val, zero); // local alignment clamp

            // Finally, mask results by active lanes. Do NOT mask f_vec here, as
            // the AVX2 manual kernel carries F across columns unmasked (only
            // band-start and band-end resets apply). Masking f_vec would alter
            // carry semantics and cause parity differences.
            h_val = E::blendv_epi8(zero, h_val, active_mask);
            e_val = E::blendv_epi8(zero, e_val, active_mask);

            #[cfg(feature = "swk-debug")]
            if dbg_rows.map(|lim| i < lim).unwrap_or(true) {
                // Log activity mask bit for the selected lane (if any)
                let mut mask_buf = [0i8; W];
                E::storeu_epi8(mask_buf.as_mut_ptr(), active_mask);
                if let Some(l) = dbg_lane {
                    swk_dbg!("[swk] lane {}: active={} at (i={}, j={})", l, mask_buf[l] != 0, i, j);
                }
            }

            // Store H(i,j) into current row buffer and E(i,j)
            E::storeu_epi8(curr_ptr.add(j * row_bytes), h_val);
            E::storeu_epi8(e_ptr.add(j * row_bytes), e_val);

            // Track maxima per lane
            E::storeu_epi8(lane_buf.as_mut_ptr(), h_val);
            for lane in 0..lanes {
                if lane_buf[lane] > max_scores[lane] {
                    max_scores[lane] = lane_buf[lane];
                    // Match legacy SSE semantics: target end uses 0-based row index
                    max_i[lane] = (i as i32).min(i8::MAX as i32) as i8;
                    max_j[lane] = (j as i32).min(i8::MAX as i32) as i8;
                    #[cfg(feature = "swk-debug")]
                    if dbg_lane.map(|l| l == lane).unwrap_or(true) && dbg_rows.map(|lim| i < lim).unwrap_or(true) {
                        swk_dbg!("[swk] lane {}: new max={} at (i={}, j={})", lane, lane_buf[lane], i, j);
                    }
                }
                if lane_buf[lane] > row_max[lane] { row_max[lane] = lane_buf[lane]; }
            }

            // Advance diagonals and left H
            h_left = h_val;
            h_diag = h_up;

            // If this column is the last in-band (j+1 == end), ensure we don't carry
            // leftward state into the out-of-band region on next iteration.
            let m_band_end = E::cmpeq_epi8(j_plus1_vec, end_vec);
            h_left = E::blendv_epi8(h_left, zero, m_band_end);
            // Note: do not clear F at band end either to match manual AVX2 behavior.
        }

        // Swap rows: current becomes previous for next iteration, and clear current
        std::mem::swap(&mut h_prev, &mut h_curr);
        // zero current row buffer for next use (optional for correctness but good hygiene)
        for b in h_curr.iter_mut() { *b = 0; }

        // Z-drop check per lane
        if params.zdrop > 0 {
            for lane in 0..lanes {
                let drop = (max_scores[lane] as i32) - (row_max[lane] as i32);
                if drop > params.zdrop {
                    terminated[lane] = true;
                }
            }
        }
    }

    // Assemble results per lane
    let mut out = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        out.push(OutScore {
            score: max_scores[lane] as i32,
            target_end_pos: max_i[lane] as i32,
            gtarget_end_pos: max_i[lane] as i32,
            query_end_pos: max_j[lane] as i32,
            global_score: max_scores[lane] as i32,
            max_offset: 0,
        });
    }
    // Truncate to actual batch size if provided smaller than W
    let actual = params.batch.len().min(lanes);
    out.truncate(actual);

    // Debug: print final OutScore for selected lane
    #[cfg(feature = "swk-debug")]
    if let Ok(val) = std::env::var("SWK_DEBUG_LANE") {
        if let Ok(l) = val.parse::<usize>() {
            if l < out.len() {
                swk_dbg!(
                    "[swk] FINAL lane {}: score={}, tend={}, qend={}",
                    l,
                    out[l].score,
                    out[l].target_end_pos,
                    out[l].query_end_pos
                );
            }
        }
    }
    out
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
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::cmpeq_epi8(a, b) }
    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine128::cmpgt_epi8(b, a) }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        simd::SimdEngine128::blendv_epi8(a, b, mask)
    }
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
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::cmpeq_epi8(a, b) }
    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine256::cmpgt_epi8(b, a) }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        simd::SimdEngine256::blendv_epi8(a, b, mask)
    }
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
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::cmpeq_epi8(a, b) }
    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 { simd::SimdEngine512::cmpgt_epi8(b, a) }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        simd::SimdEngine512::blendv_epi8(a, b, mask)
    }
}
