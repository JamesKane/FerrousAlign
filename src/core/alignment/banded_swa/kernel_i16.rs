use crate::alignment::banded_swa::OutScore;
use crate::core::alignment::shared_types::WorkspaceArena;
use crate::core::alignment::workspace::with_workspace;

// kernel_i16.rs
pub trait SwSimd16: Copy {
    type V16: Copy;
    const LANES: usize; // 8 (SSE/NEON), 16 (AVX2), 32 (AVX-512)
    unsafe fn setzero_epi16() -> Self::V16;
    unsafe fn set1_epi16(x: i16) -> Self::V16;
    unsafe fn loadu_epi16(ptr: *const i16) -> Self::V16;
    unsafe fn storeu_epi16(ptr: *mut i16, v: Self::V16);
    unsafe fn adds_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn subs_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn max_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn min_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn cmpeq_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn cmpgt_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn and_si128(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn or_si128(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn andnot_si128(a: Self::V16, b: Self::V16) -> Self::V16; // ~a & b
    unsafe fn blendv_epi8(a: Self::V16, b: Self::V16, mask: Self::V16) -> Self::V16;
}

pub struct KernelParams16<'a> {
    pub batch: &'a [(i32, &'a [u8], i32, &'a [u8], i32, i32)],
    pub query_soa: &'a [i16],
    pub target_soa: &'a [i16],
    pub qlen: &'a [i16],  // i16 for seqs > 127bp
    pub tlen: &'a [i16],  // i16 for seqs > 127bp
    pub h0: &'a [i16],
    pub w: &'a [i8],
    pub max_qlen: i32,
    pub max_tlen: i32,
    pub o_del: i32,
    pub e_del: i32,
    pub o_ins: i32,
    pub e_ins: i32,
    pub zdrop: i32,
    pub mat: &'a [i8; 25],
    pub m: i32,
}

/// Entry point for i16 SW kernel using thread-local workspace.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn sw_kernel_i16<const W: usize, E: SwSimd16>(
    params: &KernelParams16<'_>,
    num_jobs: usize,
) -> Vec<OutScore>
where
    <E as SwSimd16>::V16: std::fmt::Debug,
{
    // Use thread-local workspace to obtain reusable aligned rows
    with_workspace(|ws| sw_kernel_i16_with_ws::<W, E>(params, num_jobs, ws))
}

/// Shared banded SW kernel (i16) using a reusable `WorkspaceArena` for DP rows.
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn sw_kernel_i16_with_ws<const W: usize, E: SwSimd16>(
    params: &KernelParams16<'_>,
    num_jobs: usize,
    ws: &mut dyn WorkspaceArena,
) -> Vec<OutScore>
where
    <E as SwSimd16>::V16: std::fmt::Debug,
{
    debug_assert_eq!(W, E::LANES, "W generic must match engine lanes");
    debug_assert!(num_jobs <= W, "num_jobs ({}) exceeds SIMD width ({})", num_jobs, W);

    // SoA stride is always W (SIMD width), even if fewer lanes are active
    let stride = W;
    // Active lanes bounded by num_jobs and SIMD width
    let lanes = num_jobs.min(W);
    let qmax = params.max_qlen.max(0) as usize;
    let tmax = params.max_tlen.max(0) as usize;

    if qmax == 0 || tmax == 0 || lanes == 0 {
        return Vec::new();
    }

    // Ensure reusable rows from workspace (i16 path)
    ws.ensure_rows(stride, qmax, tmax, std::mem::size_of::<i16>());
    let (h_rows, e_rows, _f_rows) = ws
        .rows_u16()
        .expect("WorkspaceArena did not provide i16 rows after ensure_rows");
    debug_assert!(h_rows.len() >= qmax * stride, "H row too small");
    debug_assert!(e_rows.len() >= qmax * stride, "E row too small");

    let h_ptr = h_rows.as_mut_ptr();
    let e_ptr = e_rows.as_mut_ptr();

    // Score constants (match/mismatch) from matrix similar to MAIN_CODE16 pattern
    let match_score = params.mat[0] as i16;
    let mismatch_score = params.mat[1] as i16;
    let ambig_score = params.mat[4 * 5 + 4] as i16; // N vs N score

    let zero = E::setzero_epi16();
    let match_vec = E::set1_epi16(match_score);
    let mismatch_vec = E::set1_epi16(mismatch_score);
    let ambig_score_vec = E::set1_epi16(ambig_score);
    let three_vec = E::set1_epi16(3); // Threshold for ambiguous base detection

    // Gap penalties (saturating 16-bit arithmetic)
    let oe_del = (params.o_del + params.e_del) as i16;
    let oe_ins = (params.o_ins + params.e_ins) as i16;
    let e_del = params.e_del as i16;
    let e_ins = params.e_ins as i16;
    let oe_del_vec = E::set1_epi16(oe_del);
    let oe_ins_vec = E::set1_epi16(oe_ins);
    let e_del_vec = E::set1_epi16(e_del);
    let e_ins_vec = E::set1_epi16(e_ins);

    // Initialize first row using workspace buffers
    let h0_vec = E::loadu_epi16(params.h0.as_ptr());
    E::storeu_epi16(h_ptr, h0_vec);

    let h1_vec = E::subs_epi16(h0_vec, oe_ins_vec);
    let h1_vec = E::max_epi16(h1_vec, zero);
    E::storeu_epi16(h_ptr.add(stride), h1_vec);

    let mut h_prev = h1_vec;
    for j in 2..qmax {
        let h_curr = E::subs_epi16(h_prev, e_ins_vec);
        let h_curr = E::max_epi16(h_curr, zero);
        E::storeu_epi16(h_ptr.add(j * stride), h_curr);
        h_prev = h_curr;
    }

    // Track maxima per lane
    let mut max_scores_vec = E::loadu_epi16(params.h0.as_ptr());
    let mut max_i_vec = E::set1_epi16(-1); // Initialize with -1 for correct position tracking
    let mut max_j_vec = E::set1_epi16(-1); // Initialize with -1

    let mut beg = [0i16; W];
    let mut end = [0i16; W];
    let mut terminated = [false; W];
    let mut terminated_count = 0;
    for lane in 0..lanes {
        end[lane] = params.qlen[lane] as i16;
    }

    let qlen_i16_vec = {
        let mut qlen_i16 = [0i16; W];
        for lane in 0..lanes {
            qlen_i16[lane] = params.qlen[lane] as i16;
        }
        E::loadu_epi16(qlen_i16.as_ptr())
    };

    // Main DP loop
    for i in 0..tmax {
        if terminated_count == lanes {
            break;
        }

        let mut f_vec = zero;
        let mut h_diag = E::loadu_epi16(h_ptr);

        let s1 = E::loadu_epi16(params.target_soa.as_ptr().add(i * stride));

        let i_vec = E::set1_epi16(i as i16);
        let w_vec = {
            let mut w_i16 = [0i16; W];
            for lane in 0..lanes {
                w_i16[lane] = params.w[lane] as i16;
            }
            E::loadu_epi16(w_i16.as_ptr())
        };
        let beg_vec = E::loadu_epi16(beg.as_ptr());
        let end_vec = E::loadu_epi16(end.as_ptr());

        let one_vec = E::set1_epi16(1);

        // Update band bounds dynamically
        let i_minus_w = E::subs_epi16(i_vec, w_vec);
        let current_beg_vec = E::max_epi16(beg_vec, i_minus_w);

        let i_plus_w = E::adds_epi16(i_vec, w_vec);
        let i_plus_w_plus_1 = E::adds_epi16(i_plus_w, one_vec);
        let mut current_end_vec = E::min_epi16(end_vec, i_plus_w_plus_1);
        current_end_vec = E::min_epi16(current_end_vec, qlen_i16_vec);

        E::storeu_epi16(beg.as_mut_ptr(), current_beg_vec);
        E::storeu_epi16(end.as_mut_ptr(), current_end_vec);

        let mut term_mask_vals = [0i16; W];
        for lane in 0..lanes {
            if !terminated[lane] && i < params.tlen[lane] as usize {
                term_mask_vals[lane] = -1i16; // All bits set for mask
            }
        }
        let term_mask = E::loadu_epi16(term_mask_vals.as_ptr());

        // SIMD vector to track row maximum for Z-drop check
        let mut row_max_vec = zero;

        for j in 0..qmax {
            let h_top = E::loadu_epi16(h_ptr.add(j * stride));
            let e_prev = E::loadu_epi16(e_ptr.add(j * stride));

            let h_diag_curr = h_diag;
            h_diag = h_top;

            let s2 = E::loadu_epi16(params.query_soa.as_ptr().add(j * stride));

            // Scoring logic using compare-and-blend
            let cmp_eq = E::cmpeq_epi16(s1, s2);
            let score_if_match = E::blendv_epi8(mismatch_vec, match_vec, cmp_eq);

            let q_gt3 = E::cmpgt_epi16(s2, three_vec);
            let t_gt3 = E::cmpgt_epi16(s1, three_vec);
            let ambig_mask = E::or_si128(q_gt3, t_gt3);

            let score_vec = E::blendv_epi8(score_if_match, ambig_score_vec, ambig_mask);

            let m_vec = E::adds_epi16(h_diag_curr, score_vec);
            let m_vec = E::max_epi16(m_vec, zero);

            let e_open = E::subs_epi16(m_vec, oe_del_vec);
            let e_open = E::max_epi16(e_open, zero);
            let e_extend = E::subs_epi16(e_prev, e_del_vec);
            let e_val = E::max_epi16(e_open, e_extend);

            let f_open = E::subs_epi16(m_vec, oe_ins_vec);
            let f_open = E::max_epi16(f_open, zero);
            let f_extend = E::subs_epi16(f_vec, e_ins_vec);
            f_vec = E::max_epi16(f_open, f_extend);

            let mut h_val = E::max_epi16(m_vec, e_val);
            h_val = E::max_epi16(h_val, f_vec);
            h_val = E::max_epi16(h_val, zero);

            let j_vec = E::set1_epi16(j as i16);
            let in_band_left = E::cmpgt_epi16(j_vec, E::subs_epi16(current_beg_vec, one_vec));
            let in_band_right = E::cmpgt_epi16(current_end_vec, j_vec);
            let in_band_mask = E::and_si128(in_band_left, in_band_right);

            let combined_mask = E::and_si128(in_band_mask, term_mask);
            h_val = E::and_si128(h_val, combined_mask);
            let e_val_masked = E::and_si128(e_val, combined_mask);

            E::storeu_epi16(h_ptr.add(j * stride), h_val);
            E::storeu_epi16(e_ptr.add(j * stride), e_val_masked);

            let is_greater = E::cmpgt_epi16(h_val, max_scores_vec);
            max_scores_vec = E::max_epi16(h_val, max_scores_vec);

            max_i_vec = E::blendv_epi8(max_i_vec, i_vec, is_greater);
            max_j_vec = E::blendv_epi8(max_j_vec, j_vec, is_greater);

            row_max_vec = E::max_epi16(row_max_vec, h_val);
        }

        let mut max_score_vals = [0i16; W];
        E::storeu_epi16(max_score_vals.as_mut_ptr(), max_scores_vec);

        if params.zdrop > 0 {
            let mut row_max_vals = [0i16; W];
            E::storeu_epi16(row_max_vals.as_mut_ptr(), row_max_vec);

            for lane in 0..lanes {
                if !terminated[lane]
                    && (i as i16) > 0
                    && (i as i16) < params.tlen[lane] as i16
                    && max_score_vals[lane] - row_max_vals[lane] > params.zdrop as i16
                {
                    terminated[lane] = true;
                    terminated_count += 1;
                }
            }
        }
    }

    let mut out_scores = vec![];
    let mut max_scores = [0i16; W];
    let mut max_i = [0i16; W];
    let mut max_j = [0i16; W];

    E::storeu_epi16(max_scores.as_mut_ptr(), max_scores_vec);
    E::storeu_epi16(max_i.as_mut_ptr(), max_i_vec);
    E::storeu_epi16(max_j.as_mut_ptr(), max_j_vec);

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
