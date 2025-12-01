//! AVX‑512 int16 path (placeholder/manual for now)
// Keep minimal until an i16 shared kernel lands

#![cfg(target_arch = "x86_64")]

use crate::core::alignment::banded_swa::OutScore;
use crate::alignment::workspace::with_workspace;
use std::arch::x86_64::*; // For raw AVX-512 intrinsics

/// AVX-512-optimized banded Smith-Waterman for batches of up to 32 alignments (16-bit scores)
///
/// **SIMD Width**: 32 lanes (512-bit / 16-bit)
/// **Parallelism**: Processes 32 alignments simultaneously
/// **Score Range**: Full i16 range (-32768 to 32767) for sequences > 127bp
///
/// This is the 16-bit precision version optimized for:
/// - Sequences longer than 127bp where 8-bit scores would overflow
/// - Typical 151bp Illumina reads (max score = 151 with match=1)
///
/// **Performance**: 4x parallelism over SSE 8-wide (8 vs 32 lanes)
#[target_feature(enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch32_int16(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)], // (qlen, query, tlen, target, w, h0)
    o_del: i32,                                   // Gap open penalty (deletion)
    e_del: i32,                                   // Gap extend penalty (deletion)
    o_ins: i32,                                   // Gap open penalty (insertion)
    e_ins: i32,                                   // Gap extend penalty (insertion)
    zdrop: i32,                                   // Z-drop threshold for early termination
    mat: &[i8; 25],                               // Scoring matrix (5x5 for A, C, G, T, N)
    _m: i32,                                      // Matrix dimension (typically 5)
) -> Vec<OutScore> {
    // NOTE: This function intentionally uses raw AVX-512 intrinsics rather than SimdEngine512.
    // Reason: AVX-512's native __mmask32 operations (_mm512_cmpeq_epi16_mask,
    // _mm512_mask_blend_epi16, etc.) are a key performance advantage over vector-based masks.
    // The SimdEngine512 trait uses vector masks (Vec8/Vec16 with 0xFF/0x00 bytes) which would
    // require costly mask<->vector conversions and lose the benefits of native mask operations.
    // This is an acceptable deviation from the abstraction guideline for performance-critical
    // AVX-512-specific code paths.

    const SIMD_WIDTH: usize = 32; // 512-bit / 16-bit = 32 lanes
    const MAX_SEQ_LEN: usize = 512; // 16-bit supports longer sequences

    let batch_size = batch.len().min(SIMD_WIDTH);

    // ==================================================================
    // Step 1: Batch Padding and Parameter Extraction
    // ==================================================================

    // Pad batch to SIMD_WIDTH with dummy entries if needed
    let mut padded_batch = Vec::with_capacity(SIMD_WIDTH);
    for i in 0..SIMD_WIDTH {
        if i < batch.len() {
            padded_batch.push(batch[i]);
        } else {
            // Dummy alignment (will be ignored in results)
            padded_batch.push((0, &[][..], 0, &[][..], 0, 0));
        }
    }

    // Extract batch parameters (32 lanes, 16-bit precision)
    let mut qlen = [0i16; SIMD_WIDTH];
    let mut tlen = [0i16; SIMD_WIDTH];
    let mut h0 = [0i16; SIMD_WIDTH];
    let mut w_arr = [0i16; SIMD_WIDTH];
    let mut max_qlen = 0i32;
    let mut max_tlen = 0i32;

    for i in 0..SIMD_WIDTH {
        let (q, _, t, _, wi, h) = padded_batch[i];
        qlen[i] = q.min(MAX_SEQ_LEN as i32) as i16;
        tlen[i] = t.min(MAX_SEQ_LEN as i32) as i16;
        h0[i] = h as i16;
        w_arr[i] = wi as i16;
        if q > max_qlen {
            max_qlen = q;
        }
        if t > max_tlen {
            max_tlen = t;
        }
    }

    // Clamp to MAX_SEQ_LEN
    let _max_qlen = max_qlen.min(MAX_SEQ_LEN as i32);
    max_tlen = max_tlen.min(MAX_SEQ_LEN as i32);

    // ==================================================================
    // Step 2: Structure-of-Arrays (SoA) Layout Transformation (16-bit)
    // ==================================================================
    // Using 16-bit sequence storage enables vectorized compare-and-blend scoring
    // like C++ BWA-MEM2, instead of scalar matrix lookup.

    // Use thread-local workspace buffers to avoid per-batch allocations (~130KB saved)
    with_workspace(|ws| {
        ws.reset_sw_buffers_avx512();
        let query_soa_16 = &mut ws.sw_query_soa_32[..];
        let target_soa_16 = &mut ws.sw_target_soa_32[..];

        // Encoding:
        // - Normal bases: 0 (A), 1 (C), 2 (G), 3 (T)
        // - Ambiguous/N: 4 or higher
        // - Padding: 0x7FFF (high value for ambig detection)
        const PADDING_VALUE: i16 = 0x7FFF;

        // Transform query and target sequences to 16-bit SoA layout
        for i in 0..SIMD_WIDTH {
            let (q_len, query, t_len, target, _, _) = padded_batch[i];

            let actual_q_len = query.len().min(MAX_SEQ_LEN);
            let actual_t_len = target.len().min(MAX_SEQ_LEN);
            let safe_q_len = (q_len as usize).min(actual_q_len);
            let safe_t_len = (t_len as usize).min(actual_t_len);

            // Copy query (interleaved, widened to 16-bit)
            for j in 0..safe_q_len {
                query_soa_16[j * SIMD_WIDTH + i] = query[j] as i16;
            }
            for j in safe_q_len..MAX_SEQ_LEN {
                query_soa_16[j * SIMD_WIDTH + i] = PADDING_VALUE;
            }

            // Copy target (interleaved, widened to 16-bit)
            for j in 0..safe_t_len {
                target_soa_16[j * SIMD_WIDTH + i] = target[j] as i16;
            }
            for j in safe_t_len..MAX_SEQ_LEN {
                target_soa_16[j * SIMD_WIDTH + i] = PADDING_VALUE;
            }
        }

        // ==================================================================
        // Step 3: Use Pre-allocated DP Matrices (16-bit) from Workspace
        // ==================================================================

        let h_matrix = &mut ws.sw_h_matrix_32[..];
        let e_matrix = &mut ws.sw_e_matrix_32[..];

        // Initialize scores and tracking arrays
        let mut max_scores = vec![0i16; SIMD_WIDTH];
        let mut max_i = vec![-1i16; SIMD_WIDTH];
        let mut max_j = vec![-1i16; SIMD_WIDTH];
        let gscores = vec![0i16; SIMD_WIDTH];
        let max_ie = vec![0i16; SIMD_WIDTH];

        // SIMD constants (16-bit)
        let zero_vec = _mm512_setzero_si512();
        let oe_del = (o_del + e_del) as i16;
        let oe_ins = (o_ins + e_ins) as i16;
        let oe_del_vec = _mm512_set1_epi16(oe_del);
        let oe_ins_vec = _mm512_set1_epi16(oe_ins);
        let e_del_vec = _mm512_set1_epi16(e_del as i16);
        let e_ins_vec = _mm512_set1_epi16(e_ins as i16);

        // Vectorized scoring constants (matching BWA-MEM2 compare-and-blend approach)
        // Extract match/mismatch/ambig scores from scoring matrix
        let match_score = mat[0] as i16; // mat[0] = A vs A = match score
        let mismatch_score = mat[1] as i16; // mat[1] = A vs C = mismatch score
        let ambig_score = mat[4 * 5 + 4] as i16; // mat[24] = N vs N = ambig score
        let match_score_vec = _mm512_set1_epi16(match_score);
        let mismatch_score_vec = _mm512_set1_epi16(mismatch_score);
        let ambig_score_vec = _mm512_set1_epi16(ambig_score);
        let three_vec = _mm512_set1_epi16(3); // Threshold for ambiguous base detection

        // Band tracking
        let mut beg = [0i16; SIMD_WIDTH];
        let mut end = qlen;
        let mut terminated = [false; SIMD_WIDTH];
        // SIMD mask for terminated lanes (converted from bool array when needed)
        let mut terminated_mask: __mmask32 = 0;

        // Initialize first row: h0 for position 0, h0 - oe_ins - j*e_ins for others
        for lane in 0..SIMD_WIDTH {
            let h0_val = h0[lane];
            h_matrix[0 * SIMD_WIDTH + lane] = h0_val;
            e_matrix[0 * SIMD_WIDTH + lane] = 0;

            // Fill first row with gap penalties
            let mut prev_h = h0_val;
            for j in 1..(qlen[lane] as usize).min(MAX_SEQ_LEN) {
                let new_h = if j == 1 {
                    if prev_h > oe_ins { prev_h - oe_ins } else { 0 }
                } else if prev_h > e_ins as i16 {
                    prev_h - e_ins as i16
                } else {
                    0
                };
                h_matrix[j * SIMD_WIDTH + lane] = new_h;
                e_matrix[j * SIMD_WIDTH + lane] = 0;
                if new_h == 0 {
                    break;
                }
                prev_h = new_h;
            }
            max_scores[lane] = h0_val;
        }

        // ==================================================================
        // Step 4: Main DP Loop (16-bit SIMD)
        // ==================================================================

        let mut max_score_vec = _mm512_loadu_si512(max_scores.as_ptr() as *const __m512i);
        // SIMD vectors for max position tracking (avoids scalar loops in hot path)
        let mut max_i_vec = _mm512_set1_epi16(-1);
        let mut max_j_vec = _mm512_set1_epi16(-1);

        for i in 0..max_tlen as usize {
            // Load target bases for this row as 16-bit SIMD vector
            // (vectorized load instead of per-lane scalar loop)
            let t_vec =
                _mm512_loadu_si512(target_soa_16.as_ptr().add(i * SIMD_WIDTH) as *const __m512i);

            // Update band bounds per lane
            let mut current_beg = [0i16; SIMD_WIDTH];
            let mut current_end = [0i16; SIMD_WIDTH];
            for lane in 0..SIMD_WIDTH {
                if terminated[lane] {
                    continue;
                }
                let wi = w_arr[lane];
                let ii = i as i16;
                current_beg[lane] = beg[lane];
                current_end[lane] = end[lane];
                if current_beg[lane] < ii - wi {
                    current_beg[lane] = ii - wi;
                }
                if current_end[lane] > ii + wi + 1 {
                    current_end[lane] = ii + wi + 1;
                }
                if current_end[lane] > qlen[lane] {
                    current_end[lane] = qlen[lane];
                }
            }

            // Process columns within band
            let global_beg = *current_beg.iter().min().unwrap_or(&0) as usize;
            let global_end = *current_end.iter().max().unwrap_or(&0) as usize;

            let mut h1_vec = zero_vec; // H(i, j-1) for first column

            // Initial H value for column 0
            for lane in 0..SIMD_WIDTH {
                if terminated[lane] {
                    continue;
                }
                if current_beg[lane] == 0 {
                    let h_val = h0[lane] as i32 - (o_del + e_del * (i as i32 + 1));
                    let h_val = if h_val < 0 { 0 } else { h_val as i16 };
                    // Set lane 'lane' in h1_vec with h_val
                    let mut h1_arr: [i16; SIMD_WIDTH] = std::mem::transmute(h1_vec);
                    h1_arr[lane] = h_val;
                    h1_vec = std::mem::transmute(h1_arr);
                }
            }

            let mut f_vec = zero_vec;

            for j in global_beg..global_end.min(MAX_SEQ_LEN) {
                // Load H(i-1, j-1) from h_matrix (wavefront storage pattern)
                // and E from e_matrix
                let h00_vec =
                    _mm512_loadu_si512(h_matrix.as_ptr().add(j * SIMD_WIDTH) as *const __m512i);
                let e_vec =
                    _mm512_loadu_si512(e_matrix.as_ptr().add(j * SIMD_WIDTH) as *const __m512i);

                // ==================================================================
                // VECTORIZED SCORING (BWA-MEM2 compare-and-blend approach for AVX-512)
                // ==================================================================
                // This replaces the scalar matrix lookup loop with SIMD mask operations.
                // Reference: BWA-MEM2 MAIN_CODE16 macro for AVX-512 in bandedSWA.cpp:1883-1906

                // Load query bases for column j as 16-bit SIMD vector
                let q_vec =
                    _mm512_loadu_si512(query_soa_16.as_ptr().add(j * SIMD_WIDTH) as *const __m512i);

                // Step 1: Compare bases for equality (returns mask, not vector)
                let cmp_eq: __mmask32 = _mm512_cmpeq_epi16_mask(q_vec, t_vec);

                // Step 2: Select match or mismatch score based on comparison mask
                let score_vec =
                    _mm512_mask_blend_epi16(cmp_eq, mismatch_score_vec, match_score_vec);

                // Step 3: Handle ambiguous bases (either base > 3)
                // If q > 3 OR t > 3, use ambig_score instead
                let q_gt3: __mmask32 = _mm512_cmpgt_epi16_mask(q_vec, three_vec);
                let t_gt3: __mmask32 = _mm512_cmpgt_epi16_mask(t_vec, three_vec);
                let ambig_mask: __mmask32 = q_gt3 | t_gt3;

                // Final match/mismatch/ambig score selection
                let match_vec = _mm512_mask_blend_epi16(ambig_mask, score_vec, ambig_score_vec);

                // M = H(i-1, j-1) + match/mismatch score
                let m_vec = _mm512_add_epi16(h00_vec, match_vec);

                // H(i,j) = max(M, E, F, 0)
                let h_val_vec = _mm512_max_epi16(m_vec, e_vec);
                let h_val_vec = _mm512_max_epi16(h_val_vec, f_vec);
                let h_val_vec = _mm512_max_epi16(h_val_vec, zero_vec);

                // Store H(i,j) in h_matrix
                _mm512_storeu_si512(
                    h_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut __m512i,
                    h_val_vec,
                );

                // Compute E(i+1, j) = max(M - oe_del, E - e_del)
                let e_from_m = _mm512_subs_epi16(m_vec, oe_del_vec);
                let e_from_e = _mm512_subs_epi16(e_vec, e_del_vec);
                let new_e_vec = _mm512_max_epi16(e_from_m, e_from_e);
                let new_e_vec = _mm512_max_epi16(new_e_vec, zero_vec);
                _mm512_storeu_si512(
                    e_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut __m512i,
                    new_e_vec,
                );

                // Compute F(i, j+1) = max(M - oe_ins, F - e_ins)
                let f_from_m = _mm512_subs_epi16(m_vec, oe_ins_vec);
                let f_from_f = _mm512_subs_epi16(f_vec, e_ins_vec);
                f_vec = _mm512_max_epi16(f_from_m, f_from_f);
                f_vec = _mm512_max_epi16(f_vec, zero_vec);

                // VECTORIZED max score tracking
                let j_vec_512 = _mm512_set1_epi16(j as i16);
                let i_vec_512 = _mm512_set1_epi16(i as i16);

                let beg_vec_512 = _mm512_loadu_si512(current_beg.as_ptr() as *const __m512i);
                let end_vec_512 = _mm512_loadu_si512(current_end.as_ptr() as *const __m512i);

                let in_bounds_low: __mmask32 = _mm512_cmpge_epi16_mask(j_vec_512, beg_vec_512);
                let in_bounds_high: __mmask32 = _mm512_cmplt_epi16_mask(j_vec_512, end_vec_512);

                let better_score: __mmask32 = _mm512_cmpgt_epi16_mask(h_val_vec, max_score_vec);

                let update_mask: __mmask32 =
                    in_bounds_low & in_bounds_high & better_score & !terminated_mask;

                max_score_vec = _mm512_mask_blend_epi16(update_mask, max_score_vec, h_val_vec);
                max_i_vec = _mm512_mask_blend_epi16(update_mask, max_i_vec, i_vec_512);
                max_j_vec = _mm512_mask_blend_epi16(update_mask, max_j_vec, j_vec_512);
            }

            // Extract max scores for Z-drop check
            _mm512_storeu_si512(max_scores.as_mut_ptr() as *mut __m512i, max_score_vec);

            // Z-drop early termination
            if zdrop > 0 {
                for lane in 0..SIMD_WIDTH {
                    if !terminated[lane] && i > 0 && i < tlen[lane] as usize {
                        let mut row_max = 0i16;
                        let row_beg = current_beg[lane] as usize;
                        let row_end = current_end[lane] as usize;

                        for j in row_beg..row_end {
                            let h_val = h_matrix[j * SIMD_WIDTH + lane];
                            row_max = row_max.max(h_val);
                        }

                        if max_scores[lane] - row_max > zdrop as i16 {
                            terminated[lane] = true;
                            terminated_mask |= 1u32 << lane;
                        }
                    }
                }
            }

            beg = current_beg;
            end = current_end;
        }

        // ==================================================================
        // Step 5: Result Extraction
        // ==================================================================

        _mm512_storeu_si512(max_scores.as_mut_ptr() as *mut __m512i, max_score_vec);
        _mm512_storeu_si512(max_i.as_mut_ptr() as *mut __m512i, max_i_vec);
        _mm512_storeu_si512(max_j.as_mut_ptr() as *mut __m512i, max_j_vec);

        let mut results = Vec::with_capacity(batch_size);
        for lane in 0..batch_size {
            results.push(OutScore {
                score: max_scores[lane].max(h0[lane]) as i32,
                target_end_pos: max_i[lane] as i32,
                query_end_pos: max_j[lane] as i32,
                gtarget_end_pos: gscores[lane] as i32,
                global_score: max_ie[lane] as i32,
                max_offset: 0,
            });
        }

        results
    }) // End with_workspace closure
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_banded_swa_batch32_int16_basic() {
        // Basic test for 16-bit AVX-512 batch function
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let query: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let target: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        // Default scoring matrix (match=1, mismatch=0)
        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match score on diagonal
        }

        let results = unsafe { simd_banded_swa_batch32_int16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

        assert_eq!(results.len(), 1);
        assert!(
            results[0].score > 0,
            "Score {} should be > 0",
            results[0].score
        );
    }
}
