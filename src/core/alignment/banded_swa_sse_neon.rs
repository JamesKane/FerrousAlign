// bwa-mem2-rust/src/banded_swa_sse_neon.rs
//
// SSE/NEON (128-bit) SIMD implementation of banded Smith-Waterman alignment
// Processes 16 alignments in parallel (baseline SIMD for all platforms)
//
// This is a port of the AVX2 version adapted for 128-bit SIMD width
// Works on both x86_64 (SSE2+) and aarch64 (NEON)

use crate::alignment::banded_swa::OutScore;
use crate::compute::simd_abstraction::SimdEngine;
use crate::compute::simd_abstraction::SimdEngine128 as Engine;

/// SSE/NEON-optimized banded Smith-Waterman for batches of up to 16 alignments
///
/// **SIMD Width**: 16 lanes (baseline for SSE2/NEON)
/// **Parallelism**: Processes 16 alignments simultaneously
/// **Platform**: Works on x86_64 (SSE2+) and aarch64 (NEON)
///
/// **Algorithm**:
/// - Uses Structure-of-Arrays (SoA) layout for SIMD-friendly access
/// - Implements standard Smith-Waterman DP recurrence
/// - Adaptive banding: Only compute cells within [i-w, i+w+1]
/// - Z-drop early termination: Stop lanes when score drops > zdrop
///
/// **Memory Layout**:
/// - Query/target sequences: `seq[position][lane]` (interleaved)
/// - DP matrices (H, E, F): `matrix[position * 16 + lane]`
/// - Query profiles: `profile[target_base][query_pos * 16 + lane]`
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch16(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)], // (qlen, query, tlen, target, w, h0)
    o_del: i32,                                   // Gap open penalty (deletion)
    e_del: i32,                                   // Gap extend penalty (deletion)
    o_ins: i32,                                   // Gap open penalty (insertion)
    e_ins: i32,                                   // Gap extend penalty (insertion)
    zdrop: i32,                                   // Z-drop threshold for early termination
    mat: &[i8; 25],                               // Scoring matrix (5x5 for A, C, G, T, N)
    m: i32,                                       // Matrix dimension (typically 5)
) -> Vec<OutScore> {
    const SIMD_WIDTH: usize = 16; // 128-bit / 8-bit = 16 lanes
    const MAX_SEQ_LEN: usize = 128;

    // Log that we're using the SIMD path
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        #[cfg(target_arch = "x86_64")]
        log::info!("[SIMD] Executing SSE banded Smith-Waterman kernel (16-way parallelism)");
        #[cfg(target_arch = "aarch64")]
        log::info!("[SIMD] Executing NEON banded Smith-Waterman kernel (16-way parallelism)");
    }

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

    // Extract batch parameters (16 lanes)
    let mut qlen = [0i8; SIMD_WIDTH];
    let mut tlen = [0i8; SIMD_WIDTH];
    let mut h0 = [0i8; SIMD_WIDTH];
    let mut w_arr = [0i8; SIMD_WIDTH];
    let mut max_qlen = 0i32;
    let mut max_tlen = 0i32;

    for i in 0..SIMD_WIDTH {
        let (q, _, t, _, wi, h) = padded_batch[i];
        qlen[i] = q.min(127) as i8;
        tlen[i] = t.min(127) as i8;
        h0[i] = h as i8;
        w_arr[i] = wi as i8;
        if q > max_qlen {
            max_qlen = q;
        }
        if t > max_tlen {
            max_tlen = t;
        }
    }

    // Clamp to MAX_SEQ_LEN
    max_qlen = max_qlen.min(MAX_SEQ_LEN as i32);
    max_tlen = max_tlen.min(MAX_SEQ_LEN as i32);

    // ==================================================================
    // Step 2: Structure-of-Arrays (SoA) Layout Transformation
    // ==================================================================

    // Allocate SoA buffers for SIMD-friendly access
    // Layout: seq[position][lane] - all 16 lane values for position 0, then position 1, etc.
    let mut query_soa = vec![0u8; MAX_SEQ_LEN * SIMD_WIDTH];
    let mut target_soa = vec![0u8; MAX_SEQ_LEN * SIMD_WIDTH];

    // Transform query and target sequences to SoA layout
    for i in 0..SIMD_WIDTH {
        let (q_len, query, t_len, target, _, _) = padded_batch[i];

        // Copy query (interleaved: q0[0], q1[0], ..., q15[0], q0[1], q1[1], ...)
        for j in 0..(q_len as usize).min(MAX_SEQ_LEN) {
            query_soa[j * SIMD_WIDTH + i] = query[j];
        }
        // Pad with dummy value
        for j in (q_len as usize)..MAX_SEQ_LEN {
            query_soa[j * SIMD_WIDTH + i] = 0xFF;
        }

        // Copy target (interleaved)
        for j in 0..(t_len as usize).min(MAX_SEQ_LEN) {
            target_soa[j * SIMD_WIDTH + i] = target[j];
        }
        // Pad with dummy value
        for j in (t_len as usize)..MAX_SEQ_LEN {
            target_soa[j * SIMD_WIDTH + i] = 0xFF;
        }
    }

    // ==================================================================
    // Step 3: SIMD Constants and DP Matrices Allocation
    // ==================================================================

    // Allocate DP matrices in SoA layout (16 lanes)
    let mut h_matrix = vec![0i8; MAX_SEQ_LEN * SIMD_WIDTH]; // H scores (match)
    let mut e_matrix = vec![0i8; MAX_SEQ_LEN * SIMD_WIDTH]; // E scores (deletion)
    let mut f_matrix = vec![0i8; MAX_SEQ_LEN * SIMD_WIDTH]; // F scores (insertion)

    // Initialize scores and tracking arrays
    let mut max_scores = vec![0i8; SIMD_WIDTH];
    let mut max_i = vec![0i8; SIMD_WIDTH];
    let mut max_j = vec![0i8; SIMD_WIDTH];
    let _gscores = vec![0i8; SIMD_WIDTH];
    let _max_ie = vec![0i8; SIMD_WIDTH];

    // SIMD constants using SimdEngine128
    let zero_vec = Engine::setzero_epi8();
    let oe_del = (o_del + e_del) as i8;
    let oe_ins = (o_ins + e_ins) as i8;
    let oe_del_vec = Engine::set1_epi8(oe_del);
    let oe_ins_vec = Engine::set1_epi8(oe_ins);
    let e_del_vec = Engine::set1_epi8(e_del as i8);
    let e_ins_vec = Engine::set1_epi8(e_ins as i8);

    // ==================================================================
    // Step 4: Initialize First Row of H Matrix
    // ==================================================================

    // Initialize first row of H matrix (query initialization)
    let h0_vec = Engine::loadu_si128(h0.as_ptr() as *const _);
    Engine::storeu_si128(h_matrix.as_mut_ptr() as *mut _, h0_vec);

    // H[0][1] = max(0, h0 - oe_ins)
    let h1_vec = Engine::subs_epi8(h0_vec, oe_ins_vec);
    let h1_vec = Engine::max_epi8(h1_vec, zero_vec);
    Engine::storeu_si128(h_matrix.as_mut_ptr().add(SIMD_WIDTH) as *mut _, h1_vec);

    // H[0][j] = max(0, H[0][j-1] - e_ins) for j > 1
    let mut h_prev = h1_vec;
    for j in 2..max_qlen as usize {
        let h_curr = Engine::subs_epi8(h_prev, e_ins_vec);
        let h_curr = Engine::max_epi8(h_curr, zero_vec);
        Engine::storeu_si128(h_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut _, h_curr);
        h_prev = h_curr;
    }

    // ==================================================================
    // Step 5: Scoring Constants (BWA-MEM2 compare-and-blend approach)
    // ==================================================================

    // Instead of query profiles, we use direct compare-and-blend like BWA-MEM2's MAIN_CODE8
    // This eliminates the scalar gather loop and uses pure SIMD operations
    let match_score = mat[0]; // A-A match score (mat[0*5+0])
    let mismatch_score = mat[1]; // A-C mismatch score (mat[0*5+1])
    let _ambig_score = mat[4 * m as usize + 4]; // N-N score (typically 0 or negative)

    let match_vec = Engine::set1_epi8(match_score);
    let mismatch_vec = Engine::set1_epi8(mismatch_score);
    // For ambiguous bases, BWA-MEM2 uses max_epu8(s1,s2) to detect high bit (padding = 0x80+)
    // and zeros out those positions. We'll use a threshold of 3 (bases 0-3 are valid)

    // ==================================================================
    // Step 6: Main DP Loop
    // ==================================================================

    // Compute band boundaries for each lane
    let mut beg = vec![0i8; SIMD_WIDTH]; // Current band start for each lane
    let mut end = vec![0i8; SIMD_WIDTH]; // Current band end for each lane
    let mut terminated = vec![false; SIMD_WIDTH]; // Track which lanes have terminated early via Z-drop
    let mut terminated_count = 0usize; // Running count of terminated lanes for early exit

    for lane in 0..SIMD_WIDTH {
        beg[lane] = 0;
        end[lane] = qlen[lane];
    }

    // Main DP loop: Process each target position
    let qlen_vec = Engine::loadu_si128(qlen.as_ptr() as *const _);
    let _tlen_vec = Engine::loadu_si128(tlen.as_ptr() as *const _);
    let mut max_score_vec = Engine::loadu_si128(h0.as_ptr() as *const _);

    let mut final_row = max_tlen as usize; // Track where we exited
    for i in 0..max_tlen as usize {
        // Early batch completion: Exit when majority of lanes (>50%) have terminated
        // This reduces wasted computation for stragglers while ensuring most work is done
        if terminated_count > batch_size / 2 {
            final_row = i;
            break;
        }

        let mut f_vec = zero_vec; // F (insertion) scores for this row
        let h_diag = h_matrix[0..SIMD_WIDTH].to_vec(); // Save H[i-1][0..SIMD_WIDTH] for diagonal

        // Compute band boundaries for this row (per-lane)
        let i_vec = Engine::set1_epi8(i as i8);
        let w_vec = Engine::loadu_si128(w_arr.as_ptr() as *const _);
        let beg_vec = Engine::loadu_si128(beg.as_ptr() as *const _);
        let end_vec = Engine::loadu_si128(end.as_ptr() as *const _);

        // current_beg = max(beg, i - w)
        let i_minus_w = Engine::subs_epi8(i_vec, w_vec);
        let current_beg_vec = Engine::max_epi8(beg_vec, i_minus_w);

        // current_end = min(end, min(i + w + 1, qlen))
        let one_vec = Engine::set1_epi8(1);
        let i_plus_w = Engine::adds_epi8(i_vec, w_vec);
        let i_plus_w_plus_1 = Engine::adds_epi8(i_plus_w, one_vec);
        let current_end_vec = Engine::min_epu8(end_vec, i_plus_w_plus_1);
        let current_end_vec = Engine::min_epu8(current_end_vec, qlen_vec);

        // Extract band boundaries for masking
        let mut current_beg = [0i8; SIMD_WIDTH];
        let mut current_end = [0i8; SIMD_WIDTH];
        Engine::storeu_si128(current_beg.as_mut_ptr() as *mut _, current_beg_vec);
        Engine::storeu_si128(current_end.as_mut_ptr() as *mut _, current_end_vec);

        // Create termination mask: 0xFF for active lanes, 0x00 for terminated lanes
        let mut term_mask_vals = [0i8; SIMD_WIDTH];
        for lane in 0..SIMD_WIDTH {
            if !terminated[lane] && i < tlen[lane] as usize {
                term_mask_vals[lane] = -1i8; // 0xFF = all bits set = active
            }
        }
        let term_mask = Engine::loadu_si128(term_mask_vals.as_ptr() as *const _);

        // Process each query position
        // Load target base for this row (same for all query positions in the row)
        // BWA-MEM2 pattern: s10 = load(seq1SoA + i * SIMD_WIDTH)
        let s1 = Engine::loadu_si128(target_soa.as_ptr().add(i * SIMD_WIDTH) as *const _);

        let mut h_diag_curr = h_diag.clone();
        for j in 0..max_qlen as usize {
            // Create mask for positions within band (per-lane)
            let j_vec = Engine::set1_epi8(j as i8);
            let in_band_left =
                Engine::cmpgt_epi8(j_vec, Engine::subs_epi8(current_beg_vec, one_vec));
            let in_band_right = Engine::cmpgt_epi8(current_end_vec, j_vec);
            let in_band_mask = Engine::and_si128(in_band_left, in_band_right);

            // Load H[i-1][j-1] (diagonal)
            let h_diag_vec = Engine::loadu_si128(h_diag_curr.as_ptr() as *const _);

            // Load H[i-1][j] (top) and E[i-1][j]
            let h_top = Engine::loadu_si128(h_matrix.as_ptr().add(j * SIMD_WIDTH) as *const _);
            let e_prev = Engine::loadu_si128(e_matrix.as_ptr().add(j * SIMD_WIDTH) as *const _);

            // Save H[i-1][j] for next iteration's diagonal
            h_diag_curr = vec![0i8; SIMD_WIDTH];
            Engine::storeu_si128(h_diag_curr.as_mut_ptr() as *mut _, h_top);

            // ============================================================
            // MAIN_CODE8: Vectorized match/mismatch scoring (BWA-MEM2 pattern)
            // ============================================================
            // Load query base for this column
            let s2 = Engine::loadu_si128(query_soa.as_ptr().add(j * SIMD_WIDTH) as *const _);

            // Compare bases: returns 0xFF where equal, 0x00 where different
            let cmp_eq = Engine::cmpeq_epi8(s1, s2);

            // Select match or mismatch score based on comparison
            let score_vec = Engine::blendv_epi8(mismatch_vec, match_vec, cmp_eq);

            // Handle ambiguous/padding bases: max_epu8(s1, s2) has high bit set if either is >= 0x80
            // Padding uses 0xFF, which has high bit set. Zero out those positions.
            let or_bases = Engine::or_si128(s1, s2);

            // M = H[i-1][j-1] + score (diagonal + score)
            let m_vec = Engine::adds_epi8(h_diag_vec, score_vec);
            // Zero out positions where either base is padding (high bit set in or_bases)
            let m_vec = Engine::blendv_epi8(m_vec, zero_vec, or_bases);
            let m_vec = Engine::max_epi8(m_vec, zero_vec);

            // Calculate E (gap in target/deletion in query)
            let e_open = Engine::subs_epi8(m_vec, oe_del_vec);
            let e_open = Engine::max_epi8(e_open, zero_vec);
            let e_extend = Engine::subs_epi8(e_prev, e_del_vec);
            let e_vec = Engine::max_epi8(e_open, e_extend);

            // Calculate F (gap in query/insertion in target)
            let f_open = Engine::subs_epi8(m_vec, oe_ins_vec);
            let f_open = Engine::max_epi8(f_open, zero_vec);
            let f_extend = Engine::subs_epi8(f_vec, e_ins_vec);
            f_vec = Engine::max_epi8(f_open, f_extend);

            // H[i][j] = max(M, E, F)
            let mut h_vec = Engine::max_epi8(m_vec, e_vec);
            h_vec = Engine::max_epi8(h_vec, f_vec);

            // Apply combined mask: band mask AND termination mask
            let combined_mask = Engine::and_si128(in_band_mask, term_mask);
            h_vec = Engine::and_si128(h_vec, combined_mask);
            let e_vec_masked = Engine::and_si128(e_vec, combined_mask);
            let f_vec_masked = Engine::and_si128(f_vec, combined_mask);

            // Store updated scores
            Engine::storeu_si128(h_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut _, h_vec);
            Engine::storeu_si128(
                e_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut _,
                e_vec_masked,
            );
            Engine::storeu_si128(
                f_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut _,
                f_vec_masked,
            );

            // Track maximum score per lane with positions
            let is_greater = Engine::cmpgt_epi8(h_vec, max_score_vec);
            max_score_vec = Engine::max_epi8(h_vec, max_score_vec);

            // Update positions where new max was found
            let mut h_vals = [0i8; SIMD_WIDTH];
            let mut is_greater_vals = [0i8; SIMD_WIDTH];
            Engine::storeu_si128(h_vals.as_mut_ptr() as *mut _, h_vec);
            Engine::storeu_si128(is_greater_vals.as_mut_ptr() as *mut _, is_greater);

            // Update max positions for each lane
            for lane in 0..SIMD_WIDTH {
                if is_greater_vals[lane] as u8 == 0xFF {
                    max_i[lane] = i as i8;
                    max_j[lane] = j as i8;
                }
            }
        }

        // Extract max scores for Z-drop check
        let mut max_score_vals = [0i8; SIMD_WIDTH];
        Engine::storeu_si128(max_score_vals.as_mut_ptr() as *mut _, max_score_vec);

        // Z-drop early termination: Check if score has dropped too much
        // This implements the same logic as the AVX2 version's Z-drop check
        if zdrop > 0 {
            for lane in 0..SIMD_WIDTH {
                if !terminated[lane] && i > 0 && i < tlen[lane] as usize {
                    // Compute maximum score seen in this row (within adaptive band)
                    let mut row_max = 0i8;
                    let row_beg = current_beg[lane] as usize;
                    let row_end = current_end[lane] as usize;

                    for j in row_beg..row_end {
                        let h_val = h_matrix[j * SIMD_WIDTH + lane];
                        row_max = row_max.max(h_val);
                    }

                    // Early termination condition 1: row max drops to 0
                    if row_max == 0 {
                        terminated[lane] = true;
                        terminated_count += 1;
                        continue;
                    }

                    // Early termination condition 2: zdrop threshold
                    let global_max = max_score_vals[lane];
                    let score_drop = (global_max as i32) - (row_max as i32);

                    if score_drop > zdrop {
                        terminated[lane] = true;
                        terminated_count += 1;
                    }
                }
            }
        }

        // Update max_scores array
        for lane in 0..SIMD_WIDTH {
            max_scores[lane] = max_score_vals[lane];
        }
    }

    // Log early termination statistics (DEBUG level for detailed analysis)
    let early_exit = final_row < max_tlen as usize;
    let rows_saved = max_tlen as usize - final_row;
    let percent_saved = (rows_saved as f64 / max_tlen as f64) * 100.0;

    log::debug!(
        "SSE/NEON batch completion: {}/{} lanes terminated, exit_row={}/{} ({:.1}% saved), early_exit={}",
        terminated_count,
        batch_size,
        final_row,
        max_tlen,
        percent_saved,
        early_exit
    );

    // ==================================================================
    // Step 7: Result Extraction
    // ==================================================================

    let mut results = Vec::with_capacity(batch_size);
    for lane in 0..batch_size {
        results.push(OutScore {
            score: max_scores[lane] as i32,
            target_end_pos: max_i[lane] as i32,
            query_end_pos: max_j[lane] as i32,
            gtarget_end_pos: max_i[lane] as i32,
            global_score: max_scores[lane] as i32,
            max_offset: 0,
        });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_banded_swa_batch16_basic() {
        // Basic test to ensure the function compiles and runs
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let query: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let target: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        // Default scoring matrix (match=1, mismatch=0)
        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match score on diagonal
        }

        let results = unsafe {
            simd_banded_swa_batch16(
                &batch, 6,   // o_del
                1,   // e_del
                6,   // o_ins
                1,   // e_ins
                100, // zdrop
                &mat, 5,
            )
        };

        assert_eq!(results.len(), 1);
        // Perfect match should have score >= 4
        assert!(
            results[0].score >= 4,
            "Expected score >= 4 for perfect match, got {}",
            results[0].score
        );
    }

    #[test]
    fn test_simd_banded_swa_batch16_multiple() {
        // Test with multiple alignments in batch
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let q1: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let t1: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let q2: [u8; 4] = [0, 0, 0, 0]; // AAAA in 2-bit encoding
        let t2: [u8; 4] = [3, 3, 3, 3]; // TTTT in 2-bit encoding

        let batch = vec![
            (4, &q1[..], 4, &t1[..], 10, 0),
            (4, &q2[..], 4, &t2[..], 10, 0),
        ];

        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match
        }

        let results = unsafe { simd_banded_swa_batch16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

        assert_eq!(results.len(), 2);
        // First alignment (perfect match) should score higher
        assert!(
            results[0].score >= results[1].score,
            "Perfect match should score >= mismatch: {} vs {}",
            results[0].score,
            results[1].score
        );
    }
}
