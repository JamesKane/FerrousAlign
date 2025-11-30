// bwa-mem2-rust/src/banded_swa_avx2.rs
//
// AVX2 (256-bit) SIMD implementation of banded Smith-Waterman alignment
// Processes 32 alignments in parallel (2x speedup over SSE)
//
// This is a port of C++ bwa-mem2's smithWaterman256_8 function
// Reference: /Users/jkane/Applications/bwa-mem2/src/bandedSWA.cpp:722-1150

#![cfg(target_arch = "x86_64")]

use crate::alignment::banded_swa::OutScore;
use crate::alignment::banded_swa_shared::{pad_batch, soa_transform, pack_outscores};
use crate::alignment::workspace::with_workspace;
use crate::compute::simd_abstraction::SimdEngine256 as Engine;

/// AVX2-optimized banded Smith-Waterman for batches of up to 32 alignments
///
/// **SIMD Width**: 32 lanes (2x SSE/NEON)
/// **Parallelism**: Processes 32 alignments simultaneously
/// **Performance**: Expected 1.8-2.2x speedup over SSE (memory-bound)
///
/// **Algorithm**:
/// - Uses Structure-of-Arrays (SoA) layout for SIMD-friendly access
/// - Implements standard Smith-Waterman DP recurrence
/// - Adaptive banding: Only compute cells within [i-w, i+w+1]
/// - Z-drop early termination: Stop lanes when score drops > zdrop
///
/// **Memory Layout**:
/// - Query/target sequences: `seq[position][lane]` (interleaved)
/// - DP matrices (H, E, F): `matrix[position * 32 + lane]`
/// - Query profiles: `profile[target_base][query_pos * 32 + lane]`
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch32(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)], // (qlen, query, tlen, target, w, h0)
    o_del: i32,                                   // Gap open penalty (deletion)
    e_del: i32,                                   // Gap extend penalty (deletion)
    o_ins: i32,                                   // Gap open penalty (insertion)
    e_ins: i32,                                   // Gap extend penalty (insertion)
    zdrop: i32,                                   // Z-drop threshold for early termination
    mat: &[i8; 25],                               // Scoring matrix (5x5 for A, C, G, T, N)
    m: i32,                                       // Matrix dimension (typically 5)
) -> Vec<OutScore> {
    const SIMD_WIDTH: usize = 32; // Engine::WIDTH_8 (32-way parallelism)
    const MAX_SEQ_LEN: usize = 128;

    let batch_size = batch.len().min(SIMD_WIDTH);

    // ==================================================================
    // Step 1: Batch Padding and Parameter Extraction (shared helper)
    // ==================================================================
    let (mut qlen, mut tlen, mut h0, mut w_arr, mut max_qlen, mut max_tlen, padded) =
        pad_batch::<SIMD_WIDTH>(batch);

    // Clamp to MAX_SEQ_LEN
    max_qlen = max_qlen.min(MAX_SEQ_LEN as i32);
    max_tlen = max_tlen.min(MAX_SEQ_LEN as i32);

    // ==================================================================
    // Step 2: Structure-of-Arrays (SoA) Layout Transformation (shared helper)
    // ==================================================================
    let (mut query_soa, mut target_soa) = soa_transform::<SIMD_WIDTH, MAX_SEQ_LEN>(&padded);

    // ==================================================================
    // Step 3: SIMD Constants and DP Matrices Allocation
    // ==================================================================

    // Allocate DP matrices in SoA layout (32 lanes)
    let mut h_matrix = vec![0i8; MAX_SEQ_LEN * SIMD_WIDTH]; // H scores (match)
    let mut e_matrix = vec![0i8; MAX_SEQ_LEN * SIMD_WIDTH]; // E scores (deletion)
    let mut f_matrix = vec![0i8; MAX_SEQ_LEN * SIMD_WIDTH]; // F scores (insertion)

    // Initialize scores and tracking arrays
    let mut max_scores = [0i8; SIMD_WIDTH];
    let mut max_i = [0i8; SIMD_WIDTH];
    let mut max_j = [0i8; SIMD_WIDTH];
    let _gscores = [0i8; SIMD_WIDTH];
    let _max_ie = [0i8; SIMD_WIDTH];

    // SIMD constants using SimdEngine256
    let zero_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::setzero_epi8();
    let oe_del = (o_del + e_del) as i8;
    let oe_ins = (o_ins + e_ins) as i8;
    let oe_del_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(oe_del);
    let oe_ins_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(oe_ins);
    let e_del_vec =
        <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(e_del as i8);
    let e_ins_vec =
        <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(e_ins as i8);

    // ==================================================================
    // Step 4: Initialize First Row of H Matrix
    // ==================================================================

    // Initialize first row of H matrix (query initialization)
    let h0_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
        h0.as_ptr() as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8
    );
    <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
        h_matrix.as_mut_ptr()
            as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
        h0_vec,
    );

    // H[0][1] = max(0, h0 - oe_ins)
    let h1_vec =
        <Engine as crate::compute::simd_abstraction::SimdEngine>::subs_epi8(h0_vec, oe_ins_vec);
    let h1_vec =
        <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(h1_vec, zero_vec);
    <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
        h_matrix.as_mut_ptr().add(SIMD_WIDTH)
            as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
        h1_vec,
    );

    // H[0][j] = max(0, H[0][j-1] - e_ins) for j > 1
    let mut h_prev = h1_vec;
    for j in 2..max_qlen as usize {
        let h_curr =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::subs_epi8(h_prev, e_ins_vec);
        let h_curr =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(h_curr, zero_vec);
        <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
            h_matrix.as_mut_ptr().add(j * SIMD_WIDTH)
                as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
            h_curr,
        );
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

    let match_vec =
        <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(match_score);
    let mismatch_vec =
        <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(mismatch_score);

    // ==================================================================
    // Step 6: Main DP Loop
    // ==================================================================

    // Compute band boundaries for each lane
    let mut beg = [0i8; SIMD_WIDTH]; // Current band start for each lane
    let mut end = [0i8; SIMD_WIDTH]; // Current band end for each lane
    let mut terminated = [false; SIMD_WIDTH]; // Track which lanes have terminated early via Z-drop
    let mut terminated_count = 0usize; // Running count of terminated lanes for early exit

    for lane in 0..SIMD_WIDTH {
        beg[lane] = 0;
        end[lane] = qlen[lane];
    }

    // Main DP loop: Process each target position
    let qlen_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
        qlen.as_ptr() as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
    );
    let _tlen_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
        tlen.as_ptr() as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
    );
    let mut max_score_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
        h0.as_ptr() as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
    );

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
        let i_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(i as i8);
        let w_vec =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(w_arr.as_ptr()
                as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8);
        let beg_vec =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(beg.as_ptr()
                as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8);
        let end_vec =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(end.as_ptr()
                as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8);

        // current_beg = max(beg, i - w)
        let i_minus_w =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::subs_epi8(i_vec, w_vec);
        let current_beg_vec =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(beg_vec, i_minus_w);

        // current_end = min(end, min(i + w + 1, qlen))
        let one_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(1);
        let i_plus_w =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::adds_epi8(i_vec, w_vec);
        let i_plus_w_plus_1 =
            <Engine as crate::compute::simd_abstraction::SimdEngine>::adds_epi8(i_plus_w, one_vec);
        let current_end_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::min_epu8(
            end_vec,
            i_plus_w_plus_1,
        );
        let current_end_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::min_epu8(
            current_end_vec,
            qlen_vec,
        );

        // Extract band boundaries for masking
        let mut current_beg = [0i8; SIMD_WIDTH];
        let mut current_end = [0i8; SIMD_WIDTH];
        <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
            current_beg.as_mut_ptr()
                as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
            current_beg_vec,
        );
        <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
            current_end.as_mut_ptr()
                as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
            current_end_vec,
        );

        // Create termination mask: 0xFF for active lanes, 0x00 for terminated lanes
        let mut term_mask_vals = [0i8; SIMD_WIDTH];
        for lane in 0..SIMD_WIDTH {
            if !terminated[lane] && i < tlen[lane] as usize {
                term_mask_vals[lane] = -1i8; // 0xFF = all bits set = active
            }
        }
        let term_mask = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
            term_mask_vals.as_ptr()
                as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
        );

        // Process each query position
        // Load target base for this row (same for all query positions in the row)
        // BWA-MEM2 pattern: s10 = load(seq1SoA + i * SIMD_WIDTH)
        let s1 = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
            target_soa.as_ptr().add(i * SIMD_WIDTH)
                as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
        );

        let mut h_diag_curr = h_diag.clone();
        for j in 0..max_qlen as usize {
            // Create mask for positions within band (per-lane)
            let j_vec =
                <Engine as crate::compute::simd_abstraction::SimdEngine>::set1_epi8(j as i8);
            let in_band_left = <Engine as crate::compute::simd_abstraction::SimdEngine>::cmpgt_epi8(
                j_vec,
                <Engine as crate::compute::simd_abstraction::SimdEngine>::subs_epi8(
                    current_beg_vec,
                    one_vec,
                ),
            );
            let in_band_right =
                <Engine as crate::compute::simd_abstraction::SimdEngine>::cmpgt_epi8(
                    current_end_vec,
                    j_vec,
                );
            let in_band_mask = <Engine as crate::compute::simd_abstraction::SimdEngine>::and_si128(
                in_band_left,
                in_band_right,
            );

            // Load H[i-1][j-1] (diagonal)
            let h_diag_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
                h_diag_curr.as_ptr()
                    as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
            );

            // Load H[i-1][j] (top) and E[i-1][j]
            let h_top = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
                h_matrix.as_ptr().add(j * SIMD_WIDTH)
                    as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
            );
            let e_prev = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
                e_matrix.as_ptr().add(j * SIMD_WIDTH)
                    as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
            );

            // Save H[i-1][j] for next iteration's diagonal
            h_diag_curr = vec![0i8; SIMD_WIDTH];
            <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
                h_diag_curr.as_mut_ptr()
                    as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
                h_top,
            );

            // ============================================================
            // MAIN_CODE8: Vectorized match/mismatch scoring (BWA-MEM2 pattern)
            // ============================================================
            // Load query base for this column
            let s2 = <Engine as crate::compute::simd_abstraction::SimdEngine>::loadu_si128(
                query_soa.as_ptr().add(j * SIMD_WIDTH)
                    as *const <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
            );

            // Compare bases: returns 0xFF where equal, 0x00 where different
            let cmp_eq =
                <Engine as crate::compute::simd_abstraction::SimdEngine>::cmpeq_epi8(s1, s2);

            // Select match or mismatch score based on comparison
            let score_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::blendv_epi8(
                mismatch_vec,
                match_vec,
                cmp_eq,
            );

            // Handle ambiguous/padding bases: or(s1, s2) has high bit set if either is >= 0x80
            // Padding uses 0xFF, which has high bit set. Zero out those positions.
            let or_bases =
                <Engine as crate::compute::simd_abstraction::SimdEngine>::or_si128(s1, s2);

            // M = H[i-1][j-1] + score (diagonal + score)
            let m_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::adds_epi8(
                h_diag_vec, score_vec,
            );
            // Zero out positions where either base is padding (high bit set in or_bases)
            let m_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::blendv_epi8(
                m_vec, zero_vec, or_bases,
            );
            let m_vec =
                <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(m_vec, zero_vec);

            // Calculate E (gap in target/deletion in query)
            let e_open = <Engine as crate::compute::simd_abstraction::SimdEngine>::subs_epi8(
                m_vec, oe_del_vec,
            );
            let e_open = <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(
                e_open, zero_vec,
            );
            let e_extend = <Engine as crate::compute::simd_abstraction::SimdEngine>::subs_epi8(
                e_prev, e_del_vec,
            );
            let e_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(
                e_open, e_extend,
            );

            // Calculate F (gap in query/insertion in target)
            let f_open = <Engine as crate::compute::simd_abstraction::SimdEngine>::subs_epi8(
                m_vec, oe_ins_vec,
            );
            let f_open = <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(
                f_open, zero_vec,
            );
            let f_extend = <Engine as crate::compute::simd_abstraction::SimdEngine>::subs_epi8(
                f_vec, e_ins_vec,
            );
            f_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(
                f_open, f_extend,
            );

            // H[i][j] = max(M, E, F)
            let mut h_vec =
                <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(m_vec, e_vec);
            h_vec =
                <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(h_vec, f_vec);

            // Apply combined mask: band mask AND termination mask
            let combined_mask = <Engine as crate::compute::simd_abstraction::SimdEngine>::and_si128(
                in_band_mask,
                term_mask,
            );
            h_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::and_si128(
                h_vec,
                combined_mask,
            );
            let e_vec_masked = <Engine as crate::compute::simd_abstraction::SimdEngine>::and_si128(
                e_vec,
                combined_mask,
            );
            let f_vec_masked = <Engine as crate::compute::simd_abstraction::SimdEngine>::and_si128(
                f_vec,
                combined_mask,
            );

            // Store updated scores
            <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
                h_matrix.as_mut_ptr().add(j * SIMD_WIDTH)
                    as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
                h_vec,
            );
            <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
                e_matrix.as_mut_ptr().add(j * SIMD_WIDTH)
                    as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
                e_vec_masked,
            );
            <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
                f_matrix.as_mut_ptr().add(j * SIMD_WIDTH)
                    as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
                f_vec_masked,
            );

            // Track maximum score per lane with positions
            let is_greater = <Engine as crate::compute::simd_abstraction::SimdEngine>::cmpgt_epi8(
                h_vec,
                max_score_vec,
            );
            max_score_vec = <Engine as crate::compute::simd_abstraction::SimdEngine>::max_epi8(
                h_vec,
                max_score_vec,
            );

            // Update positions where new max was found
            let mut h_vals = [0i8; SIMD_WIDTH];
            let mut is_greater_vals = [0i8; SIMD_WIDTH];
            <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
                h_vals.as_mut_ptr()
                    as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
                h_vec,
            );
            <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
                is_greater_vals.as_mut_ptr()
                    as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
                is_greater,
            );

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
        <Engine as crate::compute::simd_abstraction::SimdEngine>::storeu_si128(
            max_score_vals.as_mut_ptr()
                as *mut <Engine as crate::compute::simd_abstraction::SimdEngine>::Vec8,
            max_score_vec,
        );

        // Z-drop early termination: Check if score has dropped too much
        // This implements the same logic as the SSE version's Z-drop check
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
        "AVX2 batch completion: {terminated_count}/{batch_size} lanes terminated, exit_row={final_row}/{max_tlen} ({percent_saved:.1}% saved), early_exit={early_exit}"
    );

    // ==================================================================
    // Step 7: Result Extraction
    // ==================================================================

    // Use shared packer to assemble lane-wise OutScores
    let mut scores32 = [0i32; SIMD_WIDTH];
    let mut qend32 = [0i32; SIMD_WIDTH];
    let mut tend32 = [0i32; SIMD_WIDTH];
    let mut gscore32 = [0i32; SIMD_WIDTH];
    let mut gtend32 = [0i32; SIMD_WIDTH];
    let mut maxoff32 = [0i32; SIMD_WIDTH];
    for i in 0..SIMD_WIDTH {
        scores32[i] = max_scores[i] as i32;
        qend32[i] = max_j[i] as i32;
        tend32[i] = max_i[i] as i32;
        gscore32[i] = max_scores[i] as i32;
        gtend32[i] = max_i[i] as i32;
        maxoff32[i] = 0;
    }
    pack_outscores::<SIMD_WIDTH>(scores32, qend32, tend32, gscore32, gtend32, maxoff32, batch_size)
}

/// AVX2-optimized banded Smith-Waterman for batches of up to 16 alignments (16-bit scores)
///
/// **SIMD Width**: 16 lanes (256-bit / 16-bit)
/// **Parallelism**: Processes 16 alignments simultaneously
/// **Score Range**: Full i16 range (-32768 to 32767) for sequences > 127bp
///
/// This is the 16-bit precision version optimized for:
/// - Sequences longer than 127bp where 8-bit scores would overflow
/// - Typical 151bp Illumina reads (max score = 151 with match=1)
///
/// **Performance**: 2x parallelism over SSE 8-wide (8 vs 16 lanes)
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch16_int16(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)], // (qlen, query, tlen, target, w, h0)
    o_del: i32,                                   // Gap open penalty (deletion)
    e_del: i32,                                   // Gap extend penalty (deletion)
    o_ins: i32,                                   // Gap open penalty (insertion)
    e_ins: i32,                                   // Gap extend penalty (insertion)
    zdrop: i32,                                   // Z-drop threshold for early termination
    mat: &[i8; 25],                               // Scoring matrix (5x5 for A, C, G, T, N)
    _m: i32,                                      // Matrix dimension (typically 5)
) -> Vec<OutScore> {
    // Use SimdEngine256 abstraction instead of raw intrinsics
    use crate::compute::simd_abstraction::SimdEngine;

    const SIMD_WIDTH: usize = 16; // 256-bit / 16-bit = 16 lanes
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

    // Extract batch parameters (16 lanes, 16-bit precision)
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
    max_qlen = max_qlen.min(MAX_SEQ_LEN as i32);  // TODO: Orphaned code?  Not read after assignment
    max_tlen = max_tlen.min(MAX_SEQ_LEN as i32);

    // ==================================================================
    // Step 2: Structure-of-Arrays (SoA) Layout Transformation (16-bit)
    // ==================================================================
    // Using 16-bit sequence storage enables vectorized compare-and-blend scoring
    // like C++ BWA-MEM2, instead of scalar matrix lookup.

    // Use thread-local workspace buffers to avoid per-batch allocations (~65KB saved)
    with_workspace(|ws| {
        // Reset SW kernel buffers to zero before use
        ws.reset_sw_buffers();

        let query_soa_16 = &mut ws.sw_query_soa_16[..];
        let target_soa_16 = &mut ws.sw_target_soa_16[..];

        // Encoding:
        // - Normal bases: 0 (A), 1 (C), 2 (G), 3 (T)
        // - Ambiguous/N: 4 or higher
        // - Padding: 0x7FFF (high value with MSB set for blend detection)
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
        // Step 3: DP Matrices (16-bit) - using pre-allocated workspace buffers
        // ==================================================================

        let h_matrix = &mut ws.sw_h_matrix_16[..];
        let e_matrix = &mut ws.sw_e_matrix_16[..];

        // Initialize scores and tracking arrays
        let mut max_scores = vec![0i16; SIMD_WIDTH];
        let mut max_i = vec![-1i16; SIMD_WIDTH];
        let mut max_j = vec![-1i16; SIMD_WIDTH];
        let gscores = [0i16; SIMD_WIDTH];
        let max_ie = [0i16; SIMD_WIDTH];

        // SIMD constants (16-bit) - using SimdEngine256 abstraction
        let zero_vec = Engine::setzero_epi16();
        let oe_del = (o_del + e_del) as i16;
        let oe_ins = (o_ins + e_ins) as i16;
        let oe_del_vec = Engine::set1_epi16(oe_del);
        let oe_ins_vec = Engine::set1_epi16(oe_ins);
        let e_del_vec = Engine::set1_epi16(e_del as i16);
        let e_ins_vec = Engine::set1_epi16(e_ins as i16);

        // Vectorized scoring constants (matching BWA-MEM2 compare-and-blend approach)
        // Extract match/mismatch/ambig scores from scoring matrix
        let match_score = mat[0] as i16; // mat[0] = A vs A = match score
        let mismatch_score = mat[1] as i16; // mat[1] = A vs C = mismatch score
        let ambig_score = mat[4 * 5 + 4] as i16; // mat[24] = N vs N = ambig score
        let match_score_vec = Engine::set1_epi16(match_score);
        let mismatch_score_vec = Engine::set1_epi16(mismatch_score);
        let ambig_score_vec = Engine::set1_epi16(ambig_score);
        let three_vec = Engine::set1_epi16(3); // Threshold for ambiguous base detection

        // Band tracking
        let mut beg = [0i16; SIMD_WIDTH];
        let mut end = qlen;
        let mut terminated = [false; SIMD_WIDTH];

        // Initialize first row: h0 for position 0, h0 - oe_ins - j*e_ins for others
        for lane in 0..SIMD_WIDTH {
            let h0_val = h0[lane];
            h_matrix[lane] = h0_val;
            e_matrix[lane] = 0;

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

        // Use type alias for brevity in pointer casts
        type Vec16 = std::arch::x86_64::__m256i;

        let mut max_score_vec = Engine::loadu_si128_16(max_scores.as_ptr() as *const Vec16);
        // SIMD vectors for position tracking (initialized to -1)
        let mut max_i_vec = Engine::set1_epi16(-1);
        let mut max_j_vec = Engine::set1_epi16(-1);

        for i in 0..max_tlen as usize {
            // Load target bases for this row as 16-bit SIMD vector
            // (vectorized load instead of per-lane scalar loop)
            let t_vec =
                Engine::loadu_si128_16(target_soa_16.as_ptr().add(i * SIMD_WIDTH) as *const Vec16);

            // Update band bounds per lane
            let mut current_beg = beg;
            let mut current_end = end;
            for lane in 0..SIMD_WIDTH {
                if terminated[lane] {
                    continue;
                }
                let wi = w_arr[lane];
                let ii = i as i16;
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
                    let h1_arr: &mut [i16; 16] = std::mem::transmute(&mut h1_vec);
                    h1_arr[lane] = h_val;
                }
            }

            let mut f_vec = zero_vec;

            // SIMD vector to track row maximum for Z-drop check
            // (accumulated during column loop, eliminates post-hoc nested loop scan)
            let mut row_max_vec = zero_vec;

            for j in global_beg..global_end.min(MAX_SEQ_LEN) {
                // Load H(i-1, j-1) from h_matrix (wavefront storage pattern)
                // and E from e_matrix
                let h00_vec =
                    Engine::loadu_si128_16(h_matrix.as_ptr().add(j * SIMD_WIDTH) as *const Vec16);
                let e_vec =
                    Engine::loadu_si128_16(e_matrix.as_ptr().add(j * SIMD_WIDTH) as *const Vec16);

                // ==================================================================
                // VECTORIZED SCORING (BWA-MEM2 compare-and-blend approach)
                // ==================================================================
                // This replaces the scalar matrix lookup loop with 8 SIMD instructions.
                // Reference: BWA-MEM2 MAIN_CODE16 macro in bandedSWA.cpp:327-346

                // Load query bases for column j as 16-bit SIMD vector
                let q_vec = Engine::loadu_si128_16(
                    query_soa_16.as_ptr().add(j * SIMD_WIDTH) as *const Vec16
                );

                // Step 1: Compare bases for equality (0xFFFF where equal, 0x0000 otherwise)
                let cmp_eq = Engine::cmpeq_epi16(q_vec, t_vec);

                // Step 2: Select match or mismatch score based on comparison
                // blendv_epi8 uses byte MSB: cmp_eq=0xFFFF -> both bytes MSB=1 -> select match
                let score_vec = Engine::blendv_epi8(mismatch_score_vec, match_score_vec, cmp_eq);

                // Step 3: Handle ambiguous bases (either base > 3)
                // If q > 3 OR t > 3, use ambig_score instead
                let q_gt3 = Engine::cmpgt_epi16(q_vec, three_vec); // 0xFFFF where q > 3
                let t_gt3 = Engine::cmpgt_epi16(t_vec, three_vec); // 0xFFFF where t > 3
                let ambig_mask = Engine::or_si128(q_gt3, t_gt3); // 0xFFFF where either > 3

                // Final match/mismatch/ambig score selection
                let match_vec = Engine::blendv_epi8(score_vec, ambig_score_vec, ambig_mask);

                // M = H(i-1, j-1) + match/mismatch score
                let m_vec = Engine::add_epi16(h00_vec, match_vec);

                // H(i,j) = max(M, E, F, 0)
                let h11_vec = Engine::max_epi16(m_vec, e_vec);
                let h11_vec = Engine::max_epi16(h11_vec, f_vec);
                let h11_vec = Engine::max_epi16(h11_vec, zero_vec);

                // Store h1_vec (H(i, j-1) from previous column) into h_matrix[j]
                // This maintains the wavefront pattern for the next row
                Engine::storeu_si128_16(
                    h_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut Vec16,
                    h1_vec,
                );

                // Compute E(i+1, j) = max(M - oe_del, E - e_del)
                let e_from_m = Engine::subs_epi16(m_vec, oe_del_vec);
                let e_from_e = Engine::subs_epi16(e_vec, e_del_vec);
                let new_e_vec = Engine::max_epi16(e_from_m, e_from_e);
                let new_e_vec = Engine::max_epi16(new_e_vec, zero_vec);
                Engine::storeu_si128_16(
                    e_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut Vec16,
                    new_e_vec,
                );

                // Compute F(i, j+1) = max(M - oe_ins, F - e_ins)
                let f_from_m = Engine::subs_epi16(m_vec, oe_ins_vec);
                let f_from_f = Engine::subs_epi16(f_vec, e_ins_vec);
                f_vec = Engine::max_epi16(f_from_m, f_from_f);
                f_vec = Engine::max_epi16(f_vec, zero_vec);

                // ==================================================================
                // VECTORIZED MAX TRACKING (eliminates scalar loop)
                // ==================================================================
                // BWA-MEM2 style: Use SIMD comparison and blend to track max position
                // This replaces the 16-iteration scalar loop with ~5 SIMD instructions

                // Compare h11 > max_score (0xFFFF where true)
                let cmp_gt = Engine::cmpgt_epi16(h11_vec, max_score_vec);

                // Update max_score_vec where h11 > current max
                max_score_vec = Engine::max_epi16(h11_vec, max_score_vec);

                // Broadcast current i and j to SIMD vectors
                let i_vec = Engine::set1_epi16(i as i16);
                let j_vec = Engine::set1_epi16(j as i16);

                // Update max_i_vec and max_j_vec where comparison was true
                // blendv_epi8 uses byte MSB: cmp_gt=0xFFFF -> both bytes MSB=1 -> select new
                max_i_vec = Engine::blendv_epi8(max_i_vec, i_vec, cmp_gt);
                max_j_vec = Engine::blendv_epi8(max_j_vec, j_vec, cmp_gt);

                // ==================================================================
                // VECTORIZED ROW MAX TRACKING (for Z-drop)
                // ==================================================================
                // Track maximum H score in current row for all 16 lanes simultaneously.
                // This replaces the nested loop that would scan h_matrix after the row.
                row_max_vec = Engine::max_epi16(row_max_vec, h11_vec);

                // h1_vec = H(i, j) for the next column
                h1_vec = h11_vec;
            }

            // ==================================================================
            // VECTORIZED Z-DROP CHECK
            // ==================================================================
            // Use pre-computed row_max_vec instead of scanning h_matrix
            // This eliminates O(16 × band_width) nested loop → O(16) scalar check

            // Extract max scores for Z-drop comparison
            let mut max_score_vals = [0i16; SIMD_WIDTH];
            Engine::storeu_si128_16(max_score_vals.as_mut_ptr() as *mut Vec16, max_score_vec);

            if zdrop > 0 {
                // Extract row maxes (computed incrementally during column loop)
                let mut row_max_vals = [0i16; SIMD_WIDTH];
                Engine::storeu_si128_16(row_max_vals.as_mut_ptr() as *mut Vec16, row_max_vec);

                // Check Z-drop condition for each lane (O(16) not O(16 × band))
                for lane in 0..SIMD_WIDTH {
                    if !terminated[lane]
                        && i > 0
                        && i < tlen[lane] as usize
                        && max_score_vals[lane] - row_max_vals[lane] > zdrop as i16
                    {
                        terminated[lane] = true;
                    }
                }
            }

            beg = current_beg;
            end = current_end;
        }

        // ==================================================================
        // Step 5: Result Extraction
        // ==================================================================

        // Extract final values from SIMD vectors
        Engine::storeu_si128_16(max_scores.as_mut_ptr() as *mut Vec16, max_score_vec);
        Engine::storeu_si128_16(max_i.as_mut_ptr() as *mut Vec16, max_i_vec);
        Engine::storeu_si128_16(max_j.as_mut_ptr() as *mut Vec16, max_j_vec);

        let mut results = Vec::with_capacity(batch_size);
        for lane in 0..batch_size {
            results.push(OutScore {
                score: max_scores[lane].max(h0[lane]) as i32,
                target_end_pos: max_i[lane] as i32,
                query_end_pos: max_j[lane] as i32,
                gtarget_end_pos: max_ie[lane] as i32,
                global_score: gscores[lane] as i32,
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
    fn test_simd_banded_swa_batch16_int16_basic() {
        // Basic test for 16-bit AVX2 batch function
        // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
        let query: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let target: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        // Default scoring matrix (match=1, mismatch=0)
        let mut mat = [0i8; 25];
        for i in 0..4 {
            mat[i * 5 + i] = 1; // Match score on diagonal
        }

        let results = unsafe { simd_banded_swa_batch16_int16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

        assert_eq!(results.len(), 1);
        assert!(
            results[0].score > 0,
            "Score {} should be > 0",
            results[0].score
        );
    }

    #[test]
    fn test_simd_banded_swa_batch32_skeleton() {
        // Basic test to ensure the function compiles and runs
        let query = b"ACGT";
        let target = b"ACGT";
        let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

        let results = unsafe {
            simd_banded_swa_batch32(
                &batch, 6,   // o_del
                1,   // e_del
                6,   // o_ins
                1,   // e_ins
                100, // zdrop
                &[0i8; 25], 5,
            )
        };

        assert_eq!(results.len(), 1);
        // TODO: Add proper assertions once implementation is complete
    }
}
