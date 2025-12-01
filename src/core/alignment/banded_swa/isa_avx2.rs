// bwa-mem2-rust/src/banded_swa_avx2.rs
//
// AVX2 (256-bit) SIMD implementation of banded Smith-Waterman alignment
// Processes 32 alignments in parallel (2x speedup over SSE)
//
// This is a port of C++ bwa-mem2's smithWaterman256_8 function
// Reference: /Users/jkane/Applications/bwa-mem2/src/bandedSWA.cpp:722-1150

#![cfg(target_arch = "x86_64")]

use crate::core::alignment::banded_swa::OutScore;
use super::engines::SwEngine256;

use crate::alignment::workspace::with_workspace;
use crate::compute::simd_abstraction::SimdEngine256 as Engine;
use crate::generate_swa_entry;

// -----------------------------------------------------------------------------
// Macro-generated wrapper calling the shared kernel (for parity/comparison)
// -----------------------------------------------------------------------------
// Note: single declaration only; used for side-by-side parity testing
generate_swa_entry!(
    name = simd_banded_swa_batch32,
    width = 32,
    engine = SwEngine256,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);

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

use crate::generate_swa_entry_soa;

generate_swa_entry_soa!(
    name = simd_banded_swa_batch32_soa,
    width = 32,
    engine = SwEngine256,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);
