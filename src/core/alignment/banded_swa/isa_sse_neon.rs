// bwa-mem2-rust/src/banded_swa_sse_neon.rs
//
// SSE/NEON (128-bit) SIMD implementation of banded Smith-Waterman alignment
// Processes 16 alignments in parallel (baseline SIMD for all platforms)
//
// This is a port of the AVX2 version adapted for 128-bit SIMD width
// Works on both x86_64 (SSE2+) and aarch64 (NEON)

use super::types::OutScore; // Updated path
use super::kernel::{KernelParams, SwEngine128, sw_kernel}; // Updated path
use super::shared::{pad_batch, soa_transform}; // Updated path

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m128i;

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
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    mat: &[i8; 25],
    m: i32,
) -> Vec<OutScore> {
    const W: usize = 16;
    const MAX_SEQ_LEN: usize = 128; // keep i8 limits aligned with AVX2

    let (qlen, tlen, h0, w_arr, max_qlen, max_tlen, padded) = pad_batch::<W>(batch);
    let (query_soa, target_soa) = soa_transform::<W, MAX_SEQ_LEN>(&padded);

    let params = KernelParams {
        batch,
        query_soa: &query_soa,
        target_soa: &target_soa,
        qlen: &qlen,
        tlen: &tlen,
        h0: &h0,
        w: &w_arr,
        max_qlen,
        max_tlen,
        o_del,
        e_del,
        o_ins,
        e_ins,
        zdrop,
        mat,
        m,
    };

    sw_kernel::<W, SwEngine128>(&params)
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
        // Perfect match should score higher
        assert!(
            results[0].score >= results[1].score,
            "Perfect match should score >= mismatch: {} vs {}",
            results[0].score,
            results[1].score
        );
    }
}

use crate::generate_swa_entry_soa; // This macro is exported at crate root by shared module

#[cfg(target_arch = "x86_64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "sse2",
);


/// 16-bit SIMD batch Smith-Waterman scoring (score-only, no CIGAR)
///
/// **Matches BWA-MEM2's getScores16() function**
///
/// Uses i16 arithmetic to handle sequences with scores > 127.
/// Processes 8 alignments in parallel per 128-bit vector.
///
/// **When to use:**
/// - Sequences where max possible score > 127
/// - Formula: seq_len * match_score >= 127
/// - For typical 151bp reads with match=1, max score = 151 > 127, so use 16-bit
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch8_int16(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)], // (qlen, query, tlen, target, w, h0)
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    mat: &[i8; 25],
    m: i32,
) -> Vec<OutScore> {
    // Removed portable_intrinsics import
    // This needs to use the appropriate Engine for 128-bit, which is SimdEngine128 from simd_abstraction
    use crate::compute::simd_abstraction::SimdEngine128 as Engine;
    use crate::compute::simd_abstraction::SimdEngine; // bring trait into scope for method resolution

    const SIMD_WIDTH: usize = 8; // Process 8 alignments in parallel (128-bit / 16-bit)
    const MAX_SEQ_LEN: usize = 512; // 16-bit supports longer sequences

    let batch_size = batch.len().min(SIMD_WIDTH);

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

    // Extract batch parameters (using i16 for 16-bit precision)
    let mut qlen = [0i16; SIMD_WIDTH];
    let mut tlen = [0i16; SIMD_WIDTH];
    let mut h0 = [0i16; SIMD_WIDTH];
    let mut w = [0i16; SIMD_WIDTH];
    let mut max_qlen = 0i32;
    let mut max_tlen = 0i32;

    for i in 0..SIMD_WIDTH {
        let (q, _, t, _, wi, h) = padded_batch[i];
        qlen[i] = q.min(MAX_SEQ_LEN as i32) as i16;
        tlen[i] = t.min(MAX_SEQ_LEN as i32) as i16;
        h0[i] = h as i16;
        w[i] = wi as i16;
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

    // Allocate Structure-of-Arrays (SoA) buffers for SIMD-friendly access
    let mut query_soa = vec![0u8; MAX_SEQ_LEN * SIMD_WIDTH];
    let mut target_soa = vec![0u8; MAX_SEQ_LEN * SIMD_WIDTH];

    // 16-bit SoA for vectorized scoring (SSE compare-and-blend approach from BWA-MEM2)
    // Using 0x7FFF as padding ensures ambiguous base detection works correctly
    const PADDING_VALUE: i16 = 0x7FFF;
    let mut query_soa_16 = vec![PADDING_VALUE; MAX_SEQ_LEN * SIMD_WIDTH];
    let mut target_soa_16 = vec![PADDING_VALUE; MAX_SEQ_LEN * SIMD_WIDTH];

    // Transform query and target sequences to SoA layout
    for i in 0..SIMD_WIDTH {
        let (q_len, query, t_len, target, _, _) = padded_batch[i];

        let actual_q_len = query.len().min(MAX_SEQ_LEN);
        let actual_t_len = target.len().min(MAX_SEQ_LEN);
        let safe_q_len = (q_len as usize).min(actual_q_len);
        let safe_t_len = (t_len as usize).min(actual_t_len);

        // Copy query (interleaved) - both 8-bit and 16-bit layouts
        for j in 0..safe_q_len {
            query_soa[j * SIMD_WIDTH + i] = query[j];
            query_soa_16[j * SIMD_WIDTH + i] = query[j] as i16;
        }
        for j in (q_len as usize)..MAX_SEQ_LEN {
            query_soa[j * SIMD_WIDTH + i] = 0xFF;
            // query_soa_16 already initialized to PADDING_VALUE
        }

        // Copy target (interleaved) - both 8-bit and 16-bit layouts
        for j in 0..safe_t_len {
            target_soa[j * SIMD_WIDTH + i] = target[j];
            target_soa_16[j * SIMD_WIDTH + i] = target[j] as i16;
        }
        for j in (t_len as usize)..MAX_SEQ_LEN {
            target_soa[j * SIMD_WIDTH + i] = 0xFF;
            // target_soa_16 already initialized to PADDING_VALUE
        }
    }

    // Allocate DP matrices in SoA layout (using i16)
    let mut h_matrix = vec![0i16; MAX_SEQ_LEN * SIMD_WIDTH];
    let mut e_matrix = vec![0i16; MAX_SEQ_LEN * SIMD_WIDTH];
    let _f_matrix = vec![0i16; MAX_SEQ_LEN * SIMD_WIDTH];

    // Initialize scores and tracking arrays (using i16)
    // Note: max_i/max_j initialized to -1 to match scalar ksw_extend2 behavior
    // When no score exceeds h0, scalar returns qle=max_j+1=0, tle=max_i+1=0
    let mut max_scores = vec![0i16; SIMD_WIDTH];
    let mut max_i = [-1i16; SIMD_WIDTH];
    let mut max_j = [-1i16; SIMD_WIDTH];
    let gscores = [0i16; SIMD_WIDTH];
    let max_ie = [0i16; SIMD_WIDTH];

    // SIMD constants (16-bit)
    let zero_vec = Engine::setzero_epi16();
    let oe_del = (o_del + e_del) as i16;
    let oe_ins = (o_ins + e_ins) as i16;
    let oe_del_vec = Engine::set1_epi16(oe_del);
    let oe_ins_vec = Engine::set1_epi16(oe_ins);
    let e_del_vec = Engine::set1_epi16(e_del as i16);
    let e_ins_vec = Engine::set1_epi16(e_ins as i16);

    // Vectorized scoring constants (BWA-MEM2 compare-and-blend approach)
    let match_score = mat[0] as i16; // A-A match score
    let mismatch_score = mat[1] as i16; // A-C mismatch score
    let ambig_score = mat[(4 * m + 4) as usize] as i16; // N-N score (ambiguous bases)
    let match_score_vec = Engine::set1_epi16(match_score);
    let mismatch_score_vec = Engine::set1_epi16(mismatch_score);
    let ambig_score_vec = Engine::set1_epi16(ambig_score);
    let three_vec = Engine::set1_epi16(3); // For detecting ambiguous bases (> 3)

    // Band tracking (16-bit)
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

    // Main DP loop using SIMD (16-bit operations)
    unsafe {
        #[cfg(target_arch = "x86_64")]
        log::trace!("[SIMD] Executing SSE2 intrinsics in DP loop (batch size: {SIMD_WIDTH})");
        #[cfg(target_arch = "aarch64")]
        log::trace!(
            "[SIMD] Executing NEON intrinsics in DP loop (batch size: {})",
            SIMD_WIDTH
        );

        let mut max_score_vec = Engine::loadu_si128(max_scores.as_ptr() as *const __m128i);

        for i in 0..max_tlen as usize {
            // Update band bounds per lane
            let mut current_beg = beg;
            let mut current_end = end;
            for lane in 0..SIMD_WIDTH {
                if terminated[lane] {
                    continue;
                }
                let wi = w[lane];
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

            // Load target base for this row as 16-bit SIMD vector (vectorized scoring)
                        let t_vec = Engine::loadu_si128(target_soa_16.as_ptr().add(i * SIMD_WIDTH) as *const __m128i);

            let mut h1_vec = zero_vec; // H(i, j-1) for first column

            // Initial H value for column 0
            for lane in 0..SIMD_WIDTH {
                if terminated[lane] {
                    continue;
                }
                if current_beg[lane] == 0 {
                    let h_val = h0[lane] as i32 - (o_del + e_del * (i as i32 + 1));
                    let h_val = if h_val < 0 { 0 } else { h_val as i16 };
                    let h1_arr: &mut [i16; 8] = std::mem::transmute(&mut h1_vec);
                    h1_arr[lane] = h_val;
                }
            }

            let mut f_vec = zero_vec;

            for j in global_beg..global_end.min(MAX_SEQ_LEN) {
                // Load H(i-1, j-1) and E(i, j)
                                let h00_vec = Engine::loadu_si128(h_matrix.as_ptr().add(j * SIMD_WIDTH) as *const __m128i);
                                let e_vec = Engine::loadu_si128(e_matrix.as_ptr().add(j * SIMD_WIDTH) as *const __m128i);

                // ==================================================================
                // VECTORIZED SCORING (BWA-MEM2 compare-and-blend approach)
                // ==================================================================
                // This replaces the scalar matrix lookup loop with 8 SIMD instructions.
                // Reference: BWA-MEM2 MAIN_CODE16 macro in bandedSWA.cpp:327-346

                // Load query bases for column j as 16-bit SIMD vector
                                let q_vec = Engine::loadu_si128(query_soa_16.as_ptr().add(j * SIMD_WIDTH) as *const __m128i);

                // Step 1: Compare bases for equality (0xFFFF where equal, 0x0000 otherwise)
                let cmp_eq = Engine::cmpeq_epi16(q_vec, t_vec);

                // Step 2: Select match or mismatch score based on comparison
                // blendv_epi8 uses byte MSB: cmp_eq=0xFFFF -> both bytes MSB=1 -> select match
                let score_vec = Engine::blendv_epi8(mismatch_score_vec, match_score_vec, cmp_eq);

                // Step 3: Handle ambiguous bases (base value > 3 indicates N or padding)
                // This matches the C++ pattern: max(q, t) > 3 means ambiguous
                let q_gt3 = Engine::cmpgt_epi16(q_vec, three_vec);
                let t_gt3 = Engine::cmpgt_epi16(t_vec, three_vec);
                let ambig_mask = Engine::or_si128(q_gt3, t_gt3);

                // Apply ambiguous score where either base is ambiguous
                let match_vec = Engine::blendv_epi8(score_vec, ambig_score_vec, ambig_mask);

                // M = H(i-1, j-1) + match/mismatch score
                let m_vec = Engine::add_epi16(h00_vec, match_vec);

                // H(i,j) = max(M, E, F, 0)
                let h11_vec = Engine::max_epi16(m_vec, e_vec);
                let h11_vec = Engine::max_epi16(h11_vec, f_vec);
                let h11_vec = Engine::max_epi16(h11_vec, zero_vec);

                // Store H(i, j-1) for next iteration
                Engine::storeu_si128(
                    h_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut __m128i,
                    h1_vec,
                );

                // Compute E(i+1, j) = max(M - oe_del, E - e_del)
                let e_from_m = Engine::subs_epi16(m_vec, oe_del_vec);
                let e_from_e = Engine::subs_epi16(e_vec, e_del_vec);
                let new_e_vec = Engine::max_epi16(e_from_m, e_from_e);
                let new_e_vec = Engine::max_epi16(new_e_vec, zero_vec);
                Engine::storeu_si128(
                    e_matrix.as_mut_ptr().add(j * SIMD_WIDTH) as *mut __m128i,
                    new_e_vec,
                );

                // Compute F(i, j+1) = max(M - oe_ins, F - e_ins)
                let f_from_m = Engine::subs_epi16(m_vec, oe_ins_vec);
                let f_from_f = Engine::subs_epi16(f_vec, e_ins_vec);
                f_vec = Engine::max_epi16(f_from_m, f_from_f);
                f_vec = Engine::max_epi16(f_vec, zero_vec);

                // Update max score and track position (per-lane)
                // Extract h11 values for comparison
                let mut h11_arr = [0i16; SIMD_WIDTH];
                Engine::storeu_si128(h11_arr.as_mut_ptr() as *mut __m128i, h11_vec);

                for lane in 0..SIMD_WIDTH {
                    if !terminated[lane]
                        && j >= current_beg[lane] as usize
                        && j < current_end[lane] as usize
                        && h11_arr[lane] > max_scores[lane]
                    {
                        max_scores[lane] = h11_arr[lane];
                        max_i[lane] = i as i16;
                        max_j[lane] = j as i16;
                    }
                }

                // Update vector from per-lane tracking
                max_score_vec = Engine::loadu_si128(max_scores.as_ptr() as *const __m128i);

                h1_vec = h11_vec;
            }

            // Z-drop check (per-lane)
            Engine::storeu_si128(max_scores.as_mut_ptr() as *mut __m128i, max_score_vec);
            for lane in 0..SIMD_WIDTH {
                if terminated[lane] {
                    continue;
                }

                let current_max = max_scores[lane] as i32;
                let row_max = max_scores[lane];
                if zdrop > 0 {
                    let score_drop = current_max - row_max as i32;
                    if score_drop > zdrop {
                        terminated[lane] = true;
                        continue;
                    }
                }

                // Adaptive band narrowing
                let mut new_beg = current_beg[lane];
                while new_beg < current_end[lane] {
                    let h_val = h_matrix[new_beg as usize * SIMD_WIDTH + lane];
                    let e_val = e_matrix[new_beg as usize * SIMD_WIDTH + lane];
                    if h_val != 0 || e_val != 0 {
                        break;
                    }
                    new_beg += 1;
                }
                beg[lane] = new_beg;

                let mut new_end = current_end[lane];
                while new_end > beg[lane] {
                    let idx = (new_end - 1) as usize;
                    if idx >= MAX_SEQ_LEN {
                        break;
                    }
                    let h_val = h_matrix[idx * SIMD_WIDTH + lane];
                    let e_val = e_matrix[idx * SIMD_WIDTH + lane];
                    if h_val != 0 || e_val != 0 {
                        break;
                    }
                    new_end -= 1;
                }
                end[lane] = (new_end + 2).min(qlen[lane]);
            }
        }

        // Extract final max scores
        Engine::storeu_si128(max_scores.as_mut_ptr() as *mut __m128i, max_score_vec);
    }

    // Extract results and convert to OutScore format
    // Note: scalar_banded_swa returns max_i+1 and max_j+1 (1-indexed extension lengths)
    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        results.push(OutScore {
            score: max_scores[i] as i32,
            target_end_pos: max_i[i] as i32 + 1, // +1 to match scalar output (1-indexed)
            query_end_pos: max_j[i] as i32 + 1,  // +1 to match scalar output (1-indexed)
            gtarget_end_pos: max_ie[i] as i32,
            global_score: gscores[i] as i32,
            max_offset: 0,
        });
    }

    results
}
#[cfg(target_arch = "aarch64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "aarch64"),
    target_feature = "neon",
);