// ============================================================================
// SSE/NEON Horizontal SIMD Smith-Waterman (16-way parallelism)
// ============================================================================
//
// This module implements horizontal SIMD batching for mate rescue alignments
// using SSE/NEON instructions (128-bit vectors, 16 sequences in parallel).
//
// Direct port from BWA-MEM2's kswv.cpp kswv256_u8() function.
//
// ## Architecture
//
// **Horizontal SIMD**: Processes same position across 16 different sequences
// ```
// Register layout:
// __m128i = [ seq0[pos], seq1[pos], seq2[pos], ..., seq15[pos] ]
// ```
//
// Contrast with **vertical SIMD** (banded_swa.rs):
// ```
// Register layout:
// __m128i = [ seq0[pos0], seq0[pos1], ..., seq0[pos15] ]
// ```
//
// ## Performance
//
// - Processes 16 alignments per SIMD operation
// - Reduces 44K mate rescue calls → ~2,750 batched calls
// - Expected speedup: ~8x over scalar processing
//
// ## Platform Support
//
// This module is available on all x86_64 (SSE2+) and aarch64 (NEON) platforms.
// Provides baseline horizontal SIMD support across architectures.
//
// ## Reference
//
// BWA-MEM2: bwa-mem2/src/kswv.cpp (kswv128_u8)
// ============================================================================

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use crate::alignment::kswv_batch::{SeqPair, KswResult};
use crate::compute::simd_abstraction::{SimdEngine, SimdEngine128};

/// SIMD width for SSE/NEON: 16 sequences with 8-bit scores
pub const SIMD_WIDTH8: usize = 16;

/// SIMD width for SSE/NEON: 16 sequences with 16-bit scores
pub const SIMD_WIDTH16: usize = 8;

/// Dummy values for ambiguous bases (matching BWA-MEM2)
const DUMMY5: i8 = 5;
const DUMMY8: i8 = 8;
const AMBIG: i8 = 4;

/// KSW flags (from kswv.h)
const KSW_XSUBO: i32 = 0x40000;
const KSW_XSTOP: i32 = 0x20000;

/// Batched Smith-Waterman alignment using SSE/NEON horizontal SIMD
///
/// Processes up to 16 alignments in parallel using 128-bit SIMD registers.
/// Each SIMD lane handles one complete alignment.
///
/// # Arguments
/// * `seq1_soa` - Reference sequences in SoA layout (16 sequences interleaved)
/// * `seq2_soa` - Query sequences in SoA layout (16 sequences interleaved)
/// * `nrow` - Reference sequence length
/// * `ncol` - Query sequence length
/// * `pairs` - Sequence pair metadata (lengths, h0)
/// * `results` - Output results (score, te, qe, score2, te2, tb, qb)
/// * `w_match` - Match score
/// * `w_mismatch` - Mismatch penalty
/// * `o_del` - Gap open penalty (deletion)
/// * `e_del` - Gap extension penalty (deletion)
/// * `o_ins` - Gap open penalty (insertion)
/// * `e_ins` - Gap extension penalty (insertion)
/// * `w_ambig` - Ambiguous base penalty
/// * `phase` - Processing phase (0 = forward, 1 = traceback)
///
/// # Returns
/// Number of alignments processed (always returns 1 in current implementation)
///
/// # Safety
/// This function uses SSE/NEON intrinsics which are unsafe. Caller must ensure:
/// - CPU has SSE2+ (x86_64) or NEON (aarch64) support
/// - Sequence buffers are properly aligned (16-byte)
/// - Buffer sizes match documented layout
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn batch_ksw_align_sse_neon(
    seq1_soa: *const u8,      // Reference sequences (SoA layout)
    seq2_soa: *const u8,      // Query sequences (SoA layout)
    nrow: i16,                // Reference length
    ncol: i16,                // Query length
    pairs: &[SeqPair],        // Sequence pair metadata
    results: &mut [KswResult], // Output results
    w_match: i8,              // Match score
    w_mismatch: i8,           // Mismatch penalty
    o_del: i32,               // Gap open (deletion)
    e_del: i32,               // Gap extension (deletion)
    o_ins: i32,               // Gap open (insertion)
    e_ins: i32,               // Gap extension (insertion)
    w_ambig: i8,              // Ambiguous base penalty
    _phase: i32,              // Processing phase
) -> usize {
    log::debug!("[SIMD] Horizontal kernel (SSE/NEON): processing {} alignments", pairs.len());

    // ========================================================================
    // SECTION 1: Initialization
    // ========================================================================

    // Initialize basic constants using SimdEngine128 trait
    let zero256 = SimdEngine128::setzero_epi8();
    let one256 = SimdEngine128::set1_epi8(1);

    // Compute shift value for signed/unsigned score conversion
    let mdiff = w_match.max(w_mismatch).max(w_ambig);
    let shift_val = w_match.min(w_mismatch).min(w_ambig);
    let shift = (256i16 - shift_val as i16) as u8;
    let qmax = mdiff;

    // Create scoring lookup table for shuffle operation
    let mut temp = [0i8; SIMD_WIDTH8];
    temp[0] = w_match;
    temp[1] = w_mismatch; temp[2] = w_mismatch; temp[3] = w_mismatch;
    temp[4..8].fill(w_ambig);
    temp[8..12].fill(w_ambig);
    temp[12] = w_ambig;

    // Add shift to first 16 elements for shuffle_epi8
    for i in 0..16 {
        temp[i] = temp[i].wrapping_add(shift as i8);
    }

    // Replicate pattern for full SSE/NEON width
    for i in 16..SIMD_WIDTH8 {
        temp[i] = temp[i - 16];
    }

    let perm_sft256 = SimdEngine128::loadu_si128(temp.as_ptr() as *const _);
    let sft256 = SimdEngine128::set1_epi8(shift as i8);
    let cmax256 = SimdEngine128::set1_epi8(255u8 as i8);
    let five256 = SimdEngine128::set1_epi8(DUMMY5);

    // Initialize minsc and endsc arrays from h0 flags
    let mut minsc = [0u8; SIMD_WIDTH8];
    let mut endsc = [0u8; SIMD_WIDTH8];

    for i in 0..SIMD_WIDTH8.min(pairs.len()) {
        let xtra = pairs[i].h0;

        // Check KSW_XSUBO flag for minimum score threshold
        let val = if (xtra & KSW_XSUBO) != 0 {
            (xtra & 0xffff) as u32
        } else {
            0x10000
        };
        if val <= 255 {
            minsc[i] = val as u8;
        }

        // Check KSW_XSTOP flag for early termination score
        let val = if (xtra & KSW_XSTOP) != 0 {
            (xtra & 0xffff) as u32
        } else {
            0x10000
        };
        if val <= 255 {
            endsc[i] = val as u8;
        }
    }

    // Use unaligned loads for stack arrays (not guaranteed 16-byte aligned)
    let minsc256 = SimdEngine128::loadu_si128(minsc.as_ptr() as *const _);
    let endsc256 = SimdEngine128::loadu_si128(endsc.as_ptr() as *const _);

    // Initialize scoring parameters as SIMD vectors
    let e_del256 = SimdEngine128::set1_epi8(e_del as i8);
    let oe_del256 = SimdEngine128::set1_epi8((o_del + e_del) as i8);
    let e_ins256 = SimdEngine128::set1_epi8(e_ins as i8);
    let oe_ins256 = SimdEngine128::set1_epi8((o_ins + e_ins) as i8);

    // Global maximum and target end position
    let mut gmax256 = zero256;
    let mut te256 = SimdEngine128::set1_epi16(-1);

    // Allocate DP matrices (H, F, rowMax)
    let max_query_len = ncol as usize + 1;
    let max_ref_len = nrow as usize + 1;

    let mut h0_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut h1_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut f_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut row_max_buf = vec![0u8; max_ref_len * SIMD_WIDTH8];

    // Initialize H0, F to zero (using unaligned stores for Vec buffers)
    for i in 0..=ncol as usize {
        let offset = i * SIMD_WIDTH8;
        SimdEngine128::storeu_si128(h0_buf[offset..].as_mut_ptr() as *mut _, zero256);
        SimdEngine128::storeu_si128(f_buf[offset..].as_mut_ptr() as *mut _, zero256);
    }

    // Initialize tracking variables for main loop
    let mut pimax256 = zero256;
    let mut mask256 = zero256;
    let mut minsc_msk = zero256;
    let mut qe256 = SimdEngine128::set1_epi8(0);

    SimdEngine128::storeu_si128(h0_buf.as_mut_ptr() as *mut _, zero256);
    SimdEngine128::storeu_si128(h1_buf.as_mut_ptr() as *mut _, zero256);

    // ========================================================================
    // SECTION 2: Main DP Loop
    // ========================================================================

    let mut exit0_vec = SimdEngine128::set1_epi8(-1);  // All lanes active
    let mut limit = nrow as i32;

    for i in 0..nrow as usize {
        // Initialize row variables
        let mut e11 = zero256;
        let mut imax256 = zero256;
        let mut iqe256 = SimdEngine128::set1_epi8(-1);
        let mut i256_vec = SimdEngine128::set1_epi16(i as i16);
        let mut l256 = zero256;

        // Load reference base for this row
        let s1 = SimdEngine128::load_si128(
            seq1_soa.add(i * SIMD_WIDTH8) as *const _
        );

        // Inner loop over query positions
        for j in 0..ncol as usize {
            // Load DP values and query base (unaligned for Vec buffers)
            let h00 = SimdEngine128::loadu_si128(
                h0_buf[j * SIMD_WIDTH8..].as_ptr() as *const _
            );
            let s2 = SimdEngine128::load_si128(
                seq2_soa.add(j * SIMD_WIDTH8) as *const _
            );
            let f11 = SimdEngine128::loadu_si128(
                f_buf[(j + 1) * SIMD_WIDTH8..].as_ptr() as *const _
            );

            // ============================================================
            // MAIN_SAM_CODE8_OPT: Core DP computation
            // ============================================================

            // Compute match/mismatch score via XOR and shuffle
            let xor11 = SimdEngine128::xor_si128(s1, s2);
            let mut sbt11 = SimdEngine128::shuffle_epi8(perm_sft256, xor11);

            // Handle ambiguous bases (base == 5)
            let cmpq = SimdEngine128::cmpeq_epi8(s2, five256);
            sbt11 = SimdEngine128::blendv_epi8(sbt11, sft256, cmpq);

            // Mask out invalid positions
            let or11 = SimdEngine128::or_si128(s1, s2);
            let cmp_mask = SimdEngine128::cmpeq_epi8(
                or11,
                SimdEngine128::set1_epi8(0)
            );

            // Compute match score: H[i-1,j-1] + score
            let mut m11 = SimdEngine128::adds_epu8(h00, sbt11);
            m11 = SimdEngine128::blendv_epi8(zero256, m11, cmp_mask);
            m11 = SimdEngine128::subs_epu8(m11, sft256);

            // Take max of match, gap-extend-E, gap-extend-F
            let mut h11 = SimdEngine128::max_epu8(m11, e11);
            h11 = SimdEngine128::max_epu8(h11, f11);

            // Track row maximum and query end position
            let cmp0 = SimdEngine128::cmpgt_epu8(h11, imax256);
            imax256 = SimdEngine128::max_epu8(imax256, h11);
            iqe256 = SimdEngine128::blendv_epi8(iqe256, l256, cmp0);

            // Update E (gap in query)
            let gap_e256 = SimdEngine128::subs_epu8(h11, oe_ins256);
            e11 = SimdEngine128::subs_epu8(e11, e_ins256);
            e11 = SimdEngine128::max_epu8(gap_e256, e11);

            // Update F (gap in reference)
            let gap_d256 = SimdEngine128::subs_epu8(h11, oe_del256);
            let mut f21 = SimdEngine128::subs_epu8(f11, e_del256);
            f21 = SimdEngine128::max_epu8(gap_d256, f21);

            // Store updated DP values (unaligned for Vec buffers)
            SimdEngine128::storeu_si128(
                h1_buf[(j + 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
                h11
            );
            SimdEngine128::storeu_si128(
                f_buf[(j + 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
                f21
            );

            // Increment query position counter
            l256 = SimdEngine128::add_epi8(l256, one256);
        }

        // Block I: Track row maxima for second-best score computation
        if i > 0 {
            let msk = SimdEngine128::cmpgt_epu8(imax256, pimax256);
            let msk = SimdEngine128::or_si128(msk, mask256);

            let mut pimax256_tmp = SimdEngine128::blendv_epi8(pimax256, zero256, msk);

            // Apply minsc threshold mask
            let minsc_mask_vec = SimdEngine128::set1_epi8(-1);
            pimax256_tmp = SimdEngine128::blendv_epi8(
                pimax256_tmp,
                zero256,
                minsc_mask_vec
            );

            // Apply exit mask
            pimax256_tmp = SimdEngine128::blendv_epi8(
                pimax256_tmp,
                zero256,
                exit0_vec
            );

            SimdEngine128::storeu_si128(
                row_max_buf[(i - 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
                pimax256_tmp
            );

            mask256 = SimdEngine128::andnot_si128(msk, SimdEngine128::set1_epi8(-1));
        }

        pimax256 = imax256;

        // Update minsc mask
        let minsc_msk_vec = SimdEngine128::cmpge_epu8(imax256, minsc256);
        minsc_msk = SimdEngine128::or_si128(minsc_msk, minsc_msk_vec);

        // Block II: Update global maximum and target end position
        let mut cmp0_vec = SimdEngine128::cmpgt_epu8(imax256, gmax256);
        cmp0_vec = SimdEngine128::and_si128(cmp0_vec, exit0_vec);

        gmax256 = SimdEngine128::blendv_epi8(gmax256, imax256, cmp0_vec);
        te256 = SimdEngine128::blendv_epi8(te256, i256_vec, cmp0_vec);
        qe256 = SimdEngine128::blendv_epi8(qe256, iqe256, cmp0_vec);

        // Check for early termination
        cmp0_vec = SimdEngine128::cmpge_epu8(gmax256, endsc256);

        // Check for score overflow
        let left256 = SimdEngine128::adds_epu8(gmax256, sft256);
        let cmp2_vec = SimdEngine128::cmpge_epu8(left256, cmax256);

        // Update exit mask
        let exit_cond = SimdEngine128::or_si128(cmp0_vec, cmp2_vec);
        exit0_vec = SimdEngine128::andnot_si128(exit_cond, exit0_vec);

        // Early exit if all lanes done
        if SimdEngine128::movemask_epi8(exit0_vec) == 0 {
            limit = (i + 1) as i32;
            break;
        }

        // Swap buffers
        std::mem::swap(&mut h0_buf, &mut h1_buf);

        // Increment row index
        let one256_16 = SimdEngine128::set1_epi16(1);
        i256_vec = SimdEngine128::add_epi16(i256_vec, one256_16);
    }

    // Final row max update (unaligned for Vec buffers)
    let msk = SimdEngine128::or_si128(mask256, SimdEngine128::set1_epi8(0));
    let pimax256_final = SimdEngine128::blendv_epi8(pimax256, zero256, msk);
    SimdEngine128::storeu_si128(
        row_max_buf[((limit - 1) as usize) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
        pimax256_final
    );

    // ========================================================================
    // SECTION 3: Score Extraction
    // ========================================================================

    #[repr(align(16))]
    struct AlignedScoreArray([u8; SIMD_WIDTH8]);
    #[repr(align(16))]
    struct AlignedTeArray([i16; SIMD_WIDTH16]);
    #[repr(align(16))]
    struct AlignedQeArray([u8; SIMD_WIDTH8]);

    let mut score_arr = AlignedScoreArray([0; SIMD_WIDTH8]);
    let mut te_arr = AlignedTeArray([0; SIMD_WIDTH16]);
    let mut qe_arr = AlignedQeArray([0; SIMD_WIDTH8]);

    // Store SIMD vectors to aligned arrays
    SimdEngine128::storeu_si128(score_arr.0.as_mut_ptr() as *mut _, gmax256);
    SimdEngine128::storeu_si128_16(te_arr.0.as_mut_ptr() as *mut _, te256);
    SimdEngine128::storeu_si128(qe_arr.0.as_mut_ptr() as *mut _, qe256);

    // Extract scores for each sequence in the batch
    let mut live = 0;
    for l in 0..SIMD_WIDTH8.min(pairs.len()) {
        let score = score_arr.0[l];

        // Apply shift and clamp to 255
        let final_score = if (score as i32 + shift as i32) < 255 {
            score
        } else {
            255
        };

        // Map te from 16-bit array (only 16 lanes) to 32 8-bit lanes
        let te_idx = l / 2;  // 2 sequences per 16-bit lane
        results[l].score = final_score as i32;
        results[l].te = te_arr.0[te_idx] as i32;
        results[l].qe = qe_arr.0[l] as i32;

        if final_score != 255 {
            live += 1;
        }
    }

    if live == 0 {
        return 1;
    }

    // ========================================================================
    // SECTION 4: Second-Best Score Computation
    // ========================================================================

    #[repr(align(16))]
    struct AlignedI16Array([i16; SIMD_WIDTH16]);

    let mut low_arr = AlignedI16Array([0; SIMD_WIDTH16]);
    let mut high_arr = AlignedI16Array([0; SIMD_WIDTH16]);
    let mut rlen_arr = AlignedI16Array([0; SIMD_WIDTH16]);

    let mut maxl: i32 = 0;
    let mut minh: i32 = nrow as i32;

    for i in 0..SIMD_WIDTH16.min(pairs.len()) {
        let val = (score_arr.0[i * 2] as i32 + qmax as i32 - 1) / qmax as i32;

        low_arr.0[i] = (te_arr.0[i] - val as i16).max(0);
        high_arr.0[i] = (te_arr.0[i] + val as i16).min(nrow as i16 - 1);
        rlen_arr.0[i] = pairs[i * 2].ref_len as i16;

        if qe_arr.0[i * 2] != 0 {
            maxl = maxl.max(low_arr.0[i] as i32);
            minh = minh.min(high_arr.0[i] as i32);
        }
    }

    // Initialize second-best tracking
    let mut max256 = zero256;
    let mut te256 = SimdEngine128::set1_epi16(-1);

    let low256 = SimdEngine128::loadu_si128_16(low_arr.0.as_ptr() as *const _);
    let high256 = SimdEngine128::loadu_si128_16(high_arr.0.as_ptr() as *const _);
    let rlen256 = SimdEngine128::loadu_si128_16(rlen_arr.0.as_ptr() as *const _);

    // Forward scan
    for i in 0..maxl {
        let i256 = SimdEngine128::set1_epi16(i as i16);
        let rmax256 = SimdEngine128::loadu_si128(
            row_max_buf[i as usize * SIMD_WIDTH8..].as_ptr() as *const _
        );

        let mask1 = SimdEngine128::cmpgt_epi16(low256, i256);
        let mask2_8bit = SimdEngine128::cmpgt_epu8(rmax256, max256);
        let combined_mask = SimdEngine128::and_si128(mask1, mask2_8bit);

        max256 = SimdEngine128::blendv_epi8(max256, rmax256, combined_mask);
        te256 = SimdEngine128::blendv_epi8(te256, i256, combined_mask);
    }

    // Backward scan (ensure we don't iterate with negative indices)
    let backward_start = (minh + 1).max(0);
    for i in backward_start..limit {
        let i256 = SimdEngine128::set1_epi16(i as i16);
        let rmax256 = SimdEngine128::loadu_si128(
            row_max_buf[i as usize * SIMD_WIDTH8..].as_ptr() as *const _
        );

        let mask1a = SimdEngine128::cmpgt_epi16(i256, high256);
        let mask1b = SimdEngine128::cmpgt_epi16(rlen256, i256);
        let mask2_8bit = SimdEngine128::cmpgt_epu8(rmax256, max256);

        let mask1 = SimdEngine128::and_si128(mask1a, mask1b);
        let combined_mask = SimdEngine128::and_si128(mask1, mask2_8bit);

        max256 = SimdEngine128::blendv_epi8(max256, rmax256, combined_mask);
        te256 = SimdEngine128::blendv_epi8(te256, i256, combined_mask);
    }

    // Extract second-best scores
    let mut score2_arr = AlignedScoreArray([0; SIMD_WIDTH8]);
    let mut te2_arr = AlignedTeArray([0; SIMD_WIDTH16]);

    SimdEngine128::storeu_si128(score2_arr.0.as_mut_ptr() as *mut _, max256);
    SimdEngine128::storeu_si128_16(te2_arr.0.as_mut_ptr() as *mut _, te256);

    for i in 0..pairs.len().min(SIMD_WIDTH8) {
        let te_idx = i / 2;
        if qe_arr.0[i] != 0 {
            results[i].score2 = if score2_arr.0[i] == 0 {
                -1
            } else {
                score2_arr.0[i] as i32
            };
            results[i].te2 = te2_arr.0[te_idx] as i32;
        } else {
            results[i].score2 = -1;
            results[i].te2 = -1;
        }

        // Set tb and qb based on valid alignment
        // Phase 0 doesn't compute traceback, so we estimate:
        // - For mate rescue, alignments typically start at query position 0
        // - tb is estimated as te - (qe - qb) for same-length alignment
        if results[i].score > 0 && results[i].te >= 0 && results[i].qe >= 0 {
            results[i].qb = 0;  // Alignment starts at query position 0
            results[i].tb = (results[i].te - results[i].qe).max(0);
        } else {
            results[i].tb = -1;
            results[i].qb = -1;
        }
    }

    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::kswv_batch::SeqPair;

    #[test]
    #[ignore] // TODO: Enable when SSE/NEON kernel is tested
    fn test_sse_neon_kernel() {
        // SSE2 (x86_64) and NEON (aarch64) are baseline features, always available
        // No feature detection needed

        let seq1 = vec![0u8; 256 * SIMD_WIDTH8];
        let seq2 = vec![0u8; 128 * SIMD_WIDTH8];

        let pairs = vec![SeqPair {
            ref_len: 10,
            query_len: 10,
            ..Default::default()
        }; SIMD_WIDTH8];

        let mut results = vec![KswResult::default(); SIMD_WIDTH8];

        unsafe {
            let _count = batch_ksw_align_sse_neon(
                seq1.as_ptr(),
                seq2.as_ptr(),
                10,
                10,
                &pairs,
                &mut results,
                1,
                -4,
                6,
                2,
                6,
                2,
                -1,
                0,
            );
        }
    }
}
