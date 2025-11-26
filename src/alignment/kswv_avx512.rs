// ============================================================================
// AVX-512 Horizontal SIMD Smith-Waterman (64-way parallelism)
// ============================================================================
//
// FAITHFUL PORT of BWA-MEM2's kswv.cpp kswv512_u8() function (lines 371-709).
//
// This implementation uses RAW AVX-512 intrinsics to match C++ semantics exactly.
// Key insight: AVX-512 uses k-masks (__mmask64) for comparisons and blends,
// NOT vector masks like SSE/AVX2.
//
// ## Feature Gate
//
// This module requires `--features avx512` and x86_64 with AVX-512BW.
// ============================================================================

#![cfg(all(target_arch = "x86_64", feature = "avx512"))]

use crate::alignment::kswv_batch::{KswResult, SeqPair};

// Raw AVX-512 intrinsics - faithful to C++
use std::arch::x86_64::{
    __m512i, __mmask32, __mmask64,
    // Vector creation
    _mm512_setzero_si512, _mm512_set1_epi8, _mm512_set1_epi16,
    // Vector loads/stores (unaligned for safety)
    _mm512_loadu_si512, _mm512_storeu_si512,
    // Vector arithmetic
    _mm512_add_epi8,
    _mm512_adds_epu8, _mm512_subs_epu8,
    _mm512_max_epu8,
    // Vector logic
    _mm512_xor_si512, _mm512_or_si512,
    _mm512_shuffle_epi8,
    // Mask comparisons (return __mmask64 or __mmask32)
    _mm512_cmpeq_epu8_mask, _mm512_cmpgt_epu8_mask, _mm512_cmpge_epu8_mask,
    _mm512_cmpgt_epi16_mask,
    _mm512_movepi8_mask,
    // Mask blends
    _mm512_mask_blend_epi8, _mm512_mask_blend_epi16,
};

/// SIMD width for AVX-512: 64 sequences with 8-bit scores
pub const SIMD_WIDTH8: usize = 64;

/// SIMD width for AVX-512: 32 sequences with 16-bit scores
pub const SIMD_WIDTH16: usize = 32;

/// Dummy value for ambiguous bases (matching BWA-MEM2)
const DUMMY5: i8 = 5;

/// KSW flags (from kswv.h)
const KSW_XSUBO: i32 = 0x40000;
const KSW_XSTOP: i32 = 0x20000;

/// Batched Smith-Waterman alignment using AVX-512 horizontal SIMD
///
/// Direct port from BWA-MEM2 kswv512_u8() using raw intrinsics.
///
/// # Safety
/// Requires AVX-512BW CPU support. Caller must ensure sequence buffers
/// are properly sized for 64-way parallelism.
#[target_feature(enable = "avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn batch_ksw_align_avx512(
    seq1_soa: *const u8,       // Reference sequences (SoA layout, 64 interleaved)
    seq2_soa: *const u8,       // Query sequences (SoA layout, 64 interleaved)
    nrow: i16,                 // Reference length
    ncol: i16,                 // Query length
    pairs: &[SeqPair],         // Sequence pair metadata
    results: &mut [KswResult], // Output results
    w_match: i8,               // Match score
    w_mismatch: i8,            // Mismatch penalty
    o_del: i32,                // Gap open (deletion)
    e_del: i32,                // Gap extension (deletion)
    o_ins: i32,                // Gap open (insertion)
    e_ins: i32,                // Gap extension (insertion)
    w_ambig: i8,               // Ambiguous base penalty
    _phase: i32,               // Processing phase (unused in forward pass)
) -> usize {
    // ========================================================================
    // SECTION 1: Initialization (C++ lines 387-478)
    // ========================================================================

    // Basic vectors
    let zero512: __m512i = _mm512_setzero_si512();
    let one512: __m512i = _mm512_set1_epi8(1);

    // Compute shift for signed/unsigned score conversion
    // C++: shift = 256 - min(w_match, w_mismatch, w_ambig)
    let shift_val = w_match.min(w_mismatch).min(w_ambig);
    let shift = (256i16 - shift_val as i16) as u8;
    let qmax = w_match.max(w_mismatch).max(w_ambig);

    // Build scoring lookup table for shuffle_epi8
    // Pattern repeats every 16 bytes (shuffle works per 128-bit lane)
    #[repr(align(64))]
    struct AlignedTemp([i8; SIMD_WIDTH8]);
    let mut temp = AlignedTemp([0i8; SIMD_WIDTH8]);

    temp.0[0] = w_match;                              // Match (XOR = 0)
    temp.0[1] = w_mismatch;                           // Mismatch
    temp.0[2] = w_mismatch;
    temp.0[3] = w_mismatch;
    temp.0[4..8].fill(w_ambig);                       // Beyond boundary
    temp.0[8..12].fill(w_ambig);                      // SSE2 region
    temp.0[12] = w_ambig;                             // Ambiguous

    // Add shift to first 16 elements
    for i in 0..16 {
        temp.0[i] = temp.0[i].wrapping_add(shift as i8);
    }

    // Replicate pattern for all 64 bytes
    for i in 16..SIMD_WIDTH8 {
        temp.0[i] = temp.0[i % 16];
    }

    let perm_sft512: __m512i = _mm512_loadu_si512(temp.0.as_ptr() as *const _);
    let sft512: __m512i = _mm512_set1_epi8(shift as i8);
    let cmax512: __m512i = _mm512_set1_epi8(255u8 as i8);
    let five512: __m512i = _mm512_set1_epi8(DUMMY5);

    // Initialize minsc/endsc arrays and masks from h0 flags
    #[repr(align(64))]
    struct AlignedU8([u8; SIMD_WIDTH8]);
    let mut minsc = AlignedU8([0u8; SIMD_WIDTH8]);
    let mut endsc = AlignedU8([0u8; SIMD_WIDTH8]);  // C++ initializes to 0!
    let mut minsc_msk_a: __mmask64 = 0;
    let mut endsc_msk_a: __mmask64 = 0;

    for i in 0..SIMD_WIDTH8.min(pairs.len()) {
        let xtra = pairs[i].h0;

        // KSW_XSUBO: minimum score threshold
        let val = if (xtra & KSW_XSUBO) != 0 {
            (xtra & 0xffff) as u32
        } else {
            0x10000
        };
        if val <= 255 {
            minsc.0[i] = val as u8;
            minsc_msk_a |= 1u64 << i;
        }

        // KSW_XSTOP: early termination score
        let val = if (xtra & KSW_XSTOP) != 0 {
            (xtra & 0xffff) as u32
        } else {
            0x10000
        };
        if val <= 255 {
            endsc.0[i] = val as u8;
            endsc_msk_a |= 1u64 << i;
        }
    }

    let minsc512: __m512i = _mm512_loadu_si512(minsc.0.as_ptr() as *const _);
    let endsc512: __m512i = _mm512_loadu_si512(endsc.0.as_ptr() as *const _);

    // Scoring parameters
    let e_del512: __m512i = _mm512_set1_epi8(e_del as i8);
    let oe_del512: __m512i = _mm512_set1_epi8((o_del + e_del) as i8);
    let e_ins512: __m512i = _mm512_set1_epi8(e_ins as i8);
    let oe_ins512: __m512i = _mm512_set1_epi8((o_ins + e_ins) as i8);

    // Global maximum tracking
    let mut gmax512: __m512i = zero512;
    let mut te512: __m512i = _mm512_set1_epi16(-1);   // Sequences 0-31 (16-bit)
    let mut te512_: __m512i = _mm512_set1_epi16(-1);  // Sequences 32-63 (16-bit)
    let mut qe512: __m512i = _mm512_set1_epi8(0);

    // K-masks (NOT vectors!)
    let mut exit0: __mmask64 = 0xFFFFFFFFFFFFFFFF;  // All lanes active
    let mut mask512: __mmask64 = 0;
    let mut minsc_msk: __mmask64 = 0;

    // Allocate DP matrices
    let max_query_len = ncol as usize + 1;
    let max_ref_len = nrow as usize + 1;

    let mut h0_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut h1_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut f_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut row_max_buf = vec![0u8; max_ref_len * SIMD_WIDTH8];

    // Initialize H0, F to zero
    for i in 0..=ncol as usize {
        let offset = i * SIMD_WIDTH8;
        _mm512_storeu_si512(h0_buf[offset..].as_mut_ptr() as *mut _, zero512);
        _mm512_storeu_si512(f_buf[offset..].as_mut_ptr() as *mut _, zero512);
    }

    let mut pimax512: __m512i = zero512;

    // ========================================================================
    // SECTION 2: Main DP Loop (C++ lines 480-547)
    // ========================================================================

    let mut limit = nrow as i32;
    let mut i = 0i32;

    while i < nrow as i32 {
        // Row variables
        let mut e11: __m512i = zero512;
        let mut imax512: __m512i = zero512;
        let mut iqe512: __m512i = _mm512_set1_epi8(-1);
        let i512: __m512i = _mm512_set1_epi16(i as i16);
        let mut l512: __m512i = zero512;

        // Load reference base for this row
        let s1: __m512i = _mm512_loadu_si512(seq1_soa.add(i as usize * SIMD_WIDTH8) as *const _);

        // Inner loop over query positions
        for j in 0..ncol as usize {
            // Load DP values and query base
            let h00: __m512i = _mm512_loadu_si512(h0_buf[j * SIMD_WIDTH8..].as_ptr() as *const _);
            let s2: __m512i = _mm512_loadu_si512(seq2_soa.add(j * SIMD_WIDTH8) as *const _);
            let f11: __m512i = _mm512_loadu_si512(f_buf[(j + 1) * SIMD_WIDTH8..].as_ptr() as *const _);

            // ================================================================
            // MAIN_SAM_CODE8_OPT (C++ lines 63-86)
            // ================================================================

            // Compute match/mismatch via XOR + shuffle
            let xor11: __m512i = _mm512_xor_si512(s1, s2);
            let mut sbt11: __m512i = _mm512_shuffle_epi8(perm_sft512, xor11);

            // Handle ambiguous bases (base == 5)
            let cmpq: __mmask64 = _mm512_cmpeq_epu8_mask(s2, five512);
            sbt11 = _mm512_mask_blend_epi8(cmpq, sbt11, sft512);

            // Mask out invalid positions via high bit of OR(s1, s2)
            let or11: __m512i = _mm512_or_si512(s1, s2);
            let cmp: __mmask64 = _mm512_movepi8_mask(or11);

            // Match score: H[i-1,j-1] + score
            let mut m11: __m512i = _mm512_adds_epu8(h00, sbt11);
            m11 = _mm512_mask_blend_epi8(cmp, m11, zero512);  // Zero where invalid
            m11 = _mm512_subs_epu8(m11, sft512);

            // h11 = max(match, E, F)
            let mut h11: __m512i = _mm512_max_epu8(m11, e11);
            h11 = _mm512_max_epu8(h11, f11);

            // Track row maximum and query end position
            let cmp0: __mmask64 = _mm512_cmpgt_epu8_mask(h11, imax512);
            imax512 = _mm512_max_epu8(imax512, h11);
            iqe512 = _mm512_mask_blend_epi8(cmp0, iqe512, l512);

            // Update E (gap in query)
            let gap_e512: __m512i = _mm512_subs_epu8(h11, oe_ins512);
            e11 = _mm512_subs_epu8(e11, e_ins512);
            e11 = _mm512_max_epu8(gap_e512, e11);

            // Update F (gap in reference)
            let gap_d512: __m512i = _mm512_subs_epu8(h11, oe_del512);
            let mut f21: __m512i = _mm512_subs_epu8(f11, e_del512);
            f21 = _mm512_max_epu8(gap_d512, f21);

            // Store updated DP values
            _mm512_storeu_si512(h1_buf[(j + 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _, h11);
            _mm512_storeu_si512(f_buf[(j + 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _, f21);

            // Increment query position
            l512 = _mm512_add_epi8(l512, one512);
        }

        // ================================================================
        // Block I: Track row maxima for second-best score (C++ lines 508-519)
        // ================================================================
        if i > 0 {
            let msk64: __mmask64 = _mm512_cmpgt_epu8_mask(imax512, pimax512);
            let combined: __mmask64 = msk64 | mask512;

            // pimax512 = where(combined, zero512, pimax512)
            let mut pimax512_tmp: __m512i = _mm512_mask_blend_epi8(combined, pimax512, zero512);
            // pimax512 = where(!minsc_msk, zero512, pimax512)
            pimax512_tmp = _mm512_mask_blend_epi8(minsc_msk, zero512, pimax512_tmp);
            // pimax512 = where(!exit0, zero512, pimax512)
            pimax512_tmp = _mm512_mask_blend_epi8(exit0, zero512, pimax512_tmp);

            _mm512_storeu_si512(
                row_max_buf[(i as usize - 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
                pimax512_tmp,
            );

            mask512 = !combined;
        }

        pimax512 = imax512;

        // Update minsc mask
        minsc_msk = _mm512_cmpge_epu8_mask(imax512, minsc512) & minsc_msk_a;

        // ================================================================
        // Block II: Update global maximum and target end (C++ lines 524-543)
        // ================================================================
        let cmp0: __mmask64 = _mm512_cmpgt_epu8_mask(imax512, gmax512) & exit0;

        gmax512 = _mm512_mask_blend_epi8(cmp0, gmax512, imax512);

        // te512/te512_ are 16-bit, need to split the 64-bit mask
        te512 = _mm512_mask_blend_epi16(cmp0 as __mmask32, te512, i512);
        te512_ = _mm512_mask_blend_epi16((cmp0 >> SIMD_WIDTH16) as __mmask32, te512_, i512);

        qe512 = _mm512_mask_blend_epi8(cmp0, qe512, iqe512);

        // Early termination check
        let cmp0_term: __mmask64 = _mm512_cmpge_epu8_mask(gmax512, endsc512) & endsc_msk_a;
        let left512: __m512i = _mm512_adds_epu8(gmax512, sft512);
        let cmp2: __mmask64 = _mm512_cmpge_epu8_mask(left512, cmax512);

        exit0 = (!(cmp0_term | cmp2)) & exit0;

        if exit0 == 0 {
            // C++: limit = i++; break;
            // Note: i is incremented post-break to match C++ semantics but
            // the incremented value isn't used. We keep the semantic comment.
            limit = i;
            break;
        }

        // Swap H0/H1 buffers
        std::mem::swap(&mut h0_buf, &mut h1_buf);

        i += 1;
    }

    // Final row max update (C++ lines 549-552)
    let pimax512_final: __m512i = {
        let tmp = _mm512_mask_blend_epi8(mask512, pimax512, zero512);
        let tmp = _mm512_mask_blend_epi8(minsc_msk, zero512, tmp);
        _mm512_mask_blend_epi8(exit0, zero512, tmp)
    };

    if limit > 0 {
        _mm512_storeu_si512(
            row_max_buf[(limit as usize - 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
            pimax512_final,
        );
    }

    // ========================================================================
    // SECTION 3: Score Extraction (C++ lines 556-606)
    // ========================================================================

    #[repr(align(64))]
    struct AlignedScore([u8; SIMD_WIDTH8]);
    #[repr(align(64))]
    struct AlignedTe([i16; SIMD_WIDTH8]);
    #[repr(align(64))]
    struct AlignedQe([u8; SIMD_WIDTH8]);

    let mut score_arr = AlignedScore([0; SIMD_WIDTH8]);
    let mut te_arr = AlignedTe([0; SIMD_WIDTH8]);
    let mut qe_arr = AlignedQe([0; SIMD_WIDTH8]);

    _mm512_storeu_si512(score_arr.0.as_mut_ptr() as *mut _, gmax512);
    _mm512_storeu_si512(te_arr.0.as_mut_ptr() as *mut _, te512);
    _mm512_storeu_si512(te_arr.0[SIMD_WIDTH16..].as_mut_ptr() as *mut _, te512_);
    _mm512_storeu_si512(qe_arr.0.as_mut_ptr() as *mut _, qe512);

    let mut live = 0;
    for l in 0..SIMD_WIDTH8.min(pairs.len()) {
        let score = score_arr.0[l];
        let final_score = if (score as u32 + shift as u32) < 255 {
            score
        } else {
            255
        };

        results[l].score = final_score as i32;
        results[l].te = te_arr.0[l] as i32;
        results[l].qe = qe_arr.0[l] as i32;

        if final_score != 255 {
            qe_arr.0[l] = 1;  // Mark as live for second-best computation
            live += 1;
        } else {
            qe_arr.0[l] = 0;
        }
    }

    if live == 0 {
        return 1;
    }

    // ========================================================================
    // SECTION 4: Second-Best Score Computation (C++ lines 608-677)
    // ========================================================================

    let mut low_arr = AlignedTe([0; SIMD_WIDTH8]);
    let mut high_arr = AlignedTe([0; SIMD_WIDTH8]);
    let mut rlen_arr = AlignedTe([0; SIMD_WIDTH8]);

    let mut maxl: i32 = 0;
    let mut minh: i32 = nrow as i32;

    for idx in 0..SIMD_WIDTH8.min(pairs.len()) {
        let val = ((score_arr.0[idx] as i32) + (qmax as i32) - 1) / (qmax as i32);

        low_arr.0[idx] = (te_arr.0[idx] - val as i16).max(0);
        high_arr.0[idx] = (te_arr.0[idx] + val as i16).min(nrow as i16 - 1);
        rlen_arr.0[idx] = pairs[idx].ref_len as i16;

        if qe_arr.0[idx] != 0 {
            maxl = maxl.max(low_arr.0[idx] as i32);
            minh = minh.min(high_arr.0[idx] as i32);
        }
    }

    let mut max512: __m512i = zero512;
    let mut te512_2nd: __m512i = _mm512_set1_epi16(-1);
    let mut te512_2nd_: __m512i = _mm512_set1_epi16(-1);

    // Load exclusion boundaries (16-bit values, split for sequences 0-31 and 32-63)
    let low512: __m512i = _mm512_loadu_si512(low_arr.0.as_ptr() as *const _);
    let low512_: __m512i = _mm512_loadu_si512(low_arr.0[SIMD_WIDTH16..].as_ptr() as *const _);
    let high512: __m512i = _mm512_loadu_si512(high_arr.0.as_ptr() as *const _);
    let high512_: __m512i = _mm512_loadu_si512(high_arr.0[SIMD_WIDTH16..].as_ptr() as *const _);
    let rlen512: __m512i = _mm512_loadu_si512(rlen_arr.0.as_ptr() as *const _);
    let rlen512_: __m512i = _mm512_loadu_si512(rlen_arr.0[SIMD_WIDTH16..].as_ptr() as *const _);

    // Forward scan: rows [0, maxl) - below exclusion zone
    for row in 0..maxl {
        let i512: __m512i = _mm512_set1_epi16(row as i16);

        let rmax512: __m512i = _mm512_loadu_si512(
            row_max_buf[row as usize * SIMD_WIDTH8..].as_ptr() as *const _
        );

        // mask1: i < low[seq] (16-bit comparison, combined to 64-bit)
        let mask11: __mmask64 = _mm512_cmpgt_epi16_mask(low512, i512) as __mmask64;
        let mask12: __mmask64 = _mm512_cmpgt_epi16_mask(low512_, i512) as __mmask64;
        let mask1: __mmask64 = mask11 | (mask12 << SIMD_WIDTH16);

        // mask2: rmax > max512 (8-bit comparison)
        let mask2: __mmask64 = _mm512_cmpgt_epu8_mask(rmax512, max512);

        let combined: __mmask64 = mask2 & mask1;

        max512 = _mm512_mask_blend_epi8(combined, max512, rmax512);
        te512_2nd = _mm512_mask_blend_epi16(combined as __mmask32, te512_2nd, i512);
        te512_2nd_ = _mm512_mask_blend_epi16((combined >> SIMD_WIDTH16) as __mmask32, te512_2nd_, i512);
    }

    // Backward scan: rows [minh+1, limit) - above exclusion zone
    for row in (minh + 1).max(0)..limit {
        let i512: __m512i = _mm512_set1_epi16(row as i16);

        let rmax512: __m512i = _mm512_loadu_si512(
            row_max_buf[row as usize * SIMD_WIDTH8..].as_ptr() as *const _
        );

        // mask1: i > high[seq]
        let mask11: __mmask64 = _mm512_cmpgt_epi16_mask(i512, high512) as __mmask64;
        let mask12: __mmask64 = _mm512_cmpgt_epi16_mask(i512, high512_) as __mmask64;
        let mask1: __mmask64 = mask11 | (mask12 << SIMD_WIDTH16);

        // mask2: rmax > max512
        let mask2: __mmask64 = _mm512_cmpgt_epu8_mask(rmax512, max512);

        // mask1_: i < rlen[seq] (within bounds)
        let mask11_: __mmask64 = _mm512_cmpgt_epi16_mask(rlen512, i512) as __mmask64;
        let mask12_: __mmask64 = _mm512_cmpgt_epi16_mask(rlen512_, i512) as __mmask64;
        let mask1_: __mmask64 = mask11_ | (mask12_ << SIMD_WIDTH16);

        let combined: __mmask64 = mask2 & mask1 & mask1_;

        max512 = _mm512_mask_blend_epi8(combined, max512, rmax512);
        te512_2nd = _mm512_mask_blend_epi16(combined as __mmask32, te512_2nd, i512);
        te512_2nd_ = _mm512_mask_blend_epi16((combined >> SIMD_WIDTH16) as __mmask32, te512_2nd_, i512);
    }

    // Extract second-best scores
    let mut score2_arr = AlignedScore([0; SIMD_WIDTH8]);
    let mut te2_arr = AlignedTe([0; SIMD_WIDTH8]);

    _mm512_storeu_si512(score2_arr.0.as_mut_ptr() as *mut _, max512);
    _mm512_storeu_si512(te2_arr.0.as_mut_ptr() as *mut _, te512_2nd);
    _mm512_storeu_si512(te2_arr.0[SIMD_WIDTH16..].as_mut_ptr() as *mut _, te512_2nd_);

    for idx in 0..pairs.len().min(SIMD_WIDTH8) {
        if qe_arr.0[idx] != 0 {
            results[idx].score2 = if score2_arr.0[idx] == 0 {
                -1
            } else {
                score2_arr.0[idx] as i32
            };
            results[idx].te2 = te2_arr.0[idx] as i32;
        } else {
            results[idx].score2 = -1;
            results[idx].te2 = -1;
        }

        results[idx].tb = -1;
        results[idx].qb = -1;
    }

    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_compiles() {
        // Just verify the module compiles with avx512 feature
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping: AVX-512BW not available");
            return;
        }

        // Basic sanity check that constants are correct
        assert_eq!(SIMD_WIDTH8, 64);
        assert_eq!(SIMD_WIDTH16, 32);
    }
}
