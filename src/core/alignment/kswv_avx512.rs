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
    __m512i,
    __mmask32,
    __mmask64,
    // Vector arithmetic
    _mm512_add_epi8,
    _mm512_adds_epu8,
    // Mask comparisons (return __mmask64 or __mmask32)
    _mm512_cmpeq_epu8_mask,
    _mm512_cmpge_epu8_mask,
    _mm512_cmpgt_epi16_mask,
    _mm512_cmpgt_epu8_mask,
    // Vector loads/stores (64-byte aligned for performance)
    _mm512_load_si512,
    // Mask blends
    _mm512_mask_blend_epi8,
    _mm512_mask_blend_epi16,
    _mm512_max_epu8,
    _mm512_movepi8_mask,
    _mm512_or_si512,
    _mm512_set1_epi8,
    _mm512_set1_epi16,
    // Vector creation
    _mm512_setzero_si512,
    _mm512_shuffle_epi8,
    _mm512_store_si512,
    _mm512_subs_epu8,
    // Vector logic
    _mm512_xor_si512,
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
/// If workspace_buffers provided, they must be large enough for the sequences.
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
    _debug: bool,              // Debug flag (unused, for API consistency)
    workspace_buffers: Option<(&mut [u8], &mut [u8], &mut [u8], &mut [u8])>,
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

    temp.0[0] = w_match; // Match (XOR = 0)
    temp.0[1] = w_mismatch; // Mismatch
    temp.0[2] = w_mismatch;
    temp.0[3] = w_mismatch;
    temp.0[4..8].fill(w_ambig); // Beyond boundary
    temp.0[8..12].fill(w_ambig); // SSE2 region
    temp.0[12] = w_ambig; // Ambiguous

    // Add shift to first 16 elements
    for i in 0..16 {
        temp.0[i] = temp.0[i].wrapping_add(shift as i8);
    }

    // Replicate pattern for all 64 bytes
    for i in 16..SIMD_WIDTH8 {
        temp.0[i] = temp.0[i % 16];
    }

    let perm_sft512: __m512i = _mm512_load_si512(temp.0.as_ptr() as *const _);
    let sft512: __m512i = _mm512_set1_epi8(shift as i8);
    let cmax512: __m512i = _mm512_set1_epi8(255u8 as i8);
    let five512: __m512i = _mm512_set1_epi8(DUMMY5);

    // Initialize minsc/endsc arrays and masks from h0 flags
    #[repr(align(64))]
    struct AlignedU8([u8; SIMD_WIDTH8]);
    let mut minsc = AlignedU8([0u8; SIMD_WIDTH8]);
    let mut endsc = AlignedU8([0u8; SIMD_WIDTH8]); // C++ initializes to 0!
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

    let minsc512: __m512i = _mm512_load_si512(minsc.0.as_ptr() as *const _);
    let endsc512: __m512i = _mm512_load_si512(endsc.0.as_ptr() as *const _);

    // Scoring parameters
    let e_del512: __m512i = _mm512_set1_epi8(e_del as i8);
    let oe_del512: __m512i = _mm512_set1_epi8((o_del + e_del) as i8);
    let e_ins512: __m512i = _mm512_set1_epi8(e_ins as i8);
    let oe_ins512: __m512i = _mm512_set1_epi8((o_ins + e_ins) as i8);

    // Global maximum tracking
    let mut gmax512: __m512i = zero512;
    let mut te512: __m512i = _mm512_set1_epi16(-1); // Sequences 0-31 (16-bit)
    let mut te512_: __m512i = _mm512_set1_epi16(-1); // Sequences 32-63 (16-bit)
    let mut qe512: __m512i = _mm512_set1_epi8(0);

    // K-masks (NOT vectors!)
    let mut exit0: __mmask64 = 0xFFFFFFFFFFFFFFFF; // All lanes active
    let mut mask512: __mmask64 = 0;
    let mut minsc_msk: __mmask64 = 0;

    // Allocate DP matrices
    // Use workspace pre-allocation when provided to avoid ~132KB allocation per call
    let max_query_len = ncol as usize + 1;
    let max_ref_len = nrow as usize + 1;
    let required_h_size = max_query_len * SIMD_WIDTH8;
    let required_row_max_size = max_ref_len * SIMD_WIDTH8;

    // Check if workspace buffers are provided and large enough
    let (mut h0_buf_owned, mut h1_buf_owned, mut f_buf_owned, mut row_max_buf_owned);
    let (mut h0_buf, mut h1_buf, mut f_buf, mut row_max_buf): (
        &mut [u8],
        &mut [u8],
        &mut [u8],
        &mut [u8],
    ) = if let Some((ws_h0, ws_h1, ws_f, ws_row_max)) = workspace_buffers {
        // Use workspace buffers if they're large enough
        if ws_h0.len() >= required_h_size
            && ws_h1.len() >= required_h_size
            && ws_f.len() >= required_h_size
            && ws_row_max.len() >= required_row_max_size
        {
            // Zero out the portion we'll use (workspace already allocated)
            ws_h0[..required_h_size].fill(0);
            ws_h1[..required_h_size].fill(0);
            ws_f[..required_h_size].fill(0);
            ws_row_max[..required_row_max_size].fill(0);
            (
                &mut ws_h0[..required_h_size],
                &mut ws_h1[..required_h_size],
                &mut ws_f[..required_h_size],
                &mut ws_row_max[..required_row_max_size],
            )
        } else {
            // Workspace too small, fall back to allocation
            log::trace!(
                "AVX512 kswv: workspace too small (need {}+{}, have {}+{}), allocating",
                required_h_size,
                required_row_max_size,
                ws_h0.len(),
                ws_row_max.len()
            );
            h0_buf_owned = vec![0u8; required_h_size];
            h1_buf_owned = vec![0u8; required_h_size];
            f_buf_owned = vec![0u8; required_h_size];
            row_max_buf_owned = vec![0u8; required_row_max_size];
            (
                &mut h0_buf_owned,
                &mut h1_buf_owned,
                &mut f_buf_owned,
                &mut row_max_buf_owned,
            )
        }
    } else {
        // No workspace provided, allocate locally
        h0_buf_owned = vec![0u8; required_h_size];
        h1_buf_owned = vec![0u8; required_h_size];
        f_buf_owned = vec![0u8; required_h_size];
        row_max_buf_owned = vec![0u8; required_row_max_size];
        (
            &mut h0_buf_owned,
            &mut h1_buf_owned,
            &mut f_buf_owned,
            &mut row_max_buf_owned,
        )
    };

    // Initialize H0, F to zero
    // Use pointer arithmetic to preserve 64-byte alignment (allows LLVM to optimize to aligned ops)
    let h0_ptr = h0_buf.as_mut_ptr();
    let f_ptr = f_buf.as_mut_ptr();
    for i in 0..=ncol as usize {
        let offset = i * SIMD_WIDTH8;
        _mm512_store_si512(h0_ptr.add(offset) as *mut _, zero512);
        _mm512_store_si512(f_ptr.add(offset) as *mut _, zero512);
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
        let s1: __m512i = _mm512_load_si512(seq1_soa.add(i as usize * SIMD_WIDTH8) as *const _);

        // Inner loop over query positions
        // Use pointer arithmetic to preserve alignment (allows LLVM to optimize to aligned ops)
        let h0_ptr = h0_buf.as_ptr();
        let f_ptr = f_buf.as_ptr();
        for j in 0..ncol as usize {
            // Load DP values and query base
            let h00: __m512i = _mm512_load_si512(h0_ptr.add(j * SIMD_WIDTH8) as *const _);
            let s2: __m512i = _mm512_load_si512(seq2_soa.add(j * SIMD_WIDTH8) as *const _);
            let f11: __m512i = _mm512_load_si512(f_ptr.add((j + 1) * SIMD_WIDTH8) as *const _);

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
            m11 = _mm512_mask_blend_epi8(cmp, m11, zero512); // Zero where invalid
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

            // Store updated DP values using pointer arithmetic (allows LLVM to optimize to aligned ops)
            let h1_ptr = h1_buf.as_mut_ptr();
            _mm512_store_si512(h1_ptr.add((j + 1) * SIMD_WIDTH8) as *mut _, h11);
            let f_mut_ptr = f_buf.as_mut_ptr();
            _mm512_store_si512(f_mut_ptr.add((j + 1) * SIMD_WIDTH8) as *mut _, f21);

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

            // Store row max using pointer arithmetic (allows LLVM to optimize to aligned ops)
            let row_max_ptr = row_max_buf.as_mut_ptr();
            _mm512_store_si512(
                row_max_ptr.add((i as usize - 1) * SIMD_WIDTH8) as *mut _,
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

        // Swap H0/H1 buffers (h0_buf and h1_buf are already &mut [u8])
        std::ptr::swap(&mut h0_buf, &mut h1_buf);

        i += 1;
    }

    // Final row max update (C++ lines 549-552)
    let pimax512_final: __m512i = {
        let tmp = _mm512_mask_blend_epi8(mask512, pimax512, zero512);
        let tmp = _mm512_mask_blend_epi8(minsc_msk, zero512, tmp);
        _mm512_mask_blend_epi8(exit0, zero512, tmp)
    };

    if limit > 0 {
        // Store final row max using pointer arithmetic (allows LLVM to optimize to aligned ops)
        let row_max_ptr = row_max_buf.as_mut_ptr();
        _mm512_store_si512(
            row_max_ptr.add((limit as usize - 1) * SIMD_WIDTH8) as *mut _,
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

    // Score extraction: arrays are #[repr(align(64))], LLVM will optimize to aligned ops
    _mm512_store_si512(score_arr.0.as_mut_ptr() as *mut _, gmax512);
    _mm512_store_si512(te_arr.0.as_mut_ptr() as *mut _, te512);
    _mm512_store_si512(te_arr.0[SIMD_WIDTH16..].as_mut_ptr() as *mut _, te512_); // Uses slicing
    _mm512_store_si512(qe_arr.0.as_mut_ptr() as *mut _, qe512);

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

        // Debug logging for first sequence to compare with AVX2
        if l == 0 && final_score > 0 {
            log::trace!(
                "AVX512 kswv seq0: score={}, te={}, qe={}, nrow={}, ncol={}, shift={}",
                final_score,
                te_arr.0[l],
                qe_arr.0[l],
                nrow,
                ncol,
                shift
            );
        }

        if final_score != 255 {
            qe_arr.0[l] = 1; // Mark as live for second-best computation
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
    let low512: __m512i = _mm512_load_si512(low_arr.0.as_ptr() as *const _);
    let low512_: __m512i = _mm512_load_si512(low_arr.0[SIMD_WIDTH16..].as_ptr() as *const _);
    let high512: __m512i = _mm512_load_si512(high_arr.0.as_ptr() as *const _);
    let high512_: __m512i = _mm512_load_si512(high_arr.0[SIMD_WIDTH16..].as_ptr() as *const _);
    let rlen512: __m512i = _mm512_load_si512(rlen_arr.0.as_ptr() as *const _);
    let rlen512_: __m512i = _mm512_load_si512(rlen_arr.0[SIMD_WIDTH16..].as_ptr() as *const _);

    // Forward scan: rows [0, maxl) - below exclusion zone
    for row in 0..maxl {
        let i512: __m512i = _mm512_set1_epi16(row as i16);

        let rmax512: __m512i =
            _mm512_load_si512(row_max_buf[row as usize * SIMD_WIDTH8..].as_ptr() as *const _);

        // mask1: i < low[seq] (16-bit comparison, combined to 64-bit)
        let mask11: __mmask64 = _mm512_cmpgt_epi16_mask(low512, i512) as __mmask64;
        let mask12: __mmask64 = _mm512_cmpgt_epi16_mask(low512_, i512) as __mmask64;
        let mask1: __mmask64 = mask11 | (mask12 << SIMD_WIDTH16);

        // mask2: rmax > max512 (8-bit comparison)
        let mask2: __mmask64 = _mm512_cmpgt_epu8_mask(rmax512, max512);

        let combined: __mmask64 = mask2 & mask1;

        max512 = _mm512_mask_blend_epi8(combined, max512, rmax512);
        te512_2nd = _mm512_mask_blend_epi16(combined as __mmask32, te512_2nd, i512);
        te512_2nd_ =
            _mm512_mask_blend_epi16((combined >> SIMD_WIDTH16) as __mmask32, te512_2nd_, i512);
    }

    // Backward scan: rows [minh+1, limit) - above exclusion zone
    for row in (minh + 1).max(0)..limit {
        let i512: __m512i = _mm512_set1_epi16(row as i16);

        let rmax512: __m512i =
            _mm512_load_si512(row_max_buf[row as usize * SIMD_WIDTH8..].as_ptr() as *const _);

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
        te512_2nd_ =
            _mm512_mask_blend_epi16((combined >> SIMD_WIDTH16) as __mmask32, te512_2nd_, i512);
    }

    // Extract second-best scores
    let mut score2_arr = AlignedScore([0; SIMD_WIDTH8]);
    let mut te2_arr = AlignedTe([0; SIMD_WIDTH8]);

    _mm512_store_si512(score2_arr.0.as_mut_ptr() as *mut _, max512);
    _mm512_store_si512(te2_arr.0.as_mut_ptr() as *mut _, te512_2nd);
    _mm512_store_si512(te2_arr.0[SIMD_WIDTH16..].as_mut_ptr() as *mut _, te512_2nd_);

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

        // Set tb and qb based on valid alignment (matching AVX2 behavior)
        // Phase 0 doesn't compute traceback, so we estimate:
        // - For mate rescue, alignments typically start at query position 0
        // - tb is estimated as te - (qe - qb) for same-length alignment
        if results[idx].score > 0 && results[idx].te >= 0 && results[idx].qe >= 0 {
            results[idx].qb = 0; // Alignment starts at query position 0
            results[idx].tb = (results[idx].te - results[idx].qe).max(0);
        } else {
            results[idx].tb = -1;
            results[idx].qb = -1;
        }
    }

    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::kswv_avx2;
    use crate::alignment::kswv_batch::{KswResult, SeqPair};

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

    /// Compare AVX-512 and AVX2 kernels with identical input
    #[test]
    fn test_avx512_vs_avx2_identical_scores() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping: AVX-512BW not available");
            return;
        }
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping: AVX2 not available");
            return;
        }

        // Create test sequences: simple matching sequences
        let ref_seq: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]; // ACGTACGTAC
        let query_seq: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]; // ACGTACGTAC (match)

        let nrow = ref_seq.len() as i16;
        let ncol = query_seq.len() as i16;

        // Create SoA buffers for AVX-512 (64 sequences) and AVX2 (32 sequences)
        let mut ref_soa_512 = vec![0x80u8; nrow as usize * 64];
        let mut query_soa_512 = vec![0x80u8; ncol as usize * 64];
        let mut ref_soa_256 = vec![0x80u8; nrow as usize * 32];
        let mut query_soa_256 = vec![0x80u8; ncol as usize * 32];

        // Transpose sequence into SoA for first slot
        for (i, &base) in ref_seq.iter().enumerate() {
            ref_soa_512[i * 64] = base;
            ref_soa_256[i * 32] = base;
        }
        for (i, &base) in query_seq.iter().enumerate() {
            query_soa_512[i * 64] = base;
            query_soa_256[i * 32] = base;
        }

        // Create pairs and results
        let pair = SeqPair {
            ref_len: nrow as i32,
            query_len: ncol as i32,
            h0: 0,
            ..Default::default()
        };
        let pairs_512 = vec![pair.clone(); 64];
        let pairs_256 = vec![pair.clone(); 32];

        let mut results_512 = vec![KswResult::default(); 64];
        let mut results_256 = vec![KswResult::default(); 32];

        // Run both kernels
        let match_score: i8 = 1;
        let mismatch: i8 = -4;
        let gap_open: i32 = 6;
        let gap_ext: i32 = 1;

        unsafe {
            batch_ksw_align_avx512(
                ref_soa_512.as_ptr(),
                query_soa_512.as_ptr(),
                nrow,
                ncol,
                &pairs_512,
                &mut results_512,
                match_score,
                mismatch,
                gap_open,
                gap_ext,
                gap_open,
                gap_ext,
                -1,    // w_ambig
                0,     // phase
                false, // debug
                None,  // workspace_buffers
            );

            kswv_avx2::batch_ksw_align_avx2(
                ref_soa_256.as_ptr(),
                query_soa_256.as_ptr(),
                nrow,
                ncol,
                &pairs_256,
                &mut results_256,
                match_score,
                mismatch,
                gap_open,
                gap_ext,
                gap_open,
                gap_ext,
                -1,    // w_ambig
                0,     // phase
                false, // debug
                None,  // workspace_buffers
            );
        }

        // Compare first sequence results
        let avx512_score = results_512[0].score;
        let avx2_score = results_256[0].score;

        eprintln!(
            "AVX-512 seq0: score={}, te={}, qe={}",
            results_512[0].score, results_512[0].te, results_512[0].qe
        );
        eprintln!(
            "AVX2 seq0:    score={}, te={}, qe={}",
            results_256[0].score, results_256[0].te, results_256[0].qe
        );

        // Scores should match for identical input
        assert_eq!(
            avx512_score, avx2_score,
            "AVX-512 ({}) and AVX2 ({}) scores differ for identical input!",
            avx512_score, avx2_score
        );
    }

    /// Test with longer sequences and mismatches
    #[test]
    fn test_avx512_vs_avx2_with_mismatches() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping: AVX-512BW not available");
            return;
        }
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping: AVX2 not available");
            return;
        }

        // Create longer sequences with some mismatches
        // Reference: 150bp
        let ref_seq: Vec<u8> = (0..150).map(|i| (i % 4) as u8).collect();
        // Query with 5 mismatches
        let mut query_seq = ref_seq.clone();
        query_seq[10] = (query_seq[10] + 1) % 4; // mismatch
        query_seq[50] = (query_seq[50] + 1) % 4; // mismatch
        query_seq[80] = (query_seq[80] + 1) % 4; // mismatch
        query_seq[100] = (query_seq[100] + 1) % 4; // mismatch
        query_seq[140] = (query_seq[140] + 1) % 4; // mismatch

        let nrow = ref_seq.len() as i16;
        let ncol = query_seq.len() as i16;

        // Create SoA buffers
        let mut ref_soa_512 = vec![0x80u8; nrow as usize * 64];
        let mut query_soa_512 = vec![0x80u8; ncol as usize * 64];
        let mut ref_soa_256 = vec![0x80u8; nrow as usize * 32];
        let mut query_soa_256 = vec![0x80u8; ncol as usize * 32];

        for (i, &base) in ref_seq.iter().enumerate() {
            ref_soa_512[i * 64] = base;
            ref_soa_256[i * 32] = base;
        }
        for (i, &base) in query_seq.iter().enumerate() {
            query_soa_512[i * 64] = base;
            query_soa_256[i * 32] = base;
        }

        let pair = SeqPair {
            ref_len: nrow as i32,
            query_len: ncol as i32,
            h0: 0,
            ..Default::default()
        };
        let pairs_512 = vec![pair.clone(); 64];
        let pairs_256 = vec![pair.clone(); 32];

        let mut results_512 = vec![KswResult::default(); 64];
        let mut results_256 = vec![KswResult::default(); 32];

        let match_score: i8 = 1;
        let mismatch: i8 = -4;
        let gap_open: i32 = 6;
        let gap_ext: i32 = 1;

        unsafe {
            batch_ksw_align_avx512(
                ref_soa_512.as_ptr(),
                query_soa_512.as_ptr(),
                nrow,
                ncol,
                &pairs_512,
                &mut results_512,
                match_score,
                mismatch,
                gap_open,
                gap_ext,
                gap_open,
                gap_ext,
                -1,
                0,
                false,
                None,
            );

            kswv_avx2::batch_ksw_align_avx2(
                ref_soa_256.as_ptr(),
                query_soa_256.as_ptr(),
                nrow,
                ncol,
                &pairs_256,
                &mut results_256,
                match_score,
                mismatch,
                gap_open,
                gap_ext,
                gap_open,
                gap_ext,
                -1,
                0,
                false,
                None,
            );
        }

        eprintln!("Long seq with mismatches:");
        eprintln!(
            "  AVX-512: score={}, te={}, qe={}",
            results_512[0].score, results_512[0].te, results_512[0].qe
        );
        eprintln!(
            "  AVX2:    score={}, te={}, qe={}",
            results_256[0].score, results_256[0].te, results_256[0].qe
        );

        // Expected: 145 matches + 5 mismatches = 145 * 1 + 5 * (-4) = 145 - 20 = 125
        let expected_approx = 145 - 20; // 125

        assert_eq!(
            results_512[0].score, results_256[0].score,
            "AVX-512 ({}) and AVX2 ({}) scores differ!",
            results_512[0].score, results_256[0].score
        );

        // Sanity check: score should be close to expected
        assert!(
            (results_512[0].score - expected_approx).abs() < 10,
            "Score {} too far from expected ~{}",
            results_512[0].score,
            expected_approx
        );
    }

    /// Test AVX-512 slots 32-63 (uses te512_ for 16-bit target end tracking)
    /// This is critical because AVX-512 splits 64 sequences into two 32-seq groups
    #[test]
    fn test_avx512_upper_slots_32_to_63() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping: AVX-512BW not available");
            return;
        }

        // Real-world sequences from failing read HISEQ1:18:H8VC6ADXX:1:1101:10009:11965
        // Reference from chr5:49956951-49957098 (148bp)
        let ref_seq: Vec<u8> = vec![
            3, 2, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0, 3, 0, 2, 0, 2, 1, 0, 2, 3, 3, 3, 3, 2,
            0, 0, 0, 1, 0, 1, 3, 1, 3, 2, 3, 0, 2, 0, 0, 3, 1, 3, 2, 0, 0, 0, 2, 3, 2, 2, 0, 3, 0,
            3, 3, 3, 2, 2, 0, 2, 1, 3, 1, 3, 3, 1, 2, 0, 2, 2, 2, 1, 3, 0, 3, 2, 2, 1, 2, 2, 0, 0,
            0, 0, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 3, 1, 0, 1, 0, 3, 3, 0, 0, 0, 1, 3, 0, 2, 0, 1, 0,
            2, 1, 0, 2, 1, 0, 3, 3, 1, 3, 1, 0, 2, 0, 0, 0, 1, 3, 3, 1, 3, 3, 3, 0, 2, 2, 0, 3, 2,
            3, 3, 3,
        ];
        // Read reverse complement (148bp) - 1 mismatch at position 130
        let query_seq: Vec<u8> = vec![
            3, 2, 0, 0, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 2, 0, 3, 0, 2, 0, 2, 1, 0, 2, 3, 3, 3, 3, 2,
            0, 0, 0, 1, 0, 1, 3, 1, 3, 2, 3, 0, 2, 0, 0, 3, 1, 3, 2, 0, 0, 0, 2, 3, 2, 2, 0, 3, 0,
            3, 3, 3, 2, 2, 0, 2, 1, 3, 1, 3, 3, 1, 2, 0, 2, 2, 2, 1, 3, 0, 3, 2, 2, 1, 2, 2, 0, 0,
            0, 0, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 3, 1, 0, 1, 0, 3, 3, 0, 0, 0, 1, 3, 0, 2, 0, 1, 0,
            2, 1, 0, 2, 1, 0, 3, 3, 1, 3, 1, 0, 2, 0, 2, 0, 1, 3, 3, 1, 3, 3, 3, 0, 2, 2, 0, 3, 2,
            3, 3, 3,
        ];

        let nrow = ref_seq.len() as i16;
        let ncol = query_seq.len() as i16;

        // Test slots 0, 32, and 63
        let test_slots = [0usize, 32, 63];

        let mut ref_soa_512 = vec![0x80u8; nrow as usize * 64];
        let mut query_soa_512 = vec![0x80u8; ncol as usize * 64];

        // Place test data in multiple slots
        for &slot in &test_slots {
            for (i, &base) in ref_seq.iter().enumerate() {
                ref_soa_512[i * 64 + slot] = base;
            }
            for (i, &base) in query_seq.iter().enumerate() {
                query_soa_512[i * 64 + slot] = base;
            }
        }

        let pair = SeqPair {
            ref_len: nrow as i32,
            query_len: ncol as i32,
            h0: 0,
            ..Default::default()
        };
        let pairs_512 = vec![pair.clone(); 64];
        let mut results_512 = vec![KswResult::default(); 64];

        let match_score: i8 = 1;
        let mismatch: i8 = -4;
        let gap_open: i32 = 6;
        let gap_ext: i32 = 1;

        unsafe {
            batch_ksw_align_avx512(
                ref_soa_512.as_ptr(),
                query_soa_512.as_ptr(),
                nrow,
                ncol,
                &pairs_512,
                &mut results_512,
                match_score,
                mismatch,
                gap_open,
                gap_ext,
                gap_open,
                gap_ext,
                -1,
                0,
                false,
                None,
            );
        }

        // Expected: 147 matches + 1 mismatch = 147*1 + 1*(-4) = 143
        let expected = 143;

        eprintln!("Testing slots 0, 32, 63 with real-world 148bp sequence (1 mismatch):");
        for &slot in &test_slots {
            eprintln!(
                "  Slot {}: score={}, te={}, qe={}",
                slot, results_512[slot].score, results_512[slot].te, results_512[slot].qe
            );
        }

        // All slots should produce identical scores
        let score_0 = results_512[0].score;
        for &slot in &test_slots {
            assert_eq!(
                results_512[slot].score, score_0,
                "Slot {} score ({}) differs from slot 0 score ({})!",
                slot, results_512[slot].score, score_0
            );
        }

        // Score should match expected
        assert_eq!(
            score_0, expected,
            "Score {} doesn't match expected {} for 148bp with 1 mismatch",
            score_0, expected
        );
    }

    /// Test AVX-512 vs AVX2 with DIFFERENT sequences per slot
    /// This is the critical test - real batches have diverse sequences
    #[test]
    fn test_avx512_vs_avx2_diverse_batch() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping: AVX-512BW not available");
            return;
        }
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping: AVX2 not available");
            return;
        }

        // Create 32 unique sequence pairs with varying characteristics
        // This tests whether AVX-512 handles mixed batches correctly
        let mut test_cases: Vec<(Vec<u8>, Vec<u8>, &str)> = Vec::new();

        // Case 0: Perfect match 100bp
        let seq100: Vec<u8> = (0..100).map(|i| (i % 4) as u8).collect();
        test_cases.push((seq100.clone(), seq100.clone(), "perfect_100bp"));

        // Case 1: 148bp with 1 mismatch (real-world read length)
        let seq148: Vec<u8> = (0..148).map(|i| (i % 4) as u8).collect();
        let mut seq148_mm = seq148.clone();
        seq148_mm[50] = (seq148_mm[50] + 1) % 4;
        test_cases.push((seq148.clone(), seq148_mm, "148bp_1mm"));

        // Case 2: 50bp with 2 mismatches
        let seq50: Vec<u8> = (0..50).map(|i| (i % 4) as u8).collect();
        let mut seq50_mm = seq50.clone();
        seq50_mm[10] = (seq50_mm[10] + 1) % 4;
        seq50_mm[30] = (seq50_mm[30] + 1) % 4;
        test_cases.push((seq50.clone(), seq50_mm, "50bp_2mm"));

        // Case 3: 200bp with 5 mismatches (longer than typical)
        let seq200: Vec<u8> = (0..200).map(|i| (i % 4) as u8).collect();
        let mut seq200_mm = seq200.clone();
        for pos in [20, 60, 100, 140, 180] {
            seq200_mm[pos] = (seq200_mm[pos] + 1) % 4;
        }
        test_cases.push((seq200.clone(), seq200_mm, "200bp_5mm"));

        // Case 4: Perfect match 75bp
        let seq75: Vec<u8> = (0..75).map(|i| ((i * 3) % 4) as u8).collect();
        test_cases.push((seq75.clone(), seq75.clone(), "perfect_75bp"));

        // Case 5-31: Generate more diverse cases
        for i in 5..32 {
            let len = 50 + (i * 5);
            let ref_seq: Vec<u8> = (0..len).map(|j| ((i + j) % 4) as u8).collect();
            let mut query_seq = ref_seq.clone();
            // Add i mismatches
            for k in 0..i.min(len / 10) {
                let pos = (k * len) / i.max(1);
                if pos < len {
                    query_seq[pos] = (query_seq[pos] + 1) % 4;
                }
            }
            test_cases.push((ref_seq, query_seq, "generated"));
        }

        // Find max lengths across all test cases
        let max_ref_len = test_cases.iter().map(|(r, _, _)| r.len()).max().unwrap();
        let max_query_len = test_cases.iter().map(|(_, q, _)| q.len()).max().unwrap();

        // Create SoA buffers
        let mut ref_soa_512 = vec![0x80u8; max_ref_len * 64];
        let mut query_soa_512 = vec![0x80u8; max_query_len * 64];
        let mut ref_soa_256 = vec![0x80u8; max_ref_len * 32];
        let mut query_soa_256 = vec![0x80u8; max_query_len * 32];

        // Transpose first 32 cases for both AVX-512 (slots 0-31) and AVX2
        for (slot, (ref_seq, query_seq, _)) in test_cases.iter().enumerate().take(32) {
            for (i, &base) in ref_seq.iter().enumerate() {
                ref_soa_512[i * 64 + slot] = base;
                ref_soa_256[i * 32 + slot] = base;
            }
            for (i, &base) in query_seq.iter().enumerate() {
                query_soa_512[i * 64 + slot] = base;
                query_soa_256[i * 32 + slot] = base;
            }
        }

        // Also populate slots 32-63 in AVX-512 with cases 0-31 again
        // to test the upper half handling
        for (slot, (ref_seq, query_seq, _)) in test_cases.iter().enumerate().take(32) {
            let slot_upper = slot + 32;
            for (i, &base) in ref_seq.iter().enumerate() {
                ref_soa_512[i * 64 + slot_upper] = base;
            }
            for (i, &base) in query_seq.iter().enumerate() {
                query_soa_512[i * 64 + slot_upper] = base;
            }
        }

        // Create pairs metadata
        let mut pairs_512: Vec<SeqPair> = (0..64)
            .map(|i| {
                let case_idx = i % 32;
                SeqPair {
                    ref_len: test_cases[case_idx].0.len() as i32,
                    query_len: test_cases[case_idx].1.len() as i32,
                    h0: 0,
                    ..Default::default()
                }
            })
            .collect();

        let pairs_256: Vec<SeqPair> = (0..32)
            .map(|i| SeqPair {
                ref_len: test_cases[i].0.len() as i32,
                query_len: test_cases[i].1.len() as i32,
                h0: 0,
                ..Default::default()
            })
            .collect();

        // Set nrow/ncol to max for the kernel
        pairs_512[0].ref_len = max_ref_len as i32;
        pairs_512[0].query_len = max_query_len as i32;

        let mut pairs_256_copy = pairs_256.clone();
        pairs_256_copy[0].ref_len = max_ref_len as i32;
        pairs_256_copy[0].query_len = max_query_len as i32;

        let mut results_512 = vec![KswResult::default(); 64];
        let mut results_256 = vec![KswResult::default(); 32];

        let match_score: i8 = 1;
        let mismatch: i8 = -4;
        let gap_open: i32 = 6;
        let gap_ext: i32 = 1;

        unsafe {
            batch_ksw_align_avx512(
                ref_soa_512.as_ptr(),
                query_soa_512.as_ptr(),
                max_ref_len as i16,
                max_query_len as i16,
                &pairs_512,
                &mut results_512,
                match_score,
                mismatch,
                gap_open,
                gap_ext,
                gap_open,
                gap_ext,
                -1,
                0,
                false,
                None,
            );

            kswv_avx2::batch_ksw_align_avx2(
                ref_soa_256.as_ptr(),
                query_soa_256.as_ptr(),
                max_ref_len as i16,
                max_query_len as i16,
                &pairs_256_copy,
                &mut results_256,
                match_score,
                mismatch,
                gap_open,
                gap_ext,
                gap_open,
                gap_ext,
                -1,
                0,
                false,
                None,
            );
        }

        // Compare results for slots 0-31 (AVX-512 vs AVX2)
        let mut mismatches = 0;
        for slot in 0..32 {
            let avx512 = &results_512[slot];
            let avx2 = &results_256[slot];

            if avx512.score != avx2.score || avx512.te != avx2.te || avx512.qe != avx2.qe {
                mismatches += 1;
                eprintln!(
                    "MISMATCH slot {}: AVX-512(score={}, te={}, qe={}) vs AVX2(score={}, te={}, qe={}) - {}",
                    slot,
                    avx512.score,
                    avx512.te,
                    avx512.qe,
                    avx2.score,
                    avx2.te,
                    avx2.qe,
                    test_cases[slot].2
                );
            }
        }

        // Also verify slots 32-63 match slots 0-31 in AVX-512
        let mut upper_mismatches = 0;
        for slot in 0..32 {
            let lower = &results_512[slot];
            let upper = &results_512[slot + 32];

            if lower.score != upper.score || lower.te != upper.te || lower.qe != upper.qe {
                upper_mismatches += 1;
                eprintln!(
                    "UPPER MISMATCH: slot {} (score={}, te={}, qe={}) vs slot {} (score={}, te={}, qe={})",
                    slot,
                    lower.score,
                    lower.te,
                    lower.qe,
                    slot + 32,
                    upper.score,
                    upper.te,
                    upper.qe
                );
            }
        }

        eprintln!("\n=== Diverse Batch Test Summary ===");
        eprintln!("AVX-512 vs AVX2 mismatches (slots 0-31): {}/32", mismatches);
        eprintln!(
            "AVX-512 lower vs upper half mismatches: {}/32",
            upper_mismatches
        );

        assert_eq!(
            mismatches, 0,
            "AVX-512 vs AVX2 produced {} mismatches!",
            mismatches
        );
        assert_eq!(
            upper_mismatches, 0,
            "AVX-512 upper half (32-63) produced {} mismatches vs lower half!",
            upper_mismatches
        );
    }
}

// === SoA entry point (adapter-first) ===
use crate::generate_ksw_entry_soa;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
generate_ksw_entry_soa!(
    name = kswv_batch64_soa,
    callee = batch_ksw_align_avx512,
    width = 64,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw",
);
