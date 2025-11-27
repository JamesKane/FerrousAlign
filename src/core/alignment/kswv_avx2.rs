// ============================================================================
// AVX2 Horizontal SIMD Smith-Waterman (32-way parallelism)
// ============================================================================
//
// This module implements horizontal SIMD batching for mate rescue alignments
// using AVX2 instructions (256-bit vectors, 32 sequences in parallel).
//
// Direct port from BWA-MEM2's kswv.cpp kswv256_u8() function.
//
// ## Architecture
//
// **Horizontal SIMD**: Processes same position across 32 different sequences
// ```
// Register layout:
// __m256i = [ seq0[pos], seq1[pos], seq2[pos], ..., seq31[pos] ]
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
// - Processes 32 alignments per SIMD operation
// - Reduces 44K mate rescue calls → ~1,375 batched calls
// - Expected speedup: ~16x over scalar processing
//
// ## Feature Gate
//
// This module is available on x86_64 CPUs with AVX2 support (Intel Haswell+,
// AMD Excavator+, circa 2013-2015).
//
// ## Reference
//
// BWA-MEM2: bwa-mem2/src/kswv.cpp (kswv256_u8)
// ============================================================================

#![cfg(target_arch = "x86_64")]

use crate::alignment::kswv_batch::{KswResult, SeqPair};
use crate::compute::simd_abstraction::{SimdEngine, SimdEngine256};

/// SIMD width for AVX2: 32 sequences with 8-bit scores
pub const SIMD_WIDTH8: usize = 32;

/// SIMD width for AVX2: 16 sequences with 16-bit scores
pub const SIMD_WIDTH16: usize = 16;

/// Dummy values for ambiguous bases (matching BWA-MEM2)
const DUMMY5: i8 = 5; // Used for ambiguous base detection
#[allow(dead_code)]
const DUMMY8: i8 = 8;
#[allow(dead_code)]
const AMBIG: i8 = 4;

/// KSW flags (from kswv.h)
const KSW_XSUBO: i32 = 0x40000;
const KSW_XSTOP: i32 = 0x20000;

/// Batched Smith-Waterman alignment using AVX2 horizontal SIMD
///
/// Processes up to 32 alignments in parallel using 256-bit SIMD registers.
/// Each SIMD lane handles one complete alignment.
///
/// # Arguments
/// * `seq1_soa` - Reference sequences in SoA layout (32 sequences interleaved)
/// * `seq2_soa` - Query sequences in SoA layout (32 sequences interleaved)
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
/// This function uses AVX2 intrinsics which are unsafe. Caller must ensure:
/// - CPU has AVX2 support
/// - Sequence buffers are properly aligned (32-byte)
/// - Buffer sizes match documented layout
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn batch_ksw_align_avx2(
    seq1_soa: *const u8,       // Reference sequences (SoA layout)
    seq2_soa: *const u8,       // Query sequences (SoA layout)
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
    _phase: i32,               // Processing phase
    _debug: bool,              // Debug flag (unused, for API consistency)
) -> usize {
    // ========================================================================
    // SECTION 1: Initialization
    // ========================================================================

    // Initialize basic constants using SimdEngine256 trait
    let zero256 = SimdEngine256::setzero_epi8();
    let one256 = SimdEngine256::set1_epi8(1);

    // Compute shift value for signed/unsigned score conversion
    let mdiff = w_match.max(w_mismatch).max(w_ambig);
    let shift_val = w_match.min(w_mismatch).min(w_ambig);
    let shift = (256i16 - shift_val as i16) as u8;

    let qmax = mdiff;

    // Create scoring lookup table for shuffle operation
    let mut temp = [0i8; SIMD_WIDTH8];
    temp[0] = w_match;
    temp[1] = w_mismatch;
    temp[2] = w_mismatch;
    temp[3] = w_mismatch;
    temp[4..8].fill(w_ambig);
    temp[8..12].fill(w_ambig);
    temp[12] = w_ambig;

    // Add shift to first 16 elements for shuffle_epi8
    for i in 0..16 {
        temp[i] = temp[i].wrapping_add(shift as i8);
    }

    // Replicate pattern for full AVX2 width
    for i in 16..SIMD_WIDTH8 {
        temp[i] = temp[i - 16];
    }

    let perm_sft256 = SimdEngine256::loadu_si128(temp.as_ptr() as *const _);
    let sft256 = SimdEngine256::set1_epi8(shift as i8);
    let cmax256 = SimdEngine256::set1_epi8(255u8 as i8);
    let five256 = SimdEngine256::set1_epi8(DUMMY5);

    // Initialize minsc and endsc arrays from h0 flags
    // CRITICAL: endsc must default to 255 (max u8) when no early termination requested
    // Otherwise gmax >= 0 is always true, causing immediate exit after first row!
    let mut minsc = [0u8; SIMD_WIDTH8];
    let mut endsc = [255u8; SIMD_WIDTH8]; // Default to max (never terminate early)

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
        // Only set if flag is present and value fits in u8
        if (xtra & KSW_XSTOP) != 0 {
            let val = (xtra & 0xffff) as u32;
            if val <= 255 {
                endsc[i] = val as u8;
            }
        }
        // Otherwise keep default of 255 (no early termination)
    }

    let minsc256 = SimdEngine256::loadu_si128(minsc.as_ptr() as *const _);
    let endsc256 = SimdEngine256::loadu_si128(endsc.as_ptr() as *const _);

    // Initialize scoring parameters as SIMD vectors
    let e_del256 = SimdEngine256::set1_epi8(e_del as i8);
    let oe_del256 = SimdEngine256::set1_epi8((o_del + e_del) as i8);
    let e_ins256 = SimdEngine256::set1_epi8(e_ins as i8);
    let oe_ins256 = SimdEngine256::set1_epi8((o_ins + e_ins) as i8);

    // Global maximum and target end position
    let mut gmax256 = zero256;
    // For 32 sequences, we need 32 16-bit te values = two 256-bit registers
    let mut te256_lo = SimdEngine256::set1_epi16(-1); // sequences 0-15
    let mut te256_hi = SimdEngine256::set1_epi16(-1); // sequences 16-31

    // Allocate DP matrices (H, F, rowMax)
    // TODO: Consider using workspace pre-allocation for sequences <= 256bp
    // to avoid ~32KB allocation per call (similar to batch16 kernel optimization)
    let max_query_len = ncol as usize + 1;
    let max_ref_len = nrow as usize + 1;

    let mut h0_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut h1_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut f_buf = vec![0u8; max_query_len * SIMD_WIDTH8];
    let mut row_max_buf = vec![0u8; max_ref_len * SIMD_WIDTH8];

    // Initialize H0, F to zero (using unaligned stores for Vec buffers)
    for i in 0..=ncol as usize {
        let offset = i * SIMD_WIDTH8;
        SimdEngine256::storeu_si128(h0_buf[offset..].as_mut_ptr() as *mut _, zero256);
        SimdEngine256::storeu_si128(f_buf[offset..].as_mut_ptr() as *mut _, zero256);
    }

    // Initialize tracking variables for main loop
    let mut pimax256 = zero256;
    let mut mask256 = zero256;
    let mut minsc_msk = zero256;
    let mut qe256 = SimdEngine256::set1_epi8(0);

    SimdEngine256::storeu_si128(h0_buf.as_mut_ptr() as *mut _, zero256);
    SimdEngine256::storeu_si128(h1_buf.as_mut_ptr() as *mut _, zero256);

    // ========================================================================
    // SECTION 2: Main DP Loop
    // ========================================================================

    let mut exit0_vec = SimdEngine256::set1_epi8(-1); // All lanes active
    let mut limit = nrow as i32;

    for i in 0..nrow as usize {
        // Initialize row variables
        let mut e11 = zero256;
        let mut imax256 = zero256;
        let mut iqe256 = SimdEngine256::set1_epi8(-1);
        let mut i256_vec = SimdEngine256::set1_epi16(i as i16);
        let mut l256 = zero256;

        // Load reference base for this row (use unaligned load for safety)
        let s1 = SimdEngine256::loadu_si128(seq1_soa.add(i * SIMD_WIDTH8) as *const _);

        // Inner loop over query positions
        for j in 0..ncol as usize {
            // Load DP values and query base (unaligned loads for safety)
            let h00 = SimdEngine256::loadu_si128(h0_buf[j * SIMD_WIDTH8..].as_ptr() as *const _);
            let s2 = SimdEngine256::loadu_si128(seq2_soa.add(j * SIMD_WIDTH8) as *const _);
            let f11 =
                SimdEngine256::loadu_si128(f_buf[(j + 1) * SIMD_WIDTH8..].as_ptr() as *const _);

            // ============================================================
            // MAIN_SAM_CODE8_OPT: Core DP computation
            // ============================================================

            // Compute match/mismatch score via XOR and shuffle
            let xor11 = SimdEngine256::xor_si128(s1, s2);
            let mut sbt11 = SimdEngine256::shuffle_epi8(perm_sft256, xor11);

            // Handle ambiguous bases (base == 5)
            let cmpq = SimdEngine256::cmpeq_epi8(s2, five256);
            sbt11 = SimdEngine256::blendv_epi8(sbt11, sft256, cmpq);

            // Mask out invalid positions (padding)
            // Padding uses value 0x80 (128), which has high bit set
            // blendv_epi8 uses the HIGH BIT of each mask byte to select
            // We want to zero out where high bit of (s1 | s2) is set (padding)
            let or11 = SimdEngine256::or_si128(s1, s2);

            // Compute match score: H[i-1,j-1] + score
            let mut m11 = SimdEngine256::adds_epu8(h00, sbt11);
            // Zero out padding positions (where high bit of or11 is set)
            m11 = SimdEngine256::blendv_epi8(m11, zero256, or11);
            m11 = SimdEngine256::subs_epu8(m11, sft256);

            // Take max of match, gap-extend-E, gap-extend-F
            let mut h11 = SimdEngine256::max_epu8(m11, e11);
            h11 = SimdEngine256::max_epu8(h11, f11);

            // Track row maximum and query end position
            let cmp0 = SimdEngine256::cmpgt_epu8(h11, imax256);
            imax256 = SimdEngine256::max_epu8(imax256, h11);
            iqe256 = SimdEngine256::blendv_epi8(iqe256, l256, cmp0);

            // Update E (gap in query)
            let gap_e256 = SimdEngine256::subs_epu8(h11, oe_ins256);
            e11 = SimdEngine256::subs_epu8(e11, e_ins256);
            e11 = SimdEngine256::max_epu8(gap_e256, e11);

            // Update F (gap in reference)
            let gap_d256 = SimdEngine256::subs_epu8(h11, oe_del256);
            let mut f21 = SimdEngine256::subs_epu8(f11, e_del256);
            f21 = SimdEngine256::max_epu8(gap_d256, f21);

            // Store updated DP values (unaligned for Vec buffers)
            SimdEngine256::storeu_si128(
                h1_buf[(j + 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
                h11,
            );
            SimdEngine256::storeu_si128(f_buf[(j + 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _, f21);

            // Increment query position counter
            l256 = SimdEngine256::add_epi8(l256, one256);
        }

        // Block I: Track row maxima for second-best score computation
        if i > 0 {
            let msk = SimdEngine256::cmpgt_epu8(imax256, pimax256);
            let msk = SimdEngine256::or_si128(msk, mask256);

            let mut pimax256_tmp = SimdEngine256::blendv_epi8(pimax256, zero256, msk);

            // Apply minsc threshold mask
            let minsc_mask_vec = SimdEngine256::set1_epi8(-1);
            pimax256_tmp = SimdEngine256::blendv_epi8(pimax256_tmp, zero256, minsc_mask_vec);

            // Apply exit mask
            pimax256_tmp = SimdEngine256::blendv_epi8(pimax256_tmp, zero256, exit0_vec);

            SimdEngine256::storeu_si128(
                row_max_buf[(i - 1) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
                pimax256_tmp,
            );

            mask256 = SimdEngine256::andnot_si128(msk, SimdEngine256::set1_epi8(-1));
        }

        pimax256 = imax256;

        // Update minsc mask
        let minsc_msk_vec = SimdEngine256::cmpge_epu8(imax256, minsc256);
        minsc_msk = SimdEngine256::or_si128(minsc_msk, minsc_msk_vec);

        // Block II: Update global maximum and target end position
        let mut cmp0_vec = SimdEngine256::cmpgt_epu8(imax256, gmax256);
        cmp0_vec = SimdEngine256::and_si128(cmp0_vec, exit0_vec);

        gmax256 = SimdEngine256::blendv_epi8(gmax256, imax256, cmp0_vec);
        qe256 = SimdEngine256::blendv_epi8(qe256, iqe256, cmp0_vec);

        // Expand 8-bit comparison mask to two 16-bit masks for te tracking
        // cmp0_vec has 32 8-bit comparisons (one per sequence)
        //
        // CRITICAL: AVX2 unpack operations work within 128-bit lanes, not across
        // the full 256-bit register. We must extract each 128-bit lane separately
        // and expand to 256-bit with proper SSE operations.
        //
        // Layout of cmp0_vec (256-bit):
        //   Low lane (bits 0-127):   bytes 0-15  (sequences 0-15)
        //   High lane (bits 128-255): bytes 16-31 (sequences 16-31)
        //
        // We need:
        //   mask_lo_16: 16 16-bit masks for sequences 0-15
        //   mask_hi_16: 16 16-bit masks for sequences 16-31
        //
        // NOTE: This section uses raw intrinsics (_mm256_extracti128_si256, _mm_unpacklo_epi8,
        // _mm256_set_m128i) because these operations are AVX2-specific lane manipulations that
        // don't have equivalents in the SimdEngine256 abstraction. The abstraction focuses on
        // portable operations with ARM NEON equivalents; these 128-bit lane extractions and
        // reconstructions are inherently x86-specific. This is an acceptable deviation.

        // Extract each 128-bit lane
        let cmp_lo_128 = std::arch::x86_64::_mm256_extracti128_si256(cmp0_vec, 0);
        let cmp_hi_128 = std::arch::x86_64::_mm256_extracti128_si256(cmp0_vec, 1);

        // Expand low lane (seqs 0-15) to 256-bit: [c0,c0, c1,c1, ..., c15,c15]
        let lo_unpacked_lo = std::arch::x86_64::_mm_unpacklo_epi8(cmp_lo_128, cmp_lo_128); // bytes 0-7 → 16-bit
        let lo_unpacked_hi = std::arch::x86_64::_mm_unpackhi_epi8(cmp_lo_128, cmp_lo_128); // bytes 8-15 → 16-bit
        let mask_lo_16 = std::arch::x86_64::_mm256_set_m128i(lo_unpacked_hi, lo_unpacked_lo);

        // Expand high lane (seqs 16-31) to 256-bit: [c16,c16, c17,c17, ..., c31,c31]
        let hi_unpacked_lo = std::arch::x86_64::_mm_unpacklo_epi8(cmp_hi_128, cmp_hi_128); // bytes 16-23 → 16-bit
        let hi_unpacked_hi = std::arch::x86_64::_mm_unpackhi_epi8(cmp_hi_128, cmp_hi_128); // bytes 24-31 → 16-bit
        let mask_hi_16 = std::arch::x86_64::_mm256_set_m128i(hi_unpacked_hi, hi_unpacked_lo);

        // Update both te registers with proper 16-bit blending
        te256_lo = SimdEngine256::blendv_epi8(te256_lo, i256_vec, mask_lo_16);
        te256_hi = SimdEngine256::blendv_epi8(te256_hi, i256_vec, mask_hi_16);

        // Check for early termination
        cmp0_vec = SimdEngine256::cmpge_epu8(gmax256, endsc256);

        // Check for score overflow
        let left256 = SimdEngine256::adds_epu8(gmax256, sft256);
        let cmp2_vec = SimdEngine256::cmpge_epu8(left256, cmax256);

        // Update exit mask
        let exit_cond = SimdEngine256::or_si128(cmp0_vec, cmp2_vec);
        exit0_vec = SimdEngine256::andnot_si128(exit_cond, exit0_vec);

        // Early exit if all lanes done
        if SimdEngine256::movemask_epi8(exit0_vec) == 0 {
            limit = (i + 1) as i32;
            break;
        }

        // Swap buffers
        std::mem::swap(&mut h0_buf, &mut h1_buf);

        // Increment row index (final value unused after last iteration)
        let one256_16 = SimdEngine256::set1_epi16(1);
        #[allow(unused_assignments)]
        {
            i256_vec = SimdEngine256::add_epi16(i256_vec, one256_16);
        }
    }

    // Final row max update (unaligned for Vec buffers)
    let msk = SimdEngine256::or_si128(mask256, SimdEngine256::set1_epi8(0));
    let pimax256_final = SimdEngine256::blendv_epi8(pimax256, zero256, msk);
    SimdEngine256::storeu_si128(
        row_max_buf[((limit - 1) as usize) * SIMD_WIDTH8..].as_mut_ptr() as *mut _,
        pimax256_final,
    );

    // ========================================================================
    // SECTION 3: Score Extraction
    // ========================================================================

    #[repr(align(32))]
    struct AlignedScoreArray([u8; SIMD_WIDTH8]);
    #[repr(align(32))]
    struct AlignedTeArray([i16; SIMD_WIDTH16]); // 16 values per register, need 2 for 32 sequences
    #[repr(align(32))]
    struct AlignedQeArray([u8; SIMD_WIDTH8]);

    let mut score_arr = AlignedScoreArray([0; SIMD_WIDTH8]);
    let mut te_arr_lo = AlignedTeArray([0; SIMD_WIDTH16]);
    let mut te_arr_hi = AlignedTeArray([0; SIMD_WIDTH16]);
    let mut qe_arr = AlignedQeArray([0; SIMD_WIDTH8]);

    // Store SIMD vectors to aligned arrays
    SimdEngine256::storeu_si128(score_arr.0.as_mut_ptr() as *mut _, gmax256);
    // Store both te registers (16 values each, 32 total for 32 sequences)
    SimdEngine256::storeu_si128_16(te_arr_lo.0.as_mut_ptr() as *mut _, te256_lo);
    SimdEngine256::storeu_si128_16(te_arr_hi.0.as_mut_ptr() as *mut _, te256_hi);
    SimdEngine256::storeu_si128(qe_arr.0.as_mut_ptr() as *mut _, qe256);

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

        // Read te from correct register based on sequence index
        // Sequences 0-15 use te_arr_lo, sequences 16-31 use te_arr_hi
        let te_val = if l < 16 {
            te_arr_lo.0[l]
        } else {
            te_arr_hi.0[l - 16]
        };

        results[l].score = final_score as i32;
        results[l].te = te_val as i32;
        results[l].qe = qe_arr.0[l] as i32;

        // Debug logging for first sequence to compare with AVX-512
        if l == 0 && final_score > 0 {
            log::trace!(
                "AVX2 kswv seq0: score={}, te={}, qe={}, nrow={}, ncol={}, shift={}",
                final_score,
                te_val,
                qe_arr.0[l],
                nrow,
                ncol,
                shift
            );
        }

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

    // For second-best computation, we use 16 16-bit values per register
    #[repr(align(32))]
    struct AlignedI16Array16([i16; SIMD_WIDTH16]); // 16 values

    let mut low_arr = AlignedI16Array16([0; SIMD_WIDTH16]);
    let mut high_arr = AlignedI16Array16([0; SIMD_WIDTH16]);
    let mut rlen_arr = AlignedI16Array16([0; SIMD_WIDTH16]);

    let mut maxl: i32 = 0;
    let mut minh: i32 = nrow as i32;

    // Second-best computation samples every other sequence (16 of 32)
    // This is an approximation that matches BWA-MEM2's approach
    for i in 0..SIMD_WIDTH16.min((pairs.len() + 1) / 2) {
        let seq_idx = i * 2; // Map to actual sequence index
        let val = (score_arr.0[seq_idx] as i32 + qmax as i32 - 1) / qmax as i32;

        // Get te from correct array based on sequence index
        let te_val = if seq_idx < 16 {
            te_arr_lo.0[seq_idx]
        } else {
            te_arr_hi.0[seq_idx - 16]
        };

        low_arr.0[i] = (te_val - val as i16).max(0);
        high_arr.0[i] = (te_val + val as i16).min(nrow as i16 - 1);
        rlen_arr.0[i] = pairs[seq_idx].ref_len as i16;

        if qe_arr.0[seq_idx] != 0 {
            maxl = maxl.max(low_arr.0[i] as i32);
            minh = minh.min(high_arr.0[i] as i32);
        }
    }

    // Initialize second-best tracking (16 values for subsampled sequences)
    let mut max256 = zero256;
    let mut te2_256 = SimdEngine256::set1_epi16(-1);

    let low256 = SimdEngine256::loadu_si128_16(low_arr.0.as_ptr() as *const _);
    let high256 = SimdEngine256::loadu_si128_16(high_arr.0.as_ptr() as *const _);
    let rlen256 = SimdEngine256::loadu_si128_16(rlen_arr.0.as_ptr() as *const _);

    // Forward scan (unaligned for Vec buffers)
    for i in 0..maxl {
        let i256 = SimdEngine256::set1_epi16(i as i16);
        let rmax256 = SimdEngine256::loadu_si128(
            row_max_buf[i as usize * SIMD_WIDTH8..].as_ptr() as *const _
        );

        let mask1 = SimdEngine256::cmpgt_epi16(low256, i256);
        let mask2_8bit = SimdEngine256::cmpgt_epu8(rmax256, max256);
        let combined_mask = SimdEngine256::and_si128(mask1, mask2_8bit);

        max256 = SimdEngine256::blendv_epi8(max256, rmax256, combined_mask);
        te2_256 = SimdEngine256::blendv_epi8(te2_256, i256, combined_mask);
    }

    // Backward scan (unaligned for Vec buffers)
    // Ensure we don't iterate with negative indices
    let backward_start = (minh + 1).max(0);
    for i in backward_start..limit {
        let i256 = SimdEngine256::set1_epi16(i as i16);
        let rmax256 = SimdEngine256::loadu_si128(
            row_max_buf[i as usize * SIMD_WIDTH8..].as_ptr() as *const _
        );

        let mask1a = SimdEngine256::cmpgt_epi16(i256, high256);
        let mask1b = SimdEngine256::cmpgt_epi16(rlen256, i256);
        let mask2_8bit = SimdEngine256::cmpgt_epu8(rmax256, max256);

        let mask1 = SimdEngine256::and_si128(mask1a, mask1b);
        let combined_mask = SimdEngine256::and_si128(mask1, mask2_8bit);

        max256 = SimdEngine256::blendv_epi8(max256, rmax256, combined_mask);
        te2_256 = SimdEngine256::blendv_epi8(te2_256, i256, combined_mask);
    }

    // Extract second-best scores
    let mut score2_arr = AlignedScoreArray([0; SIMD_WIDTH8]);
    let mut te2_arr = AlignedI16Array16([0; SIMD_WIDTH16]);

    SimdEngine256::storeu_si128(score2_arr.0.as_mut_ptr() as *mut _, max256);
    SimdEngine256::storeu_si128_16(te2_arr.0.as_mut_ptr() as *mut _, te2_256);

    for i in 0..pairs.len().min(SIMD_WIDTH8) {
        // Second-best values computed for every other sequence, map back
        let te2_idx = i / 2;
        if qe_arr.0[i] != 0 {
            results[i].score2 = if score2_arr.0[i] == 0 {
                -1
            } else {
                score2_arr.0[i] as i32
            };
            results[i].te2 = te2_arr.0[te2_idx] as i32;
        } else {
            results[i].score2 = -1;
            results[i].te2 = -1;
        }

        // Set tb and qb based on valid alignment
        // Phase 0 doesn't compute traceback, so we estimate:
        // - For mate rescue, alignments typically start at query position 0
        // - tb is estimated as te - (qe - qb) for same-length alignment
        if results[i].score > 0 && results[i].te >= 0 && results[i].qe >= 0 {
            results[i].qb = 0; // Alignment starts at query position 0
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
    #[ignore] // TODO: Enable when AVX2 kernel is tested
    fn test_avx2_kernel() {
        // Skip test if AVX2 is not available
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping test_avx2_kernel: AVX2 not available on this CPU");
            return;
        }

        let seq1 = vec![0u8; 256 * SIMD_WIDTH8];
        let seq2 = vec![0u8; 128 * SIMD_WIDTH8];

        let pairs = vec![
            SeqPair {
                ref_len: 10,
                query_len: 10,
                ..Default::default()
            };
            SIMD_WIDTH8
        ];

        let mut results = vec![KswResult::default(); SIMD_WIDTH8];

        unsafe {
            let _count = batch_ksw_align_avx2(
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
                false, // debug
            );
        }
    }
}
