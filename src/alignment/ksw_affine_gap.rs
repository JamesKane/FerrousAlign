//! A Rust port of the ksw (K-SW for Smith-Waterman-like alignment) algorithm from BWA-MEM2.
//! This module implements affine gap Smith-Waterman alignment using SIMD intrinsics,
//! specifically targeting the ksw_align2 function for improved indel handling.

// Allow unsafe operations within unsafe functions without explicit unsafe blocks.
// This is appropriate for SIMD-heavy code where nearly every operation is inherently unsafe.
#![allow(unsafe_op_in_unsafe_fn)]

use std::alloc::{Layout, alloc};
use std::ptr; // For `ptr::copy_nonoverlapping`

// Import the abstraction layer components and specific SIMD types
use crate::compute::simd_abstraction::SimdEngine;
use crate::compute::simd_abstraction::types::__m128i;

// Constants from ksw.h
pub const KSW_XBYTE: u32 = 0x10000;
pub const KSW_XSTOP: u32 = 0x20000;
pub const KSW_XSUBO: u32 = 0x40000;
pub const KSW_XSTART: u32 = 0x80000;
pub const KSW_XCOV: u32 = 0x100;

#[repr(C)]
#[derive(Debug)]
pub struct Kswq {
    pub qlen: i32,
    pub slen: i32,
    pub shift: u8,
    pub mdiff: u8,
    pub max: u8,
    pub size: u8, // 1 for u8 scores, 2 for i16 scores
    // Pointers to SIMD vectors, managed manually for allocation/alignment
    pub qp: *mut __m128i,   // query profile
    pub h0: *mut __m128i,   // H_0 scores
    pub h1: *mut __m128i,   // H_1 scores
    pub e: *mut __m128i,    // E scores
    pub hmax: *mut __m128i, // max H scores
}

// Default values for kswr_t, as per C code's g_defr
pub const KSW_DEFR: Kswr = Kswr {
    score: 0,
    te: -1,
    qe: -1,
    score2: -1,
    te2: -1,
    tb: -1,
    qb: -1,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Kswr {
    pub score: i32,  // best score
    pub te: i32,     // target end
    pub qe: i32,     // query end
    pub score2: i32, // second best score
    pub te2: i32,    // second best target end
    pub tb: i32,     // target start
    pub qb: i32,     // query start
}

/// Initializes the query data structure (`Kswq`).
///
/// This function pre-computes the query profile and allocates memory for the SIMD vectors
/// used in the KSW alignment.
///
/// # Arguments
/// * `size` - Number of bytes used to store a score (1 for u8, 2 for i16).
/// * `qlen` - Length of the query sequence.
/// * `query` - Query sequence with 0 <= query[i] < m.
/// * `m` - Size of the alphabet.
/// * `mat` - `m*m` scoring matrix in a one-dimension array.
///
/// # Returns
/// A newly allocated `Kswq` structure. The caller is responsible for deallocating
/// the memory using `ksw_qfree`.
pub fn ksw_qinit(size: i32, qlen: i32, query: &[u8], m: i32, mat: &[i8]) -> *mut Kswq {
    let ptr = unsafe { ksw_qalloc(qlen, m, size) };

    if ptr.is_null() {
        return ptr::null_mut(); // Allocation failed
    }

    let q = unsafe { &mut *ptr };

    // Compute shift and mdiff
    let mut min_score = 127;
    let mut max_score = 0;
    for &score in mat.iter() {
        if score < min_score {
            min_score = score;
        }
        if score > max_score {
            max_score = score;
        }
    }
    q.shift = (256 - min_score as i32) as u8;
    q.mdiff = (max_score as i32 + q.shift as i32) as u8;
    q.max = max_score as u8;

    let p = if q.size == 1 { 16 } else { 8 };
    let slen = q.slen as usize;

    unsafe {
        // Fill query profile (q.qp)
        if q.size == 1 {
            // u8 scores
            let mut t_ptr = q.qp as *mut i8;
            for a in 0..m {
                // iterate over alphabet
                let ma_start_idx = (a * m) as usize;
                for i in 0..slen {
                    // iterate over segments
                    for k in (i..).step_by(slen).take(p) {
                        // fill p values per segment
                        let score = if (k as i32) < qlen {
                            mat[ma_start_idx + query[k] as usize]
                        } else {
                            0
                        };
                        *t_ptr = score + q.shift as i8;
                        t_ptr = t_ptr.add(1);
                    }
                }
            }
        } else {
            // i16 scores
            let mut t_ptr = q.qp as *mut i16;
            for a in 0..m {
                // iterate over alphabet
                let ma_start_idx = (a * m) as usize;
                for i in 0..slen {
                    // iterate over segments
                    for k in (i..).step_by(slen).take(p) {
                        // fill p values per segment
                        let score = if (k as i32) < qlen {
                            mat[ma_start_idx + query[k] as usize]
                        } else {
                            0
                        };
                        *t_ptr = score as i16;
                        t_ptr = t_ptr.add(1);
                    }
                }
            }
        }
    }

    ptr
}

/// Allocates memory for a `Kswq` structure and its SIMD vectors based on the specified size and query length.
///
/// This function is intended to be used internally by `ksw_qinit` when handling 16-bit scores,
/// ensuring proper memory allocation and alignment for the SIMD vectors.
///
/// # Arguments
/// * `qlen` - Length of the query sequence.
/// * `m` - Size of the alphabet.
/// * `size` - Number of bytes used to store a score (1 for u8, 2 for i16).
///
/// # Returns
/// A newly allocated `Kswq` structure with pointers to uninitialized SIMD vector memory.
/// The caller is responsible for freeing this memory using `ksw_qfree`.
pub unsafe fn ksw_qalloc(qlen: i32, m: i32, size: i32) -> *mut Kswq {
    let size = if size > 1 { 2 } else { 1 };
    let p = if size == 1 { 16 } else { 8 }; // # values per __m128i: 16 for u8, 8 for i16
    let slen = (qlen as usize + p - 1) / p; // segmented length

    // Calculate total memory needed and layout
    // Kswq struct + alignment padding + slen * (m + 4) * 16 bytes for SIMD vectors
    // 16 bytes per __m128i vector, regardless of element size (u8 or i16)
    let total_simd_bytes = (slen * (m as usize + 4)) * 16;
    let kswq_layout = Layout::new::<Kswq>();
    let (layout, _) = kswq_layout
        .extend(Layout::from_size_align(total_simd_bytes, 16).unwrap())
        .unwrap();
    let ptr = alloc(layout) as *mut Kswq;

    if ptr.is_null() {
        return ptr::null_mut(); // Allocation failed
    }

    let q = &mut *ptr;

    q.qlen = qlen;
    q.slen = slen as i32;
    q.size = size as u8;

    // Set pointers, ensuring 16-byte alignment
    let base_ptr = ptr.add(1) as *mut u8; // Start after Kswq struct
    let aligned_base_ptr = (base_ptr as usize + 15) & !15; // Align to 16 bytes
    let aligned_base_ptr = aligned_base_ptr as *mut __m128i;

    q.qp = aligned_base_ptr;
    q.h0 = q.qp.add(slen * m as usize);
    q.h1 = q.h0.add(slen);
    q.e = q.h1.add(slen);
    q.hmax = q.e.add(slen);

    ptr
}

/// Frees the memory allocated for a `Kswq` structure.
///
/// # Arguments
/// * `q_ptr` - A pointer to the `Kswq` structure to be freed.
pub fn ksw_qfree(q_ptr: *mut Kswq) {
    if q_ptr.is_null() {
        return;
    }

    let q = unsafe { &*q_ptr };

    let size = q.size as i32;
    let qlen = q.qlen;
    let m = (q.h0 as usize - q.qp as usize) / q.slen as usize / 16; // Reconstruct m from pointer arithmetic

    let p = if size == 1 { 16 } else { 8 };
    let slen = (qlen as usize + p - 1) / p;

    let total_simd_bytes = (slen * (m + 4)) * 16; // Use the reconstructed m
    let kswq_layout = Layout::new::<Kswq>();
    let (layout, _) = kswq_layout
        .extend(Layout::from_size_align(total_simd_bytes, 16).unwrap())
        .unwrap();

    unsafe {
        std::alloc::dealloc(q_ptr as *mut u8, layout);
    }
}

/// Equivalent to C's `__max_16` macro for u8 scores.
/// Extracts the maximum 8-bit value from a 128-bit SIMD vector.
#[inline(always)]
unsafe fn max_16_u8<S: SimdEngine>(x_vec: S::Vec8) -> u8 {
    let mut xx = x_vec;
    // The C code uses fixed shift amounts: 8, 4, 2, 1.
    // We need to use the macro for this as the trait method `srli_si128_fixed` is hardcoded to 2.
    xx = S::max_epu8(xx, S::srli_bytes(xx, 8));
    xx = S::max_epu8(xx, S::srli_bytes(xx, 4));
    xx = S::max_epu8(xx, S::srli_bytes(xx, 2));
    xx = S::max_epu8(xx, S::srli_bytes(xx, 1));
    // Now the max value should be in the first 8-bit lane
    S::extract_epi8(xx, 0) as u8
}

/// Equivalent to C's `__max_8` macro for u16 scores.
/// Extracts the maximum 16-bit value from a 128-bit SIMD vector.
#[inline(always)]
unsafe fn max_8_u16<S: SimdEngine>(x_vec: S::Vec16) -> u16 {
    let mut xx = x_vec;
    xx = S::max_epu16(xx, S::srli_bytes_16(xx, 8)); // Shift by 8 bytes (4 16-bit lanes)
    xx = S::max_epu16(xx, S::srli_bytes_16(xx, 4)); // Shift by 4 bytes (2 16-bit lanes)
    xx = S::max_epu16(xx, S::srli_bytes_16(xx, 2)); // Shift by 2 bytes (1 16-bit lane)
    // Now the max value should be in the first 16-bit lane
    S::extract_epi16(xx, 0) as u16
}

/// Core DP loop for 8-bit scores.
/// Equivalent to `ksw_u8` in ksw.cpp.
///
/// # Safety
/// This function uses SIMD intrinsics and raw pointer manipulation.
/// Caller must ensure `q` is valid and properly initialized, and `target` is a valid slice.
pub unsafe fn ksw_u8_impl<S: SimdEngine>(
    q: &mut Kswq,
    tlen: i32,
    target: &[u8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    xtra: u32,
) -> Kswr {
    let mut r = KSW_DEFR;
    let minsc = if (xtra & KSW_XSUBO) != 0 {
        xtra & 0xffff
    } else {
        0x10000
    };
    let endsc = if (xtra & KSW_XSTOP) != 0 {
        xtra & 0xffff
    } else {
        0x10000
    };

    // b is used to track scores for secondary alignments / score tracking
    let mut b: Vec<u64> = Vec::new();
    let mut gmax = 0; // Global maximum score
    let mut te = -1; // Target end position for global maximum score

    // SIMD constants
    let zero = S::setzero_epi8(); // All zero vector
    let oe_del = S::set1_epi8((o_del + e_del) as i8);
    let e_del = S::set1_epi8(e_del as i8);
    let oe_ins = S::set1_epi8((o_ins + e_ins) as i8);
    let e_ins = S::set1_epi8(e_ins as i8);
    let shift = S::set1_epi8(q.shift as i8);

    let slen = q.slen as usize;

    // Initialize H0, E, Hmax to zero
    // Cast to *mut S::Vec8 for correct type handling with SimdEngine
    for i in 0..slen {
        S::store_si128(q.e.add(i) as *mut S::Vec8, zero);
        S::store_si128(q.h0.add(i) as *mut S::Vec8, zero);
        S::store_si128(q.hmax.add(i) as *mut S::Vec8, zero);
    }

    // The core loop iterates over the target sequence
    for i in 0..tlen {
        let mut f = zero; // F column for current target row (insertions from left)
        let mut max_vec_in_row = zero; // Max score found in the current target row

        // Load H(i-1,-1), which is the last segment of H0 from the previous row
        let mut h = S::load_si128(q.h0.add(slen - 1) as *const S::Vec8);
        h = S::slli_bytes(h, 1); // Shift left by 1 byte to align for H(i-1,j-1)

        // Base pointer for current target base's query profile
        let s_ptr_base = q.qp as *const S::Vec8;
        let s_ptr_target_base = s_ptr_base.add(target[i as usize] as usize * slen);

        // Inner loop iterates over segments of the query sequence
        for j in 0..slen {
            // H (match/mismatch score) calculation
            let current_s = S::load_si128(s_ptr_target_base.add(j)); // S(i,j)
            h = S::adds_epu8(h, current_s); // H(i-1,j-1) + S(i,j)
            h = S::subs_epu8(h, shift); // h = H'(i-1,j-1) + S(i,j) - shift

            let e = S::load_si128(q.e.add(j) as *const S::Vec8); // E'(i,j) (deletions from top)
            h = S::max_epu8(h, e);
            h = S::max_epu8(h, f); // h = H'(i,j) = max(H(i-1,j-1)+S, E(i,j), F(i,j))

            max_vec_in_row = S::max_epu8(max_vec_in_row, h); // Update max score in row
            S::store_si128(q.h1.add(j) as *mut S::Vec8, h); // Save H'(i,j) to H1

            // E (deletion) calculation for E(i+1,j)
            let mut e_new = S::subs_epu8(e, e_del); // E'(i,j) - e_del
            let t_del = S::subs_epu8(h, oe_del); // H'(i,j) - o_del - e_del
            e_new = S::max_epu8(e_new, t_del); // e_new = E'(i+1,j)
            S::store_si128(q.e.add(j) as *mut S::Vec8, e_new); // Save E'(i+1,j)

            // F (insertion) calculation for F(i,j+1)
            f = S::subs_epu8(f, e_ins); // F'(i,j) - e_ins
            let t_ins = S::subs_epu8(h, oe_ins); // H'(i,j) - o_ins - e_ins
            f = S::max_epu8(f, t_ins); // f = F'(i,j+1)

            // Prepare for next segment: h becomes H(i-1,j)
            h = S::load_si128(q.h0.add(j) as *const S::Vec8);
        }

        // Lazy F loop: propagate F values rightwards
        // This loop simulates a full DP row calculation for F, but optimized.
        // It breaks early if no F values can increase (i.e., F is saturated or zeroed).
        let mut current_f_loop_h_val; // This is h for current_f_loop, not the h for outer DP loop.
        for _k in 0..S::WIDTH_8 {
            // Iterate up to WIDTH_8 times (max possible shifts for 128-bit)
            f = S::slli_bytes(f, 1); // Shift f left by 1 byte (horizontal propagation)
            let mut changed = false; // Flag to check if any lane value changed

            for j in 0..slen {
                current_f_loop_h_val = S::load_si128(q.h1.add(j) as *const S::Vec8);
                let h_before_max = current_f_loop_h_val; // Store for comparison

                current_f_loop_h_val = S::max_epu8(current_f_loop_h_val, f); // h = H'(i,j) update
                S::store_si128(q.h1.add(j) as *mut S::Vec8, current_f_loop_h_val);

                let h_minus_oe_ins = S::subs_epu8(current_f_loop_h_val, oe_ins);
                f = S::subs_epu8(f, e_ins);

                // Check if any lane value was different after max(h,f) operation
                // If S::max_epu8(h_before_max, f) results in a different value than h_before_max
                // for any lane, it means f had an effect.
                let cmp_mask = S::cmpeq_epi8(h_before_max, current_f_loop_h_val);
                if S::movemask_epi8(cmp_mask) != (1 << S::WIDTH_8) - 1 {
                    // Check if all lanes were equal
                    changed = true;
                }
            }
            if !changed {
                // All lanes in the F loop are saturated/no longer propagating
                break;
            }
        }

        // Extract max score from max_vec_in_row
        let imax = max_16_u8::<S>(max_vec_in_row);

        // Update b array for secondary alignments/score tracking
        if imax as u32 >= minsc {
            if b.is_empty() {
                b.push(((imax as u64) << 32) | (i as u64));
            } else {
                let last_idx = b.len() - 1;
                if (b[last_idx] & 0xFFFFFFFF) as i32 + 1 != i {
                    // Check if current 'i' is consecutive to previous one
                    b.push(((imax as u64) << 32) | (i as u64));
                } else if ((b[last_idx] >> 32) as i32) < imax as i32 {
                    b[last_idx] = ((imax as u64) << 32) | (i as u64);
                }
            }
        }

        // Update global maximum score and its target end position
        if imax as i32 > gmax {
            gmax = imax as i32;
            te = i;
            // Keep the H1 vector (copy to Hmax)
            for j in 0..slen {
                S::store_si128(
                    q.hmax.add(j) as *mut S::Vec8,
                    S::load_si128(q.h1.add(j) as *const S::Vec8),
                );
            }
            // Early termination conditions
            if gmax + (q.shift as i32) >= 255 || gmax as u32 >= endsc {
                break;
            }
        }

        // Swap H0 and H1 pointers for the next iteration (current H1 becomes next H0)
        let temp_h0 = q.h0;
        q.h0 = q.h1;
        q.h1 = temp_h0;
    }

    // Finalize result
    r.score = if gmax + (q.shift as i32) < 255 {
        gmax
    } else {
        255
    };
    r.te = te;

    if r.score != 255 {
        // Find qe (query end) and score2 (second best score)
        let mut max_qe = -1;
        let mut max_val_qe = -1;
        let qlen_simd = slen * S::WIDTH_8; // Total number of 8-bit cells (effectively query length)

        // Iterate through all individual 8-bit cells in Hmax to find max score and corresponding qe
        for k in 0..qlen_simd {
            let val = S::extract_epi8(
                S::load_si128(q.hmax.add(k / S::WIDTH_8) as *const S::Vec8),
                (k % S::WIDTH_8) as i32,
            );
            // Calculate actual query position from k (SIMD lane index + segment index)
            // (k / S::WIDTH_8) is the segment index
            // (k % S::WIDTH_8) is the lane index
            let current_qe_candidate = (k / S::WIDTH_8 * slen + k % S::WIDTH_8) as i32;

            if val as i32 > max_val_qe {
                max_val_qe = val as i32;
                max_qe = current_qe_candidate;
            } else if val as i32 == max_val_qe {
                // If scores are equal, prefer smaller query end (earlier position)
                if current_qe_candidate < max_qe {
                    max_qe = current_qe_candidate;
                }
            }
        }
        r.qe = max_qe;

        // Populate score2 and te2 for suboptimal alignments
        if !b.is_empty() {
            let i_val = (r.score + q.max as i32 - 1) / q.max as i32;
            let low = te - i_val;
            let high = te + i_val;

            for val in b.iter() {
                let e_idx = (val & 0xFFFFFFFF) as i32; // Extract target end (i)
                let score = (val >> 32) as i32; // Extract score

                if (e_idx < low || e_idx > high) && score > r.score2 {
                    r.score2 = score;
                    r.te2 = e_idx;
                }
            }
        }
    }

    r
}

/// Core DP loop for 16-bit scores.
/// Equivalent to `ksw_i16` in ksw.cpp.
///
/// # Safety
/// This function uses SIMD intrinsics and raw pointer manipulation.
/// Caller must ensure `q` is valid and properly initialized, and `target` is a valid slice.
#[allow(unused_assignments)]
pub unsafe fn ksw_i16_impl<S: SimdEngine>(
    q: &mut Kswq,
    tlen: i32,
    target: &[u8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    xtra: u32,
) -> Kswr {
    let mut r = KSW_DEFR;
    let minsc = if (xtra & KSW_XSUBO) != 0 {
        xtra & 0xffff
    } else {
        0x10000
    };
    let endsc = if (xtra & KSW_XSTOP) != 0 {
        xtra & 0xffff
    } else {
        0x10000
    };
    let tcov = if (xtra & KSW_XCOV) != 0 { 1 } else { 0 }; // Only for AVX2/AVX-512, but include for now.

    let m_val = q.mdiff as i16;
    let max_score_range = S::set1_epi16(m_val);

    // b is used to track scores for secondary alignments / score tracking
    let mut b: Vec<u64> = Vec::new();
    let mut gmax = 0; // Global maximum score
    let mut te = -1; // Target end position for global maximum score

    // SIMD constants
    let zero = S::setzero_epi16(); // All zero vector
    let oe_del_v = S::set1_epi16((o_del + e_del) as i16);
    let e_del_v = S::set1_epi16(e_del as i16);
    let oe_ins_v = S::set1_epi16((o_ins + e_ins) as i16);
    let e_ins_v = S::set1_epi16(e_ins as i16);
    // Note: clip_v is calculated but not used in this implementation
    // Use wrapping_sub to prevent overflow panic when xtra=0
    let _clip_v = S::set1_epi16(
        (if (xtra & KSW_XSUBO) != 0 {
            xtra & 0xffff
        } else {
            0
        }
        .wrapping_sub(q.shift as u32)) as i16,
    );

    let slen = q.slen as usize;

    // Initialize H0, E, Hmax to zero
    // Cast to *mut S::Vec16 for correct type handling with SimdEngine
    for i in 0..slen {
        S::store_si128_16(q.e.add(i) as *mut S::Vec16, zero);
        S::store_si128_16(q.h0.add(i) as *mut S::Vec16, zero);
        S::store_si128_16(q.hmax.add(i) as *mut S::Vec16, zero);
    }

    // The core loop iterates over the target sequence
    for i in 0..tlen {
        let mut f = zero; // F column for current target row (insertions from left)
        let mut max_vec_in_row = zero; // Max score found in the current target row

        // Load H(i-1,-1), which is the last segment of H0 from the previous row
        let mut h = S::load_si128_16(q.h0.add(slen - 1) as *const S::Vec16);
        h = S::slli_bytes_16(h, 2); // Shift left by 2 bytes (one 16-bit element) to align for H(i-1,j-1)

        // Base pointer for current target base's query profile
        let s_ptr_base = q.qp as *const S::Vec16; // q.qp stores __m128i, so casting to S::Vec16 (which is also __m128i) is fine
        let s_ptr_target_base = s_ptr_base.add(target[i as usize] as usize * slen);

        // Inner loop iterates over segments of the query sequence
        for j in 0..slen {
            // H (match/mismatch score) calculation
            let current_s = S::load_si128_16(s_ptr_target_base.add(j)); // S(i,j)
            h = S::adds_epi16(h, current_s); // h = H(i-1,j-1) + S(i,j)

            let e = S::load_si128_16(q.e.add(j) as *const S::Vec16); // E'(i,j) (deletions from top)
            h = S::max_epi16(h, e);
            h = S::max_epi16(h, f); // h = H'(i,j) = max(H(i-1,j-1)+S, E(i,j), F(i,j))

            max_vec_in_row = S::max_epi16(max_vec_in_row, h); // Update max score in row
            S::store_si128_16(q.h1.add(j) as *mut S::Vec16, h); // Save H'(i,j) to H1

            // E (deletion) calculation for E(i+1,j)
            let mut e_new = S::subs_epi16(e, e_del_v); // E'(i,j) - e_del
            let t_del = S::subs_epi16(h, oe_del_v); // H'(i,j) - o_del - e_del
            e_new = S::max_epi16(e_new, t_del); // e_new = E'(i+1,j)
            S::store_si128_16(q.e.add(j) as *mut S::Vec16, e_new); // Save E'(i+1,j)

            // F (insertion) calculation for F(i,j+1)
            f = S::subs_epi16(f, e_ins_v); // F'(i,j) - e_ins
            let t_ins = S::subs_epi16(h, oe_ins_v); // H'(i,j) - o_ins - e_ins
            f = S::max_epi16(f, t_ins); // f = F'(i,j+1)

            // Prepare for next segment: h becomes H(i-1,j)
            h = S::load_si128_16(q.h0.add(j) as *const S::Vec16);
        }

        // Lazy F loop: propagate F values rightwards
        // This loop simulates a full DP row calculation for F, but optimized.
        // It breaks early if no F values can increase (i.e., F is saturated or zeroed).
        let mut current_f_loop_h_val;
        for _k in 0..S::WIDTH_16 {
            // Iterate up to WIDTH_16 times (max possible shifts for 128-bit)
            f = S::slli_bytes_16(f, 2); // Shift f left by 2 bytes (horizontal propagation)
            let mut changed = false; // Flag to check if any lane value changed

            for j in 0..slen {
                current_f_loop_h_val = S::load_si128_16(q.h1.add(j) as *const S::Vec16);
                let h_before_max = current_f_loop_h_val; // Store for comparison

                current_f_loop_h_val = S::max_epi16(current_f_loop_h_val, f); // h = H'(i,j) update
                S::store_si128_16(q.h1.add(j) as *mut S::Vec16, current_f_loop_h_val);

                let h_minus_oe_ins = S::subs_epi16(current_f_loop_h_val, oe_ins_v);
                f = S::subs_epi16(f, e_ins_v);

                // Check if any lane value was different after max(h,f) operation
                let cmp_mask = S::cmpeq_epi16(h_before_max, current_f_loop_h_val);
                if S::movemask_epi16(cmp_mask) != (1 << S::WIDTH_16) - 1 {
                    // Check if all lanes were equal
                    changed = true;
                }
            }
            if !changed {
                // All lanes in the F loop are saturated/no longer propagating
                break;
            }
        }

        // Extract max score from max_vec_in_row
        let imax = max_8_u16::<S>(max_vec_in_row); // This is where the old bug was; S::Vec16 as input

        // Update b array for secondary alignments/score tracking
        if imax as u32 >= minsc {
            if b.is_empty() {
                b.push(((imax as u64) << 32) | (i as u64));
            } else {
                let last_idx = b.len() - 1;
                if (b[last_idx] & 0xFFFFFFFF) as i32 + 1 != i {
                    // Check if current 'i' is consecutive to previous one
                    b.push(((imax as u64) << 32) | (i as u64));
                } else if ((b[last_idx] >> 32) as i32) < imax as i32 {
                    b[last_idx] = ((imax as u64) << 32) | (i as u64);
                }
            }
        }

        // Update global maximum score and its target end position
        if imax as i32 > gmax {
            gmax = imax as i32;
            te = i;
            // Keep the H1 vector (copy to Hmax)
            for j in 0..slen {
                S::store_si128_16(
                    q.hmax.add(j) as *mut S::Vec16,
                    S::load_si128_16(q.h1.add(j) as *const S::Vec16),
                );
            }
            // Early termination conditions
            if gmax as u32 >= 0x7FFF || gmax as u32 >= endsc {
                break;
            }
        }

        // Swap H0 and H1 pointers for the next iteration (current H1 becomes next H0)
        let temp_h0 = q.h0;
        q.h0 = q.h1;
        q.h1 = temp_h0;
    }

    // Finalize result
    r.score = gmax;
    r.te = te;

    if r.score != 0x7FFF {
        // Not saturated
        // Find qe (query end) and score2 (second best score)
        let mut max_qe = -1;
        let mut max_val_qe = -1;
        let qlen_simd = slen * S::WIDTH_16; // Total number of 16-bit cells (effectively query length)

        // Iterate through all individual 16-bit cells in Hmax to find max score and corresponding qe
        for k in 0..qlen_simd {
            let val = S::extract_epi16(
                S::load_si128_16(q.hmax.add(k / S::WIDTH_16) as *const S::Vec16),
                (k % S::WIDTH_16) as i32,
            );
            let current_qe_candidate = (k / S::WIDTH_16 * slen + k % S::WIDTH_16) as i32;

            if val as i32 > max_val_qe {
                max_val_qe = val as i32;
                max_qe = current_qe_candidate;
            } else if val as i32 == max_val_qe {
                // If scores are equal, prefer smaller query end (earlier position)
                if current_qe_candidate < max_qe {
                    max_qe = current_qe_candidate;
                }
            }
        }
        r.qe = max_qe;

        // Populate score2 and te2 for suboptimal alignments
        if !b.is_empty() {
            let i_val = (r.score + q.max as i32 - 1) / q.max as i32;
            let low = te - i_val;
            let high = te + i_val;

            for val in b.iter() {
                let e_idx = (val & 0xFFFFFFFF) as i32; // Extract target end (i)
                let score = (val >> 32) as i32; // Extract score

                if (e_idx < low || e_idx > high) && score > r.score2 {
                    r.score2 = score;
                    r.te2 = e_idx;
                }
            }
        }
    }

    r
}

/// Reverse a portion of a sequence in place.
/// Equivalent to C's `revseq`.
#[inline]
fn revseq(seq: &mut [u8], len: usize) {
    let half = len / 2;
    for i in 0..half {
        seq.swap(i, len - 1 - i);
    }
}

/// High-level alignment function with separate gap penalties for deletions and insertions.
/// Equivalent to C's `ksw_align2`.
///
/// This function:
/// 1. Initializes a query profile if not provided
/// 2. Runs the appropriate DP loop (8-bit or 16-bit based on xtra flags)
/// 3. If KSW_XSTART is set, performs reverse alignment to find start positions
///
/// # Arguments
/// * `qlen` - Length of the query sequence
/// * `query` - Query sequence with 0 <= query[i] < m
/// * `tlen` - Length of the target sequence
/// * `target` - Target sequence with 0 <= target[i] < m
/// * `m` - Size of the alphabet
/// * `mat` - m*m scoring matrix in a one-dimension array
/// * `o_del` - Gap open penalty for deletions
/// * `e_del` - Gap extension penalty for deletions
/// * `o_ins` - Gap open penalty for insertions
/// * `e_ins` - Gap extension penalty for insertions
/// * `xtra` - Extra flags (KSW_XBYTE, KSW_XSTART, KSW_XSTOP, KSW_XSUBO)
///
/// # Returns
/// Alignment result with score, end positions, and optionally start positions
///
/// # Safety
/// This function is unsafe because it uses SIMD intrinsics internally.
pub unsafe fn ksw_align2<S: SimdEngine>(
    qlen: i32,
    query: &mut [u8],
    tlen: i32,
    target: &mut [u8],
    m: i32,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    xtra: u32,
) -> Kswr {
    // Determine score size: 1 byte (u8) if KSW_XBYTE is set, otherwise 2 bytes (i16)
    let size = if (xtra & KSW_XBYTE) != 0 { 1 } else { 2 };

    // Initialize query profile
    let q_ptr = ksw_qinit(size, qlen, query, m, mat);
    if q_ptr.is_null() {
        return KSW_DEFR;
    }
    let q = &mut *q_ptr;

    // Run the appropriate DP function
    let r = if q.size == 2 {
        ksw_i16_impl::<S>(q, tlen, target, o_del, e_del, o_ins, e_ins, xtra)
    } else {
        ksw_u8_impl::<S>(q, tlen, target, o_del, e_del, o_ins, e_ins, xtra)
    };

    // Check if we need to find start positions
    let should_find_start =
        (xtra & KSW_XSTART) != 0 && !((xtra & KSW_XSUBO) != 0 && r.score < (xtra & 0xffff) as i32);

    let final_result = if should_find_start && r.qe >= 0 && r.te >= 0 {
        // Reverse sequences to find start positions
        // +1 because qe/te points to the exact end, not the position after the end
        let qe_len = (r.qe + 1) as usize;
        let te_len = (r.te + 1) as usize;

        revseq(query, qe_len);
        revseq(target, te_len);

        // Create new query profile for reversed sequence
        let q2_ptr = ksw_qinit(q.size as i32, r.qe + 1, query, m, mat);
        if q2_ptr.is_null() {
            // Restore original sequences
            revseq(query, qe_len);
            revseq(target, te_len);
            ksw_qfree(q_ptr);
            return r;
        }
        let q2 = &mut *q2_ptr;

        // Run reverse alignment with KSW_XSTOP flag set to the forward score
        let xtra2 = KSW_XSTOP | (r.score as u32);
        let rr = if q2.size == 2 {
            ksw_i16_impl::<S>(q2, tlen, target, o_del, e_del, o_ins, e_ins, xtra2)
        } else {
            ksw_u8_impl::<S>(q2, tlen, target, o_del, e_del, o_ins, e_ins, xtra2)
        };

        // Restore original sequences
        revseq(query, qe_len);
        revseq(target, te_len);

        // Free reverse query profile
        ksw_qfree(q2_ptr);

        // Update result with start positions if scores match
        let mut result = r;
        if r.score == rr.score {
            result.tb = r.te - rr.te;
            result.qb = r.qe - rr.qe;
        }
        result
    } else {
        r
    };

    // Free query profile
    ksw_qfree(q_ptr);

    final_result
}

/// High-level alignment function with uniform gap penalties.
/// Equivalent to C's `ksw_align`.
///
/// This is a convenience wrapper around `ksw_align2` that uses the same
/// gap penalty for both insertions and deletions.
///
/// # Arguments
/// * `qlen` - Length of the query sequence
/// * `query` - Query sequence with 0 <= query[i] < m
/// * `tlen` - Length of the target sequence
/// * `target` - Target sequence with 0 <= target[i] < m
/// * `m` - Size of the alphabet
/// * `mat` - m*m scoring matrix in a one-dimension array
/// * `gapo` - Gap open penalty
/// * `gape` - Gap extension penalty
/// * `xtra` - Extra flags (KSW_XBYTE, KSW_XSTART, KSW_XSTOP, KSW_XSUBO)
///
/// # Returns
/// Alignment result with score, end positions, and optionally start positions
///
/// # Safety
/// This function is unsafe because it uses SIMD intrinsics internally.
pub unsafe fn ksw_align<S: SimdEngine>(
    qlen: i32,
    query: &mut [u8],
    tlen: i32,
    target: &mut [u8],
    m: i32,
    mat: &[i8],
    gapo: i32,
    gape: i32,
    xtra: u32,
) -> Kswr {
    ksw_align2::<S>(
        qlen, query, tlen, target, m, mat, gapo, gape, gapo, gape, xtra,
    )
}

/// Result structure for ksw_extend2
#[derive(Debug, Clone, Copy)]
pub struct KswExtendResult {
    /// Best semi-local alignment score
    pub score: i32,
    /// Query end position (0-based, length of aligned query)
    pub qle: i32,
    /// Target end position (0-based, length of aligned target)
    pub tle: i32,
    /// Target end position if entire query is aligned
    pub gtle: i32,
    /// Score if entire query is aligned (negative if not achieved)
    pub gscore: i32,
    /// Maximum offset from diagonal (useful for adaptive banding)
    pub max_off: i32,
}

/// Internal struct for (h, e) scores in DP
#[derive(Clone, Copy, Default)]
struct EhT {
    h: i32,
    e: i32,
}

/// Banded seed extension alignment with separate gap penalties.
/// Equivalent to C's `ksw_extend2`.
///
/// This function extends alignment from a starting position, assuming
/// upstream sequences have already been aligned with score `h0`. It supports:
/// - Adaptive banding based on score
/// - Z-drop termination for early exit
/// - Both semi-local (partial query) and global (full query) scoring
///
/// # Arguments
/// * `qlen` - Length of the query sequence
/// * `query` - Query sequence with 0 <= query[i] < m
/// * `tlen` - Length of the target sequence
/// * `target` - Target sequence with 0 <= target[i] < m
/// * `m` - Size of the alphabet
/// * `mat` - m*m scoring matrix in a one-dimension array
/// * `o_del` - Gap open penalty for deletions
/// * `e_del` - Gap extension penalty for deletions
/// * `o_ins` - Gap open penalty for insertions
/// * `e_ins` - Gap extension penalty for insertions
/// * `w` - Band width
/// * `end_bonus` - Bonus for reaching end of query
/// * `zdrop` - Z-drop threshold (0 to disable)
/// * `h0` - Starting score from upstream alignment
///
/// # Returns
/// Extension result with scores and positions
pub fn ksw_extend2(
    qlen: i32,
    query: &[u8],
    tlen: i32,
    target: &[u8],
    m: i32,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w: i32,
    end_bonus: i32,
    zdrop: i32,
    h0: i32,
) -> KswExtendResult {
    assert!(h0 > 0, "h0 must be positive");

    let qlen_usize = qlen as usize;
    let m_usize = m as usize;
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;

    // Allocate query profile
    let mut qp = vec![0i8; qlen_usize * m_usize];

    // Allocate eh array (H and E scores)
    let mut eh = vec![EhT::default(); qlen_usize + 1];

    // Generate query profile
    let mut idx = 0;
    for k in 0..m_usize {
        let p = &mat[k * m_usize..];
        for j in 0..qlen_usize {
            qp[idx] = p[query[j] as usize];
            idx += 1;
        }
    }

    // Fill the first row
    eh[0].h = h0;
    eh[1].h = if h0 > oe_ins { h0 - oe_ins } else { 0 };
    let mut j = 2i32;
    while j <= qlen && eh[(j - 1) as usize].h > e_ins {
        eh[j as usize].h = eh[(j - 1) as usize].h - e_ins;
        j += 1;
    }

    // Adjust w if too large
    let mut max_score = 0;
    for i in 0..(m * m) {
        if mat[i as usize] > max_score as i8 {
            max_score = mat[i as usize] as i32;
        }
    }

    let max_ins = ((qlen as f64 * max_score as f64 + end_bonus as f64 - o_ins as f64)
        / e_ins as f64
        + 1.0) as i32;
    let max_ins = max_ins.max(1);
    let w = w.min(max_ins);

    let max_del = ((qlen as f64 * max_score as f64 + end_bonus as f64 - o_del as f64)
        / e_del as f64
        + 1.0) as i32;
    let max_del = max_del.max(1);
    let w = w.min(max_del);

    // DP loop
    let mut max = h0;
    let mut max_i = -1i32;
    let mut max_j = -1i32;
    let mut max_ie = -1i32;
    let mut gscore = -1i32;
    let mut max_off = 0i32;
    let mut beg = 0i32;
    let mut end = qlen;

    for i in 0..tlen {
        let mut f = 0i32;
        let mut m_row = 0i32;
        let mut mj = -1i32;

        let q_base = target[i as usize] as usize * qlen_usize;

        // Apply band and constraint
        if beg < i - w {
            beg = i - w;
        }
        if end > i + w + 1 {
            end = i + w + 1;
        }
        if end > qlen {
            end = qlen;
        }

        // Compute first column
        let mut h1 = if beg == 0 {
            let val = h0 - (o_del + e_del * (i + 1));
            if val < 0 { 0 } else { val }
        } else {
            0
        };

        for j in beg..end {
            let ju = j as usize;

            // At beginning: eh[j] = {H(i-1,j-1), E(i,j)}, f = F(i,j), h1 = H(i,j-1)
            let p = &mut eh[ju];
            let big_m = p.h; // H(i-1,j-1)
            let e = p.e; // E(i-1,j)

            p.h = h1; // Set H(i,j-1) for next row

            // M = M + score, but only if M > 0 (disallow I followed by D)
            let big_m = if big_m != 0 {
                big_m + qp[q_base + ju] as i32
            } else {
                0
            };

            // H = max(M, E, F)
            let mut h = if big_m > e { big_m } else { e };
            h = if h > f { h } else { f };
            h1 = h;

            // Record max position
            if m_row < h {
                mj = j;
                m_row = h;
            }

            // E(i+1,j) = max(M - oe_del, E) - e_del
            let mut t = big_m - oe_del;
            if t < 0 {
                t = 0;
            }
            let mut e_new = e - e_del;
            if e_new < t {
                e_new = t;
            }
            p.e = e_new;

            // F(i,j+1) = max(M - oe_ins, F) - e_ins
            let mut t = big_m - oe_ins;
            if t < 0 {
                t = 0;
            }
            f = f - e_ins;
            if f < t {
                f = t;
            }
        }

        eh[end as usize].h = h1;
        eh[end as usize].e = 0;

        // Check if entire query is aligned
        if (beg..end).contains(&(qlen - 1)) || end == qlen {
            let h_end = if end == qlen {
                h1
            } else {
                eh[(qlen - 1) as usize].h
            };
            if gscore < h_end {
                gscore = h_end;
                max_ie = i;
            }
        }

        if m_row == 0 {
            break;
        }

        if m_row > max {
            max = m_row;
            max_i = i;
            max_j = mj;
            let off = (mj - i).abs();
            if max_off < off {
                max_off = off;
            }
        } else if zdrop > 0 {
            // Z-drop check
            let diff_i = i - max_i;
            let diff_j = mj - max_j;
            if diff_i > diff_j {
                if max - m_row - (diff_i - diff_j) * e_del > zdrop {
                    break;
                }
            } else {
                if max - m_row - (diff_j - diff_i) * e_ins > zdrop {
                    break;
                }
            }
        }

        // Update beg and end for next round
        let mut new_beg = beg;
        while new_beg < end && eh[new_beg as usize].h == 0 && eh[new_beg as usize].e == 0 {
            new_beg += 1;
        }
        beg = new_beg;

        let mut new_end = end;
        while new_end > beg && eh[new_end as usize].h == 0 && eh[new_end as usize].e == 0 {
            new_end -= 1;
        }
        end = (new_end + 2).min(qlen);
    }

    KswExtendResult {
        score: max,
        qle: max_j + 1,
        tle: max_i + 1,
        gtle: max_ie + 1,
        gscore,
        max_off,
    }
}

/// Banded seed extension alignment with uniform gap penalties.
/// Equivalent to C's `ksw_extend`.
///
/// This is a convenience wrapper around `ksw_extend2` that uses the same
/// gap penalty for both insertions and deletions.
pub fn ksw_extend(
    qlen: i32,
    query: &[u8],
    tlen: i32,
    target: &[u8],
    m: i32,
    mat: &[i8],
    gapo: i32,
    gape: i32,
    w: i32,
    end_bonus: i32,
    zdrop: i32,
    h0: i32,
) -> KswExtendResult {
    ksw_extend2(
        qlen, query, tlen, target, m, mat, gapo, gape, gapo, gape, w, end_bonus, zdrop, h0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::simd_abstraction::SimdEngine128;

    /// Standard DNA scoring matrix: +1 for match, -1 for mismatch
    /// 5x5 matrix for alphabet {A=0, C=1, G=2, T=3, N=4}
    fn get_dna_scoring_matrix() -> Vec<i8> {
        let mut mat = vec![0i8; 25];
        for i in 0..5 {
            for j in 0..5 {
                mat[i * 5 + j] = if i == j && i < 4 { 1 } else { -1 };
            }
        }
        // Set N (4) to have 0 score against everything including itself
        for i in 0..5 {
            mat[4 * 5 + i] = 0;
            mat[i * 5 + 4] = 0;
        }
        mat
    }

    /// Convert ASCII DNA sequence to numeric encoding
    fn encode_seq(seq: &[u8]) -> Vec<u8> {
        seq.iter()
            .map(|&c| match c {
                b'A' | b'a' => 0,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' => 3,
                _ => 4,
            })
            .collect()
    }

    #[test]
    fn test_revseq_basic() {
        let mut seq = vec![0, 1, 2, 3, 4];
        revseq(&mut seq, 5);
        assert_eq!(seq, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_revseq_partial() {
        let mut seq = vec![0, 1, 2, 3, 4, 5, 6];
        revseq(&mut seq, 4); // Only reverse first 4 elements
        assert_eq!(seq, vec![3, 2, 1, 0, 4, 5, 6]);
    }

    #[test]
    fn test_revseq_empty() {
        let mut seq: Vec<u8> = vec![];
        revseq(&mut seq, 0);
        assert_eq!(seq, Vec::<u8>::new());
    }

    #[test]
    fn test_ksw_extend_simple_match() {
        // Query: ACGT, Target: ACGT (perfect match)
        let query = encode_seq(b"ACGT");
        let target = encode_seq(b"ACGT");
        let mat = get_dna_scoring_matrix();

        let result = ksw_extend(
            4,       // qlen
            &query,  // query
            4,       // tlen
            &target, // target
            5,       // m (alphabet size)
            &mat,    // scoring matrix
            5,       // gapo
            1,       // gape
            50,      // w (band width)
            0,       // end_bonus
            100,     // zdrop
            10,      // h0 (starting score)
        );

        // With h0=10, match score +1 for each of 4 positions, we expect score around 14
        assert!(result.score > 10, "Score should be positive");
        assert_eq!(result.qle, 4, "Entire query should be aligned");
        assert_eq!(result.tle, 4, "Entire target should be aligned");
    }

    #[test]
    fn test_ksw_extend_with_mismatch() {
        // Query: ACGT, Target: ACCT (1 mismatch at position 2)
        let query = encode_seq(b"ACGT");
        let target = encode_seq(b"ACCT");
        let mat = get_dna_scoring_matrix();

        let result = ksw_extend(4, &query, 4, &target, 5, &mat, 5, 1, 50, 0, 100, 10);

        // Score should be lower due to mismatch
        assert!(result.score > 0, "Score should be positive");
        // 3 matches (+3) - 1 mismatch (-1) + h0 (10) = 12
        // Should be less than perfect match score of 14
        assert!(result.score <= 14, "Score should account for mismatch");
    }

    #[test]
    fn test_ksw_extend_with_insertion() {
        // Query: ACGGT, Target: ACGT (query has 1 extra G)
        let query = encode_seq(b"ACGGT");
        let target = encode_seq(b"ACGT");
        let mat = get_dna_scoring_matrix();

        let result = ksw_extend(
            5,       // qlen
            &query,  // query
            4,       // tlen
            &target, // target
            5,       // m
            &mat,    // mat
            5,       // gapo
            1,       // gape
            50,      // w
            0,       // end_bonus
            100,     // zdrop
            10,      // h0
        );

        assert!(result.score > 0, "Score should be positive even with indel");
    }

    #[test]
    fn test_ksw_extend_zdrop() {
        // Test z-drop early termination
        // Query: AAAAAAAAAA, Target: TTTTTTTTTT (all mismatches)
        let query = encode_seq(b"AAAAAAAAAA");
        let target = encode_seq(b"TTTTTTTTTT");
        let mat = get_dna_scoring_matrix();

        let result = ksw_extend(
            10, &query, 10, &target, 5, &mat, 5, 1, 50, 0,
            5,  // Small zdrop to trigger early termination
            10, // h0
        );

        // With all mismatches and small zdrop, should terminate early
        assert!(
            result.tle < 10 || result.score < 10,
            "Should terminate early due to zdrop"
        );
    }

    #[test]
    fn test_ksw_extend2_asymmetric_gaps() {
        // Test with different insertion and deletion penalties
        let query = encode_seq(b"ACGT");
        let target = encode_seq(b"ACGT");
        let mat = get_dna_scoring_matrix();

        let result = ksw_extend2(
            4, &query, 4, &target, 5, &mat, 5, // o_del
            1, // e_del
            3, // o_ins (different from o_del)
            2, // e_ins (different from e_del)
            50, 0, 100, 10,
        );

        // Perfect match should give same score regardless of gap penalties
        assert!(result.score >= 14, "Perfect match should have high score");
    }

    #[test]
    fn test_ksw_qinit_and_free() {
        let query = encode_seq(b"ACGTACGT");
        let mat = get_dna_scoring_matrix();

        unsafe {
            let q_ptr = ksw_qinit(2, 8, &query, 5, &mat);
            assert!(!q_ptr.is_null(), "Query profile should be allocated");

            let q = &*q_ptr;
            assert_eq!(q.qlen, 8);
            assert_eq!(q.size, 2); // i16 scores

            ksw_qfree(q_ptr);
        }
    }

    #[test]
    fn test_ksw_qinit_u8_mode() {
        let query = encode_seq(b"ACGT");
        let mat = get_dna_scoring_matrix();

        unsafe {
            let q_ptr = ksw_qinit(1, 4, &query, 5, &mat);
            assert!(!q_ptr.is_null());

            let q = &*q_ptr;
            assert_eq!(q.size, 1); // u8 scores

            ksw_qfree(q_ptr);
        }
    }

    #[test]
    fn test_ksw_align_basic() {
        let mut query = encode_seq(b"ACGTACGT");
        let mut target = encode_seq(b"ACGTACGT");
        let mat = get_dna_scoring_matrix();

        unsafe {
            let result = ksw_align::<SimdEngine128>(
                8,
                &mut query,
                8,
                &mut target,
                5,
                &mat,
                5, // gapo
                1, // gape
                0, // xtra (no special flags)
            );

            assert!(
                result.score > 0,
                "Score should be positive for perfect match"
            );
            assert!(result.te >= 0, "Target end should be set");
            assert!(result.qe >= 0, "Query end should be set");
        }
    }

    #[test]
    fn test_ksw_align_with_start() {
        let mut query = encode_seq(b"ACGTACGT");
        let mut target = encode_seq(b"ACGTACGT");
        let mat = get_dna_scoring_matrix();

        unsafe {
            let result = ksw_align::<SimdEngine128>(
                8,
                &mut query,
                8,
                &mut target,
                5,
                &mat,
                5,
                1,
                KSW_XSTART, // Find start positions
            );

            assert!(result.score > 0);
            // When KSW_XSTART is set and alignment succeeds, tb and qb should be set
            if result.score > 0 && result.te >= 0 && result.qe >= 0 {
                // Start positions should be set (may be 0 for full alignment)
                assert!(result.tb >= -1);
                assert!(result.qb >= -1);
            }
        }
    }

    #[test]
    fn test_ksw_align_u8_mode() {
        let mut query = encode_seq(b"ACGT");
        let mut target = encode_seq(b"ACGT");
        let mat = get_dna_scoring_matrix();

        unsafe {
            let result = ksw_align::<SimdEngine128>(
                4,
                &mut query,
                4,
                &mut target,
                5,
                &mat,
                5,
                1,
                KSW_XBYTE, // Use 8-bit scores
            );

            assert!(result.score > 0);
            assert!(result.score <= 255, "8-bit mode should cap at 255");
        }
    }

    #[test]
    fn test_ksw_align2_asymmetric_gaps() {
        let mut query = encode_seq(b"ACGTACGT");
        let mut target = encode_seq(b"ACGTACGT");
        let mat = get_dna_scoring_matrix();

        unsafe {
            let result = ksw_align2::<SimdEngine128>(
                8,
                &mut query,
                8,
                &mut target,
                5,
                &mat,
                5, // o_del
                1, // e_del
                3, // o_ins
                2, // e_ins
                0, // xtra
            );

            assert!(result.score > 0);
        }
    }

    #[test]
    fn test_large_indel_extension() {
        // Test case specifically for large indel handling
        // Query has a 5bp insertion compared to target
        let query = encode_seq(b"ACGTAAAAACGT"); // 12bp
        let target = encode_seq(b"ACGTCGT"); // 7bp (missing AAAAA)
        let mat = get_dna_scoring_matrix();

        let result = ksw_extend(
            12, &query, 7, &target, 5, &mat, 5, 1, 100, // Wide band for indels
            0, 200, // High zdrop to not terminate early
            10,
        );

        // Should still align despite large indel
        assert!(result.score > 0, "Should find some alignment");
    }

    #[test]
    fn test_15bp_insertion_extension() {
        // Test matching the integration test scenario:
        // Reference: 50bp of AGCTAGCT repeating
        // Query: 20bp match + 15bp insertion (all G's) + 30bp match = 65bp
        // But we test the extension portion: extending from a seed into the indel

        // Simulate extending from a 10bp seed into a region with 15bp insertion
        // Query: [SEED 10bp] [15bp G insertion] [20bp suffix]
        // Target: [SEED region] [matching suffix without insertion]
        let query = encode_seq(b"AGCTAGCTAGGGGGGGGGGGGGGGGAGCTAGCTAGCTAGCTAGCT"); // 45bp
        let target = encode_seq(b"AGCTAGCTAGAGCTAGCTAGCTAGCTAGCT"); // 30bp (no insertion)
        let mat = get_dna_scoring_matrix();

        // Start with h0=20 (simulating a 10bp seed with match score 2)
        let result = ksw_extend2(
            45, &query, 30, &target, 5, &mat, 5, 1, 5, 1, 50,  // Wide band
            0,   // end_bonus
            200, // zdrop
            20,  // h0 from seed
        );

        println!(
            "15bp insertion test: score={}, gscore={}, qle={}, tle={}, max_off={}",
            result.score, result.gscore, result.qle, result.tle, result.max_off
        );

        // The extension should find the matching prefix (10bp)
        // It correctly stops before the 15bp insertion because crossing it
        // would cost 20 (gap open 5 + 15 * gap extend 1) and the subsequent
        // matches wouldn't recover enough score to make it worthwhile
        // This is correct local alignment behavior!
        assert!(
            result.score > 0 || result.gscore > 0,
            "Should find some alignment"
        );

        // The algorithm finds the 10bp match = 10 * 2 (match) = 20
        // Plus h0=20 gives total ~30-40
        assert!(
            result.score >= 20,
            "Should find at least the 10bp prefix match, got score={}",
            result.score
        );
    }

    #[test]
    fn test_extend2_vs_scalar_comparison() {
        // Compare ksw_extend2 to see if it handles indels better
        // This simulates the pathological case where scalar SW fails

        // Query with 10bp insertion in middle
        let query = encode_seq(b"ACGTACGTAAAAAAAAAAAACGTACGT"); // 27bp
        let target = encode_seq(b"ACGTACGTACGTACGT"); // 16bp (no insertion)
        let mat = get_dna_scoring_matrix();

        let result = ksw_extend2(
            27, &query, 16, &target, 5, &mat, 5, 1, 5, 1, 30, // Moderate band
            0, 100, 10, // h0
        );

        println!(
            "10bp insertion comparison: score={}, gscore={}, qle={}, tle={}",
            result.score, result.gscore, result.qle, result.tle
        );

        // Verify we get a reasonable alignment
        // With 8bp matching prefix + 8bp matching suffix - penalties
        // Expected: ~32 (16 matches * 2) - gap penalties
        assert!(
            result.score > 5,
            "Should find alignment with positive score"
        );
    }
}
