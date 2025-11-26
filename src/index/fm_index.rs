// FM-Index operations for BWT-based sequence search
//
// This module contains the core FM-Index functionality including:
// - Occurrence counting with hardware-optimized popcount
// - Backward and forward extension for BWT search
// - Checkpoint data structures for efficient occurrence queries
//
// PERFORMANCE OPTIMIZATIONS (Session 51):
// - Compile-time const mask array (eliminates lazy_static overhead)
// - Inlined forward_ext (avoids function call and struct copy)
// - Removed debug logging checks from hot path

use super::index::BwaIndex;
use crate::alignment::seeding::SMEM;

// Constants from FMI_search.h
const CP_MASK: u64 = 63;
pub const CP_SHIFT: u64 = 6; // Public for external use

/// Checkpoint occurrence structure for FM-Index
/// Corresponds to C++ CP_OCC (FMI_search.h:54-58)
/// Stores occurrence counts and encoded BWT at 64-base checkpoints
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CpOcc {
    /// Occurrence counts for each base (A,C,G,T) at this checkpoint
    pub checkpoint_counts: [i64; 4],
    /// One-hot encoded BWT bits for fast popcount-based occurrence queries
    pub bwt_encoding_bits: [u64; 4],
}

// Compile-time const one_hot_mask_array
// Matches C++ bwa-mem2: one_hot_mask_array[i] has the top i bits set
// This eliminates lazy_static overhead on every access
const ONE_HOT_MASK_ARRAY: [u64; 64] = {
    let base: u64 = 0x8000000000000000;
    let mut array = [0u64; 64];
    // array[0] = 0 (already initialized)
    array[1] = base;
    let mut i = 2;
    while i < 64 {
        array[i] = (array[i - 1] >> 1) | base;
        i += 1;
    }
    array
};

// Hardware-optimized popcount for 64-bit integers
// Uses NEON on ARM64 and POPCNT on x86_64
#[inline(always)]
pub fn popcount64(x: u64) -> i64 {
    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON implementation using vcnt (count bits in 8-bit lanes)
        // This is the equivalent of __builtin_popcountl on ARM
        unsafe {
            use std::arch::aarch64::*;

            // Load the 64-bit value into a NEON register
            let vec = vreinterpret_u8_u64(vcreate_u64(x));

            // Count bits in each 8-bit lane (vcnt_u8)
            let cnt = vcnt_u8(vec);

            // Sum all 8 lanes using horizontal add (pairwise additions)
            // vcnt gives us 8 bytes, each with bit count of that byte
            // We need to sum them all to get total popcount
            let sum16 = vpaddl_u8(cnt); // Pairwise add to 4x u16
            let sum32 = vpaddl_u16(sum16); // Pairwise add to 2x u32
            let sum64 = vpaddl_u32(sum32); // Pairwise add to 1x u64

            vget_lane_u64::<0>(sum64) as i64
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // x86_64 POPCNT instruction
        unsafe {
            use std::arch::x86_64::_popcnt64;
            _popcnt64(x as i64) as i64
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        // Fallback for other architectures
        x.count_ones() as i64
    }
}

// Helper function to get occurrences, translating GET_OCC macro
// Now uses hardware-optimized popcount
#[inline(always)]
pub fn get_occ(bwa_idx: &BwaIndex, k: i64, c: u8) -> i64 {
    let occ_id_k = (k >> CP_SHIFT) as usize;
    let y_k = (k & CP_MASK as i64) as usize;

    let cp_occ = unsafe { bwa_idx.cp_occ.get_unchecked(occ_id_k) };
    let occ_k = cp_occ.checkpoint_counts[c as usize];
    let one_hot_bwt_str_c_k = cp_occ.bwt_encoding_bits[c as usize];

    // Use direct array indexing (bounds checked at compile time for const array)
    let match_mask_k = one_hot_bwt_str_c_k & ONE_HOT_MASK_ARRAY[y_k];
    occ_k + popcount64(match_mask_k)
}

/// Vectorized get_occ for all 4 bases simultaneously
///
/// This function processes all 4 bases (A, C, G, T) in parallel, eliminating
/// the need for 4 sequential get_occ calls. This provides significant speedup
/// by reducing memory access overhead and enabling better CPU pipelining.
///
/// Returns [i64; 4] containing occurrence counts for bases 0, 1, 2, 3 (A, C, G, T)
#[inline(always)]
pub fn get_occ_all_bases(bwa_idx: &BwaIndex, k: i64) -> [i64; 4] {
    let occ_id_k = (k >> CP_SHIFT) as usize;
    let y_k = (k & CP_MASK as i64) as usize;

    // SAFETY: occ_id_k is computed from valid BWT positions
    let cp_occ = unsafe { bwa_idx.cp_occ.get_unchecked(occ_id_k) };
    let mask = ONE_HOT_MASK_ARRAY[y_k];

    // Unroll loop for better instruction-level parallelism
    [
        cp_occ.checkpoint_counts[0] + popcount64(cp_occ.bwt_encoding_bits[0] & mask),
        cp_occ.checkpoint_counts[1] + popcount64(cp_occ.bwt_encoding_bits[1] & mask),
        cp_occ.checkpoint_counts[2] + popcount64(cp_occ.bwt_encoding_bits[2] & mask),
        cp_occ.checkpoint_counts[3] + popcount64(cp_occ.bwt_encoding_bits[3] & mask),
    ]
}

/// Backward extension matching C++ bwa-mem2 FMI_search::backwardExt()
///
/// CRITICAL: This uses a cumulative sum approach for computing l[] values,
/// NOT the simple l = k + s formula! The l field encodes reverse complement
/// BWT information, so the standard BWT interval invariant s = l - k does NOT hold.
///
/// C++ reference: FMI_search.cpp lines 1025-1052
#[inline(always)]
pub fn backward_ext(bwa_idx: &BwaIndex, mut smem: SMEM, a: u8) -> SMEM {
    // Compute occurrence counts for all 4 bases at start and end positions
    let sp = smem.bwt_interval_start as i64;
    let ep = (smem.bwt_interval_start + smem.interval_size) as i64;

    let occ_sp = get_occ_all_bases(bwa_idx, sp);
    let occ_ep = get_occ_all_bases(bwa_idx, ep);

    // Compute k[] and s[] for all 4 bases
    let cumulative = &bwa_idx.bwt.cumulative_count;
    let k0 = cumulative[0] as i64 + occ_sp[0];
    let k1 = cumulative[1] as i64 + occ_sp[1];
    let k2 = cumulative[2] as i64 + occ_sp[2];
    let k3 = cumulative[3] as i64 + occ_sp[3];

    let s0 = occ_ep[0] - occ_sp[0];
    let s1 = occ_ep[1] - occ_sp[1];
    let s2 = occ_ep[2] - occ_sp[2];
    let s3 = occ_ep[3] - occ_sp[3];

    // Sentinel handling (matching C++ lines 1041-1042)
    let sentinel_idx = bwa_idx.sentinel_index as u64;
    let sentinel_offset =
        ((smem.bwt_interval_start <= sentinel_idx) & ((smem.bwt_interval_start + smem.interval_size) > sentinel_idx)) as i64;

    // CRITICAL: Cumulative sum computation for l[] (matching C++ lines 1043-1046)
    let l3 = smem.bwt_interval_end as i64 + sentinel_offset;
    let l2 = l3 + s3;
    let l1 = l2 + s2;
    let l0 = l1 + s1;

    // Select results for the requested base 'a'
    let (k_a, l_a, s_a) = match a {
        0 => (k0, l0, s0),
        1 => (k1, l1, s1),
        2 => (k2, l2, s2),
        _ => (k3, l3, s3), // 3 or any other value
    };

    smem.bwt_interval_start = k_a as u64;
    smem.bwt_interval_end = l_a as u64;
    smem.interval_size = s_a as u64;

    smem
}

/// Forward extension matching C++ bwa-mem2 pattern
///
/// Forward extension is implemented as:
/// 1. Swap k and l
/// 2. Call backwardExt with complement base (3 - a)
/// 3. Swap k and l back
///
/// C++ reference: FMI_search.cpp lines 546-554
///
/// OPTIMIZED: Inlined swap operations, eliminated debug logging overhead
#[inline(always)]
pub fn forward_ext(bwa_idx: &BwaIndex, smem: SMEM, a: u8) -> SMEM {
    // Create swapped SMEM in place (k and l swapped)
    let smem_swapped = SMEM {
        read_id: smem.read_id,
        bwt_interval_start: smem.bwt_interval_end,
        bwt_interval_end: smem.bwt_interval_start,
        interval_size: smem.interval_size,
        query_start: smem.query_start,
        query_end: smem.query_end,
        is_reverse_complement: smem.is_reverse_complement,
    };

    // Backward extension with complement base (3 - a)
    let mut result = backward_ext(bwa_idx, smem_swapped, 3 - a);

    // Swap k and l back
    std::mem::swap(&mut result.bwt_interval_start, &mut result.bwt_interval_end);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_popcount64_neon() {
        // Test the hardware-optimized popcount implementation
        // This ensures our NEON implementation matches the software version

        // Test basic cases
        assert_eq!(popcount64(0), 0);
        assert_eq!(popcount64(1), 1);
        assert_eq!(popcount64(0xFFFFFFFFFFFFFFFF), 64);
        assert_eq!(popcount64(0x8000000000000000), 1);

        // Test various bit patterns
        assert_eq!(popcount64(0b1010101010101010), 8);
        assert_eq!(popcount64(0b11111111), 8);
        assert_eq!(popcount64(0xFF00FF00FF00FF00), 32);
        assert_eq!(popcount64(0x0F0F0F0F0F0F0F0F), 32);

        // Test random patterns that match expected popcount
        assert_eq!(popcount64(0x123456789ABCDEF0), 32);
        assert_eq!(popcount64(0xAAAAAAAAAAAAAAAA), 32); // Alternating bits
        assert_eq!(popcount64(0x5555555555555555), 32); // Alternating bits (complement)
    }
}
