// FM-Index operations for BWT-based sequence search
//
// This module contains the core FM-Index functionality including:
// - Occurrence counting with hardware-optimized popcount
// - Backward and forward extension for BWT search
// - Checkpoint data structures for efficient occurrence queries

use crate::alignment::seeding::SMEM;
use crate::index::BwaIndex;

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

// Global one_hot_mask_array (initialized once)
// Matches C++ bwa-mem2: one_hot_mask_array[i] has the top i bits set
lazy_static::lazy_static! {
    static ref ONE_HOT_MASK_ARRAY: Vec<u64> = {
        let mut array = vec![0u64; 64]; // Size 64 to match C++ (indices 0-63)
        // array[0] is already 0
        let base = 0x8000000000000000u64;
        array[1] = base;  // Explicitly set like C++ does
        for i in 2..64 {
            array[i] = (array[i - 1] >> 1) | base;
        }
        array
    };
}

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
pub fn get_occ(bwa_idx: &BwaIndex, k: i64, c: u8) -> i64 {
    let cp_shift = CP_SHIFT as i64;
    let cp_mask = CP_MASK as i64;

    let occ_id_k = k >> cp_shift;
    let y_k = k & cp_mask;

    let occ_k = bwa_idx.cp_occ[occ_id_k as usize].checkpoint_counts[c as usize];
    let one_hot_bwt_str_c_k = bwa_idx.cp_occ[occ_id_k as usize].bwt_encoding_bits[c as usize];

    let match_mask_k = one_hot_bwt_str_c_k & ONE_HOT_MASK_ARRAY[y_k as usize];
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
    let cp_shift = CP_SHIFT as i64;
    let cp_mask = CP_MASK as i64;

    let occ_id_k = k >> cp_shift;
    let y_k = k & cp_mask;

    let cp_occ = &bwa_idx.cp_occ[occ_id_k as usize];
    let mask = ONE_HOT_MASK_ARRAY[y_k as usize];

    // Process all 4 bases in parallel
    let mut result = [0i64; 4];
    for i in 0..4 {
        let match_mask = cp_occ.bwt_encoding_bits[i] & mask;
        result[i] = cp_occ.checkpoint_counts[i] + popcount64(match_mask);
    }

    result
}

/// Backward extension matching C++ bwa-mem2 FMI_search::backwardExt()
///
/// CRITICAL: This uses a cumulative sum approach for computing l[] values,
/// NOT the simple l = k + s formula! The l field encodes reverse complement
/// BWT information, so the standard BWT interval invariant s = l - k does NOT hold.
///
/// C++ reference: FMI_search.cpp lines 1025-1052
pub fn backward_ext(bwa_idx: &BwaIndex, mut smem: SMEM, a: u8) -> SMEM {
    let debug_enabled = log::log_enabled!(log::Level::Trace);

    if debug_enabled {
        log::trace!(
            "backward_ext: input smem(k={}, l={}, s={}), a={}",
            smem.bwt_interval_start,
            smem.bwt_interval_end,
            smem.interval_size,
            a
        );
    }

    let mut k = [0i64; 4];
    let mut l = [0i64; 4];
    let mut s = [0i64; 4];

    // Compute k[] and s[] for all 4 bases (matching C++ lines 1030-1039)
    // OPTIMIZATION: Use vectorized get_occ_all_bases to process all 4 bases at once
    let sp = smem.bwt_interval_start as i64;
    let ep = (smem.bwt_interval_start + smem.interval_size) as i64;

    let occ_sp = get_occ_all_bases(bwa_idx, sp);
    let occ_ep = get_occ_all_bases(bwa_idx, ep);

    for b in 0..4usize {
        k[b] = bwa_idx.bwt.cumulative_count[b] as i64 + occ_sp[b];
        s[b] = occ_ep[b] - occ_sp[b];

        if debug_enabled && b == a as usize {
            log::trace!(
                "backward_ext: base {}: sp={}, ep={}, occ_sp={}, occ_ep={}, k={}, s={}",
                b,
                sp,
                ep,
                occ_sp[b],
                occ_ep[b],
                k[b],
                s[b]
            );
        }
    }

    // Sentinel handling (matching C++ lines 1041-1042)
    let sentinel_offset = if smem.bwt_interval_start <= bwa_idx.sentinel_index as u64
        && (smem.bwt_interval_start + smem.interval_size) > bwa_idx.sentinel_index as u64
    {
        1i64
    } else {
        0i64
    };

    if debug_enabled {
        log::trace!(
            "backward_ext: sentinel_offset={}, sentinel_index={}",
            sentinel_offset,
            bwa_idx.sentinel_index
        );
    }

    // CRITICAL: Cumulative sum computation for l[] (matching C++ lines 1043-1046)
    // This is NOT l[b] = k[b] + s[b]!
    // Instead: l[3] = smem.bwt_interval_end + offset, then l[2] = l[3] + s[3], etc.
    l[3] = smem.bwt_interval_end as i64 + sentinel_offset;
    l[2] = l[3] + s[3];
    l[1] = l[2] + s[2];
    l[0] = l[1] + s[1];

    if debug_enabled {
        log::trace!(
            "backward_ext: cumulative l[] = [{}, {}, {}, {}]",
            l[0],
            l[1],
            l[2],
            l[3]
        );
        log::trace!(
            "backward_ext: k[] = [{}, {}, {}, {}]",
            k[0],
            k[1],
            k[2],
            k[3]
        );
        log::trace!(
            "backward_ext: s[] = [{}, {}, {}, {}]",
            s[0],
            s[1],
            s[2],
            s[3]
        );
    }

    // Update SMEM with results for base 'a' (matching C++ lines 1048-1050)
    smem.bwt_interval_start = k[a as usize] as u64;
    smem.bwt_interval_end = l[a as usize] as u64;
    smem.interval_size = s[a as usize] as u64;

    if debug_enabled {
        log::trace!(
            "backward_ext: output smem(k={}, l={}, s={}) for base {}",
            smem.bwt_interval_start,
            smem.bwt_interval_end,
            smem.interval_size,
            a
        );
    }

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
#[inline]
pub fn forward_ext(bwa_idx: &BwaIndex, smem: SMEM, a: u8) -> SMEM {
    // Debug logging for forward extension
    let debug_enabled = log::log_enabled!(log::Level::Trace);

    if debug_enabled {
        log::trace!(
            "forward_ext: input smem(k={}, l={}, s={}), a={}",
            smem.bwt_interval_start,
            smem.bwt_interval_end,
            smem.interval_size,
            a
        );
    }

    // Step 1: Swap k and l (lines 547-548)
    let mut smem_swapped = smem;
    smem_swapped.bwt_interval_start = smem.bwt_interval_end;
    smem_swapped.bwt_interval_end = smem.bwt_interval_start;

    if debug_enabled {
        log::trace!(
            "forward_ext: after swap smem_swapped(k={}, l={})",
            smem_swapped.bwt_interval_start,
            smem_swapped.bwt_interval_end
        );
    }

    // Step 2: Backward extension with complement base (line 549)
    let mut result = backward_ext(bwa_idx, smem_swapped, 3 - a);

    if debug_enabled {
        log::trace!(
            "forward_ext: after backward_ext result(k={}, l={}, s={})",
            result.bwt_interval_start,
            result.bwt_interval_end,
            result.interval_size
        );
    }

    // Step 3: Swap k and l back (lines 552-553)
    // NOTE: We swap k and l but KEEP s unchanged (matches C++ behavior)
    // The s value is still valid because it represents interval size
    let k_temp = result.bwt_interval_start;
    result.bwt_interval_start = result.bwt_interval_end;
    result.bwt_interval_end = k_temp;

    if debug_enabled {
        log::trace!(
            "forward_ext: after swap back result(k={}, l={}, s={})",
            result.bwt_interval_start,
            result.bwt_interval_end,
            result.interval_size
        );
    }

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
