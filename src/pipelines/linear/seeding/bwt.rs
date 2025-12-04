//! BWT and suffix array helper functions.
//!
//! Low-level functions for BWT navigation and suffix array access
//! used during FM-Index search and seed extension.

use crate::pipelines::linear::index::fm_index::{get_occ, CP_SHIFT, CpOcc};
use crate::pipelines::linear::index::index::BwaIndex;

/// Get BWT base from cp_occ format (for loaded indices).
///
/// Returns 0-3 for bases A/C/G/T, or 4 for sentinel.
///
/// This is an ultra-hot path called millions of times during seeding.
/// Optimized to match C++ BWA-MEM2 FMI_search.cpp lines 1131-1140:
/// - No bounds checking (caller must ensure valid pos)
/// - If-else chain instead of loop (better branch prediction)
/// - Inline hint for compiler
#[inline(always)]
pub fn get_bwt_base_from_cp_occ(cp_occ: &[CpOcc], pos: u64) -> u8 {
    let cp_block = (pos >> CP_SHIFT) as usize;

    #[cfg(debug_assertions)]
    {
        if cp_block >= cp_occ.len() {
            log::warn!(
                "get_bwt_base_from_cp_occ: cp_block {} out of bounds (cp_occ.len()={})",
                cp_block,
                cp_occ.len()
            );
            return 4;
        }
    }

    let offset_in_block = pos & ((1 << CP_SHIFT) - 1);
    let bit_position = 63 - offset_in_block;

    #[cfg(not(debug_assertions))]
    let one_hot = unsafe { cp_occ.get_unchecked(cp_block) };
    #[cfg(debug_assertions)]
    let one_hot = &cp_occ[cp_block];

    // If-else chain matching C++ BWA-MEM2 pattern (better branch prediction than loop)
    if (one_hot.bwt_encoding_bits[0] >> bit_position) & 1 == 1 {
        0
    } else if (one_hot.bwt_encoding_bits[1] >> bit_position) & 1 == 1 {
        1
    } else if (one_hot.bwt_encoding_bits[2] >> bit_position) & 1 == 1 {
        2
    } else if (one_hot.bwt_encoding_bits[3] >> bit_position) & 1 == 1 {
        3
    } else {
        4 // Sentinel
    }
}

/// Get the next BWT position from a BWT coordinate.
///
/// Returns None if we hit the sentinel (which should not be navigated).
pub fn get_bwt(bwa_idx: &BwaIndex, pos: u64) -> Option<u64> {
    let base = if !bwa_idx.bwt.bwt_data.is_empty() {
        // Index was just built, use raw bwt_data
        bwa_idx.bwt.get_bwt_base(pos)
    } else {
        // Index was loaded from disk, use cp_occ format
        get_bwt_base_from_cp_occ(&bwa_idx.cp_occ, pos)
    };

    // If we hit the sentinel (base == 4), return None
    if base == 4 {
        return None;
    }

    Some(bwa_idx.bwt.cumulative_count[base as usize] + get_occ(bwa_idx, pos as i64, base) as u64)
}

/// Get suffix array entry at a BWT position.
///
/// Navigates through BWT until reaching a sampled SA position,
/// then returns the reference position.
pub fn get_sa_entry(bwa_idx: &BwaIndex, mut pos: u64) -> u64 {
    let original_pos = pos;
    let mut count = 0;
    const MAX_ITERATIONS: u64 = 10000;

    while pos % bwa_idx.bwt.sa_sample_interval as u64 != 0 {
        if count >= MAX_ITERATIONS {
            log::error!(
                "get_sa_entry exceeded MAX_ITERATIONS ({MAX_ITERATIONS}) - possible infinite loop!"
            );
            log::error!("  original_pos={original_pos}, current_pos={pos}, count={count}");
            log::error!(
                "  sa_intv={}, seq_len={}",
                bwa_idx.bwt.sa_sample_interval,
                bwa_idx.bwt.seq_len
            );
            return count;
        }

        let _old_pos = pos;
        match get_bwt(bwa_idx, pos) {
            Some(new_pos) => {
                pos = new_pos;
                count += 1;
            }
            None => {
                return count;
            }
        }
    }

    let sa_index = (pos / bwa_idx.bwt.sa_sample_interval as u64) as usize;
    let sa_ms_byte = bwa_idx.bwt.sa_high_bytes[sa_index] as u64;
    let sa_ls_word = bwa_idx.bwt.sa_low_words[sa_index] as u64;
    let sa_val = (sa_ms_byte << 32) | sa_ls_word;

    // Handle sentinel: SA values can point to the sentinel position (seq_len)
    let sentinel_pos = bwa_idx.bns.packed_sequence_length << 1;
    let adjusted_sa_val = if sa_val >= sentinel_pos {
        log::debug!("SA value {sa_val} is at/past sentinel {sentinel_pos} - wrapping to 0");
        0
    } else {
        sa_val
    };

    let result = adjusted_sa_val + count;

    log::debug!(
        "get_sa_entry: original_pos={}, final_pos={}, count={}, sa_index={}, sa_val={}, adjusted={}, result={}, l_pac={}, sentinel={}",
        original_pos,
        pos,
        count,
        sa_index,
        sa_val,
        adjusted_sa_val,
        result,
        bwa_idx.bns.packed_sequence_length,
        bwa_idx.bns.packed_sequence_length << 1
    );
    result
}

/// Get multiple suffix array entries from a BWT interval.
///
/// Samples evenly across the interval using integer step to match BWA-MEM2.
///
/// Matches C++ FMI_search.cpp:1200-1206:
/// ```c
/// int64_t step = (smem.s > max_occ) ? smem.s / max_occ : 1;
/// for(j = smem.k; (j < hi) && (c < max_occ); j+=step, c++)
/// ```
pub fn get_sa_entries(
    bwa_idx: &BwaIndex,
    bwt_interval_start: u64,
    interval_size: u64,
    max_occurrences: u32,
) -> Vec<u64> {
    let num_to_retrieve = (interval_size as u32).min(max_occurrences);
    let mut ref_positions = Vec::with_capacity(num_to_retrieve as usize);

    if num_to_retrieve == 0 {
        return ref_positions;
    }

    // Use integer step matching BWA-MEM2
    let step = if interval_size > num_to_retrieve as u64 {
        interval_size / num_to_retrieve as u64
    } else {
        1
    };

    let hi = bwt_interval_start + interval_size;
    let mut j = bwt_interval_start;
    let mut c = 0;

    while j < hi && c < num_to_retrieve {
        let ref_pos = get_sa_entry(bwa_idx, j);
        ref_positions.push(ref_pos);
        j += step;
        c += 1;
    }

    ref_positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_get_sa_entry_basic() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_basic - could not load index");
                return;
            }
        };

        let sa_entry = get_sa_entry(&bwa_idx, 0);
        assert!(
            sa_entry < bwa_idx.bwt.seq_len,
            "SA entry {} should be less than seq_len {}",
            sa_entry,
            bwa_idx.bwt.seq_len
        );
    }

    #[test]
    fn test_get_sa_entry_sampled_position() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_sampled_position - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_sampled_position - could not load index");
                return;
            }
        };

        let sampled_pos = bwa_idx.bwt.sa_sample_interval as u64;
        let sa_entry = get_sa_entry(&bwa_idx, sampled_pos);

        assert!(
            sa_entry < bwa_idx.bwt.seq_len,
            "SA entry at sampled position should be within sequence length"
        );
    }

    #[test]
    fn test_get_sa_entry_consistency() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_consistency - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_consistency - could not load index");
                return;
            }
        };

        let pos = 5u64;
        let sa_entry1 = get_sa_entry(&bwa_idx, pos);
        let sa_entry2 = get_sa_entry(&bwa_idx, pos);

        assert_eq!(
            sa_entry1, sa_entry2,
            "get_sa_entry should return consistent results for the same position"
        );
    }

    #[test]
    fn test_get_bwt_basic() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_bwt_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_bwt_basic - could not load index");
                return;
            }
        };

        for pos in 0..10u64 {
            let bwt_result = get_bwt(&bwa_idx, pos);

            if let Some(new_pos) = bwt_result {
                assert!(
                    new_pos < bwa_idx.bwt.seq_len,
                    "BWT position should be within sequence length"
                );
            }
        }
    }
}
