//! SMEM generation algorithms.
//!
//! Contains the bidirectional FM-index search for generating Super Maximal
//! Exact Matches (SMEMs) from query sequences.

use crate::core::compute::simd_abstraction::portable_intrinsics;
use crate::pipelines::linear::index::fm_index::{CP_SHIFT, backward_ext, forward_ext};
use crate::pipelines::linear::index::index::BwaIndex;

use super::types::SMEM;

/// Prefetch BWT data for next iteration.
#[inline(always)]
unsafe fn prefetch_bwt(bwa_idx: &BwaIndex, smem: &SMEM) {
    let prefetch_addr = bwa_idx
        .cp_occ
        .as_ptr()
        .add((smem.bwt_interval_start >> CP_SHIFT) as usize) as *const i8;
    portable_intrinsics::prefetch_read_t0(prefetch_addr);
}

/// Generate SMEMs for a single strand (forward or reverse complement).
///
/// The `prev_array_buf` and `curr_array_buf` parameters are working buffers
/// that should be pre-allocated and reused across calls to avoid allocation overhead.
pub fn generate_smems_for_strand<'a>(
    bwa_idx: &BwaIndex,
    _query_name: &str,
    query_len: usize,
    encoded_query: &[u8],
    is_reverse_complement: bool,
    min_seed_len: i32,
    min_intv: u64,
    all_smems: &mut Vec<SMEM>,
    max_smem_count: &mut usize,
    prev_array_buf: &'a mut Vec<SMEM>,
    curr_array_buf: &'a mut Vec<SMEM>,
) {
    prev_array_buf.clear();
    curr_array_buf.clear();

    let mut x = 0;
    while x < query_len {
        let a = encoded_query[x];
        if a >= 4 {
            x += 1;
            continue;
        }

        let mut smem = SMEM {
            read_id: 0,
            query_start: x as i32,
            query_end: x as i32,
            bwt_interval_start: bwa_idx.bwt.cumulative_count[a as usize],
            bwt_interval_end: bwa_idx.bwt.cumulative_count[(3 - a) as usize],
            interval_size: bwa_idx.bwt.cumulative_count[(a + 1) as usize]
                - bwa_idx.bwt.cumulative_count[a as usize],
            is_reverse_complement,
        };

        // Phase 1: Forward extension
        prev_array_buf.clear();
        prev_array_buf.reserve(query_len.saturating_sub(x));
        let mut next_x = x + 1;

        for j in (x + 1)..query_len {
            let a = encoded_query[j];
            next_x = j + 1;

            if a >= 4 {
                next_x = j;
                break;
            }

            let new_smem = forward_ext(bwa_idx, smem, a);

            // Branchless append - matches C++ FMI_search.cpp:556-559
            let mask = (new_smem.interval_size != smem.interval_size) as usize;
            unsafe {
                std::ptr::write(prev_array_buf.as_mut_ptr().add(prev_array_buf.len()), smem);
                prev_array_buf.set_len(prev_array_buf.len() + mask);
            }

            if new_smem.interval_size < min_intv {
                next_x = j;
                break;
            }

            smem = new_smem;
            smem.query_end = j as i32;
        }

        // Branchless append for final SMEM
        let mask = (smem.interval_size >= min_intv) as usize;
        unsafe {
            std::ptr::write(prev_array_buf.as_mut_ptr().add(prev_array_buf.len()), smem);
            prev_array_buf.set_len(prev_array_buf.len() + mask);
        }

        // Phase 2: Backward search
        prev_array_buf.reverse();

        for j in (0..x).rev() {
            let a = encoded_query[j];
            if a >= 4 {
                break;
            }

            curr_array_buf.clear();
            let mut curr_s: i64 = -1;
            let num_prev = prev_array_buf.len();

            // First loop: process elements until we find one to output or keep
            let mut p = 0;
            while p < num_prev {
                let smem = prev_array_buf[p];
                let mut new_smem = backward_ext(bwa_idx, smem, a);
                new_smem.query_start = j as i32;

                // Output if interval dropped below threshold and length sufficient
                if new_smem.interval_size < min_intv
                    && (smem.query_end - smem.query_start + 1) >= min_seed_len
                {
                    all_smems.push(smem);
                    break;
                }

                // Keep if above threshold and unique interval size
                if new_smem.interval_size >= min_intv && (new_smem.interval_size as i64) != curr_s {
                    curr_s = new_smem.interval_size as i64;
                    unsafe {
                        prefetch_bwt(bwa_idx, &new_smem);
                    }
                    curr_array_buf.push(new_smem);
                    break;
                }

                p += 1;
            }

            // Second loop: continue from p+1 to process remaining elements
            p += 1;
            while p < num_prev {
                let smem = prev_array_buf[p];
                let mut new_smem = backward_ext(bwa_idx, smem, a);
                new_smem.query_start = j as i32;

                if new_smem.interval_size >= min_intv && (new_smem.interval_size as i64) != curr_s {
                    curr_s = new_smem.interval_size as i64;
                    unsafe {
                        prefetch_bwt(bwa_idx, &new_smem);
                    }
                    curr_array_buf.push(new_smem);
                }

                p += 1;
            }

            std::mem::swap(prev_array_buf, curr_array_buf);
            *max_smem_count = (*max_smem_count).max(prev_array_buf.len());

            if prev_array_buf.is_empty() {
                break;
            }
        }

        // Output remaining SMEM (first element after reversal = longest SMEM)
        if !prev_array_buf.is_empty() {
            let smem = prev_array_buf[0];
            let len = smem.query_end - smem.query_start + 1;
            if len >= min_seed_len {
                all_smems.push(smem);
            }
        }

        x = next_x;
    }
}

/// Generate SMEMs from a single starting position with custom min_intv.
///
/// This is used for re-seeding long unique MEMs to find split alignments.
pub fn generate_smems_from_position<'a>(
    bwa_idx: &BwaIndex,
    _query_name: &str,
    query_len: usize,
    encoded_query: &[u8],
    is_reverse_complement: bool,
    min_seed_len: i32,
    min_intv: u64,
    start_pos: usize,
    all_smems: &mut Vec<SMEM>,
    prev_array_buf: &'a mut Vec<SMEM>,
    curr_array_buf: &'a mut Vec<SMEM>,
) {
    prev_array_buf.clear();
    curr_array_buf.clear();
    prev_array_buf.reserve(query_len.saturating_sub(start_pos));

    if start_pos >= query_len {
        return;
    }

    let a = encoded_query[start_pos];
    if a >= 4 {
        return;
    }

    let mut smem = SMEM {
        read_id: 0,
        query_start: start_pos as i32,
        query_end: start_pos as i32,
        bwt_interval_start: bwa_idx.bwt.cumulative_count[a as usize],
        bwt_interval_end: bwa_idx.bwt.cumulative_count[(3 - a) as usize],
        interval_size: bwa_idx.bwt.cumulative_count[(a + 1) as usize]
            - bwa_idx.bwt.cumulative_count[a as usize],
        is_reverse_complement,
    };

    // Phase 1: Forward extension
    for j in (start_pos + 1)..query_len {
        let a = encoded_query[j];
        if a >= 4 {
            break;
        }

        let new_smem = forward_ext(bwa_idx, smem, a);

        let mask = (new_smem.interval_size != smem.interval_size) as usize;
        unsafe {
            std::ptr::write(prev_array_buf.as_mut_ptr().add(prev_array_buf.len()), smem);
            prev_array_buf.set_len(prev_array_buf.len() + mask);
        }

        if new_smem.interval_size < min_intv {
            break;
        }

        smem = new_smem;
        smem.query_end = j as i32;
    }

    let mask = (smem.interval_size >= min_intv) as usize;
    unsafe {
        std::ptr::write(prev_array_buf.as_mut_ptr().add(prev_array_buf.len()), smem);
        prev_array_buf.set_len(prev_array_buf.len() + mask);
    }

    // Phase 2: Backward search
    prev_array_buf.reverse();

    for j in (0..start_pos).rev() {
        let a = encoded_query[j];
        if a >= 4 {
            break;
        }

        curr_array_buf.clear();
        let mut curr_s: i64 = -1;
        let num_prev = prev_array_buf.len();

        let mut p = 0;
        while p < num_prev {
            let smem = prev_array_buf[p];
            let mut new_smem = backward_ext(bwa_idx, smem, a);
            new_smem.query_start = j as i32;

            if new_smem.interval_size < min_intv
                && (smem.query_end - smem.query_start + 1) >= min_seed_len
            {
                all_smems.push(smem);
                break;
            }

            if new_smem.interval_size >= min_intv && (new_smem.interval_size as i64) != curr_s {
                curr_s = new_smem.interval_size as i64;
                unsafe {
                    prefetch_bwt(bwa_idx, &new_smem);
                }
                curr_array_buf.push(new_smem);
                break;
            }

            p += 1;
        }

        p += 1;
        while p < num_prev {
            let smem = prev_array_buf[p];
            let mut new_smem = backward_ext(bwa_idx, smem, a);
            new_smem.query_start = j as i32;

            if new_smem.interval_size >= min_intv && (new_smem.interval_size as i64) != curr_s {
                curr_s = new_smem.interval_size as i64;
                unsafe {
                    prefetch_bwt(bwa_idx, &new_smem);
                }
                curr_array_buf.push(new_smem);
            }

            p += 1;
        }

        std::mem::swap(prev_array_buf, curr_array_buf);

        if prev_array_buf.is_empty() {
            break;
        }
    }

    if !prev_array_buf.is_empty() {
        let smem = prev_array_buf[0];
        let len = smem.query_end - smem.query_start + 1;
        if len >= min_seed_len {
            all_smems.push(smem);
        }
    }
}

/// Forward-only seed strategy matching BWA-MEM2's bwtSeedStrategyAllPosOneThread.
///
/// This is a simpler seeding algorithm that:
/// 1. Iterates through all positions in the query
/// 2. Does forward extension only (no backward phase)
/// 3. Outputs seeds when interval drops BELOW max_intv (specific enough)
/// 4. Uses min_seed_len + 1 as the minimum seed length
pub fn forward_only_seed_strategy(
    bwa_idx: &BwaIndex,
    _query_name: &str,
    query_len: usize,
    encoded_query: &[u8],
    is_reverse_complement: bool,
    min_seed_len: i32,
    max_intv: u64,
    all_smems: &mut Vec<SMEM>,
) {
    let min_len = min_seed_len + 1;
    let mut x = 0;

    while x < query_len {
        let a = encoded_query[x];
        let mut next_x = x + 1;

        if a >= 4 {
            x = next_x;
            continue;
        }

        let mut smem = SMEM {
            read_id: 0,
            query_start: x as i32,
            query_end: x as i32,
            bwt_interval_start: bwa_idx.bwt.cumulative_count[a as usize],
            bwt_interval_end: bwa_idx.bwt.cumulative_count[(3 - a) as usize],
            interval_size: bwa_idx.bwt.cumulative_count[(a + 1) as usize]
                - bwa_idx.bwt.cumulative_count[a as usize],
            is_reverse_complement,
        };

        for j in (x + 1)..query_len {
            next_x = j + 1;
            let a = encoded_query[j];

            if a >= 4 {
                break;
            }

            let new_smem = forward_ext(bwa_idx, smem, a);
            smem = new_smem;
            smem.query_end = j as i32;

            let len = smem.query_end - smem.query_start + 1;
            if smem.interval_size < max_intv && len >= min_len {
                if smem.interval_size > 0 {
                    all_smems.push(smem);
                }
                break;
            }
        }

        x = next_x;
    }
}
