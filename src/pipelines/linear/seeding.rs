use super::index::fm_index::CP_SHIFT;
use super::index::fm_index::CpOcc;
use super::index::fm_index::backward_ext;
use super::index::fm_index::forward_ext;
use super::index::fm_index::get_occ;
use super::index::index::BwaIndex;
use crate::core::compute::simd_abstraction::portable_intrinsics;

// Define a struct to represent a seed
#[derive(Debug, Clone)]
pub struct Seed {
    pub query_pos: i32,     // Position in the query
    pub ref_pos: u64,       // Position in the reference
    pub len: i32,           // Length of the seed
    pub is_rev: bool,       // Is it on the reverse strand?
    pub interval_size: u64, // BWT interval size (occurrence count)
    pub rid: i32,           // Reference sequence ID (chromosome), -1 if spans boundaries
}

// Define a struct to represent a Super Maximal Exact Match (SMEM)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SMEM {
    /// Read identifier (for batch processing)
    pub read_id: i32,
    /// Start position in query sequence (0-based, inclusive)
    pub query_start: i32,
    /// End position in query sequence (0-based, exclusive)
    pub query_end: i32,
    /// Start of BWT interval in suffix array
    pub bwt_interval_start: u64,
    /// End of BWT interval in suffix array
    pub bwt_interval_end: u64,
    /// Size of BWT interval (bwt_interval_end - bwt_interval_start)
    pub interval_size: u64,
    /// Whether this SMEM is from the reverse complement strand
    pub is_reverse_complement: bool,
}

/// Generate SMEMs for a single strand (forward or reverse complement)
///
/// The `prev_array_buf` and `curr_array_buf` parameters are working buffers
/// that should be pre-allocated and reused across calls to avoid allocation overhead.
pub fn generate_smems_for_strand<'a>(
    bwa_idx: &BwaIndex,
    query_name: &str,
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
    // Clear buffers for reuse (retains capacity)
    prev_array_buf.clear();
    curr_array_buf.clear();

    let mut x = 0;
    while x < query_len {
        let a = encoded_query[x];

        if a >= 4 {
            // Skip 'N' bases
            x += 1;
            continue;
        }

        // Initialize SMEM at position x
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

        if x == 0 && !is_reverse_complement {
            log::debug!(
                "{}: Initial SMEM at x={}: a={}, k={}, l={}, s={}, l2[{}]={}, l2[{}]={}",
                query_name,
                x,
                a,
                smem.bwt_interval_start,
                smem.bwt_interval_end,
                smem.interval_size,
                a,
                bwa_idx.bwt.cumulative_count[a as usize],
                3 - a,
                bwa_idx.bwt.cumulative_count[(3 - a) as usize]
            );
        }

        // Phase 1: Forward extension
        prev_array_buf.clear();
        // Pre-reserve capacity for branchless append (max possible SMEMs is remaining positions)
        prev_array_buf.reserve(query_len.saturating_sub(x));
        let mut next_x = x + 1;

        for j in (x + 1)..query_len {
            let a = encoded_query[j];
            next_x = j + 1;

            if a >= 4 {
                if x == 0 && !is_reverse_complement && log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "{query_name}: x={x}, forward extension stopped at j={j} due to N base"
                    );
                }
                next_x = j;
                break;
            }

            let new_smem = forward_ext(bwa_idx, smem, a);

            if x == 0 && j <= 12 && !is_reverse_complement {
                log::debug!(
                    "{}: x={}, j={}, a={}, old_smem.interval_size={}, new_smem(k={}, l={}, s={})",
                    query_name,
                    x,
                    j,
                    a,
                    smem.interval_size,
                    new_smem.bwt_interval_start,
                    new_smem.bwt_interval_end,
                    new_smem.interval_size
                );
            }

            // Debug logging (only for first few positions on forward strand)
            if x < 3 && !is_reverse_complement && new_smem.interval_size != smem.interval_size {
                let s_from_lk = smem
                    .bwt_interval_end
                    .saturating_sub(smem.bwt_interval_start);
                log::debug!(
                    "{}: x={}, j={}, pushing smem to prev_array_buf: s={}, l-k={}, match={}",
                    query_name,
                    x,
                    j,
                    smem.interval_size,
                    s_from_lk,
                    smem.interval_size == s_from_lk
                );
            }
            // Branchless append - matches C++ FMI_search.cpp:556-559
            // int32_t s_neq_mask = newSmem.s != smem.s;
            // prevArray[numPrev] = smem;
            // numPrev += s_neq_mask;
            // SAFETY: capacity was pre-reserved for query_len elements
            let mask = (new_smem.interval_size != smem.interval_size) as usize;
            unsafe {
                std::ptr::write(prev_array_buf.as_mut_ptr().add(prev_array_buf.len()), smem);
                prev_array_buf.set_len(prev_array_buf.len() + mask);
            }

            if new_smem.interval_size < min_intv {
                if x == 0 && !is_reverse_complement && log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "{}: x={}, forward extension stopped at j={} because new_smem.interval_size={} < min_intv={}",
                        query_name,
                        x,
                        j,
                        new_smem.interval_size,
                        min_intv
                    );
                }
                // Match BWA-MEM2: set next_x = j (not j+1) when interval drops below min_intv
                // This ensures we try position j as a starting point in the next iteration
                next_x = j;
                break;
            }

            smem = new_smem;
            smem.query_end = j as i32;
        }

        // Branchless append for final SMEM after forward loop
        // SAFETY: capacity was pre-reserved for query_len elements
        let mask = (smem.interval_size >= min_intv) as usize;
        unsafe {
            std::ptr::write(prev_array_buf.as_mut_ptr().add(prev_array_buf.len()), smem);
            prev_array_buf.set_len(prev_array_buf.len() + mask);
        }

        if x < 3 && !is_reverse_complement {
            log::debug!(
                "{}: Position x={}, prev_array_buf.len()={}, smem.interval_size={}, min_intv={}",
                query_name,
                x,
                prev_array_buf.len(),
                smem.interval_size,
                min_intv
            );
        }

        // Phase 2: Backward search
        // BWA-MEM2 reverses prev array in-place before backward iteration (FMI_search.cpp:587-592)
        // This puts the longest SMEM (last added during forward) at index 0
        prev_array_buf.reverse();

        if !is_reverse_complement {
            log::debug!(
                "{}: [RUST Phase 2] Starting backward search from x={}, prev_array_buf.len()={} (reversed)",
                query_name,
                x,
                prev_array_buf.len()
            );
        }

        for j in (0..x).rev() {
            let a = encoded_query[j];
            if a >= 4 {
                if !is_reverse_complement {
                    log::debug!("{query_name}: [RUST Phase 2] Hit 'N' base at j={j}, stopping");
                }
                break;
            }

            curr_array_buf.clear();
            let mut curr_s: i64 = -1; // BWA-MEM2 uses int curr_s = -1
            let num_prev = prev_array_buf.len();

            if !is_reverse_complement {
                log::debug!("{query_name}: [RUST Phase 2] j={j}, base={a}, num_prev={num_prev}");
            }

            // First loop: process elements until we find one to output or keep
            // BWA-MEM2: FMI_search.cpp lines 607-630
            let mut p = 0;
            while p < num_prev {
                let smem = prev_array_buf[p];
                let mut new_smem = backward_ext(bwa_idx, smem, a);
                new_smem.query_start = j as i32;

                if !is_reverse_complement {
                    let old_len = smem.query_end - smem.query_start + 1;
                    let new_len = new_smem.query_end - new_smem.query_start + 1;
                    log::debug!(
                        "{}: [RUST Phase 2] x={}, j={}, p={}: old_smem(m={},n={},len={},k={},l={},s={}), new_smem(m={},n={},len={},k={},l={},s={}), min_intv={}",
                        query_name,
                        x,
                        j,
                        p,
                        smem.query_start,
                        smem.query_end,
                        old_len,
                        smem.bwt_interval_start,
                        smem.bwt_interval_end,
                        smem.interval_size,
                        new_smem.query_start,
                        new_smem.query_end,
                        new_len,
                        new_smem.bwt_interval_start,
                        new_smem.bwt_interval_end,
                        new_smem.interval_size,
                        min_intv
                    );
                }

                // Output condition: interval dropped below threshold AND length sufficient
                if new_smem.interval_size < min_intv
                    && (smem.query_end - smem.query_start + 1) >= min_seed_len
                {
                    if !is_reverse_complement {
                        log::debug!(
                            "{}: [RUST SMEM OUTPUT] Phase2: smem(m={},n={},k={},l={},s={}) newSmem.s={} < min_intv={}",
                            query_name,
                            smem.query_start,
                            smem.query_end,
                            smem.bwt_interval_start,
                            smem.bwt_interval_end,
                            smem.interval_size,
                            new_smem.interval_size,
                            min_intv
                        );
                    }
                    all_smems.push(smem);
                    break; // BWA-MEM2: break after output
                }

                // Keep condition: interval still above threshold AND unique interval size
                if new_smem.interval_size >= min_intv && (new_smem.interval_size as i64) != curr_s {
                    curr_s = new_smem.interval_size as i64;
                    // OPTIMIZATION: Manually prefetch data for the next likely BWT access
                    unsafe {
                        let prefetch_addr = bwa_idx
                            .cp_occ
                            .as_ptr()
                            .add((new_smem.bwt_interval_start >> CP_SHIFT) as usize)
                            as *const i8;
                        #[cfg(target_arch = "x86_64")]
                        {
                            core::arch::x86_64::_mm_prefetch(
                                prefetch_addr,
                                core::arch::x86_64::_MM_HINT_T0,
                            );
                        }
                        #[cfg(target_arch = "aarch64")]
                        {
                            // ARM prefetch intrinsics are unstable in stable Rust.
                            // Use inline assembly for prefetch on aarch64, which is stable.
                            // PRFM PLDL1KEEP is equivalent to prefetch for read with high locality.
                            core::arch::asm!(
                                "prfm pldl1keep, [{addr}]",
                                addr = in(reg) prefetch_addr,
                                options(nostack, preserves_flags)
                            );
                        }
                    }
                    curr_array_buf.push(new_smem);
                    if !is_reverse_complement {
                        log::debug!(
                            "{}: [RUST Phase 2] Keeping new_smem (s={} >= min_intv={}), breaking",
                            query_name,
                            new_smem.interval_size,
                            min_intv
                        );
                    }
                    break; // BWA-MEM2: break after keeping first valid interval
                }

                if !is_reverse_complement {
                    log::debug!(
                        "{}: [RUST Phase 2] Rejecting (s={} < min_intv={} OR already_seen={})",
                        query_name,
                        new_smem.interval_size,
                        min_intv,
                        (new_smem.interval_size as i64) == curr_s
                    );
                }

                p += 1;
            }

            // Second loop: continue from p+1 to process remaining elements
            // BWA-MEM2: FMI_search.cpp lines 631-649: "p++; for(; p < numPrev; p++)"
            p += 1;
            while p < num_prev {
                let smem = prev_array_buf[p];
                let mut new_smem = backward_ext(bwa_idx, smem, a);
                new_smem.query_start = j as i32;

                if !is_reverse_complement {
                    let new_len = new_smem.query_end - new_smem.query_start + 1;
                    log::debug!(
                        "{}: [RUST Phase 2] x={}, j={}, remaining_p={}: smem(m={},n={},s={}), new_smem(m={},n={},len={},s={}), will_push={}",
                        query_name,
                        x,
                        j,
                        p,
                        smem.query_start,
                        smem.query_end,
                        smem.interval_size,
                        new_smem.query_start,
                        new_smem.query_end,
                        new_len,
                        new_smem.interval_size,
                        new_smem.interval_size >= min_intv
                            && (new_smem.interval_size as i64) != curr_s
                    );
                }

                // Keep if above threshold and unique interval size
                if new_smem.interval_size >= min_intv && (new_smem.interval_size as i64) != curr_s {
                    curr_s = new_smem.interval_size as i64;
                    // OPTIMIZATION: Manually prefetch data for the next likely BWT access
                    // The prefetch_read_t0 function handles architecture-specific intrinsics
                    unsafe {
                        let prefetch_addr = bwa_idx
                            .cp_occ
                            .as_ptr()
                            .add((new_smem.bwt_interval_start >> CP_SHIFT) as usize)
                            as *const i8;
                        portable_intrinsics::prefetch_read_t0(prefetch_addr);
                    }
                    curr_array_buf.push(new_smem);
                }

                p += 1;
            }

            std::mem::swap(prev_array_buf, curr_array_buf);
            *max_smem_count = (*max_smem_count).max(prev_array_buf.len());

            if !is_reverse_complement {
                log::debug!(
                    "{}: [RUST Phase 2] After j={}, prev_array_buf.len()={}",
                    query_name,
                    j,
                    prev_array_buf.len()
                );
            }

            if prev_array_buf.is_empty() {
                if !is_reverse_complement {
                    log::debug!(
                        "{query_name}: [RUST Phase 2] prev_array_buf empty, breaking at j={j}"
                    );
                }
                break;
            }
        }

        // Output remaining SMEM: BWA-MEM2 takes prev[0] (longest SMEM after reversal)
        // FMI_search.cpp lines 656-664
        if !prev_array_buf.is_empty() {
            let smem = prev_array_buf[0]; // First element after in-place reversal = longest SMEM
            let len = smem.query_end - smem.query_start + 1;
            if len >= min_seed_len {
                if !is_reverse_complement {
                    log::debug!(
                        "{}: [RUST SMEM OUTPUT] Phase2 final: smem(m={},n={},k={},l={},s={}), len={}, next_x={}",
                        query_name,
                        smem.query_start,
                        smem.query_end,
                        smem.bwt_interval_start,
                        smem.bwt_interval_end,
                        smem.interval_size,
                        len,
                        next_x
                    );
                }
                all_smems.push(smem);
            } else if !is_reverse_complement {
                log::debug!(
                    "{}: [RUST Phase 2] Rejecting final SMEM: m={}, n={}, len={} < min_seed_len={}, s={}",
                    query_name,
                    smem.query_start,
                    smem.query_end,
                    len,
                    min_seed_len,
                    smem.interval_size
                );
            }
        } else if !is_reverse_complement {
            log::debug!(
                "{query_name}: [RUST Phase 2] No remaining SMEMs at end of backward search for x={x}"
            );
        }

        x = next_x;
    }
}

/// Generate SMEMs from a single starting position with custom min_intv
/// This is used for re-seeding long unique MEMs to find split alignments
///
/// The `prev_array_buf` and `curr_array_buf` parameters are working buffers
/// that should be pre-allocated and reused across calls to avoid allocation overhead.
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
    // Clear buffers for reuse (retains capacity)
    prev_array_buf.clear();
    curr_array_buf.clear();
    // Pre-reserve capacity for branchless append
    prev_array_buf.reserve(query_len.saturating_sub(start_pos));

    if start_pos >= query_len {
        return;
    }

    let a = encoded_query[start_pos];
    if a >= 4 {
        return; // Skip if starting on 'N' base
    }

    // Initialize SMEM at start_pos
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

    // Phase 1: Forward extension (uses prev_array_buf)

    for j in (start_pos + 1)..query_len {
        let a = encoded_query[j];

        if a >= 4 {
            break;
        }

        let new_smem = forward_ext(bwa_idx, smem, a);

        // Branchless append - matches C++ FMI_search.cpp:556-559
        // SAFETY: capacity was pre-reserved
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

    // Branchless append for final SMEM after forward loop
    // SAFETY: capacity was pre-reserved
    let mask = (smem.interval_size >= min_intv) as usize;
    unsafe {
        std::ptr::write(prev_array_buf.as_mut_ptr().add(prev_array_buf.len()), smem);
        prev_array_buf.set_len(prev_array_buf.len() + mask);
    }

    // Phase 2: Backward search
    // BWA-MEM2 reverses prev array in-place before backward iteration
    prev_array_buf.reverse();

    // curr_array_buf is already cleared at function start

    for j in (0..start_pos).rev() {
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

            if new_smem.interval_size < min_intv
                && (smem.query_end - smem.query_start + 1) >= min_seed_len
            {
                all_smems.push(smem);
                break;
            }

            if new_smem.interval_size >= min_intv && (new_smem.interval_size as i64) != curr_s {
                curr_s = new_smem.interval_size as i64;
                // OPTIMIZATION: Manually prefetch data for the next likely BWT access
                unsafe {
                    let prefetch_addr = bwa_idx
                        .cp_occ
                        .as_ptr()
                        .add((new_smem.bwt_interval_start >> CP_SHIFT) as usize)
                        as *const i8;
                    portable_intrinsics::prefetch_read_t0(prefetch_addr);
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
                // OPTIMIZATION: Manually prefetch data for the next likely BWT access
                unsafe {
                    let prefetch_addr = bwa_idx
                        .cp_occ
                        .as_ptr()
                        .add((new_smem.bwt_interval_start >> CP_SHIFT) as usize)
                        as *const i8;
                    portable_intrinsics::prefetch_read_t0(prefetch_addr);
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

    // Output remaining SMEM: takes prev[0] (longest SMEM after reversal)
    if !prev_array_buf.is_empty() {
        let smem = prev_array_buf[0];
        let len = smem.query_end - smem.query_start + 1;
        if len >= min_seed_len {
            all_smems.push(smem);
        }
    }
}

// ============================================================================
// BWT AND SUFFIX ARRAY HELPER FUNCTIONS
// ============================================================================
//
// This section contains low-level BWT and suffix array access functions
// used during FM-Index search and seed extension
// ============================================================================

/// Get BWT base from cp_occ format (for loaded indices)
/// Returns 0-3 for bases A/C/G/T, or 4 for sentinel
///
/// This is an ultra-hot path called millions of times during seeding.
/// Optimized to match C++ BWA-MEM2 FMI_search.cpp lines 1131-1140:
/// - No bounds checking (caller must ensure valid pos)
/// - If-else chain instead of loop (better branch prediction)
/// - Inline hint for compiler
#[inline(always)]
pub fn get_bwt_base_from_cp_occ(cp_occ: &[CpOcc], pos: u64) -> u8 {
    let cp_block = (pos >> CP_SHIFT) as usize;

    // SAFETY: Caller must ensure pos is within valid BWT range.
    // In release builds we use unchecked access for performance.
    // In debug builds we still have bounds checking via normal indexing.
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

    // Use unchecked access in release builds - this eliminates bounds checks
    // that add ~5% overhead in this hot path
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

// Function to get the next BWT position from a BWT coordinate
// Returns None if we hit the sentinel (which should not be navigated)
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

pub fn get_sa_entries(
    bwa_idx: &BwaIndex,
    bwt_interval_start: u64,
    interval_size: u64,
    max_occurrences: u32,
) -> Vec<u64> {
    let mut ref_positions = Vec::new();

    // For repetitive seeds, we need to sample evenly across the ENTIRE interval
    // to ensure we cover all reference positions, not just a clustered subset.
    //
    // BWA-MEM2 uses integer step = interval_size / max_occ, but this can leave
    // gaps when interval_size is not much larger than max_occ.
    //
    // We use floating-point arithmetic to ensure even distribution across the
    // full interval range.
    let num_to_retrieve = (interval_size as u32).min(max_occurrences);

    let actual_num = num_to_retrieve;

    if num_to_retrieve == 0 {
        return ref_positions;
    }

    // Use floating-point step to ensure we cover the entire interval
    // For interval_size=713 and num_to_retrieve=250, this gives step=2.852
    // which samples positions 0, 2.852, 5.704, ... -> 0, 2, 5, 8, ...
    let step = if interval_size > actual_num as u64 {
        interval_size as f64 / actual_num as f64
    } else {
        1.0
    };

    for i in 0..actual_num {
        let k = (i as f64 * step) as u64;
        if k >= interval_size {
            break;
        }
        let sa_index = bwt_interval_start + k;
        let ref_pos = get_sa_entry(bwa_idx, sa_index);
        ref_positions.push(ref_pos);
    }

    ref_positions
}

pub fn get_sa_entry(bwa_idx: &BwaIndex, mut pos: u64) -> u64 {
    let original_pos = pos;
    let mut count = 0;
    const MAX_ITERATIONS: u64 = 10000; // Safety limit to prevent infinite loops

    // eprintln!("get_sa_entry: starting with pos={}, sa_intv={}, seq_len={}, cp_occ.len()={}",
    //           original_pos, bwa_idx.bwt.sa_sample_interval, bwa_idx.bwt.seq_len, bwa_idx.cp_occ.len());

    while pos % bwa_idx.bwt.sa_sample_interval as u64 != 0 {
        // Safety check: prevent infinite loops
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
            return count; // Return what we have so far
        }

        let _old_pos = pos;
        match get_bwt(bwa_idx, pos) {
            Some(new_pos) => {
                pos = new_pos;
                count += 1;
                // if count <= 10 {
                //     eprintln!("  BWT step {}: pos {} -> {} (count={})", count, _old_pos, pos, count);
                // }
            }
            None => {
                // Hit sentinel - return the accumulated count
                // eprintln!("  BWT step {}: pos {} -> SENTINEL (count={})", count + 1, _old_pos, count);
                // eprintln!("get_sa_entry: original_pos={}, hit_sentinel, count={}, result={}",
                //           original_pos, count, count);
                return count;
            }
        }
    }

    let sa_index = (pos / bwa_idx.bwt.sa_sample_interval as u64) as usize;
    let sa_ms_byte = bwa_idx.bwt.sa_high_bytes[sa_index] as u64;
    let sa_ls_word = bwa_idx.bwt.sa_low_words[sa_index] as u64;
    let sa_val = (sa_ms_byte << 32) | sa_ls_word;

    // Handle sentinel: SA values can point to the sentinel position (seq_len)
    // The sentinel represents the end-of-string marker, which wraps to position 0
    // seq_len = (l_pac << 1) + 1 (forward + RC + sentinel)
    // So sentinel position is seq_len - 1 = (l_pac << 1)
    let sentinel_pos = bwa_idx.bns.packed_sequence_length << 1;
    let adjusted_sa_val = if sa_val >= sentinel_pos {
        // SA points to or past sentinel - wrap to beginning (position 0)
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

/// Forward-only seed strategy matching BWA-MEM2's bwtSeedStrategyAllPosOneThread
///
/// This is a simpler seeding algorithm that:
/// 1. Iterates through all positions in the query
/// 2. Does forward extension only (no backward phase)
/// 3. Outputs seeds when interval drops BELOW max_intv (not above)
/// 4. Uses min_seed_len + 1 as the minimum seed length
///
/// This finds seeds that might be missed by the supermaximal SMEM algorithm,
/// particularly in reads with many mismatches where seeds are fragmented.
pub fn forward_only_seed_strategy(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_len: usize,
    encoded_query: &[u8],
    is_reverse_complement: bool,
    min_seed_len: i32,
    max_intv: u64,
    all_smems: &mut Vec<SMEM>,
) {
    let min_len = min_seed_len + 1; // BWA-MEM2 uses min_seed_len + 1 for 3rd round
    let mut x = 0;

    while x < query_len {
        let a = encoded_query[x];
        let mut next_x = x + 1;

        if a >= 4 {
            // Skip 'N' bases
            x = next_x;
            continue;
        }

        // Initialize SMEM at position x
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

        // Forward extension only
        for j in (x + 1)..query_len {
            next_x = j + 1;
            let a = encoded_query[j];

            if a >= 4 {
                // Hit 'N' base - stop extension
                break;
            }

            let new_smem = forward_ext(bwa_idx, smem, a);
            smem = new_smem;
            smem.query_end = j as i32;

            // Output seed when interval drops BELOW max_intv (specific enough)
            // AND length meets minimum requirement
            let len = smem.query_end - smem.query_start + 1;
            if smem.interval_size < max_intv && len >= min_len {
                if smem.interval_size > 0 {
                    log::debug!(
                        "{}: forward_only_seed: m={}, n={}, len={}, s={} (< max_intv={}), is_rc={}",
                        query_name,
                        smem.query_start,
                        smem.query_end,
                        len,
                        smem.interval_size,
                        max_intv,
                        is_reverse_complement
                    );
                    all_smems.push(smem);
                }
                break;
            }
        }

        x = next_x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_backward_ext() {
        let prefix = Path::new("test_data/test_ref.fa");

        // Skip if test data doesn't exist
        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext - could not load index");
                return;
            }
        };

        let smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };
        let new_smem = backward_ext(&bwa_idx, smem, 0); // 0 is 'A'
        assert_ne!(new_smem.interval_size, 0);
    }

    #[test]
    fn test_backward_ext_multiple_bases() {
        // Test backward extension with all four bases
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_multiple_bases - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_multiple_bases - could not load index");
                return;
            }
        };

        // Start with full range
        let initial_smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Test extending with each base
        for base in 0..4 {
            let extended = backward_ext(&bwa_idx, initial_smem, base);

            // Extended range should be smaller or equal to initial range
            assert!(
                extended.interval_size <= initial_smem.interval_size,
                "Extended range size {} should be <= initial size {} for base {}",
                extended.interval_size,
                initial_smem.interval_size,
                base
            );

            // k should be within bounds
            assert!(
                extended.bwt_interval_start < bwa_idx.bwt.seq_len,
                "Extended k should be within sequence length"
            );
        }
    }

    #[test]
    fn test_backward_ext_chain() {
        // Test chaining multiple backward extensions (like building a seed)
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_chain - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_chain - could not load index");
                return;
            }
        };

        // Start with full range
        let mut smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Build a seed by extending with ACGT
        let bases = [0u8, 1, 2, 3]; // ACGT
        let mut prev_s = smem.interval_size;

        for (i, &base) in bases.iter().enumerate() {
            smem = backward_ext(&bwa_idx, smem, base);

            // Range should generally get smaller (or stay same) with each extension
            // (though it could stay the same if the pattern is very common)
            assert!(
                smem.interval_size <= prev_s,
                "After extension {}, range size {} should be <= previous {}",
                i,
                smem.interval_size,
                prev_s
            );

            prev_s = smem.interval_size;

            // If range becomes 0, we can't extend further
            if smem.interval_size == 0 {
                break;
            }
        }
    }

    #[test]
    fn test_backward_ext_zero_range() {
        // Test backward extension when starting with zero range
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_zero_range - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_zero_range - could not load index");
                return;
            }
        };

        let smem = SMEM {
            bwt_interval_start: 0,
            interval_size: 0, // Zero range
            ..Default::default()
        };

        let extended = backward_ext(&bwa_idx, smem, 0);

        // Extending a zero range should still give zero range
        assert_eq!(
            extended.interval_size, 0,
            "Extending zero range should give zero range"
        );
    }

    #[test]
    fn test_smem_structure() {
        // Test SMEM structure creation and defaults
        let smem1 = SMEM {
            read_id: 0,
            query_start: 10,
            query_end: 20,
            bwt_interval_start: 5,
            bwt_interval_end: 15,
            interval_size: 10,
            is_reverse_complement: false,
        };

        assert_eq!(smem1.query_start, 10);
        assert_eq!(smem1.query_end, 20);
        assert_eq!(smem1.interval_size, 10);

        // Test default
        let smem2 = SMEM::default();
        assert_eq!(smem2.read_id, 0);
        assert_eq!(smem2.query_start, 0);
        assert_eq!(smem2.query_end, 0);
    }

    #[test]
    fn test_get_sa_entry_basic() {
        // This test requires an actual index file to be present
        // We'll use a simple test to verify the function doesn't crash
        let prefix = Path::new("test_data/test_ref.fa");

        // Only run if test data exists
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

        // Test getting SA entry at position 0 (should return a valid reference position)
        let sa_entry = get_sa_entry(&bwa_idx, 0);

        // SA entry should be within the reference sequence length
        assert!(
            sa_entry < bwa_idx.bwt.seq_len,
            "SA entry {} should be less than seq_len {}",
            sa_entry,
            bwa_idx.bwt.seq_len
        );
    }

    #[test]
    fn test_get_sa_entry_sampled_position() {
        // Test getting SA entry at a sampled position (divisible by sa_intv)
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

        // Test at a sampled position (should directly lookup in SA array)
        let sampled_pos = bwa_idx.bwt.sa_sample_interval as u64;
        let sa_entry = get_sa_entry(&bwa_idx, sampled_pos);

        assert!(
            sa_entry < bwa_idx.bwt.seq_len,
            "SA entry at sampled position should be within sequence length"
        );
    }

    #[test]
    fn test_get_sa_entry_multiple_positions() {
        // Test getting SA entries for multiple positions
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_multiple_positions - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_multiple_positions - could not load index");
                return;
            }
        };

        // Test several positions
        let test_positions = vec![0u64, 1, 10, 100];

        for pos in test_positions {
            if pos >= bwa_idx.bwt.seq_len {
                continue;
            }

            let sa_entry = get_sa_entry(&bwa_idx, pos);

            // All SA entries should be valid (within sequence length)
            assert!(
                sa_entry < bwa_idx.bwt.seq_len,
                "SA entry for pos {pos} should be within sequence length"
            );
        }
    }

    #[test]
    fn test_get_sa_entry_consistency() {
        // Test that get_sa_entry returns consistent results
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

        // Same position should always return same SA entry
        assert_eq!(
            sa_entry1, sa_entry2,
            "get_sa_entry should return consistent results for the same position"
        );
    }

    #[test]
    fn test_get_bwt_basic() {
        // Test get_bwt function
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

        // Test getting BWT at various positions
        for pos in 0..10u64 {
            let bwt_result = get_bwt(&bwa_idx, pos);

            // Either we get a valid position or None (sentinel)
            if let Some(new_pos) = bwt_result {
                assert!(
                    new_pos < bwa_idx.bwt.seq_len,
                    "BWT position should be within sequence length"
                );
            }
            // If None, we hit the sentinel - that's ok
        }
    }
}
