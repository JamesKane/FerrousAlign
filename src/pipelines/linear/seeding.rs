use super::index::fm_index::CP_SHIFT;
use super::index::fm_index::CpOcc;
use super::index::fm_index::backward_ext;
use super::index::fm_index::forward_ext;
use super::index::fm_index::get_occ;
use super::index::index::BwaIndex;
use super::mem_opt::MemOpt;
use crate::alignment::utils::{base_to_code, reverse_complement_code};
use crate::alignment::workspace::with_workspace;
use crate::core::compute::simd_abstraction::portable_intrinsics;
pub use crate::core::io::soa_readers::SoAReadBatch;

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

// Define a struct to represent a batch of seeds in SoA format
#[derive(Debug, Clone, Default)]
pub struct SoASeedBatch {
    pub query_pos: Vec<i32>,
    pub ref_pos: Vec<u64>,
    pub len: Vec<i32>,
    pub is_rev: Vec<bool>,
    pub interval_size: Vec<u64>,
    pub rid: Vec<i32>,
    // Store (start_idx, count) for seeds belonging to each read in the batch
    pub read_seed_boundaries: Vec<(usize, usize)>,
}

impl SoASeedBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize, num_reads: usize) -> Self {
        Self {
            query_pos: Vec::with_capacity(capacity),
            ref_pos: Vec::with_capacity(capacity),
            len: Vec::with_capacity(capacity),
            is_rev: Vec::with_capacity(capacity),
            interval_size: Vec::with_capacity(capacity),
            rid: Vec::with_capacity(capacity),
            read_seed_boundaries: Vec::with_capacity(num_reads),
        }
    }

    pub fn clear(&mut self) {
        self.query_pos.clear();
        self.ref_pos.clear();
        self.len.clear();
        self.is_rev.clear();
        self.interval_size.clear();
        self.rid.clear();
        self.read_seed_boundaries.clear();
    }

    pub fn push(&mut self, seed: &Seed, _read_idx: usize) {
        self.query_pos.push(seed.query_pos);
        self.ref_pos.push(seed.ref_pos);
        self.len.push(seed.len);
        self.is_rev.push(seed.is_rev);
        self.interval_size.push(seed.interval_size);
        self.rid.push(seed.rid);
        // This push logic will need to be handled carefully with read_seed_boundaries
        // For now, it's a direct push, boundary management will be done at batching
    }
}

// Define a struct to represent a batch of encoded queries in SoA format
#[derive(Debug, Clone, Default)]
pub struct SoAEncodedQueryBatch {
    pub encoded_seqs: Vec<u8>,
    // Store (start_offset, length) for each encoded query within `encoded_seqs`
    pub query_boundaries: Vec<(usize, usize)>,
}

impl SoAEncodedQueryBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(total_seq_len: usize, num_reads: usize) -> Self {
        Self {
            encoded_seqs: Vec::with_capacity(total_seq_len),
            query_boundaries: Vec::with_capacity(num_reads),
        }
    }

    pub fn clear(&mut self) {
        self.encoded_seqs.clear();
        self.query_boundaries.clear();
    }
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

pub fn find_seeds_batch(
    bwa_idx: &BwaIndex,
    read_batch: &SoAReadBatch,
    opt: &MemOpt,
) -> (SoASeedBatch, SoAEncodedQueryBatch, SoAEncodedQueryBatch) {
    let num_reads = read_batch.len();
    let total_query_len: usize = read_batch.read_boundaries.iter().map(|(_, len)| *len).sum();

    let mut soa_seed_batch = SoASeedBatch::with_capacity(num_reads * 50, num_reads); // Heuristic capacity
    let mut soa_encoded_query_batch =
        SoAEncodedQueryBatch::with_capacity(total_query_len, num_reads);
    let mut soa_encoded_query_rc_batch =
        SoAEncodedQueryBatch::with_capacity(total_query_len, num_reads);

    for read_idx in 0..num_reads {
        let (seq_start, query_len) = read_batch.read_boundaries[read_idx];
        let query_name = &read_batch.names[read_idx];
        let query_seq = &read_batch.seqs[seq_start..(seq_start + query_len)];

        // Create encoded versions of the query sequence
        let mut encoded_query = Vec::with_capacity(query_len);
        let mut encoded_query_rc = Vec::with_capacity(query_len); // Reverse complement
        for &base in query_seq {
            let code = base_to_code(base);
            encoded_query.push(code);
            encoded_query_rc.push(reverse_complement_code(code));
        }
        encoded_query_rc.reverse();

        // Store encoded queries in SoA batches
        let current_encoded_query_start = soa_encoded_query_batch.encoded_seqs.len();
        soa_encoded_query_batch
            .encoded_seqs
            .extend_from_slice(&encoded_query);
        soa_encoded_query_batch
            .query_boundaries
            .push((current_encoded_query_start, query_len));

        let current_encoded_query_rc_start = soa_encoded_query_rc_batch.encoded_seqs.len();
        soa_encoded_query_rc_batch
            .encoded_seqs
            .extend_from_slice(&encoded_query_rc);
        soa_encoded_query_rc_batch
            .query_boundaries
            .push((current_encoded_query_rc_start, query_len));

        if query_len == 0 {
            soa_seed_batch
                .read_seed_boundaries
                .push((soa_seed_batch.query_pos.len(), 0));
            continue;
        }

        #[cfg(feature = "debug-logging")]
        let is_debug_read = query_name.contains("1150:14380");

        #[cfg(feature = "debug-logging")]
        if is_debug_read {
            log::debug!("[DEBUG_READ] Generating seeds for: {}", query_name);
            log::debug!("[DEBUG_READ] Query length: {}", query_len);
        }

        // Pre-allocate for typical SMEM counts to avoid reallocations
        let mut all_smems: Vec<SMEM> = Vec::with_capacity(512);
        let min_seed_len = opt.min_seed_len;
        let min_intv = 1u64;

        log::debug!(
            "{query_name}: Starting SMEM generation: min_seed_len={min_seed_len}, min_intv={min_intv}, query_len={query_len}"
        );

        // PHASE 1 VALIDATION: Log SMEM generation parameters
        log::debug!(
            "SMEM_VALIDATION {}: Parameters: min_seed_len={}, max_occ={}, split_factor={:.2}, split_width={}, max_mem_intv={}",
            query_name,
            opt.min_seed_len,
            opt.max_occ,
            opt.split_factor,
            opt.split_width,
            opt.max_mem_intv
        );

        let mut max_smem_count = 0usize;

        let current_read_seed_start_idx = soa_seed_batch.query_pos.len();

        // BWA-MEM2 SMEM algorithm: Search ONLY with the original query.
        // The bidirectional FM-index automatically finds matches on both strands:
        // - Positions in [0, l_pac): forward strand alignments
        // - Positions in [l_pac, 2*l_pac): reverse strand alignments
        //
        // Searching with the reverse complement query would find DIFFERENT positions
        // (where the revcomp pattern matches), not the same alignment on the other strand.
        // The strand is determined later based on whether the FM-index position >= l_pac.
        //
        // Use thread-local workspace buffers to avoid per-read allocations
        with_workspace(|ws| {
            generate_smems_for_strand(
                bwa_idx,
                query_name,
                query_len,
                &encoded_query,
                false, // is_reverse_complement = false for all SMEMs (strand determined by position)
                min_seed_len,
                min_intv,
                &mut all_smems,
                &mut max_smem_count,
                &mut ws.smem_prev_buf,
                &mut ws.smem_curr_buf,
            );
        });

        // PHASE 1 VALIDATION: Log initial SMEMs
        let pass1_count = all_smems.len();
        log::debug!("SMEM_VALIDATION {query_name}: Pass 1 (initial) generated {pass1_count} SMEMs");
        if log::log_enabled!(log::Level::Debug) {
            for (idx, smem) in all_smems.iter().enumerate().take(10) {
                log::debug!(
                    "SMEM_VALIDATION {}:   Pass1[{}]: query[{}..{}] len={} bwt=[{}, {}) size={}",
                    query_name,
                    idx,
                    smem.query_start,
                    smem.query_end,
                    smem.query_end - smem.query_start + 1,
                    smem.bwt_interval_start,
                    smem.bwt_interval_end,
                    smem.interval_size
                );
            }
            if all_smems.len() > 10 {
                log::debug!(
                    "SMEM_VALIDATION {}:   ... ({} more SMEMs)",
                    query_name,
                    all_smems.len() - 10
                );
            }
        }

        // Re-seeding pass: For long unique SMEMs, re-seed from middle to find split alignments
        // This matches C++ bwamem.cpp:695-714
        // C++ uses: (int)(min_seed_len * split_factor + 0.499) which rounds to nearest
        let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;
        let split_width = opt.split_width as u64;

        // Collect re-seeding candidates from initial SMEMs
        // NOTE: Re-seeding always uses original query since all SMEMs come from original query search
        let mut reseed_candidates: Vec<(usize, u64)> = Vec::with_capacity(32); // (middle_pos, min_intv)

        for smem in all_smems.iter() {
            let smem_len = smem.query_end - smem.query_start + 1;
            // Re-seed if: length >= split_len AND interval_size <= split_width
            if smem_len >= split_len && smem.interval_size <= split_width {
                // Calculate middle position: (start + end + 1) >> 1 to match C++
                let middle_pos = ((smem.query_start + smem.query_end + 1) >> 1) as usize;
                let new_min_intv = smem.interval_size + 1;

                log::debug!(
                    "{}: Re-seed candidate: smem m={}, n={}, len={}, s={}, middle_pos={}, new_min_intv={}",
                    query_name,
                    smem.query_start,
                    smem.query_end,
                    smem_len,
                    smem.interval_size,
                    middle_pos,
                    new_min_intv
                );

                reseed_candidates.push((middle_pos, new_min_intv));
            }
        }

        // Execute re-seeding for each candidate (always use original query)
        // Use thread-local workspace buffers to avoid per-call allocations
        let initial_smem_count = all_smems.len();
        with_workspace(|ws| {
            for (middle_pos, new_min_intv) in &reseed_candidates {
                generate_smems_from_position(
                    bwa_idx,
                    query_name,
                    query_len,
                    &encoded_query,
                    false, // is_reverse_complement = false for all SMEMs (strand determined by position)
                    min_seed_len,
                    *new_min_intv,
                    *middle_pos,
                    &mut all_smems,
                    &mut ws.smem_prev_buf,
                    &mut ws.smem_curr_buf,
                );
            }
        });

        // PHASE 1 VALIDATION: Log Pass 2 (re-seeding) results
        let pass2_added = all_smems.len() - initial_smem_count;
        if pass2_added > 0 {
            log::debug!(
                "SMEM_VALIDATION {}: Pass 2 (re-seeding) added {} new SMEMs (total: {})",
                query_name,
                pass2_added,
                all_smems.len()
            );
            if log::log_enabled!(log::Level::Debug) {
                for (idx, smem) in all_smems
                    .iter()
                    .skip(initial_smem_count)
                    .enumerate()
                    .take(10)
                {
                    log::debug!(
                        "SMEM_VALIDATION {}:   Pass2[{}]: query[{}..{}] len={} bwt=[{}, {}) size={}",
                        query_name,
                        idx,
                        smem.query_start,
                        smem.query_end,
                        smem.query_end - smem.query_start + 1,
                        smem.bwt_interval_start,
                        smem.bwt_interval_end,
                        smem.interval_size
                    );
                }
                if all_smems.len() > 10 {
                    log::debug!(
                        "SMEM_VALIDATION {}:   ... ({} more SMEMs)",
                        query_name,
                        all_smems.len() - 10
                    );
                }
            }
        } else {
            log::debug!("SMEM_VALIDATION {query_name}: Pass 2 (re-seeding) added 0 new SMEMs");
        }

        // 3rd round seeding: Additional seeding pass with forward-only strategy
        // BWA-MEM2 runs this unconditionally when max_mem_intv > 0 (default 20)
        // Uses min_seed_len + 1 as minimum length and max_mem_intv as the interval threshold
        // This finds seeds that might be missed by the supermaximal SMEM algorithm
        let smems_before_3rd_round = all_smems.len();
        let mut used_3rd_round_seeding = false;

        // Match BWA-MEM2: run 3rd round seeding unconditionally when max_mem_intv > 0
        // (Previously required all SMEMs to exceed max_occ, which was incorrect)
        if opt.max_mem_intv > 0 {
            used_3rd_round_seeding = true;
            log::debug!(
                "{}: Running 3rd round seeding (max_mem_intv={}) with {} existing SMEMs",
                query_name,
                opt.max_mem_intv,
                all_smems.len()
            );

            // Use forward-only seed strategy matching BWA-MEM2's bwtSeedStrategyAllPosOneThread
            // This iterates through ALL positions, doing forward extension only,
            // and outputs seeds when interval drops BELOW max_mem_intv
            // NOTE: Only search with original query - bidirectional index handles both strands
            forward_only_seed_strategy(
                bwa_idx,
                query_name,
                query_len,
                &encoded_query,
                false, // is_reverse_complement = false for all SMEMs (strand determined by position)
                min_seed_len,
                opt.max_mem_intv,
                &mut all_smems,
            );

            // PHASE 1 VALIDATION: Log Pass 3 (forward-only) results
            let pass3_added = all_smems.len() - smems_before_3rd_round;
            if pass3_added > 0 {
                log::debug!(
                    "SMEM_VALIDATION {}: Pass 3 (forward-only) added {} new SMEMs (total: {})",
                    query_name,
                    pass3_added,
                    all_smems.len()
                );
                if log::log_enabled!(log::Level::Debug) {
                    for (idx, smem) in all_smems
                        .iter()
                        .skip(smems_before_3rd_round)
                        .enumerate()
                        .take(10)
                    {
                        log::debug!(
                            "SMEM_VALIDATION {}:   Pass3[{}]: query[{}..{}] len={} bwt=[{}, {}) size={}",
                            query_name,
                            idx,
                            smem.query_start,
                            smem.query_end,
                            smem.query_end - smem.query_start + 1,
                            smem.bwt_interval_start,
                            smem.bwt_interval_end,
                            smem.interval_size
                        );
                    }
                    if all_smems.len() > 10 {
                        log::debug!(
                            "SMEM_VALIDATION {}:   ... ({} more SMEMs)",
                            query_name,
                            all_smems.len() - 10
                        );
                    }
                }
            } else {
                log::debug!(
                    "SMEM_VALIDATION {query_name}: Pass 3 (forward-only) added 0 new SMEMs"
                );
            }
        } else {
            log::debug!(
                "SMEM_VALIDATION {query_name}: Pass 3 (forward-only) skipped (max_mem_intv=0)"
            );
        }

        // Filter SMEMs
        let mut unique_filtered_smems: Vec<SMEM> = Vec::new();
        all_smems.sort_by_key(|smem| {
            (
                smem.query_start,
                smem.query_end,
                smem.bwt_interval_start,
                smem.is_reverse_complement,
            )
        });

        // NOTE: split_factor and split_width control RE-SEEDING for chimeric detection,
        // NOT seed filtering. The basic filter (min_seed_len + max_occ) is sufficient.
        // The previous "chimeric filter" was incorrectly discarding valid seeds.
        // See C++ bwamem.cpp:639-695 - split logic is for creating additional sub-seeds,
        // not for removing seeds that pass the basic quality checks.

        // For 3rd round seeding: if all SMEMs still exceed max_occ, use a much higher threshold
        // to allow some seeds through. This is the fallback for highly repetitive regions.
        // BWA-MEM2 uses seed_occurrence_3rd parameter for this purpose.
        let effective_max_occ = if used_3rd_round_seeding {
            // Find the minimum occurrence among all SMEMs and use that as the threshold
            // This ensures at least some seeds pass through
            let min_occ = all_smems
                .iter()
                .map(|s| s.interval_size)
                .min()
                .unwrap_or(opt.max_occ as u64);
            // Use min_occ + 1 to ensure seeds pass
            let relaxed_threshold = (min_occ + 1).max(opt.max_occ as u64);
            log::debug!(
                "{}: 3rd round seeding used, relaxing max_occ filter from {} to {} (min_occ={})",
                query_name,
                opt.max_occ,
                relaxed_threshold,
                min_occ
            );
            relaxed_threshold
        } else {
            opt.max_occ as u64
        };

        // CRITICAL FIX: DO NOT filter duplicate SMEMs!
        // BWA-MEM2 preserves all SMEMs including duplicates from different passes.
        // Filtering duplicates here causes chains to have fewer seeds, leading to
        // different extension boundaries and lower proper pairing rates.
        for smem in all_smems.iter() {
            let seed_len = smem.query_end - smem.query_start + 1;
            let occurrences = smem.interval_size;
            // Keep seeds that pass basic quality filter (min_seed_len AND max_occ)
            if seed_len >= opt.min_seed_len && occurrences <= effective_max_occ {
                unique_filtered_smems.push(*smem);
            }
        }

        // PHASE 1 VALIDATION: Log filtering summary
        log::debug!(
            "SMEM_VALIDATION {}: Filtering summary: {} total SMEMs -> {} unique (min_seed_len={}, max_occ={})",
            query_name,
            all_smems.len(),
            unique_filtered_smems.len(),
            opt.min_seed_len,
            effective_max_occ
        );

        log::debug!(
            "{}: Generated {} SMEMs, filtered to {} unique",
            query_name,
            all_smems.len(),
            unique_filtered_smems.len()
        );

        // SMEM OVERLAP DEBUG: Log ALL SMEMs to identify overlapping/duplicate SMEMs
        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "SMEM_OVERLAP {}: {} SMEMs after filtering:",
                query_name,
                unique_filtered_smems.len()
            );
            for (idx, smem) in unique_filtered_smems.iter().enumerate().take(10) {
                log::debug!(
                    "SMEM_OVERLAP {}:   SMEM[{}]: query[{}..{}] len={} bwt=[{}, {}) size={}",
                    query_name,
                    idx,
                    smem.query_start,
                    smem.query_end,
                    smem.query_end - smem.query_start + 1,
                    smem.bwt_interval_start,
                    smem.bwt_interval_end,
                    smem.interval_size
                );
            }
            if unique_filtered_smems.len() > 10 {
                log::debug!(
                    "SMEM_OVERLAP {}:   ... ({} more SMEMs)",
                    query_name,
                    unique_filtered_smems.len() - 10
                );
            }
        }

        let mut sorted_smems = unique_filtered_smems;
        sorted_smems.sort_by_key(|smem| -(smem.query_end - smem.query_start + 1));

        // Match C++ SEEDS_PER_READ limit (see bwa-mem2/src/macro.h)
        const SEEDS_PER_READ: usize = 500;

        // For highly repetitive reads (small number of SMEMs all covering full query),
        // allow more seeds per SMEM to get full coverage of the reference range.
        // Otherwise divide SEEDS_PER_READ among all SMEMs.
        let is_highly_repetitive = sorted_smems.len() <= 4
            && sorted_smems
                .iter()
                .all(|s| s.query_end - s.query_start > (query_len as i32 * 3 / 4));

        let seeds_per_smem = if sorted_smems.is_empty() {
            SEEDS_PER_READ
        } else if is_highly_repetitive {
            // For highly repetitive: use full max_occ per SMEM
            SEEDS_PER_READ
        } else {
            (SEEDS_PER_READ / sorted_smems.len()).max(1)
        };

        let mut current_read_seeds: Vec<Seed> = Vec::new();
        let mut seeds_per_smem_count = Vec::new(); // Track seeds generated per SMEM for Phase 2 validation

        for (smem_idx, smem) in sorted_smems.iter().enumerate() {
            let seeds_before = current_read_seeds.len();

            // Limit positions per SMEM to ensure coverage from multiple SMEMs
            // The get_sa_entries function will sample evenly across the interval
            let max_positions_this_smem = (seeds_per_smem as u32).min(opt.max_occ as u32);

            // Use the new get_sa_entries function to get multiple reference positions
            // It samples evenly across the entire BWT interval using floating-point step
            let ref_positions = get_sa_entries(
                bwa_idx,
                smem.bwt_interval_start,
                smem.interval_size,
                max_positions_this_smem,
            );

            let seed_len = smem.query_end - smem.query_start;
            let l_pac = bwa_idx.bns.packed_sequence_length;

            // PHASE 2 VALIDATION: Log ALL SMEMs for exhaustive comparison
            if log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "SEED_CONVERSION {}: SMEM[{}] query[{}..{}] → {} ref positions (requested: {})",
                    query_name,
                    smem_idx,
                    smem.query_start,
                    smem.query_end,
                    ref_positions.len(),
                    max_positions_this_smem
                );
            }

            let mut skipped_boundary = 0;
            for (pos_idx, ref_pos) in ref_positions.iter().enumerate() {
                // Compute rid (chromosome ID) - skip seeds that span chromosome boundaries
                // Matches C++ bwamem.cpp:911-914
                let rid = bwa_idx.bns.pos_to_rid(*ref_pos, *ref_pos + seed_len as u64);
                if rid < 0 {
                    // Seed spans multiple chromosomes or forward-reverse boundary - skip
                    skipped_boundary += 1;
                    continue;
                }

                // BWA-MEM2 determines strand from reference position, not SMEM flag:
                // - Positions in [0, l_pac): forward strand (read matches forward ref)
                // - Positions in [l_pac, 2*l_pac): reverse strand (read matches revcomp ref)
                // See bns_depos() in bntseq.h: (*is_rev = (pos >= bns->l_pac))
                let is_rev = *ref_pos >= l_pac;

                // PHASE 2 VALIDATION: Log ALL seeds for exhaustive comparison
                if log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "SEED_CONVERSION {}:   Seed[{}]: ref_pos={} is_rev={} rid={} chr={}",
                        query_name,
                        pos_idx,
                        ref_pos,
                        is_rev,
                        rid,
                        if rid >= 0 {
                            bwa_idx.bns.annotations[rid as usize].name.as_str()
                        } else {
                            "N/A"
                        }
                    );
                }

                let seed = Seed {
                    query_pos: smem.query_start,
                    ref_pos: *ref_pos,
                    len: seed_len,
                    is_rev,
                    interval_size: smem.interval_size,
                    rid,
                };
                current_read_seeds.push(seed);

                // Hard limit on seeds per read to prevent memory explosion
                if current_read_seeds.len() >= SEEDS_PER_READ {
                    log::debug!(
                        "{query_name}: Hit SEEDS_PER_READ limit ({SEEDS_PER_READ}), truncating"
                    );
                    break;
                }
            }

            let seeds_added = current_read_seeds.len() - seeds_before;
            seeds_per_smem_count.push((smem_idx, seeds_added, skipped_boundary));

            if current_read_seeds.len() >= SEEDS_PER_READ {
                break;
            }
        }

        // PHASE 2 VALIDATION: Log seed conversion summary
        log::debug!(
            "SEED_CONVERSION {}: Total {} SMEMs → {} seeds ({} seeds/SMEM limit)",
            query_name,
            sorted_smems.len(),
            current_read_seeds.len(),
            seeds_per_smem
        );

        // Log per-SMEM breakdown for first few SMEMs
        if log::log_enabled!(log::Level::Debug) {
            for &(idx, count, skipped) in seeds_per_smem_count.iter().take(10) {
                if count > 0 || skipped > 0 {
                    log::debug!(
                        "SEED_CONVERSION {query_name}:   SMEM[{idx}] → {count} seeds ({skipped} skipped at boundary)"
                    );
                }
            }
            if seeds_per_smem_count.len() > 10 {
                log::debug!(
                    "SEED_CONVERSION {}:   ... ({} more SMEMs)",
                    query_name,
                    seeds_per_smem_count.len() - 10
                );
            }
        }

        if max_smem_count > query_len {
            log::debug!(
                "{query_name}: SMEM buffer grew beyond initial capacity! max_smem_count={max_smem_count} > query_len={query_len}"
            );
        }

        log::debug!(
            "{}: Created {} seeds from {} SMEMs",
            query_name,
            current_read_seeds.len(),
            sorted_smems.len()
        );

        // Populate SoASeedBatch for the current read
        for seed in current_read_seeds {
            soa_seed_batch.query_pos.push(seed.query_pos);
            soa_seed_batch.ref_pos.push(seed.ref_pos);
            soa_seed_batch.len.push(seed.len);
            soa_seed_batch.is_rev.push(seed.is_rev);
            soa_seed_batch.interval_size.push(seed.interval_size);
            soa_seed_batch.rid.push(seed.rid);
        }
        let num_seeds_for_read = soa_seed_batch.query_pos.len() - current_read_seed_start_idx;
        soa_seed_batch
            .read_seed_boundaries
            .push((current_read_seed_start_idx, num_seeds_for_read));
    }

    (
        soa_seed_batch,
        soa_encoded_query_batch,
        soa_encoded_query_rc_batch,
    )
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
