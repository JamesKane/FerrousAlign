// Define a struct to represent a seed
#[derive(Debug, Clone)]
pub struct Seed {
    pub query_pos: i32,     // Position in the query
    pub ref_pos: u64,       // Position in the reference
    pub len: i32,           // Length of the seed
    pub is_rev: bool,       // Is it on the reverse strand?
    pub interval_size: u64, // BWT interval size (occurrence count)
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
fn generate_smems_for_strand(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_len: usize,
    encoded_query: &[u8],
    is_reverse_complement: bool,
    min_seed_len: i32,
    min_intv: u64,
    all_smems: &mut Vec<SMEM>,
    max_smem_count: &mut usize,
) {
    // OPTIMIZATION: Pre-allocate buffers to avoid repeated allocations
    let mut prev_array_buf: Vec<SMEM> = Vec::with_capacity(query_len);
    let mut curr_array_buf: Vec<SMEM> = Vec::with_capacity(query_len);

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
        let mut next_x = x + 1;

        for j in (x + 1)..query_len {
            let a = encoded_query[j];
            next_x = j + 1;

            if a >= 4 {
                if x == 0 && !is_reverse_complement && log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "{}: x={}, forward extension stopped at j={} due to N base",
                        query_name,
                        x,
                        j
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

            if new_smem.interval_size != smem.interval_size {
                if x < 3 && !is_reverse_complement {
                    let s_from_lk = if smem.bwt_interval_end > smem.bwt_interval_start {
                        smem.bwt_interval_end - smem.bwt_interval_start
                    } else {
                        0
                    };
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
                prev_array_buf.push(smem);
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
                break;
            }

            smem = new_smem;
            smem.query_end = j as i32;
        }

        if smem.interval_size >= min_intv {
            prev_array_buf.push(smem);
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
        if !is_reverse_complement {
            log::debug!(
                "{}: [RUST Phase 2] Starting backward search from x={}, prev_array_buf.len()={}",
                query_name,
                x,
                prev_array_buf.len()
            );
        }

        for j in (0..x).rev() {
            let a = encoded_query[j];
            if a >= 4 {
                if !is_reverse_complement {
                    log::debug!(
                        "{}: [RUST Phase 2] Hit 'N' base at j={}, stopping",
                        query_name,
                        j
                    );
                }
                break;
            }

            curr_array_buf.clear();
            let curr_array = &mut curr_array_buf;
            let mut curr_s = None;

            if !is_reverse_complement {
                log::debug!(
                    "{}: [RUST Phase 2] j={}, base={}, prev_array_buf.len()={}",
                    query_name,
                    j,
                    a,
                    prev_array_buf.len()
                );
            }

            for (i, smem) in prev_array_buf.iter().rev().enumerate() {
                let mut new_smem = backward_ext(bwa_idx, *smem, a);
                new_smem.query_start = j as i32;

                if !is_reverse_complement {
                    let old_len = smem.query_end - smem.query_start + 1;
                    let new_len = new_smem.query_end - new_smem.query_start + 1;
                    log::debug!(
                        "{}: [RUST Phase 2] x={}, j={}, i={}: old_smem(m={},n={},len={},k={},l={},s={}), new_smem(m={},n={},len={},k={},l={},s={}), min_intv={}",
                        query_name,
                        x,
                        j,
                        i,
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

                if new_smem.interval_size < min_intv
                    && (smem.query_end - smem.query_start + 1) >= min_seed_len
                {
                    if !is_reverse_complement {
                        let s_from_lk = if smem.bwt_interval_end > smem.bwt_interval_start {
                            smem.bwt_interval_end - smem.bwt_interval_start
                        } else {
                            0
                        };
                        let s_matches = smem.interval_size == s_from_lk;
                        log::debug!(
                            "{}: [RUST SMEM OUTPUT] Phase2 line 617: smem(m={},n={},k={},l={},s={}) newSmem.s={} < min_intv={}, l-k={}, s_match={}",
                            query_name,
                            smem.query_start,
                            smem.query_end,
                            smem.bwt_interval_start,
                            smem.bwt_interval_end,
                            smem.interval_size,
                            new_smem.interval_size,
                            min_intv,
                            s_from_lk,
                            s_matches
                        );
                    }
                    all_smems.push(*smem);
                    break;
                }

                if new_smem.interval_size >= min_intv && curr_s != Some(new_smem.interval_size) {
                    curr_s = Some(new_smem.interval_size);
                    curr_array.push(new_smem);
                    if !is_reverse_complement {
                        log::debug!(
                            "{}: [RUST Phase 2] Keeping new_smem (s={} >= min_intv={}), breaking",
                            query_name,
                            new_smem.interval_size,
                            min_intv
                        );
                    }
                    break;
                }

                if !is_reverse_complement {
                    log::debug!(
                        "{}: [RUST Phase 2] Rejecting new_smem (s={} < min_intv={} OR already_seen={})",
                        query_name,
                        new_smem.interval_size,
                        min_intv,
                        curr_s == Some(new_smem.interval_size)
                    );
                }
            }

            for (i, smem) in prev_array_buf.iter().rev().skip(1).enumerate() {
                let mut new_smem = backward_ext(bwa_idx, *smem, a);
                new_smem.query_start = j as i32;

                if !is_reverse_complement {
                    let new_len = new_smem.query_end - new_smem.query_start + 1;
                    log::debug!(
                        "{}: [RUST Phase 2] x={}, j={}, remaining_i={}: smem(m={},n={},s={}), new_smem(m={},n={},len={},s={}), will_push={}",
                        query_name,
                        x,
                        j,
                        i + 1,
                        smem.query_start,
                        smem.query_end,
                        smem.interval_size,
                        new_smem.query_start,
                        new_smem.query_end,
                        new_len,
                        new_smem.interval_size,
                        new_smem.interval_size >= min_intv
                            && curr_s != Some(new_smem.interval_size)
                    );
                }

                if new_smem.interval_size >= min_intv && curr_s != Some(new_smem.interval_size) {
                    curr_s = Some(new_smem.interval_size);
                    curr_array.push(new_smem);
                }
            }

            std::mem::swap(&mut prev_array_buf, &mut curr_array_buf);
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
                        "{}: [RUST Phase 2] prev_array_buf empty, breaking at j={}",
                        query_name,
                        j
                    );
                }
                break;
            }
        }

        if !prev_array_buf.is_empty() {
            let smem = prev_array_buf[prev_array_buf.len() - 1];
            let len = smem.query_end - smem.query_start + 1;
            if len >= min_seed_len {
                if !is_reverse_complement {
                    let s_from_lk = if smem.bwt_interval_end > smem.bwt_interval_start {
                        smem.bwt_interval_end - smem.bwt_interval_start
                    } else {
                        0
                    };
                    let s_matches = smem.interval_size == s_from_lk;
                    log::debug!(
                        "{}: [RUST SMEM OUTPUT] Phase2 line 671: smem(m={},n={},k={},l={},s={}), len={}, l-k={}, s_match={}, next_x={}",
                        query_name,
                        smem.query_start,
                        smem.query_end,
                        smem.bwt_interval_start,
                        smem.bwt_interval_end,
                        smem.interval_size,
                        len,
                        s_from_lk,
                        s_matches,
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
                "{}: [RUST Phase 2] No remaining SMEMs at end of backward search for x={}",
                query_name,
                x
            );
        }

        x = next_x;
    }
}

// ============================================================================
// BWT AND SUFFIX ARRAY HELPER FUNCTIONS
// ============================================================================
//
// This section contains low-level BWT and suffix array access functions
// used during FM-Index search and seed extension
// ============================================================================

// Function to get BWT base from cp_occ format (for loaded indices)
// Returns 0-3 for bases A/C/G/T, or 4 for sentinel
pub fn get_bwt_base_from_cp_occ(cp_occ: &[CpOcc], pos: u64) -> u8 {
    let cp_block = (pos >> CP_SHIFT) as usize;

    // Safety: check bounds
    if cp_block >= cp_occ.len() {
        log::warn!(
            "get_bwt_base_from_cp_occ: cp_block {} >= cp_occ.len() {}",
            cp_block,
            cp_occ.len()
        );
        return 4; // Return sentinel for out-of-bounds
    }

    let offset_in_block = pos & ((1 << CP_SHIFT) - 1);
    let bit_position = 63 - offset_in_block;

    // Check which of the 4 one-hot encoded arrays has a 1 at this position
    for base in 0..4 {
        if (cp_occ[cp_block].bwt_encoding_bits[base] >> bit_position) & 1 == 1 {
            return base as u8;
        }
    }
    4 // Return 4 for sentinel (no bit set means sentinel position)
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
                "get_sa_entry exceeded MAX_ITERATIONS ({}) - possible infinite loop!",
                MAX_ITERATIONS
            );
            log::error!(
                "  original_pos={}, current_pos={}, count={}",
                original_pos,
                pos,
                count
            );
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
        log::debug!(
            "SA value {} is at/past sentinel {} - wrapping to 0",
            sa_val,
            sentinel_pos
        );
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
        (bwa_idx.bns.packed_sequence_length << 1)
    );
    result
}