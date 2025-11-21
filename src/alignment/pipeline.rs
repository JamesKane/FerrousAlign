use crate::alignment::banded_swa::BandedPairWiseSW;
use crate::alignment::banded_swa::merge_cigar_operations;
use crate::alignment::chaining::cal_max_gap;
use crate::alignment::chaining::chain_seeds;
use crate::alignment::chaining::filter_chains;
use crate::alignment::extension::AlignmentJob;
use crate::alignment::extension::execute_adaptive_alignments;
use crate::alignment::extension::execute_scalar_alignments;
use crate::alignment::finalization::Alignment;
use crate::alignment::finalization::generate_sa_tags;
use crate::alignment::finalization::generate_xa_tags;
use crate::alignment::finalization::mark_secondary_alignments;
use crate::alignment::finalization::sam_flags;
use crate::alignment::seeding::SMEM;
use crate::alignment::seeding::Seed;
use crate::alignment::seeding::generate_smems_for_strand;
use crate::alignment::seeding::get_sa_entry;
use crate::alignment::utils::base_to_code;
use crate::alignment::utils::reverse_complement_code;
use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;
use crate::utils::hash_64;

// ============================================================================
// SEED GENERATION (SMEM EXTRACTION)
// ============================================================================
//
// This section contains the main seed generation pipeline:
// - SMEM (Supermaximal Exact Match) extraction using FM-Index
// - Bidirectional search (forward and reverse complement)
// - Seed extension and filtering
// ============================================================================

pub fn generate_seeds(
    bwa_idx: &BwaIndex,
    pac_data: &[u8], // Pre-loaded PAC data for MD tag generation
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
) -> Vec<Alignment> {
    generate_seeds_with_mode(
        bwa_idx, pac_data, query_name, query_seq, query_qual, true, opt,
    )
}

// Internal implementation with option to use batched SIMD
fn generate_seeds_with_mode(
    bwa_idx: &BwaIndex,
    pac_data: &[u8], // Pre-loaded PAC data (loaded once, not per-read!)
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    use_batched_simd: bool,
    _opt: &MemOpt, // TODO: Use opt parameters in Phase 2+
) -> Vec<Alignment> {
    let query_len = query_seq.len();
    if query_len == 0 {
        return Vec::new();
    }

    #[cfg(feature = "debug-logging")]
    let is_debug_read = query_name.contains("1150:14380");

    #[cfg(feature = "debug-logging")]
    if is_debug_read {
        log::debug!("[DEBUG_READ] Generating seeds for: {}", query_name);
        log::debug!("[DEBUG_READ] Query length: {}", query_len);
    }

    // Instantiate BandedPairWiseSW with parameters from MemOpt
    let sw_params = BandedPairWiseSW::new(
        _opt.o_del,      // Gap open deletion penalty
        _opt.e_del,      // Gap extension deletion penalty
        _opt.o_ins,      // Gap open insertion penalty
        _opt.e_ins,      // Gap extension insertion penalty
        _opt.zdrop,      // Z-dropoff
        5,               // end_bonus (reserved for future use)
        _opt.pen_clip5,  // 5' clipping penalty (default=5)
        _opt.pen_clip3,  // 3' clipping penalty (default=5)
        _opt.mat,        // Scoring matrix (generated from -A/-B)
        _opt.a as i8,    // Match score
        -(_opt.b as i8), // Mismatch penalty (negative)
    );

    let mut encoded_query = Vec::with_capacity(query_len);
    let mut encoded_query_rc = Vec::with_capacity(query_len); // Reverse complement
    for &base in query_seq {
        let code = base_to_code(base);
        encoded_query.push(code);
        encoded_query_rc.push(reverse_complement_code(code));
    }
    encoded_query_rc.reverse(); // Reverse the reverse complement to get the sequence in correct order

    let mut all_smems: Vec<SMEM> = Vec::new();

    // --- Generate SMEMs using C++ two-phase algorithm ---
    // C++ reference: FMI_search.cpp getSMEMsOnePosOneThread (lines 496-670)
    // For each starting position x:
    //   Phase 1: Forward extension (collect intermediate SMEMs)
    //   Phase 2: Backward extension (generate final bidirectional SMEMs)

    let min_seed_len = _opt.min_seed_len;
    // CRITICAL FIX: C++ bwa-mem2 uses min_intv=1 during SMEM generation (not max_occ!)
    // See bwamem.cpp:661 - min_intv_ar[l] = 1;
    // The max_occ filter is applied LATER during SMEM filtering, not during generation
    let min_intv = 1u64;

    log::debug!(
        "{}: Starting SMEM generation: min_seed_len={}, min_intv={}, query_len={}",
        query_name,
        min_seed_len,
        min_intv,
        query_len
    );

    let mut max_smem_count = 0usize;

    // Process forward strand
    generate_smems_for_strand(
        bwa_idx,
        query_name,
        query_len,
        &encoded_query,
        false, // is_rev_comp
        min_seed_len,
        min_intv,
        &mut all_smems,
        &mut max_smem_count,
    );

    // Process reverse complement strand
    generate_smems_for_strand(
        bwa_idx,
        query_name,
        query_len,
        &encoded_query_rc,
        true, // is_rev_comp
        min_seed_len,
        min_intv,
        &mut all_smems,
        &mut max_smem_count,
    );

    // eprintln!("all_smems: {:?}", all_smems);

    // --- Filtering SMEMs ---
    let mut unique_filtered_smems: Vec<SMEM> = Vec::new();
    let mut filtered_too_short = 0;
    let mut filtered_too_many_occ = 0;
    let mut duplicates = 0;

    // Sort SMEMs to process unique ones efficiently
    all_smems.sort_by_key(|smem| {
        (
            smem.query_start,
            smem.query_end,
            smem.bwt_interval_start,
            smem.is_reverse_complement,
        )
    });

    let split_len_threshold = (_opt.min_seed_len as f32 * _opt.split_factor) as i32;

    if let Some(mut prev_smem) = all_smems.first().cloned() {
        let seed_len = prev_smem.query_end - prev_smem.query_start + 1;
        let occurrences = prev_smem.interval_size;

        // Standard filter (min_seed_len, max_occ)
        let mut keep_smem = seed_len >= _opt.min_seed_len && occurrences <= _opt.max_occ as u64;

        // Chimeric filter (from BWA-MEM2 bwamem.cpp:685)
        // If SMEM is too short for splitting OR too repetitive, it's NOT considered for chimeric re-seeding
        if seed_len < split_len_threshold || occurrences > _opt.split_width as u64 {
            keep_smem = false; // Mark as not suitable for chimeric processing
        }

        if keep_smem {
            unique_filtered_smems.push(prev_smem);
        } else {
            if seed_len < _opt.min_seed_len {
                filtered_too_short += 1;
            }
            if occurrences > _opt.max_occ as u64 {
                filtered_too_many_occ += 1;
            }
        }

        for i in 1..all_smems.len() {
            let current_smem = all_smems[i];
            if current_smem != prev_smem {
                // Use PartialEq for comparison
                let seed_len = current_smem.query_end - current_smem.query_start + 1;
                let occurrences = current_smem.interval_size;

                // Standard filter (min_seed_len, max_occ)
                let mut keep_smem_current =
                    seed_len >= _opt.min_seed_len && occurrences <= _opt.max_occ as u64;

                // Chimeric filter
                if seed_len < split_len_threshold || occurrences > _opt.split_width as u64 {
                    keep_smem_current = false;
                }

                if keep_smem_current {
                    unique_filtered_smems.push(current_smem);
                } else {
                    if seed_len < _opt.min_seed_len {
                        filtered_too_short += 1;
                    }
                    if occurrences > _opt.max_occ as u64 {
                        filtered_too_many_occ += 1;
                    }
                }
            } else {
                duplicates += 1;
            }
            prev_smem = current_smem;
        }
    }

    if unique_filtered_smems.is_empty() && all_smems.len() > 0 {
        log::debug!(
            "{}: All SMEMs filtered out - too_short={}, too_many_occ={}, duplicates={}, min_len={}, max_occ={}",
            query_name,
            filtered_too_short,
            filtered_too_many_occ,
            duplicates,
            _opt.min_seed_len,
            _opt.max_occ
        );

        // Sample first few SMEMs to see actual values
        for (i, smem) in all_smems.iter().take(5).enumerate() {
            let len = smem.query_end - smem.query_start + 1;
            let occ = smem.bwt_interval_end - smem.bwt_interval_start;
            log::debug!(
                "{}: Sample SMEM {}: len={}, occ={}, m={}, n={}, k={}, l={}",
                query_name,
                i,
                len,
                occ,
                smem.query_start,
                smem.query_end,
                smem.bwt_interval_start,
                smem.bwt_interval_end
            );
        }
    }
    // --- End Filtering SMEMs ---

    log::debug!(
        "{}: Generated {} SMEMs, filtered to {} unique",
        query_name,
        all_smems.len(),
        unique_filtered_smems.len()
    );

    // Convert SMEMs to Seed structs and perform seed extension
    // FIXED: Remove artificial SMEM limit - process ALL seeds like C++ bwa-mem2
    let mut sorted_smems = unique_filtered_smems;
    sorted_smems.sort_by_key(|smem| -(smem.query_end - smem.query_start + 1)); // Sort by length, descending

    let useful_smems = sorted_smems;

    log::debug!(
        "{}: Using {} SMEMs for alignment",
        query_name,
        useful_smems.len()
    );

    let mut seeds = Vec::new();
    let mut alignment_jobs = Vec::new(); // Collect alignment jobs for batching

    // Prepare query segment once - use the FULL query for alignment
    let query_segment_encoded: Vec<u8> = query_seq.iter().map(|&b| base_to_code(b)).collect();

    // Also prepare reverse complement for RC SMEMs
    // CRITICAL FIX: Use reverse_complement_code() to properly handle 'N' bases (code 4)
    // The XOR trick (b ^ 3) breaks for N: 4 ^ 3 = 7 (invalid!)
    let mut query_segment_encoded_rc: Vec<u8> = query_segment_encoded
        .iter()
        .map(|&b| reverse_complement_code(b)) // Properly handles N (4 → 4)
        .collect();
    query_segment_encoded_rc.reverse();

    log::debug!(
        "{}: query_segment_encoded (FWD) first_10={:?}",
        query_name,
        &query_segment_encoded[..10.min(query_segment_encoded.len())]
    );
    log::debug!(
        "{}: query_segment_encoded_rc (RC) first_10={:?}",
        query_name,
        &query_segment_encoded_rc[..10.min(query_segment_encoded_rc.len())]
    );

    for (idx, smem) in useful_smems.iter().enumerate() {
        let smem = *smem;

        // Get SA position and log the reconstruction process
        log::debug!(
            "{}: SMEM {}: BWT interval [k={}, l={}, s={}], query range [m={}, n={}], is_rev_comp={}",
            query_name,
            idx,
            smem.bwt_interval_start,
            smem.bwt_interval_end,
            smem.interval_size,
            smem.query_start,
            smem.query_end,
            smem.is_reverse_complement
        );

        // Try multiple positions in the BWT interval to find which one is correct
        let ref_pos_at_k = get_sa_entry(bwa_idx, smem.bwt_interval_start);
        let ref_pos_at_l_minus_1 = if smem.bwt_interval_end > 0 {
            get_sa_entry(bwa_idx, smem.bwt_interval_end - 1)
        } else {
            ref_pos_at_k
        };
        log::debug!(
            "{}: SMEM {}: SA at k={} -> ref_pos {}, SA at l-1={} -> ref_pos {}",
            query_name,
            idx,
            smem.bwt_interval_start,
            ref_pos_at_k,
            smem.bwt_interval_end - 1,
            ref_pos_at_l_minus_1
        );

        let mut ref_pos = ref_pos_at_k;

        let mut is_rev = smem.is_reverse_complement;

        // CRITICAL FIX: Use correct query orientation based on is_rev_comp flag
        // If SMEM is from RC search, use RC query bases for comparison
        let query_for_smem = if smem.is_reverse_complement {
            &query_segment_encoded_rc
        } else {
            &query_segment_encoded
        };
        let smem_query_bases = &query_for_smem
            [smem.query_start as usize..=(smem.query_end as usize).min(query_for_smem.len() - 1)];
        let smem_len = smem.query_end - smem.query_start + 1;
        log::debug!(
            "{}: SMEM {}: Query bases at [{}..{}] (len={}): {:?}",
            query_name,
            idx,
            smem.query_start,
            smem.query_end,
            smem_len,
            &smem_query_bases[..10.min(smem_query_bases.len())]
        );

        // CRITICAL: Keep seed positions in BIDIRECTIONAL coordinates
        // C++ keeps seeds in bidirectional coords [0, 2*l_pac) and only converts at SAM output
        // - Forward strand: ref_pos in [0, l_pac)
        // - Reverse strand: ref_pos in [l_pac, 2*l_pac)
        // get_reference_segment() handles bidirectional coords automatically

        log::debug!(
            "{}: SMEM {}: Seed ref_pos={} (bidirectional), is_rev={}, l_pac={}",
            query_name,
            idx,
            ref_pos,
            is_rev,
            bwa_idx.bns.packed_sequence_length
        );

        // Diagnostic validation removed - was causing 79,000% performance regression
        // The SMEM validation loop with log::info/warn was being called millions of times

        // Use query position in the coordinate system of the query we're aligning
        // For RC seeds, use smem.query_start directly (RC query coordinates)
        // For forward seeds, use smem.query_start directly (forward query coordinates)
        let query_pos = smem.query_start;

        let seed = Seed {
            query_pos,
            ref_pos, // Keep bidirectional coordinates!
            // CRITICAL: smem.query_end is EXCLUSIVE (see line 187), so len = end - start, NOT +1
            // C++ bwamem uses qend = qbeg + len (exclusive end), matching this calculation
            len: smem.query_end - smem.query_start,
            is_rev,
            interval_size: smem.interval_size, // Propagate interval_size from SMEM to Seed
        };

        // DEBUG: Log each seed creation with SMEM bounds
        log::debug!(
            "[SEED_CREATE] {}: ref_pos={}, qpos={}, len={}, strand={}, SMEM.query=[{}, {}) (query_end is exclusive)",
            query_name,
            seed.ref_pos,
            seed.query_pos,
            seed.len,
            if seed.is_rev { "rev" } else { "fwd" },
            smem.query_start,
            smem.query_end
        );

        // === CHAIN-BASED ALIGNMENT STRATEGY (C++ bwa-mem2) ===
        // Instead of aligning each seed individually, we now:
        // 1. Create all seeds first
        // 2. Chain them together
        // 3. Create alignment jobs using CHAIN bounds (not per-seed bounds)
        // This prevents N-rich regions from being included in DP alignment
        // (Per-seed bounds were too conservative and still included N bases)

        seeds.push(seed);
    }

    log::debug!(
        "{}: Found {} seeds (alignment jobs will be created from chains)",
        query_name,
        seeds.len()
    );

    // === NEW FLOW: Chain seeds FIRST, then create alignment jobs from chains ===
    // This matches C++ bwa-mem2 mem_align1_core() flow exactly

    // --- Seed Chaining ---
    // IMPORTANT: Pass seeds by value (not clone) so chain_seeds() sorts in place.
    // The chain seed indices will then correctly refer to the sorted seeds array.
    // chain_seeds() returns both the chains and the sorted seeds array.
    let (mut chained_results, seeds) = chain_seeds(seeds, _opt);
    log::debug!(
        "{}: Chaining produced {} chains",
        query_name,
        chained_results.len()
    );

    // --- Chain Filtering ---
    // Implements bwa-mem2 mem_chain_flt logic (bwamem.cpp:506-624)
    let filtered_chains = filter_chains(&mut chained_results, &seeds, _opt, query_len as i32);
    log::debug!(
        "{}: Chain filtering kept {} chains (from {} total)",
        query_name,
        filtered_chains.len(),
        chained_results.len()
    );

    // === CREATE ALIGNMENT JOBS FROM FILTERED CHAINS (C++ Strategy) ===
    // Create separate LEFT and RIGHT extension jobs per chain
    // This matches C++ bwa-mem2 separate extension model (bwamem.cpp:2229-2418)
    // CRITICAL: Process ALL seeds per chain, not just one (C++ line 2206)

    // Track left and right job indices for each seed in a chain
    #[derive(Debug, Clone)]
    struct SeedJobMapping {
        seed_idx: usize,              // Index into chain.seeds
        left_job_idx: Option<usize>,  // LEFT extension job index
        right_job_idx: Option<usize>, // RIGHT extension job index
    }

    #[derive(Debug, Clone)]
    struct ChainJobMapping {
        seed_jobs: Vec<SeedJobMapping>, // Multiple seeds per chain
    }

    let mut chain_to_job_map: Vec<ChainJobMapping> = Vec::new();

    for (chain_idx, chain) in filtered_chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            chain_to_job_map.push(ChainJobMapping {
                seed_jobs: Vec::new(),
            });
            continue;
        }

        // CRITICAL: Use correct query orientation based on chain strand
        // For reverse-strand chains: use RC query (matches reference in RC region)
        // For forward-strand chains: use forward query (matches reference in forward region)
        // C++ bwamem.cpp:2116 uses seq_[l].seq, but this is context-dependent
        let full_query = if chain.is_rev {
            &query_segment_encoded_rc
        } else {
            &query_segment_encoded
        };
        let query_len = full_query.len() as i32;

        log::debug!(
            "{}: Chain {}: is_rev={}, query_len={}, full_query[0..10]={:?}",
            query_name,
            chain_idx,
            chain.is_rev,
            query_len,
            full_query.iter().take(10).copied().collect::<Vec<u8>>()
        );

        // Calculate max possible reference span for this chain (C++ bwamem.cpp:2144-2166)
        // This covers all seeds with margins for gaps
        let l_pac = bwa_idx.bns.packed_sequence_length;
        let mut rmax_0 = l_pac << 1; // Start with max possible
        let mut rmax_1 = 0u64; // Start with min possible

        for &seed_idx in &chain.seeds {
            let seed = &seeds[seed_idx];

            // Calculate reference bounds with gap margins
            // b = t->rbeg - (t->qbeg + cal_max_gap(opt, t->qbeg))
            let left_margin = seed.query_pos + cal_max_gap(_opt, seed.query_pos);
            let b = if left_margin as u64 > seed.ref_pos {
                0
            } else {
                seed.ref_pos - left_margin as u64
            };

            // e = t->rbeg + t->len + (remaining_query + cal_max_gap(opt, remaining_query))
            let remaining_query = query_len - seed.query_pos - seed.len;
            let right_margin = remaining_query + cal_max_gap(_opt, remaining_query);
            let e = seed.ref_pos + seed.len as u64 + right_margin as u64;

            // Take min of all b values, max of all e values
            rmax_0 = rmax_0.min(b);
            rmax_1 = rmax_1.max(e);
        }

        // Clamp to valid range
        rmax_0 = rmax_0.max(0);
        rmax_1 = rmax_1.min(l_pac << 1);

        // If span crosses l_pac boundary, clamp to one side (C++ lines 2162-2166)
        if rmax_0 < l_pac && l_pac < rmax_1 {
            let first_seed = &seeds[chain.seeds[0]];
            if first_seed.ref_pos < l_pac {
                rmax_1 = l_pac;
            } else {
                rmax_0 = l_pac;
            }
        }

        log::debug!(
            "{}: Chain {}: rmax=[{}, {}) span={} (l_pac={})",
            query_name,
            chain_idx,
            rmax_0,
            rmax_1,
            rmax_1 - rmax_0,
            l_pac
        );

        // Fetch single large reference buffer covering entire chain
        let rseq = match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
            Ok(seq) => seq,
            Err(e) => {
                log::error!(
                    "{}: Chain {}: Error fetching reference segment: {}",
                    query_name,
                    chain_idx,
                    e
                );
                chain_to_job_map.push(ChainJobMapping {
                    seed_jobs: Vec::new(),
                });
                continue;
            }
        };

        // CRITICAL: Iterate through ALL seeds in REVERSE order (C++ bwamem.cpp:2206)
        // C++: for (int k=c->n-1; k >= 0; k--) { s = &c->seeds[(uint32_t)srt[k]]; ... }
        // Each seed gets its own LEFT/RIGHT extension pair
        let mut seed_job_mappings = Vec::new();

        log::debug!(
            "{}: Chain {}: Processing {} seeds (rmax=[{}, {}) span={}, l_pac={})",
            query_name,
            chain_idx,
            chain.seeds.len(),
            rmax_0,
            rmax_1,
            rmax_1 - rmax_0,
            bwa_idx.bns.packed_sequence_length
        );

        // Show first 10 bases of rseq buffer
        let rseq_first_10: Vec<u8> = rseq.iter().take(10).copied().collect();
        log::debug!(
            "{}: Chain {}: rseq[0..10]={:?}",
            query_name,
            chain_idx,
            rseq_first_10
        );

        // Iterate seeds in reverse order (matching C++)
        for &seed_chain_idx in chain.seeds.iter().rev() {
            let seed = &seeds[seed_chain_idx];

            let seed_query_start = seed.query_pos;
            let seed_query_end = seed.query_pos + seed.len;

            log::debug!(
                "{}: Chain {}: Processing seed {} - query_pos={}, len={}, ref_pos={} (bidirectional)",
                query_name,
                chain_idx,
                seed_chain_idx,
                seed_query_start,
                seed.len,
                seed.ref_pos
            );

            // Verify seed matches: check if query[seed_query_start..seed_query_end] matches rseq[seed_buffer_pos..seed_buffer_pos+seed.len]
            let seed_buffer_pos = (seed.ref_pos - rmax_0) as usize;
            let seed_query_slice: Vec<u8> = full_query
                [seed_query_start as usize..seed_query_end as usize]
                .iter()
                .take(10.min(seed.len as usize))
                .copied()
                .collect();
            let seed_ref_slice: Vec<u8> = if seed_buffer_pos < rseq.len()
                && seed_buffer_pos + seed.len as usize <= rseq.len()
            {
                rseq[seed_buffer_pos..seed_buffer_pos + seed.len as usize]
                    .iter()
                    .take(10.min(seed.len as usize))
                    .copied()
                    .collect()
            } else {
                vec![]
            };
            log::debug!(
                "{}: Chain {}: Seed {}: SEED MATCH CHECK: buffer_pos={}, query_slice[0..10]={:?}, ref_slice[0..10]={:?}",
                query_name,
                chain_idx,
                seed_chain_idx,
                seed_buffer_pos,
                seed_query_slice,
                seed_ref_slice
            );

            // Show seed boundary: last few bases of seed and first few bases after seed
            let seed_end_buffer_pos = seed_buffer_pos + seed.len as usize;
            let boundary_start = seed_end_buffer_pos.saturating_sub(5);
            let boundary_end = (seed_end_buffer_pos + 5).min(rseq.len());
            if boundary_start < rseq.len() {
                let boundary_ref: Vec<u8> = rseq[boundary_start..boundary_end].to_vec();
                log::debug!(
                    "{}: Chain {}: Seed {}: SEED BOUNDARY: rseq[{}..{}]={:?} (pos {} is last seed base, pos {} is first RIGHT base)",
                    query_name,
                    chain_idx,
                    seed_chain_idx,
                    boundary_start,
                    boundary_end,
                    boundary_ref,
                    seed_end_buffer_pos - 1,
                    seed_end_buffer_pos
                );
            }
            let query_boundary_start = seed_query_end.saturating_sub(5) as usize;
            let query_boundary_end = ((seed_query_end + 5).min(query_len)) as usize;
            let boundary_query: Vec<u8> =
                full_query[query_boundary_start..query_boundary_end].to_vec();
            log::debug!(
                "{}: Chain {}: Seed {}: QUERY BOUNDARY: full_query[{}..{}]={:?} (pos {} is last seed base, pos {} is first RIGHT base)",
                query_name,
                chain_idx,
                seed_chain_idx,
                query_boundary_start,
                query_boundary_end,
                boundary_query,
                seed_query_end - 1,
                seed_query_end
            );

            let mut left_job_idx = None;
            let mut right_job_idx = None;

            // --- LEFT Extension: seed start → query position 0 ---
            if seed_query_start > 0 {
                let left_query_len = seed_query_start as usize;
                let left_query: Vec<u8> = full_query[0..left_query_len].to_vec();

                // Index into rseq buffer (C++ bwamem.cpp:2280, 2298)
                // tmp = s->rbeg - rmax[0];
                // for (int64_t i = 0; i < tmp; ++i) rs[i] = rseq[tmp - 1 - i];
                let tmp = (seed.ref_pos - rmax_0) as usize;

                if tmp > 0 && tmp <= rseq.len() {
                    // Extract rseq[0..tmp] for LEFT extension
                    // Note: scalar_banded_swa_directional will reverse this
                    let left_target: Vec<u8> = rseq[0..tmp].to_vec();

                    left_job_idx = Some(alignment_jobs.len());

                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT extension qlen={} tlen={} tmp={}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        left_query_len,
                        left_target.len(),
                        tmp
                    );

                    // DETAILED LEFT DEBUG
                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT query extraction: seed_query_start={}, left_query_len={}, full_query[0..10]={:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        seed_query_start,
                        left_query_len,
                        full_query
                            .iter()
                            .take(10.min(left_query_len))
                            .copied()
                            .collect::<Vec<u8>>()
                    );
                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT target extraction: tmp={}, rseq[0..10]={:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        tmp,
                        rseq.iter().take(10.min(tmp)).copied().collect::<Vec<u8>>()
                    );
                    let left_query_first_10: Vec<u8> =
                        left_query.iter().take(10).copied().collect();
                    let left_target_first_10: Vec<u8> =
                        left_target.iter().take(10).copied().collect();
                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT left_query[0..10]={:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        left_query_first_10
                    );
                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT left_target[0..10]={:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        left_target_first_10
                    );

                    alignment_jobs.push(AlignmentJob {
                        seed_idx: chain_idx,
                        query: left_query,
                        target: left_target,
                        band_width: _opt.w,
                        query_offset: 0,
                        direction: Some(crate::alignment::banded_swa::ExtensionDirection::Left),
                        seed_len: seed.len, // For h0 calculation
                    });
                } else {
                    log::warn!(
                        "{}: Chain {}: Seed {}: Invalid LEFT tmp={} (seed.ref_pos={}, rmax_0={}, rseq.len={})",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        tmp,
                        seed.ref_pos,
                        rmax_0,
                        rseq.len()
                    );
                }
            }

            // --- RIGHT Extension: seed end → query end ---
            if seed_query_end < query_len {
                let right_query_start = seed_query_end as usize;
                let right_query_len = (query_len - seed_query_end) as usize;

                log::debug!(
                    "{}: Chain {}: Seed {}: RIGHT query extraction: seed_query_start={}, seed_query_end={}, right_query_start={}, full_query.len()={}",
                    query_name,
                    chain_idx,
                    seed_chain_idx,
                    seed_query_start,
                    seed_query_end,
                    right_query_start,
                    full_query.len()
                );
                log::debug!(
                    "{}: Chain {}: Seed {}: RIGHT query full_query[{}..{}]={:?}",
                    query_name,
                    chain_idx,
                    seed_chain_idx,
                    right_query_start,
                    right_query_start + 10.min(full_query.len() - right_query_start),
                    full_query[right_query_start..]
                        .iter()
                        .take(10)
                        .copied()
                        .collect::<Vec<u8>>()
                );

                let right_query: Vec<u8> = full_query[right_query_start..].to_vec();

                // Index into rseq buffer (C++ bwamem.cpp:2327, 2354)
                // re = s->rbeg + s->len - rmax[0];
                // sp.len1 = rmax[1] - rmax[0] - re;
                let re = ((seed.ref_pos + seed.len as u64) - rmax_0) as usize;

                if re < rseq.len() {
                    // Extract rseq[re..] for RIGHT extension (forward direction)
                    let right_target: Vec<u8> = rseq[re..].to_vec();

                    right_job_idx = Some(alignment_jobs.len());

                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT extension qlen={} tlen={} buffer_idx={}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        right_query_len,
                        right_target.len(),
                        re
                    );

                    // DETAILED INDEXING DEBUG
                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT indexing: seed.ref_pos={}, seed.len={}, rmax_0={}, re={} (calculation: {} + {} - {} = {})",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        seed.ref_pos,
                        seed.len,
                        rmax_0,
                        re,
                        seed.ref_pos,
                        seed.len,
                        rmax_0,
                        seed.ref_pos + seed.len as u64 - rmax_0
                    );

                    // Show first 10 bases of rseq at re
                    let rseq_at_re: Vec<u8> = rseq[re..].iter().take(10).copied().collect();
                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT buffer rseq[{}..{}]: {:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        re,
                        re + 10.min(rseq.len() - re),
                        rseq_at_re
                    );

                    // Show first 10 bases of right_target
                    let right_target_first_10: Vec<u8> =
                        right_target.iter().take(10).copied().collect();
                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT right_target[0..10]: {:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        right_target_first_10
                    );

                    // Show first 10 bases of right_query
                    let right_query_first_10: Vec<u8> =
                        right_query.iter().take(10).copied().collect();
                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT right_query[0..10]: {:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        right_query_first_10
                    );

                    alignment_jobs.push(AlignmentJob {
                        seed_idx: chain_idx,
                        query: right_query,
                        target: right_target,
                        band_width: _opt.w,
                        query_offset: seed_query_end,
                        direction: Some(crate::alignment::banded_swa::ExtensionDirection::Right),
                        seed_len: seed.len, // For h0 calculation
                    });
                }
            }

            // Store this seed's job mapping
            seed_job_mappings.push(SeedJobMapping {
                seed_idx: seed_chain_idx,
                left_job_idx,
                right_job_idx,
            });
        } // End of seed iteration loop

        // Store all seed job mappings for this chain
        chain_to_job_map.push(ChainJobMapping {
            seed_jobs: seed_job_mappings,
        });
    }

    log::debug!(
        "{}: Created {} alignment jobs from {} filtered chains",
        query_name,
        alignment_jobs.len(),
        filtered_chains.len()
    );

    // --- Execute Alignments (Adaptive Strategy) ---
    // Use adaptive routing strategy that combines:
    // 1. Divergence-based routing (high divergence → scalar, low → SIMD)
    // 2. Adaptive batch sizing (8-32 based on sequence characteristics)
    // 3. Fallback to scalar for small batches (<8 jobs)
    let extended_cigars = if use_batched_simd && alignment_jobs.len() >= 8 {
        // Use adaptive strategy for 8+ jobs
        // This provides optimal performance by routing jobs to the best execution path
        execute_adaptive_alignments(&sw_params, &alignment_jobs)
    } else {
        // Fall back to scalar processing for very small batches
        execute_scalar_alignments(&sw_params, &alignment_jobs)
    };

    // Separate scores, CIGARs, and aligned sequences
    let alignment_scores: Vec<i32> = extended_cigars
        .iter()
        .map(|(score, _, _, _)| *score)
        .collect();
    let alignment_cigars: Vec<Vec<(u8, i32)>> = extended_cigars
        .iter()
        .map(|(_, cigar, _, _)| cigar.clone())
        .collect();
    let ref_aligned_seqs: Vec<Vec<u8>> = extended_cigars
        .iter()
        .map(|(_, _, ref_aligned, _)| ref_aligned.clone())
        .collect();
    let query_aligned_seqs: Vec<Vec<u8>> = extended_cigars
        .into_iter()
        .map(|(_, _, _, query_aligned)| query_aligned)
        .collect();

    // Extract query offsets for soft-clipping (C++ strategy: match mem_reg2aln)
    let query_offsets: Vec<i32> = alignment_jobs.iter().map(|job| job.query_offset).collect();

    log::debug!(
        "{}: Extended {} chains, {} CIGARs produced",
        query_name,
        filtered_chains.len(),
        alignment_cigars.len()
    );

    // === MULTI-ALIGNMENT GENERATION FROM FILTERED CHAINS ===
    // Generate multiple alignment candidates per chain (one per seed)
    // Match C++ bwa-mem2: each seed gets LEFT+SEED+RIGHT, then select best
    let mut alignments = Vec::new();

    for (chain_idx, chain) in filtered_chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            continue;
        }

        // Get all seed job mappings for this chain
        let mapping = &chain_to_job_map[chain_idx];

        // Select appropriate query based on strand
        let full_query = if chain.is_rev {
            &query_segment_encoded_rc
        } else {
            &query_segment_encoded
        };
        let query_len = full_query.len() as i32;

        // Process ALL seeds for this chain, creating alignment candidates
        let mut best_score = 0;
        let mut best_alignment_data: Option<(Vec<(u8, i32)>, i32, u64, i32, i32)> = None;

        log::debug!(
            "{}: Chain {}: Processing {} seed alignments",
            query_name,
            chain_idx,
            mapping.seed_jobs.len()
        );

        for seed_job in &mapping.seed_jobs {
            let seed = &seeds[seed_job.seed_idx];

            // === COMBINE LEFT + SEED + RIGHT EXTENSIONS FOR THIS SEED ===
            let mut combined_cigar = Vec::new();
            let mut combined_score = 0;
            let mut alignment_start_pos = seed.ref_pos;
            let mut query_start_aligned = 0;
            let mut query_end_aligned = query_len;

            // LEFT extension
            if let Some(left_idx) = seed_job.left_job_idx {
                let left_cigar = &alignment_cigars[left_idx];
                let left_score = alignment_scores[left_idx];

                log::debug!(
                    "{}: Chain {}: Seed {}: LEFT extension score={}, CIGAR={:?}",
                    query_name,
                    chain_idx,
                    seed_job.seed_idx,
                    left_score,
                    left_cigar
                );

                // Add left CIGAR operations
                combined_cigar.extend(left_cigar.iter().cloned());
                combined_score += left_score;

                // Update alignment start position (move back by left extension)
                let left_ref_len: i32 = left_cigar
                    .iter()
                    .filter_map(|&(op, len)| match op as char {
                        'M' | 'D' => Some(len),
                        _ => None,
                    })
                    .sum();
                if left_ref_len as u64 <= alignment_start_pos {
                    alignment_start_pos -= left_ref_len as u64;
                } else {
                    alignment_start_pos = 0;
                }
            } else if seed.query_pos > 0 {
                // No left extension job, soft-clip 5' end
                combined_cigar.push((b'S', seed.query_pos));
                query_start_aligned = seed.query_pos;
            }

            // SEED match (perfect match for seed length)
            combined_cigar.push((b'M', seed.len));
            combined_score += seed.len; // Add seed score

            // RIGHT extension
            let seed_end = seed.query_pos + seed.len;
            if let Some(right_idx) = seed_job.right_job_idx {
                let right_cigar = &alignment_cigars[right_idx];
                let right_score = alignment_scores[right_idx];

                log::debug!(
                    "{}: Chain {}: Seed {}: RIGHT extension score={}, CIGAR={:?}",
                    query_name,
                    chain_idx,
                    seed_job.seed_idx,
                    right_score,
                    right_cigar
                );

                // Add right CIGAR operations
                combined_cigar.extend(right_cigar.iter().cloned());
                combined_score += right_score;
            } else if seed_end < query_len {
                // No right extension job, soft-clip 3' end
                combined_cigar.push((b'S', query_len - seed_end));
                query_end_aligned = seed_end;
            }

            // Merge consecutive CIGAR operations (e.g., M+M → M)
            let cigar_for_candidate = merge_cigar_operations(combined_cigar);

            log::debug!(
                "{}: Chain {}: Seed {}: combined_score={} bounds=[{}, {})",
                query_name,
                chain_idx,
                seed_job.seed_idx,
                combined_score,
                query_start_aligned,
                query_end_aligned
            );

            // Store as candidate if better than current best
            if combined_score > best_score {
                best_score = combined_score;
                best_alignment_data = Some((
                    cigar_for_candidate,
                    combined_score,
                    alignment_start_pos,
                    query_start_aligned,
                    query_end_aligned,
                ));
            }
        } // End of seed_job iteration

        // Use the best alignment candidate for this chain
        if let Some((
            mut cigar_for_alignment,
            combined_score,
            alignment_start_pos,
            query_start_aligned,
            query_end_aligned,
        )) = best_alignment_data
        {
            log::debug!(
                "{}: Chain {} (weight={}, kept={}): BEST score={} from {} candidates",
                query_name,
                chain_idx,
                chain.weight,
                chain.kept,
                combined_score,
                mapping.seed_jobs.len()
            );

            // CRITICAL: Validate and correct CIGAR length to match query length
            // Safety check for systematic off-by-one errors in CIGAR generation
            // Only count operations that consume query bases (M, I, S, =, X)
            let cigar_len: i32 = cigar_for_alignment
                .iter()
                .filter_map(|&(op, len)| match op as char {
                    'M' | 'I' | 'S' | '=' | 'X' => Some(len),
                    _ => None, // D, H, N, P don't consume query
                })
                .sum();

            if cigar_len != query_len {
                log::debug!(
                    "{}: Chain {}: CIGAR length mismatch: cigar={}, query={}, diff={} - adjusting last query-consuming operation",
                    query_name,
                    chain_idx,
                    cigar_len,
                    query_len,
                    query_len - cigar_len
                );

                // Adjust the last query-consuming operation (S or M) to match query length
                let diff = query_len - cigar_len;
                for op in cigar_for_alignment.iter_mut().rev() {
                    if op.0 == b'S' || op.0 == b'M' {
                        let old_len = op.1;
                        op.1 += diff;
                        if op.1 > 0 {
                            log::debug!(
                                "{}: Adjusted CIGAR op {} from {} to {}",
                                query_name,
                                op.0 as char,
                                old_len,
                                op.1
                            );
                            break;
                        } else {
                            log::warn!(
                                "{}: CIGAR op {} became negative ({})! Setting to 0 and continuing...",
                                query_name,
                                op.0 as char,
                                op.1
                            );
                            op.1 = 0;
                        }
                    }
                }
            }

            // Use the alignment start position calculated from extensions
            let adjusted_ref_start = alignment_start_pos;

            // Convert global position to chromosome-specific position
            let global_pos = adjusted_ref_start as i64;
            let (pos_f, _is_rev_depos) = bwa_idx.bns.bns_depos(global_pos);
            let rid = bwa_idx.bns.bns_pos2rid(pos_f);

            let (ref_name, ref_id, chr_pos) =
                if rid >= 0 && (rid as usize) < bwa_idx.bns.annotations.len() {
                    let ann = &bwa_idx.bns.annotations[rid as usize];
                    let chr_relative_pos = pos_f - ann.offset as i64;
                    (ann.name.clone(), rid as usize, chr_relative_pos as u64)
                } else {
                    log::warn!(
                        "{}: Invalid reference ID {} for position {}",
                        query_name,
                        rid,
                        global_pos
                    );
                    ("unknown_ref".to_string(), 0, 0)
                };

            // Calculate query bounds from aligned region
            let query_start = query_start_aligned;
            let query_end = query_end_aligned;
            let seed_coverage = chain.weight; // Use chain weight as seed coverage for MAPQ

            // Generate hash for tie-breaking (based on position and strand)
            let hash = hash_64((chr_pos << 1) | (if chain.is_rev { 1 } else { 0 }));

            // Generate MD tag by comparing actual reference and query sequences
            let md_tag = if !pac_data.is_empty() {
                // Calculate reference length from CIGAR (M and D consume reference)
                let ref_len: i32 = cigar_for_alignment
                    .iter()
                    .filter_map(|&(op, len)| match op as char {
                        'M' | 'D' => Some(len),
                        _ => None,
                    })
                    .sum();

                // Extract reference sequence for aligned region
                let ref_start = adjusted_ref_start as i64;
                let ref_end = ref_start + ref_len as i64;
                let ref_aligned = bwa_idx.bns.bns_get_seq(&pac_data, ref_start, ref_end);

                // Extract query sequence for aligned region (already in 2-bit encoding)
                let query_aligned = &full_query[query_start as usize..query_end as usize];

                // Generate MD tag by comparing sequences
                Alignment::generate_md_tag(&ref_aligned, query_aligned, &cigar_for_alignment)
            } else {
                // Fallback if PAC file not available: simple MD tag from CIGAR
                let match_len: i32 = cigar_for_alignment
                    .iter()
                    .filter_map(|&(op, len)| if op == b'M' { Some(len) } else { None })
                    .sum();
                format!("{}", match_len)
            };

            // Calculate exact NM (edit distance) from MD tag and CIGAR
            let nm = Alignment::calculate_exact_nm(&md_tag, &cigar_for_alignment);

            alignments.push(Alignment {
                query_name: query_name.to_string(),
                flag: if chain.is_rev { sam_flags::REVERSE } else { 0 },
                ref_name,
                ref_id,
                pos: chr_pos,
                mapq: 60,              // Will be calculated by mark_secondary_alignments
                score: combined_score, // Use combined score from left + seed + right
                cigar: cigar_for_alignment,
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(query_seq).to_string(),
                qual: query_qual.to_string(),
                tags: vec![
                    ("AS".to_string(), format!("i:{}", combined_score)),
                    ("NM".to_string(), format!("i:{}", nm)),
                    ("MD".to_string(), format!("Z:{}", md_tag)),
                ],
                // Internal fields for alignment selection
                query_start,
                query_end,
                seed_coverage,
                hash,
                frac_rep: chain.frac_rep,
            });
        } else {
            log::warn!(
                "{}: Chain {} has no valid alignment candidates from {} seeds",
                query_name,
                chain_idx,
                mapping.seed_jobs.len()
            );
        }
    } // End of chain iteration

    // Log buffer capacity validation
    if max_smem_count > query_len {
        log::debug!(
            "{}: SMEM buffer grew beyond initial capacity! max_smem_count={} > query_len={} (growth factor: {:.2}x)",
            query_name,
            max_smem_count,
            query_len,
            max_smem_count as f64 / query_len as f64
        );
    } else {
        log::debug!(
            "{}: SMEM buffer stayed within capacity. max_smem_count={} <= query_len={} (utilization: {:.1}%)",
            query_name,
            max_smem_count,
            query_len,
            (max_smem_count as f64 / query_len as f64) * 100.0
        );
    }

    #[cfg(feature = "debug-logging")]
    if is_debug_read {
        log::debug!("[DEBUG_READ] Generated {} SMEM(s)", all_smems.len());
        log::debug!("[DEBUG_READ] Created {} alignment(s)", alignments.len());
        for (i, aln) in alignments.iter().enumerate() {
            log::debug!(
                "[DEBUG_READ] Alignment[{}]: {}:{} MAPQ={} Score={} CIGAR={}",
                i,
                aln.ref_name,
                aln.pos,
                aln.mapq,
                aln.score,
                aln.cigar_string()
            );
        }
    }

    // === ALIGNMENT SELECTION: Sort, Mark Secondary, Calculate MAPQ ===
    // Implements bwa-mem2 mem_mark_primary_se logic (bwamem.cpp:1420-1464)
    if !alignments.is_empty() {
        // Sort alignments by score (descending), then by hash (for tie-breaking)
        // Matches C++ alnreg_hlt comparator in bwamem.cpp:155
        alignments.sort_by(|a, b| {
            match b.score.cmp(&a.score) {
                std::cmp::Ordering::Equal => a.hash.cmp(&b.hash), // Tie-breaker
                other => other,
            }
        });

        // Mark secondary alignments and calculate MAPQ
        // This modifies alignments in-place:
        // - Sets sam_flags::SECONDARY flag for secondary alignments
        // - Calculates proper MAPQ values (0-60) based on score differences
        mark_secondary_alignments(&mut alignments, _opt);

        log::debug!(
            "{}: After alignment selection: {} alignments ({} primary, {} secondary)",
            query_name,
            alignments.len(),
            alignments
                .iter()
                .filter(|a| a.flag & sam_flags::SECONDARY == 0)
                .count(),
            alignments
                .iter()
                .filter(|a| a.flag & sam_flags::SECONDARY != 0)
                .count()
        );

        let xa_tags = generate_xa_tags(&alignments, _opt);
        let sa_tags = generate_sa_tags(&alignments);

        for aln in alignments.iter_mut() {
            // Only consider non-secondary alignments for SA/XA tags
            if aln.flag & sam_flags::SECONDARY == 0 {
                if let Some(sa_tag) = sa_tags.get(&aln.query_name) {
                    // This is part of a chimeric read, add SA tag
                    aln.tags.push(("SA".to_string(), sa_tag.clone()));
                    log::debug!(
                        "{}: Added SA tag for read {}",
                        aln.query_name,
                        aln.query_name
                    );
                } else if let Some(xa_tag) = xa_tags.get(&aln.query_name) {
                    // Not a chimeric read, add XA tag if available
                    aln.tags.push(("XA".to_string(), xa_tag.clone()));
                    log::debug!(
                        "{}: Added XA tag with {} alternative alignments",
                        aln.query_name,
                        xa_tag.matches(';').count()
                    );
                }
            }
        }
    } else {
        // === NO ALIGNMENTS FOUND: Create unmapped read (C++ bwa-mem2 behavior) ===
        // When no seeds/chains were generated, we must still output the read as unmapped
        // (SAM flag sam_flags::UNMAPPED) to match C++ bwa-mem2 behavior and avoid silently dropping reads
        log::debug!(
            "{}: No alignments generated (no seeds or all chains filtered), creating unmapped read",
            query_name
        );

        alignments.push(Alignment {
            query_name: query_name.to_string(),
            flag: sam_flags::UNMAPPED,
            ref_name: "*".to_string(),
            ref_id: 0, // 0 for unmapped (doesn't correspond to any real chromosome)
            pos: 0,
            mapq: 0,
            score: 0,
            cigar: Vec::new(), // Empty CIGAR = "*" in SAM format
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::from_utf8_lossy(query_seq).to_string(),
            qual: query_qual.to_string(),
            tags: vec![
                ("AS".to_string(), "i:0".to_string()),
                ("NM".to_string(), "i:0".to_string()),
            ],
            // Internal fields (not used for unmapped reads)
            query_start: 0,
            query_end: 0,
            seed_coverage: 0,
            hash: 0,
            frac_rep: 0.0, // Initial placeholder
        });
    }

    alignments
}

#[cfg(test)]
mod tests {
    use crate::alignment::pipeline::sam_flags;
    use crate::index::BwaIndex;

    #[test]
    fn test_generate_seeds_basic() {
        use crate::mem_opt::MemOpt;
        use std::path::Path;

        let prefix = Path::new("test_data/test_ref.fa");
        if !prefix.exists() {
            eprintln!("Skipping test_generate_seeds_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_generate_seeds_basic - could not load index");
                return;
            }
        };

        let mut opt = MemOpt::default();
        opt.min_seed_len = 10; // Ensure our seed is long enough

        let query_name = "test_query";
        let query_seq = b"ACGTACGTACGT"; // 12bp
        let query_qual = "IIIIIIIIIIII";

        // Test doesn't need real MD tags, use empty pac_data
        let pac_data: &[u8] = &[];
        let alignments =
            super::generate_seeds(&bwa_idx, pac_data, query_name, query_seq, query_qual, &opt);

        assert!(
            !alignments.is_empty(),
            "Expected at least one alignment for a matching query"
        );

        let primary_alignment = alignments
            .iter()
            .find(|a| a.flag & sam_flags::SECONDARY == 0);
        assert!(primary_alignment.is_some(), "Expected a primary alignment");

        let pa = primary_alignment.unwrap();
        assert_eq!(pa.ref_name, "test_sequence");
        assert!(
            pa.score > 0,
            "Expected a positive score for a good match, got {}",
            pa.score
        );
        assert!(pa.pos < 60, "Position should be within reference length");
        assert_eq!(pa.cigar_string(), "12M", "Expected a perfect match CIGAR");
    }
}
