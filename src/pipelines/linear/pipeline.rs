use super::chaining::{
    Chain, SoAChainBatch, chain_seeds, chain_seeds_batch, filter_chains, filter_chains_batch,
};
use super::index::index::BwaIndex;
use super::mem_opt::MemOpt;
use super::seeding::SMEM;
use super::seeding::Seed;
use super::seeding::forward_only_seed_strategy;
use super::seeding::generate_smems_for_strand;
use super::seeding::generate_smems_from_position;
use super::seeding::SoASeedBatch;
use crate::alignment::utils::base_to_code;
use crate::alignment::utils::reverse_complement_code;
use crate::alignment::workspace::with_workspace;
use crate::core::io::soa_readers::SoAReadBatch;

// ============================================================================
// SEED GENERATION (SMEM EXTRACTION)
// ============================================================================
//
// This section contains the main seed generation pipeline:
// - SMEM (Supermaximal Exact Match) extraction using FM-Index
// - Bidirectional search (forward and reverse complement)
// - Seed extension and filtering
// ============================================================================

/// Stage 1: Seed Finding
///
/// Generates seeds from the FM-Index using SMEM algorithm.
/// Returns seeds and encoded query sequences.
pub fn find_seeds(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_seq: &[u8],
    opt: &MemOpt,
) -> (Vec<Seed>, Vec<u8>, Vec<u8>) {
    let query_len = query_seq.len();
    if query_len == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    #[cfg(feature = "debug-logging")]
    let is_debug_read = query_name.contains("1150:14380");

    #[cfg(feature = "debug-logging")]
    if is_debug_read {
        log::debug!("[DEBUG_READ] Generating seeds for: {}", query_name);
        log::debug!("[DEBUG_READ] Query length: {}", query_len);
    }

    // Create encoded versions of the query sequence
    let mut encoded_query = Vec::with_capacity(query_len);
    let mut encoded_query_rc = Vec::with_capacity(query_len); // Reverse complement
    for &base in query_seq {
        let code = base_to_code(base);
        encoded_query.push(code);
        encoded_query_rc.push(reverse_complement_code(code));
    }
    encoded_query_rc.reverse();

    // Pre-allocate for typical SMEM counts to avoid reallocations
    // 512 accounts for initial + re-seeding + 3rd round SMEMs
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
            if pass2_added > 10 {
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
                if pass3_added > 10 {
                    log::debug!(
                        "SMEM_VALIDATION {}:   ... ({} more SMEMs)",
                        query_name,
                        pass3_added - 10
                    );
                }
            }
        } else {
            log::debug!("SMEM_VALIDATION {query_name}: Pass 3 (forward-only) added 0 new SMEMs");
        }
    } else {
        log::debug!("SMEM_VALIDATION {query_name}: Pass 3 (forward-only) skipped (max_mem_intv=0)");
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

    let mut seeds = Vec::new();
    let mut seeds_per_smem_count = Vec::new(); // Track seeds generated per SMEM for Phase 2 validation

    for (smem_idx, smem) in sorted_smems.iter().enumerate() {
        let seeds_before = seeds.len();

        // Limit positions per SMEM to ensure coverage from multiple SMEMs
        // The get_sa_entries function will sample evenly across the interval
        let max_positions_this_smem = (seeds_per_smem as u32).min(opt.max_occ as u32);

        // Use the new get_sa_entries function to get multiple reference positions
        // It samples evenly across the entire BWT interval using floating-point step
        let ref_positions = super::seeding::get_sa_entries(
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
            seeds.push(seed);

            // Hard limit on seeds per read to prevent memory explosion
            if seeds.len() >= SEEDS_PER_READ {
                log::debug!(
                    "{query_name}: Hit SEEDS_PER_READ limit ({SEEDS_PER_READ}), truncating"
                );
                break;
            }
        }

        let seeds_added = seeds.len() - seeds_before;
        seeds_per_smem_count.push((smem_idx, seeds_added, skipped_boundary));

        if seeds.len() >= SEEDS_PER_READ {
            break;
        }
    }

    // PHASE 2 VALIDATION: Log seed conversion summary
    log::debug!(
        "SEED_CONVERSION {}: Total {} SMEMs → {} seeds ({} seeds/SMEM limit)",
        query_name,
        sorted_smems.len(),
        seeds.len(),
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
        seeds.len(),
        sorted_smems.len()
    );
    (seeds, encoded_query, encoded_query_rc)
}

/// Stage 2: Chaining
///
/// Chains seeds together using O(n²) DP and filters by score.
/// Returns filtered chains and sorted seeds.
pub fn build_and_filter_chains(
    seeds: Vec<Seed>,
    opt: &MemOpt,
    query_len: usize,
    query_name: &str,
) -> (Vec<Chain>, Vec<Seed>) {
    // Chain seeds together and then filter them
    let (mut chained_results, sorted_seeds) = chain_seeds(seeds, opt);
    log::debug!(
        "{}: Chaining produced {} chains",
        query_name,
        chained_results.len()
    );

    // PHASE 3 VALIDATION: Log all chains before filtering
    if log::log_enabled!(log::Level::Debug) {
        for (idx, chain) in chained_results.iter().enumerate() {
            log::debug!(
                "CHAIN_VALIDATION {}: Chain[{}] score={} seeds={} query=[{}..{}] ref=[{}..{}] rev={} weight={} frac_rep={:.3} rid={}",
                query_name,
                idx,
                chain.score,
                chain.seeds.len(),
                chain.query_start,
                chain.query_end,
                chain.ref_start,
                chain.ref_end,
                chain.is_rev,
                chain.weight,
                chain.frac_rep,
                chain.rid
            );
        }
    }

    let filtered_chains = filter_chains(&mut chained_results, &sorted_seeds, opt, query_len as i32);
    log::debug!(
        "CHAIN_VALIDATION {}: Kept {} chains after filtering (from {} total)",
        query_name,
        filtered_chains.len(),
        chained_results.len()
    );

    // PHASE 3 VALIDATION: Log filtered chains
    if log::log_enabled!(log::Level::Debug) {
        for (idx, chain) in filtered_chains.iter().enumerate() {
            log::debug!(
                "CHAIN_VALIDATION {}: FilteredChain[{}] score={} seeds={} query=[{}..{}] ref=[{}..{}] rev={} weight={} frac_rep={:.3} rid={}",
                query_name,
                idx,
                chain.score,
                chain.seeds.len(),
                chain.query_start,
                chain.query_end,
                chain.ref_start,
                chain.ref_end,
                chain.is_rev,
                chain.weight,
                chain.frac_rep,
                chain.rid
            );
        }
    }

    (filtered_chains, sorted_seeds)
}

/// Stage 2: Chaining (Batch version)
///
/// Chains seeds together using SoA-native functions and filters by score for a batch of reads.
/// Returns SoA-friendly chained results.
pub fn build_and_filter_chains_batch(
    bwa_idx: &BwaIndex, // Added bwa_idx parameter
    soa_seed_batch: &SoASeedBatch,
    read_batch: &SoAReadBatch,
    opt: &MemOpt,
) -> SoAChainBatch {
    let l_pac = bwa_idx.bns.packed_sequence_length;

    // Chain seeds together
    let mut soa_chain_batch = chain_seeds_batch(soa_seed_batch, opt, l_pac);

    // Prepare query lengths for filtering
    let query_lengths: Vec<i32> = read_batch
        .read_boundaries
        .iter()
        .map(|(_, len)| *len as i32)
        .collect();

    // Filter chains
    filter_chains_batch(&mut soa_chain_batch, soa_seed_batch, opt, &query_lengths, &read_batch.names);

    soa_chain_batch
}
#[cfg(test)]
mod tests {
    use super::super::finalization::sam_flags;
    use super::super::index::index::BwaIndex;
    use super::super::mem_opt::MemOpt;

    // ========================================================================
    // CIGAR REFERENCE LENGTH CALCULATION TESTS
    // ========================================================================
    //
    // The bounds check at lines 1472-1489 requires accurate CIGAR reference
    // length calculation. M and D operations consume reference.
    // ========================================================================

    /// Calculate reference length from CIGAR (M, D consume reference)
    /// This matches the logic in build_candidate_alignments()
    fn calculate_cigar_ref_len(cigar: &[(u8, i32)]) -> i32 {
        cigar
            .iter()
            .filter_map(|&(op, len)| {
                if matches!(op as char, 'M' | 'D') {
                    Some(len)
                } else {
                    None
                }
            })
            .sum()
    }

    #[test]
    fn test_cigar_ref_len_matches_only() {
        // 100M = 100bp reference consumed
        let cigar = vec![(b'M', 100)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 100);
    }

    #[test]
    fn test_cigar_ref_len_with_insertions() {
        // 50M2I48M = 98bp reference (insertions don't consume reference)
        let cigar = vec![(b'M', 50), (b'I', 2), (b'M', 48)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 98);
    }

    #[test]
    fn test_cigar_ref_len_with_deletions() {
        // 50M2D48M = 100bp reference (deletions consume reference)
        let cigar = vec![(b'M', 50), (b'D', 2), (b'M', 48)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 100);
    }

    #[test]
    fn test_cigar_ref_len_with_soft_clips() {
        // 10S80M10S = 80bp reference (soft clips don't consume reference)
        let cigar = vec![(b'S', 10), (b'M', 80), (b'S', 10)];
        assert_eq!(calculate_cigar_ref_len(&cigar), 80);
    }

    #[test]
    fn test_cigar_ref_len_complex() {
        // 5S45M2I3D47M5S = 45+3+47 = 95bp reference
        let cigar = vec![
            (b'S', 5),
            (b'M', 45),
            (b'I', 2),
            (b'D', 3),
            (b'M', 47),
            (b'S', 5),
        ];
        assert_eq!(calculate_cigar_ref_len(&cigar), 95);
    }

    // ========================================================================
    // BOUNDS CHECK LOGIC TESTS
    // ========================================================================
    //
    // These tests verify the bounds checking logic that prevents
    // CIGAR_MAPS_OFF_REFERENCE errors (lines 1472-1489).
    // ========================================================================

    /// Test bounds check logic (extracted for unit testing)
    /// Returns true if alignment is WITHIN bounds (acceptable)
    fn bounds_check_passes(chr_pos: u64, cigar_ref_len: i32, ref_length: u64) -> bool {
        chr_pos + cigar_ref_len as u64 <= ref_length
    }

    #[test]
    fn test_bounds_check_well_within_bounds() {
        // Alignment at position 1000, 100bp CIGAR, on 10000bp chromosome
        assert!(bounds_check_passes(1000, 100, 10000));
    }

    #[test]
    fn test_bounds_check_exactly_at_end() {
        // Alignment at position 9900, 100bp CIGAR, on 10000bp chromosome
        // chr_pos + cigar_ref_len = 9900 + 100 = 10000 = ref_length (OK)
        assert!(bounds_check_passes(9900, 100, 10000));
    }

    #[test]
    fn test_bounds_check_one_bp_past_end() {
        // Alignment at position 9901, 100bp CIGAR, on 10000bp chromosome
        // chr_pos + cigar_ref_len = 9901 + 100 = 10001 > 10000 (FAIL)
        assert!(!bounds_check_passes(9901, 100, 10000));
    }

    #[test]
    fn test_bounds_check_far_past_end() {
        // Alignment at position 9950, 100bp CIGAR, on 10000bp chromosome
        // chr_pos + cigar_ref_len = 9950 + 100 = 10050 > 10000 (FAIL)
        assert!(!bounds_check_passes(9950, 100, 10000));
    }

    #[test]
    fn test_bounds_check_chry_end_case() {
        // chrY in GRCh38 is 57,227,415bp
        // Bug case: alignment at 57227414 with ~100bp CIGAR
        // chr_pos + cigar_ref_len = 57227414 + 100 = 57227514 > 57227415 (FAIL)
        let chry_length: u64 = 57_227_415;
        let chr_pos: u64 = 57_227_414;
        let cigar_ref_len: i32 = 100;
        assert!(!bounds_check_passes(chr_pos, cigar_ref_len, chry_length));
    }

    #[test]
    fn test_bounds_check_chry_valid_alignment() {
        // Valid alignment near chrY end: position 57227314 with 100bp CIGAR
        // chr_pos + cigar_ref_len = 57227314 + 100 = 57227414 < 57227415 (OK)
        let chry_length: u64 = 57_227_415;
        let chr_pos: u64 = 57_227_314;
        let cigar_ref_len: i32 = 100;
        assert!(bounds_check_passes(chr_pos, cigar_ref_len, chry_length));
    }

    #[test]
    fn test_bounds_check_at_position_zero() {
        // Alignment at chromosome start
        assert!(bounds_check_passes(0, 100, 10000));
    }

    #[test]
    fn test_bounds_check_single_base_chromosome() {
        // Edge case: 1bp chromosome
        assert!(bounds_check_passes(0, 1, 1)); // pos 0, 1bp CIGAR, 1bp chr = OK
        assert!(!bounds_check_passes(0, 2, 1)); // pos 0, 2bp CIGAR, 1bp chr = FAIL
        assert!(!bounds_check_passes(1, 1, 1)); // pos 1, 1bp CIGAR, 1bp chr = FAIL
    }

    #[test]
    fn test_bounds_check_real_cigar_scenario() {
        // Real scenario: 148bp read with complex CIGAR
        // CIGAR: 84S64M (84bp soft-clipped, 64bp match)
        // Reference consumed = 64bp (only M/D operations)
        let cigar = vec![(b'S', 84), (b'M', 64)];
        let ref_len = calculate_cigar_ref_len(&cigar);
        assert_eq!(ref_len, 64);

        // At chrY end (57227415bp), alignment starting at 57227400
        // would end at 57227400 + 64 = 57227464 > 57227415 (FAIL)
        let chry_length: u64 = 57_227_415;
        assert!(!bounds_check_passes(57_227_400, ref_len, chry_length));

        // Alignment starting at 57227350 would be OK
        // 57227350 + 64 = 57227414 < 57227415 (OK)
        assert!(bounds_check_passes(57_227_350, ref_len, chry_length));
    }

}
