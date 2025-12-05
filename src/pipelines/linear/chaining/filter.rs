//! Chain filtering algorithms.
//!
//! Implements C++ mem_chain_flt (bwamem.cpp:506-624) for filtering chains
//! based on weight, overlap, and drop_ratio thresholds.

use crate::pipelines::linear::mem_opt::MemOpt;
use crate::pipelines::linear::seeding::{Seed, SoASeedBatch};

use super::types::{Chain, SoAChainBatch};
use super::weight::{calculate_chain_weight, calculate_chain_weight_soa};

/// Filter chains using drop_ratio and score thresholds.
///
/// Implements C++ mem_chain_flt (bwamem.cpp:506-624).
///
/// Algorithm:
/// 1. Calculate weight for each chain
/// 2. Sort chains by weight (descending)
/// 3. Filter by min_chain_weight
/// 4. Apply drop_ratio: keep chains with weight >= best_weight * drop_ratio
/// 5. Mark overlapping chains as kept=1/2, non-overlapping as kept=3
pub fn filter_chains(
    chains: &mut Vec<Chain>,
    seeds: &[Seed],
    opt: &MemOpt,
    query_length: i32,
) -> Vec<Chain> {
    if chains.is_empty() {
        return Vec::new();
    }
    log::debug!("filter_chains: Input with {} chains", chains.len());

    // Calculate weights for all chains
    for (idx, chain) in chains.iter_mut().enumerate() {
        let (weight, l_rep) = calculate_chain_weight(chain, seeds, opt);
        chain.weight = weight;
        chain.frac_rep = if query_length > 0 {
            l_rep as f32 / query_length as f32
        } else {
            0.0
        };
        chain.kept = 0; // Initially mark as discarded
        log::debug!(
            "  filter_chains: Chain {} (score {}) weight={}, frac_rep={:.2}",
            idx,
            chain.score,
            chain.weight,
            chain.frac_rep
        );
    }

    // Sort chains by weight (descending)
    chains.sort_by(|a, b| b.weight.cmp(&a.weight));
    log::debug!("filter_chains: Chains after sorting by weight:");
    for (idx, chain) in chains.iter().enumerate() {
        log::debug!(
            "  Sorted Chain {}: score={}, weight={}, q=[{},{}), r=[{},{}), is_rev={}, kept={}",
            idx,
            chain.score,
            chain.weight,
            chain.query_start,
            chain.query_end,
            chain.ref_start,
            chain.ref_end,
            chain.is_rev,
            chain.kept
        );
    }

    // Filter by minimum weight
    let mut kept_chains: Vec<Chain> = Vec::new();

    for i in 0..chains.len() {
        let chain = &chains[i];

        // Skip if below minimum weight
        if chain.weight < opt.min_chain_weight {
            log::debug!(
                "  filter_chains: Chain {} discarded (weight {} < min_chain_weight {})",
                i,
                chain.weight,
                opt.min_chain_weight
            );
            continue;
        }

        // Check overlap with already-kept chains (matching C++ bwamem.cpp:568-589)
        let mut overlaps = false;
        let mut should_discard = false;
        let mut chain_copy = chain.clone();

        log::debug!(
            "  filter_chains: Processing chain {}: score={}, weight={}, q=[{},{}), r=[{},{}), is_rev={}",
            i,
            chain.score,
            chain.weight,
            chain.query_start,
            chain.query_end,
            chain.ref_start,
            chain.ref_end,
            chain.is_rev
        );

        for (kept_idx, kept_chain) in kept_chains.iter().enumerate() {
            // Check if chains overlap on query
            let qb_max = chain.query_start.max(kept_chain.query_start);
            let qe_min = chain.query_end.min(kept_chain.query_end);

            if qe_min > qb_max {
                // Chains overlap on query
                let overlap = qe_min - qb_max;
                let min_len = (chain.query_end - chain.query_start)
                    .min(kept_chain.query_end - kept_chain.query_start);

                log::debug!(
                    "    Overlap check: Chain {} (q=[{},{}]) vs Kept Chain {} (q=[{},{}]) -> Overlap={}, min_len={}",
                    i,
                    chain.query_start,
                    chain.query_end,
                    kept_idx,
                    kept_chain.query_start,
                    kept_chain.query_end,
                    overlap,
                    min_len
                );

                // Check if overlap is significant
                if overlap >= (min_len as f32 * opt.mask_level) as i32 {
                    overlaps = true;
                    chain_copy.kept = 1; // Shadowed by better chain

                    // C++ bwamem.cpp:580-581: Apply drop_ratio ONLY for overlapping chains
                    let weight_threshold = (kept_chain.weight as f32 * opt.drop_ratio) as i32;
                    let weight_diff = kept_chain.weight - chain.weight;

                    log::debug!(
                        "      Significant overlap: weight_threshold={}, weight_diff={}, min_seed_len={}",
                        weight_threshold,
                        weight_diff,
                        opt.min_seed_len
                    );

                    if chain.weight < weight_threshold && weight_diff >= (opt.min_seed_len << 1) {
                        log::debug!(
                            "      Chain {} dropped due to drop_ratio: weight={} < threshold={} (kept_weight={} * drop_ratio={}) AND diff={} >= (2*min_seed_len={})",
                            i,
                            chain.weight,
                            weight_threshold,
                            kept_chain.weight,
                            opt.drop_ratio,
                            weight_diff,
                            opt.min_seed_len << 1
                        );
                        should_discard = true;
                        break;
                    } else {
                        log::debug!(
                            "      Chain {i} is shadowed by kept chain {kept_idx} but NOT dropped by drop_ratio."
                        );
                    }
                    break;
                } else {
                    log::debug!(
                        "      Overlap is not significant: overlap={} < (min_len={} * mask_level={})",
                        overlap,
                        min_len,
                        opt.mask_level
                    );
                }
            } else {
                log::debug!("    No query overlap for Chain {i} vs Kept Chain {kept_idx}");
            }
        }

        // Skip discarded chains
        if should_discard {
            log::debug!(
                "  filter_chains: Chain {} (score {}) discarded.",
                i,
                chain.score
            );
            continue;
        }

        // Non-overlapping chains are always kept (C++ line 588: kept = large_ovlp? 2 : 3)
        if !overlaps {
            chain_copy.kept = 3; // Primary chain (no overlap)
            log::debug!(
                "  filter_chains: Chain {} (score {}) kept as primary (non-overlapping).",
                i,
                chain.score
            );
        } else {
            chain_copy.kept = 1; // Shadowed
            log::debug!(
                "  filter_chains: Chain {} (score {}) kept as shadowed (overlapping).",
                i,
                chain.score
            );
        }

        kept_chains.push(chain_copy);

        // Limit number of chains to extend
        if kept_chains.len() >= opt.max_chain_extend as usize {
            log::debug!(
                "Reached max_chain_extend={}, stopping chain filtering",
                opt.max_chain_extend
            );
            break;
        }
    }

    log::debug!(
        "Chain filtering: {} input chains -> {} kept chains ({} primary, {} shadowed)",
        chains.len(),
        kept_chains.len(),
        kept_chains.iter().filter(|c| c.kept == 3).count(),
        kept_chains.iter().filter(|c| c.kept == 1).count()
    );

    kept_chains
}

/// Filter chains in SoA batch format.
///
/// Implements C++ mem_chain_flt (bwamem.cpp:506-624).
///
/// Algorithm:
/// 1. Calculate weight for each chain
/// 2. Sort chains by weight (descending)
/// 3. Filter by min_chain_weight
/// 4. Apply drop_ratio: keep chains with weight >= best_weight * drop_ratio
/// 5. Mark overlapping chains as kept=1/2, non-overlapping as kept=3
pub fn filter_chains_batch(
    soa_chain_batch: &mut SoAChainBatch,
    soa_seed_batch: &SoASeedBatch,
    opt: &MemOpt,
    query_lengths: &[i32],
    query_names: &[String],
) {
    let num_reads = soa_chain_batch.read_chain_boundaries.len();

    for read_idx in 0..num_reads {
        let (chain_start_idx, num_chains_for_read) =
            soa_chain_batch.read_chain_boundaries[read_idx];
        let current_read_query_length = query_lengths[read_idx];

        if num_chains_for_read == 0 {
            continue;
        }

        // 1. Calculate weights and frac_rep for all chains of the current read
        for i in 0..num_chains_for_read {
            let global_chain_idx = chain_start_idx + i;
            let (weight, l_rep) =
                calculate_chain_weight_soa(global_chain_idx, soa_chain_batch, soa_seed_batch, opt);
            soa_chain_batch.weight[global_chain_idx] = weight;
            soa_chain_batch.frac_rep[global_chain_idx] = if current_read_query_length > 0 {
                l_rep as f32 / current_read_query_length as f32
            } else {
                0.0
            };
            soa_chain_batch.kept[global_chain_idx] = 0; // Initialize as discarded

            // DEBUG: Detailed chain logging for analysis
            let (seed_start_idx_in_chain, num_seeds_in_chain) =
                soa_chain_batch.chain_seed_boundaries[global_chain_idx];
            let query_name = if read_idx < query_names.len() {
                &query_names[read_idx]
            } else {
                "UNKNOWN"
            };
            log::debug!(
                "FERROUS_CHAIN read={} read_idx={} chain_local_idx={} weight={} n_seeds={} q=[{},{}] r=[{},{}] is_rev={}",
                query_name,
                read_idx,
                i,
                weight,
                num_seeds_in_chain,
                soa_chain_batch.query_start[global_chain_idx],
                soa_chain_batch.query_end[global_chain_idx],
                soa_chain_batch.ref_start[global_chain_idx],
                soa_chain_batch.ref_end[global_chain_idx],
                soa_chain_batch.is_rev[global_chain_idx]
            );

            // Log each seed in the chain
            for j in 0..num_seeds_in_chain {
                let seed_idx_in_soa = soa_chain_batch.seeds_indices[seed_start_idx_in_chain + j];
                log::debug!(
                    "  FERROUS_SEED chain={} seed_idx={} q=[{},{}] r=[{},{}] len={}",
                    i,
                    j,
                    soa_seed_batch.query_pos[seed_idx_in_soa],
                    soa_seed_batch.query_pos[seed_idx_in_soa] + soa_seed_batch.len[seed_idx_in_soa],
                    soa_seed_batch.ref_pos[seed_idx_in_soa],
                    soa_seed_batch.ref_pos[seed_idx_in_soa]
                        + soa_seed_batch.len[seed_idx_in_soa] as u64,
                    soa_seed_batch.len[seed_idx_in_soa]
                );
            }
        }

        // 2. Sort chains for the current read by weight (descending)
        let mut chain_global_indices_for_read: Vec<usize> = (0..num_chains_for_read)
            .map(|i| chain_start_idx + i)
            .collect();

        // CRITICAL FIX: Use stable sort to match main branch and BWA-MEM2 behavior
        chain_global_indices_for_read
            .sort_by(|&a, &b| soa_chain_batch.weight[b].cmp(&soa_chain_batch.weight[a]));

        let mut kept_chain_global_indices: Vec<usize> = Vec::new();

        // DEBUG: Log chain filtering for read_idx=10
        if read_idx == 10 {
            log::debug!(
                "FERROUS_FILTER_START read_idx=10 n_chains={} mask_level={:.2} drop_ratio={:.2}",
                num_chains_for_read,
                opt.mask_level,
                opt.drop_ratio
            );
        }

        // 3. Apply filtering logic
        for (_chain_iter_idx, &global_chain_idx) in chain_global_indices_for_read.iter().enumerate()
        {
            // Check if below minimum weight
            if soa_chain_batch.weight[global_chain_idx] < opt.min_chain_weight {
                continue;
            }

            let mut overlaps = false;
            let mut should_discard = false;

            // Check overlap with already-kept chains
            for &kept_global_chain_idx in kept_chain_global_indices.iter() {
                let qb_max = soa_chain_batch.query_start[global_chain_idx]
                    .max(soa_chain_batch.query_start[kept_global_chain_idx]);
                let qe_min = soa_chain_batch.query_end[global_chain_idx]
                    .min(soa_chain_batch.query_end[kept_global_chain_idx]);

                if qe_min > qb_max {
                    let overlap = qe_min - qb_max;
                    let min_len = (soa_chain_batch.query_end[global_chain_idx]
                        - soa_chain_batch.query_start[global_chain_idx])
                        .min(
                            soa_chain_batch.query_end[kept_global_chain_idx]
                                - soa_chain_batch.query_start[kept_global_chain_idx],
                        );

                    if overlap >= (min_len as f32 * opt.mask_level) as i32 {
                        overlaps = true;

                        let weight_threshold = (soa_chain_batch.weight[kept_global_chain_idx]
                            as f32
                            * opt.drop_ratio) as i32;
                        let weight_diff = soa_chain_batch.weight[kept_global_chain_idx]
                            - soa_chain_batch.weight[global_chain_idx];

                        let will_discard = soa_chain_batch.weight[global_chain_idx]
                            < weight_threshold
                            && weight_diff >= (opt.min_seed_len << 1);

                        // DEBUG: Log overlap decision for read_idx=10
                        if read_idx == 10 {
                            let chain_local_idx = global_chain_idx - chain_start_idx;
                            let kept_local_idx = kept_global_chain_idx - chain_start_idx;
                            log::debug!(
                                "  Chain[{}] vs Chain[{}]: ovlp={} min_l={} w_curr={} w_kept={} thresh={} diff={} DROP={}",
                                chain_local_idx,
                                kept_local_idx,
                                overlap,
                                min_len,
                                soa_chain_batch.weight[global_chain_idx],
                                soa_chain_batch.weight[kept_global_chain_idx],
                                weight_threshold,
                                weight_diff,
                                if will_discard { "YES" } else { "NO" }
                            );
                        }

                        if will_discard {
                            should_discard = true;
                            break;
                        } else {
                            soa_chain_batch.kept[global_chain_idx] = 1;
                        }
                    }
                }
            }

            if should_discard {
                if read_idx == 10 {
                    let chain_local_idx = global_chain_idx - chain_start_idx;
                    log::debug!(
                        "  Chain[{}] DROPPED weight={}",
                        chain_local_idx,
                        soa_chain_batch.weight[global_chain_idx]
                    );
                }
                continue;
            }

            if !overlaps {
                soa_chain_batch.kept[global_chain_idx] = 3; // Primary
            }

            if read_idx == 10 {
                let chain_local_idx = global_chain_idx - chain_start_idx;
                let kept_status = if soa_chain_batch.kept[global_chain_idx] == 3 {
                    "PRIMARY"
                } else {
                    "SHADOWED"
                };
                log::debug!(
                    "  Chain[{}] kept={} ({}) weight={}",
                    chain_local_idx,
                    soa_chain_batch.kept[global_chain_idx],
                    kept_status,
                    soa_chain_batch.weight[global_chain_idx]
                );
            }

            kept_chain_global_indices.push(global_chain_idx);

            if kept_chain_global_indices.len() >= opt.max_chain_extend as usize {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_chains_empty() {
        let opt = MemOpt::default();
        let seeds: Vec<Seed> = Vec::new();
        let mut chains: Vec<Chain> = Vec::new();

        let result = filter_chains(&mut chains, &seeds, &opt, 150);
        assert!(result.is_empty());
    }
}
