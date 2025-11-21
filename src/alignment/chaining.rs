use crate::alignment::seeding::Seed;
use crate::mem_opt::MemOpt;

#[derive(Debug, Clone)]
pub struct Chain {
    pub score: i32,
    pub seeds: Vec<usize>, // Indices of seeds in the original seeds vector
    pub query_start: i32,
    pub query_end: i32,
    pub ref_start: u64,
    pub ref_end: u64,
    pub is_rev: bool,
    pub weight: i32,   // Chain weight (seed coverage), calculated by mem_chain_weight
    pub kept: i32,     // Chain status: 0=discarded, 1=shadowed, 2=partial_overlap, 3=primary
    pub frac_rep: f32, // Fraction of repetitive seeds in this chain
}

// ============================================================================
// SEED CHAINING
// ============================================================================
//
// This section contains the seed chaining algorithm:
// - Dynamic programming-based chaining
// - Chain scoring and filtering
// - Extension of chains into alignments
// ============================================================================

pub fn chain_seeds(mut seeds: Vec<Seed>, opt: &MemOpt) -> (Vec<Chain>, Vec<Seed>) {
    if seeds.is_empty() {
        return (Vec::new(), seeds);
    }

    // 1. Sort seeds: by query_pos, then by ref_pos
    seeds.sort_by_key(|s| (s.query_pos, s.ref_pos));

    let num_seeds = seeds.len();
    let mut dp = vec![0; num_seeds]; // dp[i] stores the max score of a chain ending at seeds[i]
    let mut prev_seed_idx = vec![None; num_seeds]; // To reconstruct the chain

    // Use max_chain_gap from options for gap penalty calculation
    let max_gap = opt.max_chain_gap;

    // 2. Dynamic Programming
    for i in 0..num_seeds {
        dp[i] = seeds[i].len; // Initialize with the seed's own length as score
        for j in 0..i {
            // Check for compatibility: seed[j] must end before seed[i] starts in both query and reference
            // And they must be on the same strand
            if seeds[j].is_rev == seeds[i].is_rev
                && seeds[j].query_pos + seeds[j].len < seeds[i].query_pos
                && seeds[j].ref_pos + (seeds[j].len as u64) < seeds[i].ref_pos
            {
                // Calculate distances (matching C++ test_and_merge logic)
                let x = seeds[i].query_pos - seeds[j].query_pos; // query distance
                let y = seeds[i].ref_pos as i32 - seeds[j].ref_pos as i32; // reference distance
                let q_gap = x - seeds[j].len; // gap after seed[j] ends
                let r_gap = y - seeds[j].len as i32; // gap after seed[j] ends

                // C++ constraint 1: y >= 0 (new seed downstream on reference)
                if y < 0 {
                    continue;
                }

                // C++ constraint 2: Diagonal band width check (|x - y| <= w)
                // This ensures seeds stay within a diagonal band
                let diagonal_offset = x - y;
                if diagonal_offset.abs() > opt.w {
                    continue; // Seeds too far from diagonal
                }

                // C++ constraint 3 & 4: max_chain_gap check
                if q_gap > max_gap || r_gap > max_gap {
                    continue; // Gap too large
                }

                // Simple gap penalty (average of query and reference gaps)
                let current_gap_penalty = (q_gap + r_gap.abs()) / 2;

                let potential_score = dp[j] + seeds[i].len - current_gap_penalty;

                if potential_score > dp[i] {
                    dp[i] = potential_score;
                    prev_seed_idx[i] = Some(j);
                }
            }
        }
    }

    // 3. Multi-chain extraction via iterative peak finding
    // Algorithm:
    // 1. Find highest-scoring peak in DP array
    // 2. Backtrack to reconstruct chain
    // 3. Mark seeds in chain as "used"
    // 4. Repeat until no more peaks above min_chain_weight
    // This matches bwa-mem2's approach to multi-chain generation

    let mut chains = Vec::new();
    let mut used_seeds = vec![false; num_seeds]; // Track which seeds are already in chains

    // Iteratively extract chains by finding peaks
    loop {
        // Find the highest unused peak
        let mut best_chain_score = opt.min_chain_weight; // Only consider chains above minimum
        let mut best_chain_end_idx: Option<usize> = None;

        for i in 0..num_seeds {
            if !used_seeds[i] && dp[i] >= best_chain_score {
                best_chain_score = dp[i];
                best_chain_end_idx = Some(i);
            }
        }

        // Stop if no more chains above threshold
        if best_chain_end_idx.is_none() {
            break;
        }

        // Backtrack to reconstruct this chain
        let mut current_idx = best_chain_end_idx.unwrap();
        let mut chain_seeds_indices = Vec::new();
        let mut current_seed = &seeds[current_idx];

        let mut query_start = current_seed.query_pos;
        let mut query_end = current_seed.query_pos + current_seed.len;
        let mut ref_start = current_seed.ref_pos;
        let mut ref_end = current_seed.ref_pos + current_seed.len as u64;
        let is_rev = current_seed.is_rev;

        // Backtrack through the chain
        loop {
            chain_seeds_indices.push(current_idx);
            used_seeds[current_idx] = true; // Mark seed as used

            // Get previous seed in chain
            if let Some(prev_idx) = prev_seed_idx[current_idx] {
                current_idx = prev_idx;
                current_seed = &seeds[current_idx];

                // Update chain bounds
                query_start = query_start.min(current_seed.query_pos);
                query_end = query_end.max(current_seed.query_pos + current_seed.len);
                ref_start = ref_start.min(current_seed.ref_pos);
                ref_end = ref_end.max(current_seed.ref_pos + current_seed.len as u64);
            } else {
                break; // Reached start of chain
            }
        }

        chain_seeds_indices.reverse(); // Order from start to end

        log::debug!(
            "Chain extraction: chain_idx={}, num_seeds={}, score={}, query=[{}, {}), ref=[{}, {}), is_rev={}",
            chains.len(),
            chain_seeds_indices.len(),
            best_chain_score,
            query_start,
            query_end,
            ref_start,
            ref_end,
            is_rev
        );

        chains.push(Chain {
            score: best_chain_score,
            seeds: chain_seeds_indices,
            query_start,
            query_end,
            ref_start,
            ref_end,
            is_rev,
            weight: 0,     // Will be calculated by filter_chains()
            kept: 0,       // Will be set by filter_chains()
            frac_rep: 0.0, // Initial placeholder
        });

        // Safety limit: stop after extracting a reasonable number of chains
        // This prevents pathological cases from consuming too much memory
        if chains.len() >= 100 {
            log::debug!(
                "Extracted maximum of 100 chains from {} seeds, stopping",
                num_seeds
            );
            break;
        }
    }

    log::debug!(
        "Chain extraction: {} seeds → {} chains (min_weight={})",
        num_seeds,
        chains.len(),
        opt.min_chain_weight
    );

    (chains, seeds)
}

/// Filter chains using drop_ratio and score thresholds
/// Implements C++ mem_chain_flt (bwamem.cpp:506-624)
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

    // Calculate weights for all chains
    for chain in chains.iter_mut() {
        let (weight, l_rep) = calculate_chain_weight(chain, seeds, opt);
        chain.weight = weight;
        // Calculate frac_rep = l_rep / query_length
        chain.frac_rep = if query_length > 0 {
            l_rep as f32 / query_length as f32
        } else {
            0.0
        };
        chain.kept = 0; // Initially mark as discarded
    }

    // Sort chains by weight (descending)
    chains.sort_by(|a, b| b.weight.cmp(&a.weight));

    // Filter by minimum weight
    let mut kept_chains: Vec<Chain> = Vec::new();

    for i in 0..chains.len() {
        let chain = &chains[i];

        // Skip if below minimum weight
        if chain.weight < opt.min_chain_weight {
            continue;
        }

        // Check overlap with already-kept chains (matching C++ bwamem.cpp:568-589)
        // IMPORTANT: drop_ratio only applies to OVERLAPPING chains, not all chains
        let mut overlaps = false;
        let mut should_discard = false;
        let mut chain_copy = chain.clone();

        for kept_chain in &kept_chains {
            // Check if chains overlap on query
            let qb_max = chain.query_start.max(kept_chain.query_start);
            let qe_min = chain.query_end.min(kept_chain.query_end);

            if qe_min > qb_max {
                // Chains overlap on query
                let overlap = qe_min - qb_max;
                let min_len = (chain.query_end - chain.query_start)
                    .min(kept_chain.query_end - kept_chain.query_start);

                // Check if overlap is significant
                if overlap >= (min_len as f32 * opt.mask_level) as i32 {
                    overlaps = true;
                    chain_copy.kept = 1; // Shadowed by better chain

                    // C++ bwamem.cpp:580-581: Apply drop_ratio ONLY for overlapping chains
                    // Drop if weight < kept_weight * drop_ratio AND difference >= 2 * min_seed_len
                    let weight_threshold = (kept_chain.weight as f32 * opt.drop_ratio) as i32;
                    let weight_diff = kept_chain.weight - chain.weight;

                    if chain.weight < weight_threshold && weight_diff >= (opt.min_seed_len << 1) {
                        log::debug!(
                            "Chain {} dropped: overlaps with kept chain, weight={} < threshold={} (kept_weight={} * drop_ratio={})",
                            i,
                            chain.weight,
                            weight_threshold,
                            kept_chain.weight,
                            opt.drop_ratio
                        );
                        should_discard = true;
                        break;
                    }
                    break;
                }
            }
        }

        // Skip discarded chains
        if should_discard {
            continue;
        }

        // Non-overlapping chains are always kept (C++ line 588: kept = large_ovlp? 2 : 3)
        if !overlaps {
            chain_copy.kept = 3; // Primary chain (no overlap)
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
        "Chain filtering: {} input chains → {} kept chains ({} primary, {} shadowed)",
        chains.len(),
        kept_chains.len(),
        kept_chains.iter().filter(|c| c.kept == 3).count(),
        kept_chains.iter().filter(|c| c.kept == 1).count()
    );

    kept_chains
}

// ----------------------------------------------------------------------------
// Chain Scoring and Filtering
// ----------------------------------------------------------------------------

/// Calculate chain weight based on seed coverage
/// Implements C++ mem_chain_weight (bwamem.cpp:429-448)
///
/// Weight = minimum of query coverage and reference coverage
/// This accounts for non-overlapping seed lengths in the chain
pub fn calculate_chain_weight(chain: &Chain, seeds: &[Seed], opt: &MemOpt) -> (i32, i32) {
    if chain.seeds.is_empty() {
        return (0, 0);
    }

    let mut query_cov = 0;
    let mut last_qe = -1i32;
    let mut l_rep = 0; // Length of repetitive seeds

    for &seed_idx in &chain.seeds {
        let seed = &seeds[seed_idx];
        let qb = seed.query_pos;
        let qe = seed.query_pos + seed.len;

        if qb > last_qe {
            query_cov += seed.len;
        } else if qe > last_qe {
            query_cov += qe - last_qe;
        }
        last_qe = last_qe.max(qe);

        // Check for repetitive seeds: if interval_size > max_occ
        // This threshold needs to be dynamically adjusted based on context if we want to mimic BWA-MEM2's exact filtering.
        // For now, using opt.max_occ as the threshold for 'repetitive'.
        if seed.interval_size > opt.max_occ as u64 {
            // Assuming interval_size is the occurrence count of the seed
            l_rep += seed.len;
        }
    }

    let mut ref_cov = 0;
    let mut last_re = 0u64;

    for &seed_idx in &chain.seeds {
        let seed = &seeds[seed_idx];
        let rb = seed.ref_pos;
        let re = seed.ref_pos + seed.len as u64;

        if rb > last_re {
            ref_cov += seed.len;
        } else if re > last_re {
            ref_cov += (re - last_re) as i32;
        }

        last_re = last_re.max(re);
    }

    (query_cov.min(ref_cov), l_rep)
}

/// Calculate maximum gap size for a given query length
/// Matches C++ bwamem.cpp:66 cal_max_gap()
#[inline]
pub fn cal_max_gap(opt: &MemOpt, qlen: i32) -> i32 {
    let l_del = ((qlen * opt.a as i32 - opt.o_del as i32) as f64 / opt.e_del as f64 + 1.0) as i32;
    let l_ins = ((qlen * opt.a as i32 - opt.o_ins as i32) as f64 / opt.e_ins as f64 + 1.0) as i32;

    let l = if l_del > l_ins { l_del } else { l_ins };
    let l = if l > 1 { l } else { 1 };

    if l < (opt.w << 1) { l } else { opt.w << 1 }
}
