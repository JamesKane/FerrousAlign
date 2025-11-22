use crate::alignment::seeding::Seed;
use crate::mem_opt::MemOpt;
use std::collections::BTreeMap;

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
    pub rid: i32,      // Reference sequence ID (chromosome)
    pos: u64,          // B-tree key: reference position of first seed
    // Last seed info for test_and_merge (matching C++ behavior)
    last_qbeg: i32,    // Last seed's query begin
    last_rbeg: u64,    // Last seed's reference begin
    last_len: i32,     // Last seed's length
}

// ============================================================================
// SEED CHAINING - B-TREE BASED O(n log n) ALGORITHM
// ============================================================================
//
// This implementation matches C++ bwa-mem2's mem_chain_seeds() function.
// Uses a B-tree (BTreeMap) to find the closest chain for each seed in O(log n),
// giving overall O(n log n) complexity instead of O(n²) DP.
//
// Algorithm:
// 1. Sort seeds by (query_pos, ref_pos)
// 2. For each seed:
//    a. Find the chain with closest reference position using BTreeMap
//    b. Try to merge seed into that chain (test_and_merge)
//    c. If merge fails, create a new chain
// 3. Return all chains
// ============================================================================

/// Try to merge a seed into an existing chain
/// Implements C++ test_and_merge (bwamem.cpp:357-399)
///
/// Returns true if the seed was merged into the chain
fn test_and_merge(
    chain: &mut Chain,
    chain_seeds: &mut Vec<usize>,
    seed_idx: usize,
    seed: &Seed,
    opt: &MemOpt,
    l_pac: u64,
) -> bool {
    // C++ bwamem.cpp:361-363 - get last seed's end positions
    let last_qend = chain.last_qbeg + chain.last_len;
    let last_rend = chain.last_rbeg + chain.last_len as u64;

    // C++ lines 366-368: Check if seed is fully contained in existing chain
    // Uses first seed's start and last seed's end
    if seed.query_pos >= chain.query_start
        && seed.query_pos + seed.len <= last_qend
        && seed.ref_pos >= chain.ref_start
        && seed.ref_pos + seed.len as u64 <= last_rend
    {
        // Contained seed - do nothing but report success (seed is "merged" by being ignored)
        return true;
    }

    // C++ lines 370-371: Don't chain if on different strands
    // Seeds on forward strand have rbeg < l_pac, reverse have rbeg >= l_pac
    let last_on_forward = chain.last_rbeg < l_pac;
    let first_on_forward = chain.ref_start < l_pac;
    let seed_on_forward = seed.ref_pos < l_pac;
    if (last_on_forward || first_on_forward) && !seed_on_forward {
        return false;
    }

    // C++ lines 373-374: Calculate x and y from LAST SEED's position
    let x = seed.query_pos - chain.last_qbeg;           // query distance from last seed
    let y = seed.ref_pos as i64 - chain.last_rbeg as i64; // reference distance from last seed

    // C++ line 375-377: All conditions for merging
    // y >= 0: seed is downstream on reference
    // |x - y| <= w: within diagonal band
    // x - last->len < max_chain_gap: query gap from last seed end
    // y - last->len < max_chain_gap: reference gap from last seed end
    if y >= 0
        && (x as i64 - y) <= opt.w as i64
        && (y - x as i64) <= opt.w as i64
        && (x - chain.last_len) < opt.max_chain_gap
        && (y - chain.last_len as i64) < opt.max_chain_gap as i64
    {
        // All constraints passed - merge the seed into the chain
        chain_seeds.push(seed_idx);

        // Update chain bounds
        chain.query_start = chain.query_start.min(seed.query_pos);
        chain.query_end = chain.query_end.max(seed.query_pos + seed.len);
        chain.ref_start = chain.ref_start.min(seed.ref_pos);
        chain.ref_end = chain.ref_end.max(seed.ref_pos + seed.len as u64);
        chain.score += seed.len;

        // Update last seed info (C++ c->seeds[c->n++] = *p)
        chain.last_qbeg = seed.query_pos;
        chain.last_rbeg = seed.ref_pos;
        chain.last_len = seed.len;

        return true;
    }

    false // Request to add a new chain
}

/// B-tree based seed chaining - O(n log n) complexity
/// Implements C++ mem_chain_seeds (bwamem.cpp:806-974)
pub fn chain_seeds(seeds: Vec<Seed>, opt: &MemOpt) -> (Vec<Chain>, Vec<Seed>) {
    chain_seeds_with_l_pac(seeds, opt, u64::MAX / 2)
}

// Safety limits to prevent runaway memory/CPU usage
const MAX_SEEDS_PER_READ: usize = 100_000;
const MAX_CHAINS_PER_READ: usize = 10_000;

/// B-tree based seed chaining with explicit l_pac parameter
/// l_pac is the length of the packed reference (for strand detection)
pub fn chain_seeds_with_l_pac(mut seeds: Vec<Seed>, opt: &MemOpt, l_pac: u64) -> (Vec<Chain>, Vec<Seed>) {
    if seeds.is_empty() {
        return (Vec::new(), seeds);
    }

    // Runaway guard: cap seed count to prevent memory explosion
    if seeds.len() > MAX_SEEDS_PER_READ {
        log::warn!(
            "chain_seeds: Seed count {} exceeds limit {}, truncating to prevent runaway",
            seeds.len(),
            MAX_SEEDS_PER_READ
        );
        seeds.truncate(MAX_SEEDS_PER_READ);
    }

    log::debug!("chain_seeds: Input with {} seeds (B-tree algorithm)", seeds.len());

    // 1. Sort seeds by (query_pos, ref_pos) - same as C++
    seeds.sort_by_key(|s| (s.query_pos, s.ref_pos));

    // 2. Initialize B-tree for chain lookup
    // Key: reference position (chain.pos), Value: index into chains vector
    let mut tree: BTreeMap<u64, usize> = BTreeMap::new();
    let mut chains: Vec<Chain> = Vec::new();
    let mut chain_seeds_vec: Vec<Vec<usize>> = Vec::new(); // Seeds for each chain

    // 3. Process each seed
    for (seed_idx, seed) in seeds.iter().enumerate() {
        let seed_rpos = seed.ref_pos;

        // Find the chain with the closest reference position <= seed_rpos
        // This is equivalent to kb_intervalp finding the "lower" chain
        let mut merged = false;

        // Look for chains with positions close to this seed
        // Use range query to find candidates
        if let Some((&chain_pos, &chain_idx)) = tree.range(..=seed_rpos).next_back() {
            let chain = &mut chains[chain_idx];
            let chain_seeds = &mut chain_seeds_vec[chain_idx];

            // Check strand compatibility (same is_rev flag)
            if chain.is_rev == seed.is_rev {
                // Try to merge
                if test_and_merge(chain, chain_seeds, seed_idx, seed, opt, l_pac) {
                    merged = true;
                    log::trace!(
                        "  Seed {} merged into chain {} (pos={})",
                        seed_idx, chain_idx, chain_pos
                    );
                }
            }
        }

        // If merge failed, create a new chain
        if !merged {
            // Runaway guard: cap chain count
            if chains.len() >= MAX_CHAINS_PER_READ {
                log::warn!(
                    "chain_seeds: Chain count {} exceeds limit {}, skipping remaining seeds",
                    chains.len(),
                    MAX_CHAINS_PER_READ
                );
                break;
            }

            let new_chain_idx = chains.len();

            let new_chain = Chain {
                score: seed.len,
                seeds: Vec::new(), // Will be set from chain_seeds_vec later
                query_start: seed.query_pos,
                query_end: seed.query_pos + seed.len,
                ref_start: seed.ref_pos,
                ref_end: seed.ref_pos + seed.len as u64,
                is_rev: seed.is_rev,
                weight: 0,
                kept: 0,
                frac_rep: 0.0,
                rid: 0, // Will be set by caller if needed
                pos: seed_rpos,
                // Initialize last seed info to the first seed
                last_qbeg: seed.query_pos,
                last_rbeg: seed.ref_pos,
                last_len: seed.len,
            };

            chains.push(new_chain);
            chain_seeds_vec.push(vec![seed_idx]);

            // Insert into B-tree
            // Handle collision by using a unique key (add small offset if needed)
            let mut key = seed_rpos;
            while tree.contains_key(&key) {
                key += 1; // Simple collision handling
            }
            tree.insert(key, new_chain_idx);

            log::trace!(
                "  Seed {} created new chain {} (pos={})",
                seed_idx, new_chain_idx, seed_rpos
            );
        }
    }

    // 4. Finalize chains - copy seed indices into chain structs
    for (chain_idx, chain) in chains.iter_mut().enumerate() {
        chain.seeds = chain_seeds_vec[chain_idx].clone();
    }

    // 5. Filter out chains below minimum weight
    let filtered_chains: Vec<Chain> = chains
        .into_iter()
        .filter(|c| c.score >= opt.min_chain_weight)
        .collect();

    log::debug!(
        "chain_seeds: {} seeds → {} chains (B-tree, min_weight={})",
        seeds.len(),
        filtered_chains.len(),
        opt.min_chain_weight
    );

    (filtered_chains, seeds)
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
    log::debug!("filter_chains: Input with {} chains", chains.len());

    // Calculate weights for all chains
    for (idx, chain) in chains.iter_mut().enumerate() {
        let (weight, l_rep) = calculate_chain_weight(chain, seeds, opt);
        chain.weight = weight;
        // Calculate frac_rep = l_rep / query_length
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
        // IMPORTANT: drop_ratio only applies to OVERLAPPING chains, not all chains
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
                    // Drop if weight < kept_weight * drop_ratio AND difference >= 2 * min_seed_len
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
                            "      Chain {} is shadowed by kept chain {} but NOT dropped by drop_ratio.",
                            i,
                            kept_idx
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
                log::debug!(
                    "    No query overlap for Chain {} vs Kept Chain {}",
                    i,
                    kept_idx
                );
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
