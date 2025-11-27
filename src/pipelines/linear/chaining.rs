use super::mem_opt::MemOpt;
use super::seeding::Seed;
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
    #[allow(dead_code)] // B-tree key used internally
    pos: u64, // B-tree key: reference position of first seed
    // Last seed info for test_and_merge (matching C++ behavior)
    last_qbeg: i32, // Last seed's query begin
    last_rbeg: u64, // Last seed's reference begin
    last_len: i32,  // Last seed's length
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
    seed_idx: usize,
    seed: &Seed,
    opt: &MemOpt,
    l_pac: u64,
) -> bool {
    // C++ bwamem.cpp:359: Different chromosome - request a new chain
    if seed.rid != chain.rid {
        return false;
    }

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
    let x = seed.query_pos - chain.last_qbeg; // query distance from last seed
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
        chain.seeds.push(seed_idx);

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
pub fn chain_seeds_with_l_pac(
    mut seeds: Vec<Seed>,
    opt: &MemOpt,
    l_pac: u64,
) -> (Vec<Chain>, Vec<Seed>) {
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

    log::debug!(
        "chain_seeds: Input with {} seeds (B-tree algorithm)",
        seeds.len()
    );

    // 1. Sort seeds by (query_pos, query_end) - CRITICAL for overlapping seed handling!
    // BWA-MEM2 sorts SMEMs by (query_start, query_end), which ensures that when
    // multiple seeds start at the same position (e.g., len=117 and len=130 both at pos 18),
    // the SHORTER seed is processed first. This allows the LONGER seed to be added to
    // the chain via test_and_merge (since the longer seed is NOT contained).
    // If we sorted by ref_pos instead, the order would be random, and if the longer
    // seed is processed first, the shorter seed would be marked as "contained" and dropped,
    // resulting in chains with only 1 seed instead of multiple overlapping seeds.
    seeds.sort_by_key(|s| (s.query_pos, s.query_pos + s.len));

    // 2. Initialize B-tree for chain lookup
    // Key: reference position (chain.pos), Value: index into chains vector
    let mut tree: BTreeMap<u64, usize> = BTreeMap::new();
    let mut chains: Vec<Chain> = Vec::new();

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

            // Check strand compatibility (same is_rev flag)
            if chain.is_rev == seed.is_rev {
                // Try to merge (uses chain.seeds directly)
                if test_and_merge(chain, seed_idx, seed, opt, l_pac) {
                    merged = true;
                    log::trace!(
                        "  Seed {} merged into chain {} (pos={})",
                        seed_idx,
                        chain_idx,
                        chain_pos
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
                seeds: vec![seed_idx], // Initialize with first seed
                query_start: seed.query_pos,
                query_end: seed.query_pos + seed.len,
                ref_start: seed.ref_pos,
                ref_end: seed.ref_pos + seed.len as u64,
                is_rev: seed.is_rev,
                weight: 0,
                kept: 0,
                frac_rep: 0.0,
                rid: seed.rid, // Chromosome ID from seed
                pos: seed_rpos,
                // Initialize last seed info to the first seed
                last_qbeg: seed.query_pos,
                last_rbeg: seed.ref_pos,
                last_len: seed.len,
            };

            chains.push(new_chain);

            // Insert into B-tree
            // Handle collision by using a unique key (add small offset if needed)
            let mut key = seed_rpos;
            while tree.contains_key(&key) {
                key += 1; // Simple collision handling
            }
            tree.insert(key, new_chain_idx);

            log::trace!(
                "  Seed {} created new chain {} (pos={})",
                seed_idx,
                new_chain_idx,
                seed_rpos
            );
        }
    }

    // 4. Filter out chains below minimum weight
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

// ============================================================================
// UNIT TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a seed for testing
    fn make_seed(query_pos: i32, ref_pos: u64, len: i32, is_rev: bool, rid: i32) -> Seed {
        Seed {
            query_pos,
            ref_pos,
            len,
            is_rev,
            rid,
            interval_size: 1, // Low occurrence (not repetitive)
        }
    }

    /// Helper to create default MemOpt for testing
    fn default_test_opt() -> MemOpt {
        let mut opt = MemOpt::default();
        opt.w = 100; // Band width
        opt.max_chain_gap = 10000; // Max gap in chain
        opt.min_chain_weight = 0; // Keep all chains for testing
        opt.min_seed_len = 19;
        opt.drop_ratio = 0.5;
        opt.mask_level = 0.5;
        opt.max_chain_extend = 50;
        opt
    }

    // ------------------------------------------------------------------------
    // chain_seeds() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_chain_seeds_empty() {
        let opt = default_test_opt();
        let seeds: Vec<Seed> = vec![];
        let (chains, _) = chain_seeds(seeds, &opt);
        assert!(chains.is_empty());
    }

    #[test]
    fn test_chain_seeds_single_seed() {
        let opt = default_test_opt();
        let seeds = vec![make_seed(0, 1000, 20, false, 0)];
        let (chains, _) = chain_seeds(seeds, &opt);

        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].seeds.len(), 1);
        assert_eq!(chains[0].score, 20);
        assert_eq!(chains[0].query_start, 0);
        assert_eq!(chains[0].query_end, 20);
    }

    #[test]
    fn test_chain_seeds_two_compatible_seeds() {
        // Two seeds that should chain together (close on both query and ref)
        let opt = default_test_opt();
        let seeds = vec![
            make_seed(0, 1000, 20, false, 0),  // First seed
            make_seed(25, 1025, 20, false, 0), // Second seed: 5bp gap on both
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        // Should merge into one chain
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].seeds.len(), 2);
        assert_eq!(chains[0].score, 40); // 20 + 20
        assert_eq!(chains[0].query_start, 0);
        assert_eq!(chains[0].query_end, 45);
    }

    #[test]
    fn test_chain_seeds_incompatible_different_strands() {
        // Seeds on different strands should not chain
        let opt = default_test_opt();
        let seeds = vec![
            make_seed(0, 1000, 20, false, 0), // Forward
            make_seed(25, 1025, 20, true, 0), // Reverse
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        // Should create two separate chains
        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_chain_seeds_incompatible_large_gap() {
        // Seeds with large gap should not chain
        let mut opt = default_test_opt();
        opt.max_chain_gap = 100; // Small gap limit

        let seeds = vec![
            make_seed(0, 1000, 20, false, 0),
            make_seed(200, 1200, 20, false, 0), // 180bp gap > 100
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        // Should create two separate chains
        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_chain_seeds_different_chromosomes() {
        // Seeds on different chromosomes should not chain
        let opt = default_test_opt();
        let seeds = vec![
            make_seed(0, 1000, 20, false, 0),  // chr0
            make_seed(25, 1025, 20, false, 1), // chr1
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        // Should create two separate chains
        assert_eq!(chains.len(), 2);
        assert_eq!(chains[0].rid, 0);
        assert_eq!(chains[1].rid, 1);
    }

    #[test]
    fn test_chain_seeds_contained_seed_ignored() {
        // A seed fully contained in existing chain should be merged (ignored)
        let opt = default_test_opt();
        let seeds = vec![
            make_seed(0, 1000, 50, false, 0),  // Large seed
            make_seed(10, 1010, 10, false, 0), // Small seed contained in first
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        // Should have one chain (second seed contained)
        assert_eq!(chains.len(), 1);
        // The contained seed counts as merged but doesn't add to score
        // Actually it returns true but doesn't push the seed_idx
    }

    #[test]
    fn test_chain_seeds_multiple_chains() {
        // Multiple independent chains
        let opt = default_test_opt();
        let seeds = vec![
            // Chain 1: forward strand, chr0
            make_seed(0, 1000, 20, false, 0),
            make_seed(25, 1025, 20, false, 0),
            // Chain 2: reverse strand, chr0
            make_seed(50, 2000, 20, true, 0),
            make_seed(75, 2025, 20, true, 0),
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_chain_seeds_min_weight_filter() {
        let mut opt = default_test_opt();
        opt.min_chain_weight = 30; // Require score >= 30

        let seeds = vec![
            make_seed(0, 1000, 20, false, 0),   // Single seed, score=20 < 30
            make_seed(100, 5000, 40, false, 0), // Single seed, score=40 >= 30
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        // Only the second chain should pass filter
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].score, 40);
    }

    // ------------------------------------------------------------------------
    // filter_chains() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_filter_chains_empty() {
        let opt = default_test_opt();
        let seeds: Vec<Seed> = vec![];
        let mut chains: Vec<Chain> = vec![];

        let filtered = filter_chains(&mut chains, &seeds, &opt, 100);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_chains_single_chain() {
        let opt = default_test_opt();
        let seeds = vec![make_seed(0, 1000, 50, false, 0)];

        let mut chains = vec![Chain {
            score: 50,
            seeds: vec![0],
            query_start: 0,
            query_end: 50,
            ref_start: 1000,
            ref_end: 1050,
            is_rev: false,
            weight: 0,
            kept: 0,
            frac_rep: 0.0,
            rid: 0,
            pos: 1000,
            last_qbeg: 0,
            last_rbeg: 1000,
            last_len: 50,
        }];

        let filtered = filter_chains(&mut chains, &seeds, &opt, 100);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].kept, 3); // Primary (non-overlapping)
    }

    #[test]
    fn test_filter_chains_overlapping_drop_ratio() {
        let mut opt = default_test_opt();
        opt.drop_ratio = 0.5;
        opt.mask_level = 0.5;
        opt.min_seed_len = 10;

        let seeds = vec![
            make_seed(0, 1000, 80, false, 0),  // Seed for chain 1
            make_seed(10, 2000, 30, false, 0), // Seed for chain 2 (overlaps on query)
        ];

        let mut chains = vec![
            Chain {
                score: 80,
                seeds: vec![0],
                query_start: 0,
                query_end: 80,
                ref_start: 1000,
                ref_end: 1080,
                is_rev: false,
                weight: 0,
                kept: 0,
                frac_rep: 0.0,
                rid: 0,
                pos: 1000,
                last_qbeg: 0,
                last_rbeg: 1000,
                last_len: 80,
            },
            Chain {
                score: 30,
                seeds: vec![1],
                query_start: 10,
                query_end: 40,
                ref_start: 2000,
                ref_end: 2030,
                is_rev: false,
                weight: 0,
                kept: 0,
                frac_rep: 0.0,
                rid: 0,
                pos: 2000,
                last_qbeg: 10,
                last_rbeg: 2000,
                last_len: 30,
            },
        ];

        let filtered = filter_chains(&mut chains, &seeds, &opt, 100);

        // First chain should be kept as primary
        // Second chain overlaps and weight is much lower, may be dropped
        assert!(!filtered.is_empty());
        assert_eq!(filtered[0].kept, 3); // Primary
    }

    #[test]
    fn test_filter_chains_non_overlapping_kept() {
        let opt = default_test_opt();

        let seeds = vec![
            make_seed(0, 1000, 30, false, 0),
            make_seed(50, 5000, 30, false, 0), // Non-overlapping on query
        ];

        let mut chains = vec![
            Chain {
                score: 30,
                seeds: vec![0],
                query_start: 0,
                query_end: 30,
                ref_start: 1000,
                ref_end: 1030,
                is_rev: false,
                weight: 0,
                kept: 0,
                frac_rep: 0.0,
                rid: 0,
                pos: 1000,
                last_qbeg: 0,
                last_rbeg: 1000,
                last_len: 30,
            },
            Chain {
                score: 30,
                seeds: vec![1],
                query_start: 50,
                query_end: 80,
                ref_start: 5000,
                ref_end: 5030,
                is_rev: false,
                weight: 0,
                kept: 0,
                frac_rep: 0.0,
                rid: 0,
                pos: 5000,
                last_qbeg: 50,
                last_rbeg: 5000,
                last_len: 30,
            },
        ];

        let filtered = filter_chains(&mut chains, &seeds, &opt, 100);

        // Both chains should be kept (non-overlapping)
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|c| c.kept == 3));
    }

    // ------------------------------------------------------------------------
    // calculate_chain_weight() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_calculate_chain_weight_single_seed() {
        let opt = default_test_opt();
        let seeds = vec![make_seed(0, 1000, 50, false, 0)];

        let chain = Chain {
            score: 50,
            seeds: vec![0],
            query_start: 0,
            query_end: 50,
            ref_start: 1000,
            ref_end: 1050,
            is_rev: false,
            weight: 0,
            kept: 0,
            frac_rep: 0.0,
            rid: 0,
            pos: 1000,
            last_qbeg: 0,
            last_rbeg: 1000,
            last_len: 50,
        };

        let (weight, l_rep) = calculate_chain_weight(&chain, &seeds, &opt);
        assert_eq!(weight, 50);
        assert_eq!(l_rep, 0); // Not repetitive
    }

    #[test]
    fn test_calculate_chain_weight_non_overlapping_seeds() {
        let opt = default_test_opt();
        let seeds = vec![
            make_seed(0, 1000, 20, false, 0),
            make_seed(30, 1030, 20, false, 0), // Gap of 10
        ];

        let chain = Chain {
            score: 40,
            seeds: vec![0, 1],
            query_start: 0,
            query_end: 50,
            ref_start: 1000,
            ref_end: 1050,
            is_rev: false,
            weight: 0,
            kept: 0,
            frac_rep: 0.0,
            rid: 0,
            pos: 1000,
            last_qbeg: 30,
            last_rbeg: 1030,
            last_len: 20,
        };

        let (weight, _) = calculate_chain_weight(&chain, &seeds, &opt);
        assert_eq!(weight, 40); // 20 + 20, no overlap
    }

    #[test]
    fn test_calculate_chain_weight_overlapping_seeds() {
        let opt = default_test_opt();
        let seeds = vec![
            make_seed(0, 1000, 30, false, 0),
            make_seed(20, 1020, 30, false, 0), // Overlap of 10
        ];

        let chain = Chain {
            score: 60,
            seeds: vec![0, 1],
            query_start: 0,
            query_end: 50,
            ref_start: 1000,
            ref_end: 1050,
            is_rev: false,
            weight: 0,
            kept: 0,
            frac_rep: 0.0,
            rid: 0,
            pos: 1000,
            last_qbeg: 20,
            last_rbeg: 1020,
            last_len: 30,
        };

        let (weight, _) = calculate_chain_weight(&chain, &seeds, &opt);
        // Query coverage: 30 + (50-30) = 50
        // Ref coverage: 30 + (1050-1030) = 50
        assert_eq!(weight, 50);
    }

    #[test]
    fn test_calculate_chain_weight_repetitive_seeds() {
        let mut opt = default_test_opt();
        opt.max_occ = 100;

        // Create a repetitive seed (interval_size > max_occ)
        let mut seed = make_seed(0, 1000, 30, false, 0);
        seed.interval_size = 500; // > max_occ
        let seeds = vec![seed];

        let chain = Chain {
            score: 30,
            seeds: vec![0],
            query_start: 0,
            query_end: 30,
            ref_start: 1000,
            ref_end: 1030,
            is_rev: false,
            weight: 0,
            kept: 0,
            frac_rep: 0.0,
            rid: 0,
            pos: 1000,
            last_qbeg: 0,
            last_rbeg: 1000,
            last_len: 30,
        };

        let (weight, l_rep) = calculate_chain_weight(&chain, &seeds, &opt);
        assert_eq!(weight, 30);
        assert_eq!(l_rep, 30); // All 30bp are repetitive
    }

    // ------------------------------------------------------------------------
    // cal_max_gap() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_cal_max_gap_short_query() {
        let opt = default_test_opt();
        let gap = cal_max_gap(&opt, 50);
        assert!(gap >= 1);
        assert!(gap <= opt.w << 1);
    }

    #[test]
    fn test_cal_max_gap_long_query() {
        let opt = default_test_opt();
        let gap = cal_max_gap(&opt, 1000);
        // For long queries, should be capped at 2*w
        assert_eq!(gap, opt.w << 1);
    }

    // ------------------------------------------------------------------------
    // Runaway guard tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_chain_seeds_runaway_guard_seeds() {
        let opt = default_test_opt();

        // Create more seeds than MAX_SEEDS_PER_READ
        let mut seeds = Vec::new();
        for i in 0..MAX_SEEDS_PER_READ + 100 {
            seeds.push(make_seed(i as i32, i as u64 * 1000, 20, false, 0));
        }

        // Should truncate and not panic
        let (chains, returned_seeds) = chain_seeds(seeds, &opt);

        // Seeds should be truncated
        assert!(returned_seeds.len() <= MAX_SEEDS_PER_READ);
        // Chains should be created without panic
        assert!(!chains.is_empty());
    }

    // ------------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------------

    #[test]
    fn test_chain_seeds_diagonal_band() {
        // Test diagonal band constraint (|x - y| <= w)
        let mut opt = default_test_opt();
        opt.w = 10; // Small band

        let seeds = vec![
            make_seed(0, 1000, 20, false, 0),
            // Second seed: query advances 30, ref advances 50 -> |30-50| = 20 > w=10
            make_seed(30, 1050, 20, false, 0),
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        // Seeds should NOT chain due to band violation
        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_chain_seeds_seed_upstream_on_ref() {
        // Test that seed with y < 0 (upstream on ref) doesn't chain
        let opt = default_test_opt();

        let seeds = vec![
            make_seed(0, 1000, 20, false, 0),
            make_seed(25, 900, 20, false, 0), // Upstream on ref (y < 0)
        ];
        let (chains, _) = chain_seeds(seeds, &opt);

        // Should create two chains (can't chain backwards on ref)
        assert_eq!(chains.len(), 2);
    }
}
