//! B-tree based seed chaining algorithm.
//!
//! Implements O(n log n) seed chaining using a B-tree for efficient
//! nearest-neighbor queries. Matches C++ mem_chain_seeds (bwamem.cpp:806-974).

use crate::core::kbtree::KBTree;
use crate::pipelines::linear::mem_opt::MemOpt;
use crate::pipelines::linear::seeding::{Seed, SoASeedBatch};

use super::types::{Chain, SoAChainBatch, MAX_CHAINS_PER_READ, MAX_SEEDS_PER_READ};

/// B-tree based seed chaining - O(n log n) complexity.
///
/// Implements C++ mem_chain_seeds (bwamem.cpp:806-974).
pub fn chain_seeds(seeds: Vec<Seed>, opt: &MemOpt) -> (Vec<Chain>, Vec<Seed>) {
    chain_seeds_with_l_pac(seeds, opt, u64::MAX / 2)
}

/// B-tree based seed chaining with explicit l_pac parameter.
///
/// l_pac is the length of the packed reference (for strand detection).
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
    seeds.sort_by_key(|s| (s.query_pos, s.query_pos + s.len));

    // 2. Initialize KBTree for chain lookup (faster than std BTreeMap)
    let mut tree: KBTree<u64, usize> = KBTree::new();
    let mut chains: Vec<Chain> = Vec::new();

    // 3. Process each seed
    for (seed_idx, seed) in seeds.iter().enumerate() {
        let seed_rpos = seed.ref_pos;

        // Find the chain with the closest reference position <= seed_rpos
        let mut merged = false;

        let (lower, _upper) = tree.interval(&seed_rpos);
        if let Some(&(chain_pos, chain_idx)) = lower {
            let chain = &mut chains[chain_idx];

            // Check strand compatibility (same is_rev flag)
            if chain.is_rev == seed.is_rev {
                if test_and_merge(chain, seed_idx, seed, opt, l_pac) {
                    merged = true;
                    log::trace!(
                        "  Seed {seed_idx} merged into chain {chain_idx} (pos={chain_pos})"
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
                seeds: vec![seed_idx],
                query_start: seed.query_pos,
                query_end: seed.query_pos + seed.len,
                ref_start: seed.ref_pos,
                ref_end: seed.ref_pos + seed.len as u64,
                is_rev: seed.is_rev,
                weight: 0,
                kept: 0,
                frac_rep: 0.0,
                rid: seed.rid,
                pos: seed_rpos,
                last_qbeg: seed.query_pos,
                last_rbeg: seed.ref_pos,
                last_len: seed.len,
            };

            chains.push(new_chain);
            tree.insert(seed_rpos, new_chain_idx);

            log::trace!("  Seed {seed_idx} created new chain {new_chain_idx} (pos={seed_rpos})");
        }
    }

    // 4. Filter out chains below minimum weight
    let filtered_chains: Vec<Chain> = chains
        .into_iter()
        .filter(|c| c.score >= opt.min_chain_weight)
        .collect();

    log::debug!(
        "chain_seeds: {} seeds -> {} chains (B-tree, min_weight={})",
        seeds.len(),
        filtered_chains.len(),
        opt.min_chain_weight
    );

    (filtered_chains, seeds)
}

/// B-tree based seed chaining for a batch of reads, consuming SoASeedBatch.
pub fn chain_seeds_batch(soa_seed_batch: &SoASeedBatch, opt: &MemOpt, l_pac: u64) -> SoAChainBatch {
    let num_reads = soa_seed_batch.read_seed_boundaries.len();
    let mut soa_chain_batch = SoAChainBatch::with_capacity(num_reads * 10, num_reads);

    // Debug: Track reads for logging
    let mut read_names_for_debug: Vec<String> = Vec::new();

    for read_idx in 0..num_reads {
        let (seed_start_idx, num_seeds_for_read) = soa_seed_batch.read_seed_boundaries[read_idx];

        if num_seeds_for_read == 0 {
            soa_chain_batch
                .read_chain_boundaries
                .push((soa_chain_batch.score.len(), 0));
            continue;
        }

        let mut current_read_chains: Vec<Chain> = Vec::new();
        let mut tree: KBTree<u64, usize> = KBTree::new();

        // 1. Create and sort local indices for seeds of the current read
        let mut local_seed_indices: Vec<usize> = (0..num_seeds_for_read).collect();
        local_seed_indices.sort_unstable_by_key(|&local_idx| {
            let global_seed_idx = seed_start_idx + local_idx;
            (
                soa_seed_batch.query_pos[global_seed_idx],
                soa_seed_batch.query_pos[global_seed_idx] + soa_seed_batch.len[global_seed_idx],
            )
        });

        // 2. Process each seed for the current read using sorted local indices
        for &local_seed_idx in local_seed_indices.iter() {
            let global_seed_idx = seed_start_idx + local_seed_idx;
            let seed_rpos = soa_seed_batch.ref_pos[global_seed_idx];
            let mut merged = false;

            let (lower, _upper) = tree.interval(&seed_rpos);
            if let Some(&(_chain_pos, chain_local_idx)) = lower {
                let chain = &mut current_read_chains[chain_local_idx];

                if chain.is_rev == soa_seed_batch.is_rev[global_seed_idx] {
                    if test_and_merge_soa(chain, global_seed_idx, soa_seed_batch, opt, l_pac) {
                        merged = true;
                    }
                }
            }

            if !merged {
                if current_read_chains.len() >= MAX_CHAINS_PER_READ {
                    log::warn!(
                        "chain_seeds_batch: Chain count {} exceeds limit {}, skipping remaining seeds for read {}",
                        current_read_chains.len(),
                        MAX_CHAINS_PER_READ,
                        read_idx
                    );
                    break;
                }

                let new_chain_local_idx = current_read_chains.len();
                let seed_query_pos = soa_seed_batch.query_pos[global_seed_idx];
                let seed_len = soa_seed_batch.len[global_seed_idx];
                let seed_ref_pos = soa_seed_batch.ref_pos[global_seed_idx];
                let seed_is_rev = soa_seed_batch.is_rev[global_seed_idx];
                let seed_rid = soa_seed_batch.rid[global_seed_idx];

                let new_chain = Chain {
                    score: seed_len,
                    seeds: vec![global_seed_idx],
                    query_start: seed_query_pos,
                    query_end: seed_query_pos + seed_len,
                    ref_start: seed_ref_pos,
                    ref_end: seed_ref_pos + seed_len as u64,
                    is_rev: seed_is_rev,
                    weight: 0,
                    kept: 0,
                    frac_rep: 0.0,
                    rid: seed_rid,
                    pos: seed_rpos,
                    last_qbeg: seed_query_pos,
                    last_rbeg: seed_ref_pos,
                    last_len: seed_len,
                };
                current_read_chains.push(new_chain);
                tree.insert(seed_rpos, new_chain_local_idx);
            }
        }

        // Filter chains for the current read
        let filtered_chains_for_read: Vec<Chain> = current_read_chains
            .into_iter()
            .filter(|c| c.score >= opt.min_chain_weight)
            .collect();

        // Populate SoAChainBatch for the current read
        let current_read_chain_start_idx = soa_chain_batch.score.len();

        for chain in filtered_chains_for_read {
            soa_chain_batch.score.push(chain.score);
            soa_chain_batch.query_start.push(chain.query_start);
            soa_chain_batch.query_end.push(chain.query_end);
            soa_chain_batch.ref_start.push(chain.ref_start);
            soa_chain_batch.ref_end.push(chain.ref_end);
            soa_chain_batch.is_rev.push(chain.is_rev);
            soa_chain_batch.weight.push(chain.weight);
            soa_chain_batch.kept.push(chain.kept);
            soa_chain_batch.frac_rep.push(chain.frac_rep);
            soa_chain_batch.rid.push(chain.rid);
            soa_chain_batch.pos.push(chain.pos);
            soa_chain_batch.last_qbeg.push(chain.last_qbeg);
            soa_chain_batch.last_rbeg.push(chain.last_rbeg);
            soa_chain_batch.last_len.push(chain.last_len);

            let current_chain_seed_start_idx = soa_chain_batch.seeds_indices.len();
            soa_chain_batch
                .seeds_indices
                .extend_from_slice(&chain.seeds);
            soa_chain_batch
                .chain_seed_boundaries
                .push((current_chain_seed_start_idx, chain.seeds.len()));
        }

        let num_chains_for_read = soa_chain_batch.score.len() - current_read_chain_start_idx;
        soa_chain_batch
            .read_chain_boundaries
            .push((current_read_chain_start_idx, num_chains_for_read));
    }
    soa_chain_batch
}

/// Try to merge a seed into an existing chain.
///
/// Implements C++ test_and_merge (bwamem.cpp:357-399).
///
/// Returns true if the seed was merged into the chain.
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
    if seed.query_pos >= chain.query_start
        && seed.query_pos + seed.len <= last_qend
        && seed.ref_pos >= chain.ref_start
        && seed.ref_pos + seed.len as u64 <= last_rend
    {
        // Contained seed - do nothing but report success
        return true;
    }

    // C++ lines 370-371: Don't chain if on different strands
    let last_on_forward = chain.last_rbeg < l_pac;
    let first_on_forward = chain.ref_start < l_pac;
    let seed_on_forward = seed.ref_pos < l_pac;
    if (last_on_forward || first_on_forward) && !seed_on_forward {
        return false;
    }

    // C++ lines 373-374: Calculate x and y from LAST SEED's position
    let x = seed.query_pos - chain.last_qbeg;
    let y = seed.ref_pos as i64 - chain.last_rbeg as i64;

    // C++ line 375-377: All conditions for merging
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

        // Update last seed info
        chain.last_qbeg = seed.query_pos;
        chain.last_rbeg = seed.ref_pos;
        chain.last_len = seed.len;

        return true;
    }

    false
}

/// Try to merge a seed into an existing chain (SoA-aware version).
///
/// Implements C++ test_and_merge (bwamem.cpp:357-399).
///
/// Returns true if the seed was merged into the chain.
fn test_and_merge_soa(
    chain: &mut Chain,
    global_seed_idx: usize,
    soa_seed_batch: &SoASeedBatch,
    opt: &MemOpt,
    l_pac: u64,
) -> bool {
    let seed_rid = soa_seed_batch.rid[global_seed_idx];
    let seed_query_pos = soa_seed_batch.query_pos[global_seed_idx];
    let seed_len = soa_seed_batch.len[global_seed_idx];
    let seed_ref_pos = soa_seed_batch.ref_pos[global_seed_idx];

    // C++ bwamem.cpp:359: Different chromosome - request a new chain
    if seed_rid != chain.rid {
        return false;
    }

    // C++ bwamem.cpp:361-363 - get last seed's end positions
    let last_qend = chain.last_qbeg + chain.last_len;
    let last_rend = chain.last_rbeg + chain.last_len as u64;

    // C++ lines 366-368: Check if seed is fully contained in existing chain
    if seed_query_pos >= chain.query_start
        && seed_query_pos + seed_len <= last_qend
        && seed_ref_pos >= chain.ref_start
        && seed_ref_pos + seed_len as u64 <= last_rend
    {
        return true;
    }

    // C++ lines 370-371: Don't chain if on different strands
    let last_on_forward = chain.last_rbeg < l_pac;
    let first_on_forward = chain.ref_start < l_pac;
    let seed_on_forward = seed_ref_pos < l_pac;
    if (last_on_forward || first_on_forward) && !seed_on_forward {
        return false;
    }

    // C++ lines 373-374: Calculate x and y from LAST SEED's position
    let x = seed_query_pos - chain.last_qbeg;
    let y = seed_ref_pos as i64 - chain.last_rbeg as i64;

    // C++ line 375-377: All conditions for merging
    if y >= 0
        && (x as i64 - y) <= opt.w as i64
        && (y - x as i64) <= opt.w as i64
        && (x - chain.last_len) < opt.max_chain_gap
        && (y - chain.last_len as i64) < opt.max_chain_gap as i64
    {
        chain.seeds.push(global_seed_idx);

        chain.query_start = chain.query_start.min(seed_query_pos);
        chain.query_end = chain.query_end.max(seed_query_pos + seed_len);
        chain.ref_start = chain.ref_start.min(seed_ref_pos);
        chain.ref_end = chain.ref_end.max(seed_ref_pos + seed_len as u64);
        chain.score += seed_len;

        chain.last_qbeg = seed_query_pos;
        chain.last_rbeg = seed_ref_pos;
        chain.last_len = seed_len;

        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_seeds_empty() {
        let opt = MemOpt::default();
        let (chains, seeds) = chain_seeds(Vec::new(), &opt);
        assert!(chains.is_empty());
        assert!(seeds.is_empty());
    }

    #[test]
    fn test_chain_seeds_single_seed() {
        let opt = MemOpt::default();
        let seeds = vec![Seed {
            query_pos: 0,
            ref_pos: 1000,
            len: 20,
            interval_size: 1,
            is_rev: false,
            rid: 0,
        }];

        let (chains, _) = chain_seeds(seeds, &opt);
        // Single seed forms a chain if it meets min_chain_weight
        assert!(chains.len() <= 1);
    }
}
