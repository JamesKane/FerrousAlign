use super::mem_opt::MemOpt;
use super::seeding::Seed;
use crate::core::kbtree::KBTree;
use crate::pipelines::linear::seeding::SoASeedBatch;

// Safety limits to prevent runaway memory/CPU usage
const MAX_SEEDS_PER_READ: usize = 100_000;
const MAX_CHAINS_PER_READ: usize = 10_000;

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
    pub(crate) pos: u64, // B-tree key: reference position of first seed
    // Last seed info for test_and_merge (matching C++ behavior)
    pub(crate) last_qbeg: i32, // Last seed's query begin
    pub(crate) last_rbeg: u64, // Last seed's reference begin
    pub(crate) last_len: i32,  // Last seed's length
}

// Define a struct to represent a batch of chains in SoA format
#[derive(Debug, Clone, Default)]
pub struct SoAChainBatch {
    pub score: Vec<i32>,
    pub query_start: Vec<i32>,
    pub query_end: Vec<i32>,
    pub ref_start: Vec<u64>,
    pub ref_end: Vec<u64>,
    pub is_rev: Vec<bool>,
    pub weight: Vec<i32>,
    pub kept: Vec<i32>,
    pub frac_rep: Vec<f32>,
    pub rid: Vec<i32>,
    pub pos: Vec<u64>,
    pub last_qbeg: Vec<i32>,
    pub last_rbeg: Vec<u64>,
    pub last_len: Vec<i32>,
    // Boundaries for chains belonging to each read in the batch
    pub read_chain_boundaries: Vec<(usize, usize)>, // (start_idx, count)
    // To store indices of seeds in the SoASeedBatch
    pub chain_seed_boundaries: Vec<(usize, usize)>, // (start_idx_in_soa_seed_batch, count)
    pub seeds_indices: Vec<usize>, // Flattened indices into the global SoASeedBatch
}

impl SoAChainBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize, num_reads: usize) -> Self {
        Self {
            score: Vec::with_capacity(capacity),
            query_start: Vec::with_capacity(capacity),
            query_end: Vec::with_capacity(capacity),
            ref_start: Vec::with_capacity(capacity),
            ref_end: Vec::with_capacity(capacity),
            is_rev: Vec::with_capacity(capacity),
            weight: Vec::with_capacity(capacity),
            kept: Vec::with_capacity(capacity),
            frac_rep: Vec::with_capacity(capacity),
            rid: Vec::with_capacity(capacity),
            pos: Vec::with_capacity(capacity),
            last_qbeg: Vec::with_capacity(capacity),
            last_rbeg: Vec::with_capacity(capacity),
            last_len: Vec::with_capacity(capacity),
            read_chain_boundaries: Vec::with_capacity(num_reads),
            chain_seed_boundaries: Vec::with_capacity(capacity),
            seeds_indices: Vec::new(), // Starts empty, will grow with chain_seed_boundaries
        }
    }

    pub fn clear(&mut self) {
        self.score.clear();
        self.query_start.clear();
        self.query_end.clear();
        self.ref_start.clear();
        self.ref_end.clear();
        self.is_rev.clear();
        self.weight.clear();
        self.kept.clear();
        self.frac_rep.clear();
        self.rid.clear();
        self.pos.clear();
        self.last_qbeg.clear();
        self.last_rbeg.clear();
        self.last_len.clear();
        self.read_chain_boundaries.clear();
        self.chain_seed_boundaries.clear();
        self.seeds_indices.clear();
    }
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

/// Try to merge a seed into an existing chain (SoA-aware version)
/// Implements C++ test_and_merge (bwamem.cpp:357-399)
///
/// Returns true if the seed was merged into the chain
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
    // Uses first seed's start and last seed's end
    if seed_query_pos >= chain.query_start
        && seed_query_pos + seed_len <= last_qend
        && seed_ref_pos >= chain.ref_start
        && seed_ref_pos + seed_len as u64 <= last_rend
    {
        // Contained seed - do nothing but report success (seed is "merged" by being ignored)
        return true;
    }

    // C++ lines 370-371: Don't chain if on different strands
    // Seeds on forward strand have rbeg < l_pac, reverse have rbeg >= l_pac
    let last_on_forward = chain.last_rbeg < l_pac;
    let first_on_forward = chain.ref_start < l_pac;
    let seed_on_forward = seed_ref_pos < l_pac;
    if (last_on_forward || first_on_forward) && !seed_on_forward {
        return false;
    }

    // C++ lines 373-374: Calculate x and y from LAST SEED's position
    let x = seed_query_pos - chain.last_qbeg; // query distance from last seed
    let y = seed_ref_pos as i64 - chain.last_rbeg as i64; // reference distance from last seed

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
        chain.seeds.push(global_seed_idx);

        // Update chain bounds
        chain.query_start = chain.query_start.min(seed_query_pos);
        chain.query_end = chain.query_end.max(seed_query_pos + seed_len);
        chain.ref_start = chain.ref_start.min(seed_ref_pos);
        chain.ref_end = chain.ref_end.max(seed_ref_pos + seed_len as u64);
        chain.score += seed_len;

        // Update last seed info (C++ c->seeds[c->n++] = *p)
        chain.last_qbeg = seed_query_pos;
        chain.last_rbeg = seed_ref_pos;
        chain.last_len = seed_len;

        return true;
    }

    false // Request to add a new chain
}

/// B-tree based seed chaining - O(n log n) complexity
/// Implements C++ mem_chain_seeds (bwamem.cpp:806-974)
pub fn chain_seeds(seeds: Vec<Seed>, opt: &MemOpt) -> (Vec<Chain>, Vec<Seed>) {
    chain_seeds_with_l_pac(seeds, opt, u64::MAX / 2)
}

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

    // 2. Initialize KBTree for chain lookup (faster than std BTreeMap)
    // Key: reference position (chain.pos), Value: index into chains vector
    let mut tree: KBTree<u64, usize> = KBTree::new();
    let mut chains: Vec<Chain> = Vec::new();

    // 3. Process each seed
    for (seed_idx, seed) in seeds.iter().enumerate() {
        let seed_rpos = seed.ref_pos;

        // Find the chain with the closest reference position <= seed_rpos
        // This is equivalent to kb_intervalp finding the "lower" chain
        let mut merged = false;

        // Look for chains with positions close to this seed
        // Use interval query to find the closest chain
        let (lower, _upper) = tree.interval(&seed_rpos);
        if let Some(&(chain_pos, chain_idx)) = lower {
            let chain = &mut chains[chain_idx];

            // Check strand compatibility (same is_rev flag)
            if chain.is_rev == seed.is_rev {
                // Try to merge (uses chain.seeds directly)
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

            // Insert into KBTree
            // KBTree handles duplicates internally (unlike BTreeMap)
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
        "chain_seeds: {} seeds → {} chains (B-tree, min_weight={})",
        seeds.len(),
        filtered_chains.len(),
        opt.min_chain_weight
    );

    (filtered_chains, seeds)
}

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

/// B-tree based seed chaining for a batch of reads, consuming SoASeedBatch
pub fn chain_seeds_batch(soa_seed_batch: &SoASeedBatch, opt: &MemOpt, l_pac: u64) -> SoAChainBatch {
    let num_reads = soa_seed_batch.read_seed_boundaries.len();
    let mut soa_chain_batch = SoAChainBatch::with_capacity(num_reads * 10, num_reads); // Heuristic capacity

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
                    // Try to merge using the SoA-aware function
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
                    seeds: vec![global_seed_idx], // Store global index
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
        // Clear old total_seeds_in_chains_for_read
        // let mut total_seeds_in_chains_for_read = 0;

        for chain in filtered_chains_for_read {
            soa_chain_batch.score.push(chain.score);
            soa_chain_batch.query_start.push(chain.query_start);
            soa_chain_batch.query_end.push(chain.query_end);
            soa_chain_batch.ref_start.push(chain.ref_start);
            soa_chain_batch.ref_end.push(chain.ref_end);
            soa_chain_batch.is_rev.push(chain.is_rev);
            // Before pushing, calculate weight and frac_rep
            // These still require access to individual seeds. We need an SoA-aware calculate_chain_weight.
            // For now, these will be default values or based on an adapter if we keep the old `calculate_chain_weight`
            // and pass in an "on-the-fly" adapter for seeds.
            // For simplicity and initial compilation, use dummy values or re-evaluate how to calculate.
            // The existing filter_chains function relies on `calculate_chain_weight`
            // We should call filter_chains AFTER we have populated initial SoA chains.

            // The 'kept' value needs to be set by the filtering stage, not here.
            // For initial population, let's assume `kept` is initialized by `SoAChainBatch` default (0).
            soa_chain_batch.weight.push(chain.weight); // Will be 0 initially
            soa_chain_batch.kept.push(chain.kept); // Will be 0 initially
            soa_chain_batch.frac_rep.push(chain.frac_rep); // Will be 0.0 initially

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
        let re = rb + seed.len as u64;

        if rb > last_re {
            ref_cov += seed.len;
        } else if re > last_re {
            ref_cov += (re - last_re) as i32;
        }

        last_re = last_re.max(re);
    }

    (query_cov.min(ref_cov), l_rep)
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
        "Chain filtering: {} input chains → {} kept chains ({} primary, {} shadowed)",
        chains.len(),
        kept_chains.len(),
        kept_chains.iter().filter(|c| c.kept == 3).count(),
        kept_chains.iter().filter(|c| c.kept == 1).count()
    );

    kept_chains
}
/// Implements C++ mem_chain_flt (bwamem.cpp:506-624)
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
    query_lengths: &[i32], // Array of query lengths, indexed by read_idx
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
        }

        // 2. Sort chains for the current read by weight (descending)
        let mut chain_global_indices_for_read: Vec<usize> = (0..num_chains_for_read)
            .map(|i| chain_start_idx + i)
            .collect();

        chain_global_indices_for_read
            .sort_unstable_by(|&a, &b| soa_chain_batch.weight[b].cmp(&soa_chain_batch.weight[a]));

        // Use a vector to store the global indices of chains that are "kept" for this read.
        // This simulates the `kept_chains: Vec<Chain>` in the original function.
        let mut kept_chain_global_indices: Vec<usize> = Vec::new();

        // 3. Apply filtering logic
        for &global_chain_idx in chain_global_indices_for_read.iter() {
            // Check if below minimum weight
            if soa_chain_batch.weight[global_chain_idx] < opt.min_chain_weight {
                continue; // Discard
            }

            let mut overlaps = false;
            let mut should_discard = false;

            // Check overlap with already-kept chains
            for &kept_global_chain_idx in kept_chain_global_indices.iter() {
                // Check if chains overlap on query
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
                        // Mark as shadowed, but don't commit until final decision
                        // soa_chain_batch.kept[global_chain_idx] = 1;

                        let weight_threshold = (soa_chain_batch.weight[kept_global_chain_idx]
                            as f32
                            * opt.drop_ratio) as i32;
                        let weight_diff = soa_chain_batch.weight[kept_global_chain_idx]
                            - soa_chain_batch.weight[global_chain_idx];

                        if soa_chain_batch.weight[global_chain_idx] < weight_threshold
                            && weight_diff >= (opt.min_seed_len << 1)
                        {
                            should_discard = true;
                            break;
                        } else {
                            // It overlaps but is not discarded by drop_ratio, so it is shadowed
                            soa_chain_batch.kept[global_chain_idx] = 1;
                        }
                    }
                }
            }

            if should_discard {
                continue; // This chain is discarded
            }

            // If it reaches here, it's either primary or shadowed (if overlaps was true but not discarded)
            if !overlaps {
                soa_chain_batch.kept[global_chain_idx] = 3; // Primary
            }
            // If overlaps is true, and it wasn't discarded, kept is already set to 1.

            kept_chain_global_indices.push(global_chain_idx);

            // Limit number of chains to extend
            if kept_chain_global_indices.len() >= opt.max_chain_extend as usize {
                break;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Chain Scoring and Filtering
// ----------------------------------------------------------------------------

/// Calculate chain weight based on seed coverage (SoA-aware version)
/// Implements C++ mem_chain_weight (bwamem.cpp:429-448)
///
/// Weight = minimum of query coverage and reference coverage
/// This accounts for non-overlapping seed lengths in the chain
pub fn calculate_chain_weight_soa(
    chain_global_idx: usize,
    soa_chain_batch: &SoAChainBatch,
    soa_seed_batch: &SoASeedBatch,
    opt: &MemOpt,
) -> (i32, i32) {
    let (chain_seed_start_idx, num_seeds_in_chain) =
        soa_chain_batch.chain_seed_boundaries[chain_global_idx];

    if num_seeds_in_chain == 0 {
        return (0, 0);
    }

    let mut query_cov = 0;
    let mut last_qe = -1i32;
    let mut l_rep = 0; // Length of repetitive seeds

    for i in 0..num_seeds_in_chain {
        let global_seed_idx = soa_chain_batch.seeds_indices[chain_seed_start_idx + i];
        let qb = soa_seed_batch.query_pos[global_seed_idx];
        let qe = qb + soa_seed_batch.len[global_seed_idx];
        let seed_len = soa_seed_batch.len[global_seed_idx];

        if qb > last_qe {
            query_cov += seed_len;
        } else if qe > last_qe {
            query_cov += qe - last_qe;
        }
        last_qe = last_qe.max(qe);

        // Check for repetitive seeds: if interval_size > max_occ
        if soa_seed_batch.interval_size[global_seed_idx] > opt.max_occ as u64 {
            l_rep += seed_len;
        }
    }

    let mut ref_cov = 0;
    let mut last_re = 0u64;

    for i in 0..num_seeds_in_chain {
        let global_seed_idx = soa_chain_batch.seeds_indices[chain_seed_start_idx + i];
        let rb = soa_seed_batch.ref_pos[global_seed_idx];
        let seed_len = soa_seed_batch.len[global_seed_idx];
        let re = rb + seed_len as u64;

        if rb > last_re {
            ref_cov += seed_len;
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
    let l_del = ((qlen * opt.a - opt.o_del) as f64 / opt.e_del as f64 + 1.0) as i32;
    let l_ins = ((qlen * opt.a - opt.o_ins) as f64 / opt.e_ins as f64 + 1.0) as i32;

    let l = if l_del > l_ins { l_del } else { l_ins };
    let l = if l > 1 { l } else { 1 };

    if l < (opt.w << 1) { l } else { opt.w << 1 }
}
