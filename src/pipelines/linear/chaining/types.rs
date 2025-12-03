//! Core data types for seed chaining.
//!
//! Contains the `Chain` struct for AoS processing and `SoAChainBatch` for
//! batched SIMD-friendly processing.

// Safety limits to prevent runaway memory/CPU usage
pub const MAX_SEEDS_PER_READ: usize = 100_000;
pub const MAX_CHAINS_PER_READ: usize = 10_000;

/// A chain of seeds representing a potential alignment region.
///
/// Chains group compatible seeds that likely originate from the same
/// genomic location. Each chain tracks query and reference bounds,
/// as well as metadata for filtering and scoring.
#[derive(Debug, Clone)]
pub struct Chain {
    /// Chain score (sum of seed lengths)
    pub score: i32,
    /// Indices of seeds in the original seeds vector
    pub seeds: Vec<usize>,
    /// Query start position (minimum of all seeds)
    pub query_start: i32,
    /// Query end position (maximum of all seeds)
    pub query_end: i32,
    /// Reference start position
    pub ref_start: u64,
    /// Reference end position
    pub ref_end: u64,
    /// Whether chain is on reverse strand
    pub is_rev: bool,
    /// Chain weight (seed coverage), calculated by mem_chain_weight
    pub weight: i32,
    /// Chain status: 0=discarded, 1=shadowed, 2=partial_overlap, 3=primary
    pub kept: i32,
    /// Fraction of repetitive seeds in this chain
    pub frac_rep: f32,
    /// Reference sequence ID (chromosome)
    pub rid: i32,
    /// B-tree key: reference position of first seed
    #[allow(dead_code)]
    pub(crate) pos: u64,
    /// Last seed's query begin (for test_and_merge)
    pub(crate) last_qbeg: i32,
    /// Last seed's reference begin (for test_and_merge)
    pub(crate) last_rbeg: u64,
    /// Last seed's length (for test_and_merge)
    pub(crate) last_len: i32,
}

/// Batch of chains in Structure-of-Arrays (SoA) format.
///
/// This layout is optimized for SIMD processing and cache efficiency
/// when processing many chains in parallel.
#[derive(Debug, Clone, Default)]
pub struct SoAChainBatch {
    /// Chain scores
    pub score: Vec<i32>,
    /// Query start positions
    pub query_start: Vec<i32>,
    /// Query end positions
    pub query_end: Vec<i32>,
    /// Reference start positions
    pub ref_start: Vec<u64>,
    /// Reference end positions
    pub ref_end: Vec<u64>,
    /// Reverse strand flags
    pub is_rev: Vec<bool>,
    /// Chain weights
    pub weight: Vec<i32>,
    /// Chain kept status
    pub kept: Vec<i32>,
    /// Fraction of repetitive seeds
    pub frac_rep: Vec<f32>,
    /// Reference sequence IDs
    pub rid: Vec<i32>,
    /// B-tree key positions
    pub pos: Vec<u64>,
    /// Last seed query begins
    pub last_qbeg: Vec<i32>,
    /// Last seed reference begins
    pub last_rbeg: Vec<u64>,
    /// Last seed lengths
    pub last_len: Vec<i32>,
    /// Boundaries for chains belonging to each read: (start_idx, count)
    pub read_chain_boundaries: Vec<(usize, usize)>,
    /// Boundaries for seeds in each chain: (start_idx_in_seeds_indices, count)
    pub chain_seed_boundaries: Vec<(usize, usize)>,
    /// Flattened indices into the global SoASeedBatch
    pub seeds_indices: Vec<usize>,
}

impl SoAChainBatch {
    /// Create a new empty chain batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a chain batch with pre-allocated capacity.
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
            seeds_indices: Vec::new(),
        }
    }

    /// Clear all data from the batch.
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

    /// Get total number of chains in the batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.score.len()
    }

    /// Check if the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.score.is_empty()
    }

    /// Get number of reads in the batch.
    #[inline]
    pub fn num_reads(&self) -> usize {
        self.read_chain_boundaries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_chain_batch_new() {
        let batch = SoAChainBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.num_reads(), 0);
    }

    #[test]
    fn test_soa_chain_batch_with_capacity() {
        let batch = SoAChainBatch::with_capacity(100, 10);
        assert!(batch.is_empty());
        assert_eq!(batch.score.capacity(), 100);
        assert_eq!(batch.read_chain_boundaries.capacity(), 10);
    }

    #[test]
    fn test_soa_chain_batch_clear() {
        let mut batch = SoAChainBatch::with_capacity(10, 2);
        batch.score.push(100);
        batch.query_start.push(0);
        batch.read_chain_boundaries.push((0, 1));

        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.num_reads(), 0);
    }
}
