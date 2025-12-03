//! Core data types for seeding.
//!
//! Contains the `Seed`, `SMEM`, `SoASeedBatch`, and `SoAEncodedQueryBatch` types.

/// Represents a seed match between query and reference.
#[derive(Debug, Clone)]
pub struct Seed {
    /// Position in the query sequence
    pub query_pos: i32,
    /// Position in the reference sequence
    pub ref_pos: u64,
    /// Length of the seed
    pub len: i32,
    /// Whether the seed is on the reverse strand
    pub is_rev: bool,
    /// BWT interval size (occurrence count)
    pub interval_size: u64,
    /// Reference sequence ID (chromosome), -1 if spans boundaries
    pub rid: i32,
}

/// Super Maximal Exact Match (SMEM).
///
/// Represents a maximal exact match between query and reference that cannot
/// be extended in either direction while remaining unique in the reference.
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

/// Batch of seeds in Structure-of-Arrays (SoA) format.
///
/// This layout is optimized for SIMD processing and cache efficiency.
#[derive(Debug, Clone, Default)]
pub struct SoASeedBatch {
    /// Query positions
    pub query_pos: Vec<i32>,
    /// Reference positions
    pub ref_pos: Vec<u64>,
    /// Seed lengths
    pub len: Vec<i32>,
    /// Reverse strand flags
    pub is_rev: Vec<bool>,
    /// BWT interval sizes
    pub interval_size: Vec<u64>,
    /// Reference sequence IDs
    pub rid: Vec<i32>,
    /// Boundaries for seeds belonging to each read: (start_idx, count)
    pub read_seed_boundaries: Vec<(usize, usize)>,
}

impl SoASeedBatch {
    /// Create a new empty seed batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a seed batch with pre-allocated capacity.
    pub fn with_capacity(capacity: usize, num_reads: usize) -> Self {
        Self {
            query_pos: Vec::with_capacity(capacity),
            ref_pos: Vec::with_capacity(capacity),
            len: Vec::with_capacity(capacity),
            is_rev: Vec::with_capacity(capacity),
            interval_size: Vec::with_capacity(capacity),
            rid: Vec::with_capacity(capacity),
            read_seed_boundaries: Vec::with_capacity(num_reads),
        }
    }

    /// Clear all data from the batch.
    pub fn clear(&mut self) {
        self.query_pos.clear();
        self.ref_pos.clear();
        self.len.clear();
        self.is_rev.clear();
        self.interval_size.clear();
        self.rid.clear();
        self.read_seed_boundaries.clear();
    }

    /// Push a seed into the batch.
    pub fn push(&mut self, seed: &Seed, _read_idx: usize) {
        self.query_pos.push(seed.query_pos);
        self.ref_pos.push(seed.ref_pos);
        self.len.push(seed.len);
        self.is_rev.push(seed.is_rev);
        self.interval_size.push(seed.interval_size);
        self.rid.push(seed.rid);
    }

    /// Get the total number of seeds.
    #[inline]
    pub fn len(&self) -> usize {
        self.query_pos.len()
    }

    /// Check if the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.query_pos.is_empty()
    }

    /// Get the number of reads in the batch.
    #[inline]
    pub fn num_reads(&self) -> usize {
        self.read_seed_boundaries.len()
    }
}

/// Batch of encoded queries in SoA format.
///
/// Stores 2-bit encoded sequences for efficient FM-index lookup.
#[derive(Debug, Clone, Default)]
pub struct SoAEncodedQueryBatch {
    /// Concatenated encoded sequences
    pub encoded_seqs: Vec<u8>,
    /// Boundaries for each query: (start_offset, length)
    pub query_boundaries: Vec<(usize, usize)>,
}

impl SoAEncodedQueryBatch {
    /// Create a new empty encoded query batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an encoded query batch with pre-allocated capacity.
    pub fn with_capacity(total_seq_len: usize, num_reads: usize) -> Self {
        Self {
            encoded_seqs: Vec::with_capacity(total_seq_len),
            query_boundaries: Vec::with_capacity(num_reads),
        }
    }

    /// Clear all data from the batch.
    pub fn clear(&mut self) {
        self.encoded_seqs.clear();
        self.query_boundaries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smem_default() {
        let smem = SMEM::default();
        assert_eq!(smem.read_id, 0);
        assert_eq!(smem.query_start, 0);
        assert_eq!(smem.query_end, 0);
        assert_eq!(smem.interval_size, 0);
    }

    #[test]
    fn test_smem_structure() {
        let smem = SMEM {
            read_id: 0,
            query_start: 10,
            query_end: 20,
            bwt_interval_start: 5,
            bwt_interval_end: 15,
            interval_size: 10,
            is_reverse_complement: false,
        };

        assert_eq!(smem.query_start, 10);
        assert_eq!(smem.query_end, 20);
        assert_eq!(smem.interval_size, 10);
    }

    #[test]
    fn test_soa_seed_batch_new() {
        let batch = SoASeedBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.num_reads(), 0);
    }

    #[test]
    fn test_soa_seed_batch_with_capacity() {
        let batch = SoASeedBatch::with_capacity(100, 10);
        assert!(batch.is_empty());
        assert_eq!(batch.query_pos.capacity(), 100);
        assert_eq!(batch.read_seed_boundaries.capacity(), 10);
    }

    #[test]
    fn test_soa_encoded_query_batch() {
        let mut batch = SoAEncodedQueryBatch::with_capacity(100, 10);
        assert!(batch.encoded_seqs.is_empty());
        assert!(batch.query_boundaries.is_empty());

        batch.encoded_seqs.extend_from_slice(&[0, 1, 2, 3]);
        batch.query_boundaries.push((0, 4));

        batch.clear();
        assert!(batch.encoded_seqs.is_empty());
        assert!(batch.query_boundaries.is_empty());
    }
}
