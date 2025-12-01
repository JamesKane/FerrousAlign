use std::cell::RefCell;


use super::super::chaining::{Chain, SoAChainBatch};
use super::super::region::{ChainExtensionMapping};
use super::super::seeding::{Seed, SoASeedBatch};


thread_local! {
    pub static REVERSE_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(256));
}

/// Direction of extension from seed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionDirection {
    Left,
    Right,
}

/// A single extension job with metadata for result distribution
#[derive(Debug, Clone)]
pub struct BatchedExtensionJob {
    /// Index of the read in the batch (0..batch_size)
    pub read_idx: usize,
    /// Index of the chain within the read's chains
    pub chain_idx: usize,
    /// Index of the seed within the chain (used for primary seed tracking)
    pub seed_idx: usize,
    /// Direction of extension
    pub direction: ExtensionDirection,
    /// Query sequence length for this extension
    pub query_len: i32,
    /// Reference/target sequence length for this extension
    pub ref_len: i32,
    /// Initial score (h0) - seed_len * match_score for left, 0 for right
    pub h0: i32,
    /// Band width for banded alignment
    pub band_width: i32,
    /// Offset into the query sequence buffer
    pub query_offset: usize,
    /// Offset into the reference sequence buffer
    pub ref_offset: usize,
}

/// Result of a batched extension job
#[derive(Debug, Clone)]
pub struct BatchExtensionResult {
    /// Index of the read in the batch
    pub read_idx: usize,
    /// Index of the chain within the read's chains
    pub chain_idx: usize,
    /// Index of the seed within the chain
    pub seed_idx: usize,
    /// Direction of extension
    pub direction: ExtensionDirection,
    /// Alignment score
    pub score: i32,
    /// Query end position (relative to extension segment)
    pub query_end: i32,
    /// Reference end position (relative to extension segment)
    pub ref_end: i32,
    /// Global score (for clipping detection)
    pub gscore: i32,
    /// Global target end position
    pub gref_end: i32,
    /// Max offset (for band width adjustment)
    pub max_off: i32,
}

/// Batch of extension jobs collected from multiple reads
///
/// Uses Structure-of-Arrays (SoA) layout for SIMD-friendly memory access.
/// Sequences are stored contiguously in separate buffers, with per-job
/// offsets for slicing.
#[derive(Debug)]
pub struct ExtensionJobBatch {
    /// Job metadata for result distribution
    pub jobs: Vec<BatchedExtensionJob>,

    /// Contiguous buffer of all query sequences (2-bit encoded)
    pub query_seqs: Vec<u8>,

    /// Contiguous buffer of all reference sequences (2-bit encoded)
    pub ref_seqs: Vec<u8>,

    /// SoA interleaved query buffer
    pub query_soa: Vec<u8>,
    /// SoA interleaved target buffer
    pub target_soa: Vec<u8>,
    /// Number of active SIMD lanes for SoA
    pub lanes: usize,
    /// Per-job start offsets in SoA buffers (packed metadata)
    pub pos_offsets: Vec<usize>,

    /// Running offset for next query sequence
    pub query_offset: usize,

    /// Running offset for next reference sequence
    pub ref_offset: usize,
}

impl ExtensionJobBatch {
    /// Create a new empty batch with pre-allocated capacity
    pub fn with_capacity(job_capacity: usize, seq_capacity: usize) -> Self {
        Self {
            jobs: Vec::with_capacity(job_capacity),
            query_seqs: Vec::with_capacity(seq_capacity),
            ref_seqs: Vec::with_capacity(seq_capacity),
            query_soa: Vec::with_capacity(seq_capacity), // Estimate capacity
            target_soa: Vec::with_capacity(seq_capacity),
            lanes: 0,
            pos_offsets: Vec::with_capacity(job_capacity * 4), // 4 elements per chunk
            query_offset: 0,
            ref_offset: 0,
        }
    }

    /// Create a new empty batch
    pub fn new() -> Self {
        Self::with_capacity(1024, 65536)
    }

    /// Add an extension job to the batch
    ///
    /// Sequences are copied into contiguous buffers.
    pub fn add_job(
        &mut self,
        read_idx: usize,
        chain_idx: usize,
        seed_idx: usize,
        direction: ExtensionDirection,
        query_seq: &[u8],
        ref_seq: &[u8],
        h0: i32,
        band_width: i32,
    ) {
        let query_offset = self.query_offset;
        let ref_offset = self.ref_offset;

        // Copy sequences to contiguous buffers
        self.query_seqs.extend_from_slice(query_seq);
        self.ref_seqs.extend_from_slice(ref_seq);

        // Update offsets
        self.query_offset += query_seq.len();
        self.ref_offset += ref_seq.len();

        // Add job metadata
        self.jobs.push(BatchedExtensionJob {
            read_idx,
            chain_idx,
            seed_idx,
            direction,
            query_len: query_seq.len() as i32,
            ref_len: ref_seq.len() as i32,
            h0,
            band_width,
            query_offset,
            ref_offset,
        });
    }

    /// Get the query sequence slice for a job
    pub fn get_query_seq(&self, job_idx: usize) -> &[u8] {
        let job = &self.jobs[job_idx];
        &self.query_seqs[job.query_offset..job.query_offset + job.query_len as usize]
    }

    /// Get the reference sequence slice for a job
    pub fn get_ref_seq(&self, job_idx: usize) -> &[u8] {
        let job = &self.jobs[job_idx];
        &self.ref_seqs[job.ref_offset..job.ref_offset + job.ref_len as usize]
    }

    /// Number of jobs in the batch
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Clear the batch for reuse
    pub fn clear(&mut self) {
        self.jobs.clear();
        self.query_seqs.clear();
        self.ref_seqs.clear();
        self.query_offset = 0;
        self.ref_offset = 0;

        // Clear SoA data
        self.query_soa.clear();
        self.target_soa.clear();
        self.lanes = 0;
        self.pos_offsets.clear();
    }
}

impl Default for ExtensionJobBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Scores for a chain's left and right extensions
#[derive(Debug, Clone, Default)]
pub struct ChainExtensionScores {
    /// Left extension score
    pub left_score: Option<i32>,
    /// Left extension query end (relative to extension segment)
    pub left_query_end: Option<i32>,
    /// Left extension reference end (relative to extension segment)
    pub left_ref_end: Option<i32>,
    /// Left extension global score
    pub left_gscore: Option<i32>,
    /// Left extension global reference end
    pub left_gref_end: Option<i32>,

    /// Right extension score
    pub right_score: Option<i32>,
    /// Right extension query end
    pub right_query_end: Option<i32>,
    /// Right extension reference end
    pub right_ref_end: Option<i32>,
    /// Right extension global score
    pub right_gscore: Option<i32>,
    /// Right extension global reference end
    pub right_gref_end: Option<i32>,
}

/// Context for a single read during cross-read batched extension
///
/// Contains all the data needed to collect extension jobs and later
/// merge results back into AlignmentRegions.
#[derive(Debug, Clone)]
pub struct ReadExtensionContext {
    /// Read name (for debugging)
    pub query_name: String,
    /// Encoded query sequence (2-bit)
    pub encoded_query: Vec<u8>,
    /// Reverse complement of encoded query
    pub encoded_query_rc: Vec<u8>,
    /// Chains for this read
    pub chains: Vec<Chain>,
    /// Seeds for this read
    pub seeds: Vec<Seed>,
    /// Query length
    pub query_len: i32,
    /// Reference segments for each chain: (rmax_0, rmax_1, rseq)
    pub chain_ref_segments: Vec<Option<(u64, u64, Vec<u8>)>>,
}

/// Per-read mappings from chains/seeds to job indices
///
/// Uses the same SeedExtensionMapping and ChainExtensionMapping structures
/// from region.rs for compatibility with merge_extension_scores_to_regions().
#[derive(Debug, Clone, Default)]
pub struct ReadExtensionMappings {
    pub chain_mappings: Vec<ChainExtensionMapping>,
}

/// SoA-native context for cross-read batched extension (PR3)
///
/// This structure holds batch-wide SoA data from seeding and chaining stages,
/// eliminating the need for per-read AoS intermediate representations.
/// All data is stored in SoA format with per-read boundaries for indexing.
#[derive(Debug, Clone)]
pub struct SoAReadExtensionContext {
    /// Per-read boundaries into batch-wide arrays
    /// Each entry is (start_index, count) for accessing batch-wide SoA data
    pub read_boundaries: Vec<(usize, usize)>,

    /// Query names for all reads in the batch
    pub query_names: Vec<String>,

    /// Query lengths for all reads in the batch
    pub query_lengths: Vec<i32>,

    /// Encoded query sequences (flattened from SoAEncodedQueryBatch)
    /// Access via read_boundaries to get per-read slices
    pub encoded_queries: Vec<u8>,

    /// Encoded query boundaries for accessing encoded_queries
    /// Each entry is (start_offset, length) in encoded_queries buffer
    pub encoded_query_boundaries: Vec<(usize, usize)>,

    /// Reverse complement encoded queries (flattened from SoAEncodedQueryBatch)
    pub encoded_queries_rc: Vec<u8>,

    /// Encoded query RC boundaries for accessing encoded_queries_rc
    /// Each entry is (start_offset, length) in encoded_queries_rc buffer
    pub encoded_query_rc_boundaries: Vec<(usize, usize)>,

    /// SoA seed batch (from PR2's find_seeds_batch)
    pub soa_seed_batch: SoASeedBatch,

    /// SoA chain batch (from PR2's chain_seeds_batch + filter_chains_batch)
    pub soa_chain_batch: SoAChainBatch,

    /// Reference segments for each chain: Some((rmax_0, rmax_1)) or None
    /// Indexed by global chain index across all reads
    /// Size = total number of chains across all reads in batch
    pub chain_ref_segments: Vec<Option<(u64, u64)>>,
}

/// SoA-native alignment results (PR4)
///
/// Structure-of-Arrays representation of batch-wide alignment results.
/// This eliminates the need to build individual Alignment structs before SAM output,
/// allowing direct batch-wise SAM record generation.
///
/// All alignments are stored contiguously with per-read boundaries for indexing.
#[derive(Debug, Clone)]
pub struct SoAAlignmentResult {
    // Scalar SAM fields (one per alignment)
    /// Query/read names
    pub query_names: Vec<String>,
    /// SAM flags
    pub flags: Vec<u16>,
    /// Reference sequence names (chromosome)
    pub ref_names: Vec<String>,
    /// Reference sequence IDs (for paired-end scoring)
    pub ref_ids: Vec<usize>,
    /// 0-based leftmost mapping positions
    pub positions: Vec<u64>,
    /// Mapping qualities
    pub mapqs: Vec<u8>,
    /// Alignment scores
    pub scores: Vec<i32>,

    // CIGAR strings (flattened with boundaries)
    /// CIGAR operation codes (M, I, D, S, etc.)
    pub cigar_ops: Vec<u8>,
    /// CIGAR operation lengths
    pub cigar_lens: Vec<i32>,
    /// Per-alignment boundaries into cigar_ops/cigar_lens: (start_idx, count)
    pub cigar_boundaries: Vec<(usize, usize)>,

    // Mate information
    /// Mate reference names
    pub rnexts: Vec<String>,
    /// Mate positions
    pub pnexts: Vec<u64>,
    /// Template lengths
    pub tlens: Vec<i32>,

    // Sequence and quality (flattened with boundaries)
    /// Query sequences (concatenated bytes)
    pub seqs: Vec<u8>,
    /// Quality scores (concatenated bytes)
    pub quals: Vec<u8>,
    /// Per-alignment boundaries into seqs/quals: (start_offset, length)
    pub seq_boundaries: Vec<(usize, usize)>,

    // SAM tags (flattened with boundaries)
    /// Tag names (concatenated strings)
    pub tag_names: Vec<String>,
    /// Tag values (concatenated strings)
    pub tag_values: Vec<String>,
    /// Per-alignment boundaries into tag_names/tag_values: (start_idx, count)
    pub tag_boundaries: Vec<(usize, usize)>,

    // Internal fields for alignment selection (not output to SAM)
    /// Query start positions (0-based)
    pub query_starts: Vec<i32>,
    /// Query end positions (exclusive)
    pub query_ends: Vec<i32>,
    /// Seed coverage lengths
    pub seed_coverages: Vec<i32>,
    /// Hash values for deterministic tie-breaking
    pub hashes: Vec<u64>,
    /// Repetitive fraction values
    pub frac_reps: Vec<f32>,

    // Per-read boundaries
    /// Boundaries for alignments belonging to each read: (start_idx, count)
    /// Index i gives the range of alignments for read i
    pub read_alignment_boundaries: Vec<(usize, usize)>,
}

impl SoAAlignmentResult {
    /// Create a new empty result with pre-allocated capacity
    pub fn with_capacity(alignment_capacity: usize, num_reads: usize) -> Self {
        Self {
            query_names: Vec::with_capacity(alignment_capacity),
            flags: Vec::with_capacity(alignment_capacity),
            ref_names: Vec::with_capacity(alignment_capacity),
            ref_ids: Vec::with_capacity(alignment_capacity),
            positions: Vec::with_capacity(alignment_capacity),
            mapqs: Vec::with_capacity(alignment_capacity),
            scores: Vec::with_capacity(alignment_capacity),
            cigar_ops: Vec::with_capacity(alignment_capacity * 10), // Estimate ~10 ops per alignment
            cigar_lens: Vec::with_capacity(alignment_capacity * 10),
            cigar_boundaries: Vec::with_capacity(alignment_capacity),
            rnexts: Vec::with_capacity(alignment_capacity),
            pnexts: Vec::with_capacity(alignment_capacity),
            tlens: Vec::with_capacity(alignment_capacity),
            seqs: Vec::with_capacity(alignment_capacity * 150), // Estimate ~150bp per read
            quals: Vec::with_capacity(alignment_capacity * 150),
            seq_boundaries: Vec::with_capacity(alignment_capacity),
            tag_names: Vec::with_capacity(alignment_capacity * 3), // Estimate ~3 tags per alignment
            tag_values: Vec::with_capacity(alignment_capacity * 3),
            tag_boundaries: Vec::with_capacity(alignment_capacity),
            query_starts: Vec::with_capacity(alignment_capacity),
            query_ends: Vec::with_capacity(alignment_capacity),
            seed_coverages: Vec::with_capacity(alignment_capacity),
            hashes: Vec::with_capacity(alignment_capacity),
            frac_reps: Vec::with_capacity(alignment_capacity),
            read_alignment_boundaries: Vec::with_capacity(num_reads),
        }
    }

    /// Create a new empty result
    pub fn new() -> Self {
        Self::with_capacity(512, 512)
    }

    /// Number of alignments in the result
    pub fn len(&self) -> usize {
        self.flags.len()
    }

    /// Check if the result is empty
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }

    /// Number of reads in the result
    pub fn num_reads(&self) -> usize {
        self.read_alignment_boundaries.len()
    }
}

impl Default for SoAAlignmentResult {
    fn default() -> Self {
        Self::new()
    }
}

