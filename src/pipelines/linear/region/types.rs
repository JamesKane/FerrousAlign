//! Core types for alignment regions.
//!
//! Contains the `AlignmentRegion` struct and mapping types for extension jobs.

use super::super::chaining::Chain;
use super::super::seeding::Seed;

/// Alignment region with boundaries but NO CIGAR
///
/// Matches C++ `mem_alnreg_t` structure (bwamem.h:137-158).
/// CIGAR is generated later via `generate_cigar_from_region()`.
///
/// ## Performance Note
///
/// This struct is ~100 bytes vs ~300+ bytes for a full Alignment with CIGAR.
/// Since ~80-90% of regions are filtered before CIGAR generation, this
/// significantly reduces memory allocation during extension.
#[derive(Debug, Clone)]
pub struct AlignmentRegion {
    /// Index into the chains array
    pub chain_idx: usize,

    /// Index of the best seed within this chain
    pub seed_idx: usize,

    // ========================================================================
    // Reference boundaries (FM-index coordinates)
    // ========================================================================
    /// Reference start position (inclusive, FM-index space)
    /// C++ equivalent: mem_alnreg_t.rb
    pub rb: u64,

    /// Reference end position (exclusive, FM-index space)
    /// C++ equivalent: mem_alnreg_t.re
    pub re: u64,

    // ========================================================================
    // Query boundaries (0-based)
    // ========================================================================
    /// Query start position (inclusive, 0-based)
    /// C++ equivalent: mem_alnreg_t.qb
    pub qb: i32,

    /// Query end position (exclusive, 0-based)
    /// C++ equivalent: mem_alnreg_t.qe
    pub qe: i32,

    // ========================================================================
    // Alignment metrics
    // ========================================================================
    /// Best local Smith-Waterman score
    /// C++ equivalent: mem_alnreg_t.score
    pub score: i32,

    /// True score corresponding to the aligned region
    /// May be smaller than score due to clipping adjustments
    /// C++ equivalent: mem_alnreg_t.truesc
    pub truesc: i32,

    /// Actual band width used in extension
    /// C++ equivalent: mem_alnreg_t.w
    pub w: i32,

    /// Length of regions covered by seeds
    /// Used for MAPQ calculation
    /// C++ equivalent: mem_alnreg_t.seedcov
    pub seedcov: i32,

    /// Length of the starting seed
    /// C++ equivalent: mem_alnreg_t.seedlen0
    pub seedlen0: i32,

    // ========================================================================
    // Reference information
    // ========================================================================
    /// Reference sequence ID (-1 if spanning boundary)
    /// C++ equivalent: mem_alnreg_t.rid
    pub rid: i32,

    /// Reference sequence name (e.g., "chr1")
    pub ref_name: String,

    /// Chromosome position (0-based, after coordinate conversion)
    pub chr_pos: u64,

    /// Whether the alignment is on the reverse strand
    pub is_rev: bool,

    // ========================================================================
    // Paired-end and selection fields
    // ========================================================================
    /// Fraction of repetitive seeds in this alignment
    /// Used for MAPQ calculation
    /// C++ equivalent: mem_alnreg_t.frac_rep
    pub frac_rep: f32,

    /// Hash for deterministic tie-breaking
    /// C++ equivalent: mem_alnreg_t.hash
    pub hash: u64,

    /// Secondary alignment index (-1 if primary)
    /// C++ equivalent: mem_alnreg_t.secondary
    pub secondary: i32,

    /// Sub-optimal score (2nd best)
    /// C++ equivalent: mem_alnreg_t.sub
    pub sub: i32,

    /// Number of sub-alignments chained together
    /// C++ equivalent: mem_alnreg_t.n_comp
    pub n_comp: i32,
}

impl AlignmentRegion {
    /// Create a new alignment region with default values
    pub fn new(chain_idx: usize, seed_idx: usize) -> Self {
        Self {
            chain_idx,
            seed_idx,
            rb: 0,
            re: 0,
            qb: 0,
            qe: 0,
            score: 0,
            truesc: 0,
            w: 0,
            seedcov: 0,
            seedlen0: 0,
            rid: -1,
            ref_name: String::new(),
            chr_pos: 0,
            is_rev: false,
            frac_rep: 0.0,
            hash: 0,
            secondary: -1,
            sub: 0,
            n_comp: 0,
        }
    }

    /// Calculate aligned query length
    #[inline]
    pub fn query_span(&self) -> i32 {
        self.qe - self.qb
    }

    /// Calculate aligned reference length
    #[inline]
    pub fn ref_span(&self) -> u64 {
        self.re - self.rb
    }

    /// Check if this region overlaps significantly with another
    /// Used for redundant alignment removal
    pub fn overlaps_with(&self, other: &AlignmentRegion, mask_level: f32) -> bool {
        if self.rid != other.rid {
            return false;
        }

        // Check reference overlap
        let ref_overlap_start = self.rb.max(other.rb);
        let ref_overlap_end = self.re.min(other.re);
        if ref_overlap_start >= ref_overlap_end {
            return false;
        }

        // Check query overlap
        let query_overlap_start = self.qb.max(other.qb);
        let query_overlap_end = self.qe.min(other.qe);
        if query_overlap_start >= query_overlap_end {
            return false;
        }

        // Calculate overlap fraction
        let ref_overlap = (ref_overlap_end - ref_overlap_start) as f32;
        let query_overlap = (query_overlap_end - query_overlap_start) as f32;

        let min_ref_span = (self.ref_span().min(other.ref_span())) as f32;
        let min_query_span = (self.query_span().min(other.query_span())) as f32;

        let ref_frac = if min_ref_span > 0.0 {
            ref_overlap / min_ref_span
        } else {
            0.0
        };
        let query_frac = if min_query_span > 0.0 {
            query_overlap / min_query_span
        } else {
            0.0
        };

        ref_frac > mask_level || query_frac > mask_level
    }
}

/// Tracks which extension jobs belong to which seed
#[derive(Debug, Clone)]
pub struct SeedExtensionMapping {
    pub seed_idx: usize,
    pub left_job_idx: Option<usize>,
    pub right_job_idx: Option<usize>,
}

/// Tracks all seed mappings for a chain
#[derive(Debug, Clone)]
pub struct ChainExtensionMapping {
    pub seed_mappings: Vec<SeedExtensionMapping>,
}

/// Result of score-only extension phase
///
/// Contains all information needed for filtering and CIGAR regeneration,
/// but NO CIGAR strings (they're generated later for survivors only).
pub struct ScoreOnlyExtensionResult {
    /// Alignment regions with boundaries (no CIGAR)
    pub regions: Vec<AlignmentRegion>,

    /// Filtered chains (for CIGAR regeneration)
    pub chains: Vec<Chain>,

    /// Sorted seeds (for CIGAR regeneration)
    pub seeds: Vec<Seed>,

    /// Encoded query (for CIGAR regeneration)
    pub encoded_query: Vec<u8>,

    /// Reverse complement of query (for strand handling)
    pub encoded_query_rc: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_region_creation() {
        let region = AlignmentRegion::new(0, 1);
        assert_eq!(region.chain_idx, 0);
        assert_eq!(region.seed_idx, 1);
        assert_eq!(region.score, 0);
        assert_eq!(region.secondary, -1);
    }

    #[test]
    fn test_alignment_region_spans() {
        let mut region = AlignmentRegion::new(0, 0);
        region.qb = 10;
        region.qe = 60;
        region.rb = 1000;
        region.re = 1050;

        assert_eq!(region.query_span(), 50);
        assert_eq!(region.ref_span(), 50);
    }

    #[test]
    fn test_alignment_region_overlap() {
        let mut region1 = AlignmentRegion::new(0, 0);
        region1.rid = 0;
        region1.qb = 0;
        region1.qe = 100;
        region1.rb = 1000;
        region1.re = 1100;

        let mut region2 = AlignmentRegion::new(1, 0);
        region2.rid = 0;
        region2.qb = 50;
        region2.qe = 150;
        region2.rb = 1050;
        region2.re = 1150;

        // 50% overlap should be detected at 0.3 mask level
        assert!(region1.overlaps_with(&region2, 0.3));

        // Different chromosome should not overlap
        region2.rid = 1;
        assert!(!region1.overlaps_with(&region2, 0.3));
    }
}
