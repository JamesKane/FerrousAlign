// Rust equivalent of dnaSeqPair (C++ bandedSWA.h:90-99)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqPair {
    /// Offset into reference sequence buffer
    pub reference_offset: i32,
    /// Offset into query sequence buffer
    pub query_offset: i32,
    /// Sequence pair identifier
    pub pair_id: i32,
    /// Length of reference sequence
    pub reference_length: i32,
    /// Length of query sequence
    pub query_length: i32,
    /// Initial alignment score (from previous alignment)
    pub initial_score: i32,
    /// Sequence identifier (index into sequence array)
    pub sequence_id: i32,
    /// Region identifier (index into alignment region array)
    pub region_id: i32,
    /// Best alignment score
    pub score: i32,
    /// Target (reference) end position
    pub target_end_pos: i32,
    /// Global target (reference) end position
    pub global_target_end_pos: i32,
    /// Query end position
    pub query_end_pos: i32,
    /// Global alignment score
    pub global_score: i32,
    /// Maximum offset in alignment
    pub max_offset: i32,
}

// Rust equivalent of eh_t
#[derive(Debug, Clone, Copy, Default)]
pub struct EhT {
    pub h: i32, // H score (match/mismatch)
    pub e: i32, // E score (gap in target)
}

/// Extension direction for seed extension
/// Matches C++ bwa-mem2 LEFT/RIGHT extension model (bwamem.cpp:2229-2418)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionDirection {
    /// LEFT extension: seed start → query position 0 (5' direction)
    /// Sequences are reversed before alignment, pen_clip5 applied
    Left,
    /// RIGHT extension: seed end → query end (3' direction)
    /// Sequences aligned forward, pen_clip3 applied
    Right,
}

/// Result from directional extension alignment
/// Contains both local and global alignment scores for clipping penalty decision
#[derive(Debug, Clone)]
pub struct ExtensionResult {
    /// Best local alignment score (may terminate early via Z-drop)
    pub local_score: i32,
    /// Global alignment score (score at boundary: qb=0 for left, qe=qlen for right)
    pub global_score: i32,
    /// Query bases extended in local alignment
    pub query_ext_len: i32,
    /// Target bases extended in local alignment
    pub target_ext_len: i32,
    /// Target bases extended in global alignment
    pub global_target_len: i32,
    /// Should soft-clip this extension? (based on clipping penalty decision)
    pub should_clip: bool,
    /// CIGAR operations for this extension (already reversed if LEFT)
    pub cigar: Vec<(u8, i32)>,
    /// Reference aligned sequence
    pub ref_aligned: Vec<u8>,
    /// Query aligned sequence
    pub query_aligned: Vec<u8>,
}

// Rust equivalent of dnaOutScore
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutScore {
    pub score: i32,
    pub target_end_pos: i32,
    pub gtarget_end_pos: i32,
    pub query_end_pos: i32,
    pub global_score: i32,
    pub max_offset: i32,
}

// Complete alignment result including CIGAR string
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentResult {
    pub score: OutScore,
    pub cigar: Vec<(u8, i32)>,
    /// Reference bases in the alignment (for MD tag generation)
    /// Encoded as 0=A, 1=C, 2=G, 3=T, 4=N
    pub ref_aligned: Vec<u8>,
    /// Query bases in the alignment (for MD tag generation)
    /// Encoded as 0=A, 1=C, 2=G, 3=T, 4=N
    pub query_aligned: Vec<u8>,
}

// Helper function to create scoring matrix (similar to bwa_fill_scmat in main_banded.cpp)
pub fn bwa_fill_scmat(match_score: i8, mismatch_penalty: i8, ambig_penalty: i8) -> [i8; 25] {
    let mut mat = [0i8; 25];
    let mut k = 0;

    // Fill 5x5 matrix for A, C, G, T, N
    for i in 0..4 {
        for j in 0..4 {
            mat[k] = if i == j {
                match_score
            } else {
                -mismatch_penalty
            };
            k += 1;
        }
        mat[k] = ambig_penalty; // ambiguous base (N)
        k += 1;
    }

    // Last row for N
    for _ in 0..5 {
        mat[k] = ambig_penalty;
        k += 1;
    }

    mat
}

// ============================================================================
// Helper Functions for Separate Left/Right Extensions
// ============================================================================

/// Reverse a sequence for left extension alignment
/// C++ reference: bwamem.cpp:2278 reverses query for left extension
#[inline]
pub fn reverse_sequence(seq: &[u8]) -> Vec<u8> {
    seq.iter().copied().rev().collect()
}

/// Reverse a CIGAR string after left extension alignment
/// When we align reversed sequences, the CIGAR is also reversed
/// This function reverses it back to the forward orientation
#[inline]
pub fn reverse_cigar(cigar: &[(u8, i32)]) -> Vec<(u8, i32)> {
    cigar.iter().copied().rev().collect()
}

/// Merge consecutive identical CIGAR operations
/// E.g., [(M, 10), (M, 5)] → [(M, 15)]
///
/// **Deprecated**: Use `crate::alignment::cigar::normalize()` instead.
/// This function is kept for internal compatibility only.
#[inline]
#[deprecated(
    since = "0.5.3",
    note = "Use crate::alignment::cigar::normalize() instead"
)]
pub fn merge_cigar_operations(cigar: Vec<(u8, i32)>) -> Vec<(u8, i32)> {
    crate::alignment::cigar::normalize(cigar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bwa_fill_scmat() {
        let mat = bwa_fill_scmat(1, 4, -1);

        // Check diagonal (matches)
        assert_eq!(mat[0], 1); // A-A
        assert_eq!(mat[6], 1); // C-C
        assert_eq!(mat[12], 1); // G-G
        assert_eq!(mat[18], 1); // T-T

        // Check mismatches
        assert_eq!(mat[1], -4); // A-C
        assert_eq!(mat[5], -4); // C-A
    }
}
