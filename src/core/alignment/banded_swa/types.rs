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

// Constants for traceback
pub const DEFAULT_AMBIG: i8 = -1;
pub const TB_MATCH: u8 = 0;
pub const TB_DEL: u8 = 1; // Gap in target/reference
pub const TB_INS: u8 = 2; // Gap in query

// Rust equivalent of BandedPairWiseSW class
pub struct BandedPairWiseSW {
    m: i32,
    end_bonus: i32,
    zdrop: i32,
    /// Clipping penalty for 5' end (default: 5)
    pen_clip5: i32,
    /// Clipping penalty for 3' end (default: 5)
    pen_clip3: i32,
    o_del: i32,
    o_ins: i32,
    e_del: i32,
    e_ins: i32,
    mat: [i8; 25], // Assuming a 5x5 matrix for A,C,G,T,N
}

impl BandedPairWiseSW {
    pub fn new(
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        zdrop: i32,
        end_bonus: i32,
        pen_clip5: i32,
        pen_clip3: i32,
        mat: [i8; 25],
        _w_match: i8,
        _w_mismatch: i8,
    ) -> Self {
        BandedPairWiseSW {
            m: 5, // Assuming 5 bases (A, C, G, T, N)
            end_bonus,
            zdrop,
            pen_clip5,
            pen_clip3,
            o_del,
            o_ins,
            e_del,
            e_ins,
            mat,
        }
    }

    // Getter methods for ksw_affine_gap integration
    /// Returns the gap open penalty for deletions
    pub fn o_del(&self) -> i32 {
        self.o_del
    }

    /// Returns the gap extension penalty for deletions
    pub fn e_del(&self) -> i32 {
        self.e_del
    }

    /// Returns the gap open penalty for insertions
    pub fn o_ins(&self) -> i32 {
        self.o_ins
    }

    /// Returns the gap extension penalty for insertions
    pub fn e_ins(&self) -> i32 {
        self.e_ins
    }

    /// Returns the Z-drop threshold
    pub fn zdrop(&self) -> i32 {
        self.zdrop
    }

    /// Returns the end bonus
    pub fn end_bonus(&self) -> i32 {
        self.end_bonus
    }

    /// Returns the alphabet size
    pub fn alphabet_size(&self) -> i32 {
        self.m
    }

    /// Returns the scoring matrix
    pub fn scoring_matrix(&self) -> &[i8; 25] {
        &self.mat
    }

    /// Returns the 5' clipping penalty
    pub fn pen_clip5(&self) -> i32 {
        self.pen_clip5
    }

    /// Returns the 3' clipping penalty
    pub fn pen_clip3(&self) -> i32 {
        self.pen_clip3
    }
}
