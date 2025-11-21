// bwa-mem2-rust/src/align.rs

// Import BwaIndex and MemOpt
use crate::banded_swa::{BandedPairWiseSW, merge_cigar_operations};
use crate::fm_index::{CP_SHIFT, CpOcc, backward_ext, forward_ext, get_occ};
use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;
use crate::utils::hash_64;

/// Calculate maximum gap size for a given query length
/// Matches C++ bwamem.cpp:66 cal_max_gap()
#[inline]
fn cal_max_gap(opt: &MemOpt, qlen: i32) -> i32 {
    let l_del = ((qlen * opt.a as i32 - opt.o_del as i32) as f64 / opt.e_del as f64 + 1.0) as i32;
    let l_ins = ((qlen * opt.a as i32 - opt.o_ins as i32) as f64 / opt.e_ins as f64 + 1.0) as i32;

    let l = if l_del > l_ins { l_del } else { l_ins };
    let l = if l > 1 { l } else { 1 };

    if l < (opt.w << 1) { l } else { opt.w << 1 }
}

// Define a struct to represent a seed
#[derive(Debug, Clone)]
pub struct Seed {
    pub query_pos: i32,     // Position in the query
    pub ref_pos: u64,       // Position in the reference
    pub len: i32,           // Length of the seed
    pub is_rev: bool,       // Is it on the reverse strand?
    pub interval_size: u64, // BWT interval size (occurrence count)
}

// Define a struct to represent a Super Maximal Exact Match (SMEM)
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

// Scoring matrix matching C++ defaults (Match=1, Mismatch=-4)
// This matches C++ bwa_fill_scmat() with a=1, b=4
// A, C, G, T, N
// A  1 -4 -4 -4 -1
// C -4  1 -4 -4 -1
// G -4 -4  1 -4 -1
// T -4 -4 -4  1 -1
// N -1 -1 -1 -1 -1
pub const DEFAULT_SCORING_MATRIX: [i8; 25] = [
    1, -4, -4, -4, -1, // A row
    -4, 1, -4, -4, -1, // C row
    -4, -4, 1, -4, -1, // G row
    -4, -4, -4, 1, -1, // T row
    -1, -1, -1, -1, -1, // N row
];

// Function to convert a base character to its 0-3 encoding
// A=0, C=1, G=2, T=3, N=4
#[inline(always)]
pub fn base_to_code(base: u8) -> u8 {
    match base {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4, // N or any other character
    }
}

// Function to get the reverse complement of a code
// 0=A, 1=C, 2=G, 3=T, 4=N
#[inline(always)]
pub fn reverse_complement_code(code: u8) -> u8 {
    match code {
        0 => 3, // A -> T
        1 => 2, // C -> G
        2 => 1, // G -> C
        3 => 0, // T -> A
        _ => 4, // N or any other character remains N
    }
}

/// Encode a DNA sequence to numeric codes in bulk
/// Converts ASCII bases (ACGTN) to numeric codes (01234)
/// Case-insensitive: A/a -> 0, C/c -> 1, G/g -> 2, T/t -> 3, other -> 4
///
/// This is a convenience function that applies `base_to_code` to each base
/// in the sequence. Use this to avoid repetitive iterator chains.
///
/// # Example
/// ```
/// use ferrous_align::align::encode_sequence;
///
/// let seq = b"ACGTN";
/// let encoded = encode_sequence(seq);
/// assert_eq!(encoded, vec![0, 1, 2, 3, 4]);
/// ```
#[inline]
pub fn encode_sequence(seq: &[u8]) -> Vec<u8> {
    seq.iter().map(|&b| base_to_code(b)).collect()
}

/// Compute reverse complement of an encoded sequence
/// Takes numeric codes (01234) and returns reverse complement
/// Handles N bases correctly (N -> N)
///
/// This function:
/// 1. Reverses the sequence order
/// 2. Complements each base: A↔T (0↔3), C↔G (1↔2), N→N (4→4)
///
/// # Example
/// ```
/// use ferrous_align::align::{encode_sequence, reverse_complement_sequence};
///
/// let seq = b"ACG";  // Non-palindromic sequence
/// let encoded = encode_sequence(seq);  // [0, 1, 2]
/// let rc = reverse_complement_sequence(&encoded);  // CGT encoded as [1, 2, 3]
/// assert_eq!(rc, vec![1, 2, 3]);
/// ```
#[inline]
pub fn reverse_complement_sequence(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .map(|&code| reverse_complement_code(code))
        .collect()
}

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
}

/// SAM flag bit masks (SAM specification v1.6)
/// Used for setting and querying alignment flags in SAM/BAM format
pub mod sam_flags {
    pub const PAIRED: u16 = 0x1; // Template having multiple segments in sequencing
    pub const PROPER_PAIR: u16 = 0x2; // Each segment properly aligned according to the aligner
    pub const UNMAPPED: u16 = 0x4; // Segment unmapped
    pub const MATE_UNMAPPED: u16 = 0x8; // Next segment in the template unmapped
    pub const REVERSE: u16 = 0x10; // SEQ being reverse complemented
    pub const MATE_REVERSE: u16 = 0x20; // SEQ of the next segment in the template being reverse complemented
    pub const FIRST_IN_PAIR: u16 = 0x40; // The first segment in the template
    pub const SECOND_IN_PAIR: u16 = 0x80; // The last segment in the template
    pub const SECONDARY: u16 = 0x100; // Secondary alignment
    pub const QCFAIL: u16 = 0x200; // Not passing filters, such as platform/vendor quality controls
    pub const DUPLICATE: u16 = 0x400; // PCR or optical duplicate
    pub const SUPPLEMENTARY: u16 = 0x800; // Supplementary alignment
}

#[derive(Debug)]
pub struct Alignment {
    pub query_name: String,
    pub flag: u16, // SAM flag
    pub ref_name: String,
    pub ref_id: usize,         // Reference sequence ID (for paired-end scoring)
    pub pos: u64,              // 0-based leftmost mapping position
    pub mapq: u8,              // Mapping quality
    pub score: i32,            // Alignment score (for paired-end scoring)
    pub cigar: Vec<(u8, i32)>, // CIGAR string
    pub rnext: String,         // Ref. name of the mate/next read
    pub pnext: u64,            // Position of the mate/next read
    pub tlen: i32,             // Observed template length
    pub seq: String,           // Segment sequence
    pub qual: String,          // ASCII of Phred-scaled base quality+33
    // Optional SAM tags
    pub tags: Vec<(String, String)>, // Vector of (tag_name, tag_value) pairs
    // Internal fields for alignment selection (not output to SAM)
    pub(crate) query_start: i32,   // Query start position (0-based)
    pub(crate) query_end: i32,     // Query end position (exclusive)
    pub(crate) seed_coverage: i32, // Length of region covered by seeds (for MAPQ)
    pub(crate) hash: u64,          // Hash for deterministic tie-breaking
    pub(crate) frac_rep: f32,      // Fraction of repetitive seeds in this alignment
}

impl Alignment {
    /// Get CIGAR string as a formatted string (e.g., "50M2I48M")
    /// Returns "*" for empty CIGAR (unmapped reads per SAM spec)
    ///
    /// Per bwa-mem2 behavior (bwamem.cpp:1585-1586):
    /// - Soft clips (S) are converted to hard clips (H) for supplementary alignments
    /// - Primary and secondary alignments keep soft clips
    pub fn cigar_string(&self) -> String {
        if self.cigar.is_empty() {
            "*".to_string()
        } else {
            let is_supplementary = (self.flag & sam_flags::SUPPLEMENTARY) != 0;

            self.cigar
                .iter()
                .map(|&(op, len)| {
                    // Convert soft clip (S) to hard clip (H) for supplementary alignments
                    // Matches C++ bwa-mem2: c = which? 4 : 3 (where 3=S, 4=H)
                    let op_char = if is_supplementary && op == b'S' {
                        'H'
                    } else {
                        op as char
                    };
                    format!("{}{}", len, op_char)
                })
                .collect()
        }
    }

    /// Calculate the aligned length on the reference from CIGAR
    /// Sums M, D, N, =, X operations (operations that consume reference bases)
    pub fn reference_length(&self) -> i32 {
        self.cigar
            .iter()
            .filter_map(|&(op, len)| match op as char {
                'M' | 'D' | 'N' | '=' | 'X' => Some(len),
                _ => None,
            })
            .sum()
    }

    pub fn to_sam_string(&self) -> String {
        // Convert CIGAR to string format
        let cigar_string = self.cigar_string();

        // TODO: Clipping penalties (opt.pen_clip5, opt.pen_clip3, opt.pen_unpaired)
        // are used in C++ to adjust alignment scores, not SAM output directly.
        // They affect score calculation during alignment extension and pair scoring.
        // This requires deeper integration into the scoring logic in banded_swa.rs

        // Handle reverse complement for SEQ and QUAL if flag sam_flags::REVERSE (reverse strand) is set
        // Matching bwa-mem2 mem_aln2sam() behavior (bwamem.cpp:1706-1716)
        let (output_seq, output_qual) = if self.flag & sam_flags::REVERSE != 0 {
            // Reverse strand: reverse complement the sequence and reverse the quality
            let rev_comp_seq: String = self
                .seq
                .chars()
                .rev()
                .map(|c| match c {
                    'A' => 'T',
                    'T' => 'A',
                    'C' => 'G',
                    'G' => 'C',
                    'N' => 'N',
                    _ => c, // Keep any other characters as-is
                })
                .collect();
            let rev_qual: String = self.qual.chars().rev().collect();
            (rev_comp_seq, rev_qual)
        } else {
            // Forward strand: use sequence and quality as-is
            (self.seq.clone(), self.qual.clone())
        };

        // Format mandatory SAM fields
        let mut sam_line = format!(
            "{}	{}	{}	{}	{}	{}	{}	{}	{}	{}	{}",
            self.query_name,
            self.flag,
            self.ref_name,
            self.pos + 1, // SAM POS is 1-based
            self.mapq,
            cigar_string,
            self.rnext,
            self.pnext,
            self.tlen,
            output_seq,
            output_qual
        );

        // Append optional tags
        for (tag, value) in &self.tags {
            sam_line.push('\t');
            sam_line.push_str(tag);
            sam_line.push(':');
            sam_line.push_str(value);
        }

        sam_line
    }

    /// Calculate the aligned length on the query from CIGAR
    /// Sums M, I, S, =, X operations (operations that consume query bases)
    pub fn query_length(&self) -> i32 {
        self.cigar
            .iter()
            .filter_map(|&(op, len)| match op as char {
                'M' | 'I' | 'S' | '=' | 'X' => Some(len),
                _ => None,
            })
            .sum()
    }

    /// Calculate edit distance (NM tag) from CIGAR string
    /// NM = number of mismatches + insertions + deletions
    /// Approximation: counts I, D, X operations (M operations may contain mismatches)
    /// For exact NM, would need to compare query and reference sequences
    pub fn calculate_edit_distance(&self) -> i32 {
        self.cigar
            .iter()
            .filter_map(|&(op, len)| {
                match op as char {
                    'I' | 'D' | 'X' => Some(len), // Count indels and explicit mismatches
                    'M' => {
                        // M includes both matches and mismatches
                        // Approximate as 0 for now (would need sequence comparison for exact count)
                        // In practice, bwa-mem2 calculates this from alignment score
                        None
                    }
                    _ => None,
                }
            })
            .sum()
    }

    /// Calculate NM tag (edit distance) from alignment score
    /// This is a better approximation than calculate_edit_distance() since it
    /// estimates mismatches from the score, not just counting indels.
    ///
    /// Formula: NM = indels + estimated_mismatches
    /// Where: estimated_mismatches ≈ (perfect_score - actual_score) / mismatch_penalty
    ///        perfect_score = query_length * match_score (1)
    ///        mismatch_penalty = match_score (1) + mismatch_cost (4) = 5
    ///
    /// This will be replaced with exact NM calculation once MD tag is implemented.
    pub fn calculate_nm_from_score(&self) -> i32 {
        // Count indels from CIGAR
        let indels: i32 = self
            .cigar
            .iter()
            .filter_map(|&(op, len)| match op as char {
                'I' | 'D' => Some(len),
                _ => None,
            })
            .sum();

        // Estimate mismatches from score difference
        let query_len = self.query_length();
        let perfect_score = query_len * 1; // Match score = 1
        let score_diff = (perfect_score - self.score).max(0); // Clamp to non-negative
        let estimated_mismatches = (score_diff / 5).max(0); // Match(1) + Mismatch(4) = 5 penalty

        indels + estimated_mismatches
    }

    /// Generate MD tag from aligned sequences and CIGAR
    ///
    /// The MD tag is a string representing the reference bases at mismatching positions
    /// and deleted reference bases. Format:
    /// - Numbers: count of matching bases
    /// - Letters: mismatching reference base
    /// - ^LETTERS: deleted reference bases
    ///
    /// Examples:
    /// - "100" = 100 perfect matches
    /// - "50A49" = 50 matches, mismatch at ref base A, 49 matches
    /// - "25^AC25" = 25 matches, deletion of AC, 25 matches
    ///
    /// This requires ref_aligned and query_aligned sequences from Smith-Waterman traceback.
    pub fn generate_md_tag(
        ref_aligned: &[u8],
        query_aligned: &[u8],
        cigar: &[(u8, i32)],
    ) -> String {
        let mut md = String::new();
        let mut match_count = 0;

        // Base-to-char conversion for 2-bit encoding
        let base_to_char = |b: u8| -> char {
            match b {
                0 => 'A',
                1 => 'C',
                2 => 'G',
                3 => 'T',
                _ => 'N',
            }
        };

        let mut ref_idx = 0;
        let mut query_idx = 0;

        for &(op, len) in cigar {
            match op as char {
                'M' => {
                    // Match/mismatch - compare bases
                    for _ in 0..len {
                        if ref_idx >= ref_aligned.len() || query_idx >= query_aligned.len() {
                            break;
                        }

                        if ref_aligned[ref_idx] == query_aligned[query_idx] {
                            // Match
                            match_count += 1;
                        } else {
                            // Mismatch - emit match count, then mismatch base
                            if match_count > 0 {
                                md.push_str(&match_count.to_string());
                                match_count = 0;
                            }
                            md.push(base_to_char(ref_aligned[ref_idx]));
                        }

                        ref_idx += 1;
                        query_idx += 1;
                    }
                }
                'D' => {
                    // Deletion from reference - emit match count, then ^DELETED_BASES
                    if match_count > 0 {
                        md.push_str(&match_count.to_string());
                        match_count = 0;
                    }
                    md.push('^');
                    for _ in 0..len {
                        if ref_idx >= ref_aligned.len() {
                            break;
                        }
                        md.push(base_to_char(ref_aligned[ref_idx]));
                        ref_idx += 1;
                    }
                }
                'I' => {
                    // Insertion to query - skip query bases, no MD tag entry
                    query_idx += len as usize;
                }
                'S' | 'H' => {
                    // Soft/hard clip - skip query bases, no MD tag entry
                    query_idx += len as usize;
                }
                _ => {
                    // Unknown op - skip
                }
            }
        }

        // Emit final match count
        if match_count > 0 {
            md.push_str(&match_count.to_string());
        }

        // Handle empty MD tag (shouldn't happen, but be safe)
        if md.is_empty() {
            md.push('0');
        }

        md
    }

    /// Calculate exact NM (edit distance) from MD tag and CIGAR
    ///
    /// NM = mismatches + insertions + deletions
    ///
    /// Mismatches and deletions are counted from the MD tag:
    /// - Each letter (A, C, G, T, N) = 1 mismatch
    /// - Each ^LETTERS segment = length deletions
    ///
    /// Insertions are counted from the CIGAR string.
    pub fn calculate_exact_nm(md_tag: &str, cigar: &[(u8, i32)]) -> i32 {
        let mut nm = 0;

        // Count mismatches and deletions from MD tag
        let mut in_deletion = false;
        for ch in md_tag.chars() {
            if ch == '^' {
                in_deletion = true;
            } else if ch.is_ascii_digit() {
                in_deletion = false;
            } else {
                // It's a base letter (A, C, G, T, N)
                nm += 1;
            }
        }

        // Count insertions from CIGAR
        for &(op, len) in cigar {
            if op == b'I' {
                nm += len;
            }
        }

        nm
    }

    /// Generate XA tag entry for this alignment (alternative alignment format)
    /// Format: RNAME,STRAND+POS,CIGAR,NM
    /// Example: chr1,+1000,50M,2
    pub fn to_xa_entry(&self) -> String {
        let strand = if self.flag & sam_flags::REVERSE != 0 { '-' } else { '+' };
        let pos = self.pos + 1; // XA uses 1-based position
        let cigar = self.cigar_string();
        let nm = self.calculate_edit_distance();

        format!("{},{}{},{},{}", self.ref_name, strand, pos, cigar, nm)
    }

    /// Set paired-end SAM flags in a consistent manner
    /// Reduces code duplication and prevents flag-setting errors
    ///
    /// # Arguments
    /// * `is_first` - true if this is the first read in the pair (R1), false for second (R2)
    /// * `is_proper_pair` - true if the pair is properly aligned according to insert size
    /// * `mate_unmapped` - true if the mate is unmapped
    /// * `mate_reverse` - true if the mate is reverse-complemented
    ///
    /// # SAM Flags Set
    /// - Always sets PAIRED (sam_flags::PAIRED)
    /// - Sets FIRST_IN_PAIR (sam_flags::FIRST_IN_PAIR) or SECOND_IN_PAIR (sam_flags::SECOND_IN_PAIR) based on `is_first`
    /// - Conditionally sets PROPER_PAIR (sam_flags::PROPER_PAIR), MATE_UNMAPPED (sam_flags::MATE_UNMAPPED), MATE_REVERSE (sam_flags::MATE_REVERSE)
    ///
    /// # Example
    /// ```ignore
    /// use ferrous_align::align::{Alignment, sam_flags};
    ///
    /// # let mut alignment = Alignment::default();
    /// alignment.set_paired_flags(
    ///     true,   // first in pair
    ///     true,   // proper pair
    ///     false,  // mate mapped
    ///     false   // mate forward
    /// );
    /// assert_eq!(alignment.flag & sam_flags::PAIRED, sam_flags::PAIRED);
    /// assert_eq!(alignment.flag & sam_flags::FIRST_IN_PAIR, sam_flags::FIRST_IN_PAIR);
    /// assert_eq!(alignment.flag & sam_flags::PROPER_PAIR, sam_flags::PROPER_PAIR);
    /// ```
    #[inline]
    pub fn set_paired_flags(
        &mut self,
        is_first: bool,
        is_proper_pair: bool,
        mate_unmapped: bool,
        mate_reverse: bool,
    ) {
        use sam_flags::*;

        self.flag |= PAIRED;
        self.flag |= if is_first {
            FIRST_IN_PAIR
        } else {
            SECOND_IN_PAIR
        };

        if is_proper_pair {
            self.flag |= PROPER_PAIR;
        }
        if mate_unmapped {
            self.flag |= MATE_UNMAPPED;
        }
        if mate_reverse {
            self.flag |= MATE_REVERSE;
        }
    }

    /// Calculate template length (TLEN) for paired-end reads
    /// Returns signed TLEN according to SAM specification:
    /// - Positive for leftmost read (smaller coordinate)
    /// - Negative for rightmost read (larger coordinate)
    /// - 0 for unmapped or different references
    ///
    /// # Arguments
    /// * `mate_pos` - Mate's 0-based mapping position
    /// * `mate_ref_len` - Mate's reference length from CIGAR (aligned bases)
    ///
    /// # Formula
    /// - If this read is leftmost: `TLEN = (mate_pos - this_pos) + mate_ref_len`
    /// - If this read is rightmost: `TLEN = -((this_pos - mate_pos) + this_ref_len)`
    ///
    /// # SAM Specification Notes
    /// TLEN represents the signed observed template length. For reads on same reference:
    /// - Leftmost segment has positive TLEN
    /// - Rightmost segment has negative TLEN
    /// - Both segments should have equal magnitude
    /// - TLEN = rightmost_end - leftmost_start (outer coordinates)
    ///
    /// # Example
    /// ```
    /// use ferrous_align::align::Alignment;
    ///
    /// // Read1 at position 1000, 100bp aligned
    /// // Read2 at position 1200, 100bp aligned
    /// // Read1 TLEN = (1200 - 1000) + 100 = 300 (positive, leftmost)
    /// // Read2 TLEN = -((1200 - 1000) + 100) = -300 (negative, rightmost)
    /// ```
    #[inline]
    pub fn calculate_tlen(&self, mate_pos: u64, mate_ref_len: i32) -> i32 {
        let this_pos = self.pos as i64;
        let mate_pos = mate_pos as i64;

        if this_pos <= mate_pos {
            // This read is leftmost: positive TLEN
            // TLEN = (mate_start - this_start) + mate_length
            ((mate_pos - this_pos) + mate_ref_len as i64) as i32
        } else {
            // This read is rightmost: negative TLEN
            // TLEN = -((this_start - mate_start) + this_length)
            let this_ref_len = self.reference_length();
            -(((this_pos - mate_pos) + this_ref_len as i64) as i32)
        }
    }

    /// Create an unmapped alignment for paired-end reads
    /// Reduces 40+ lines of duplicate struct initialization code
    ///
    /// # Arguments
    /// * `query_name` - Read identifier
    /// * `seq` - Read sequence (will be converted to String)
    /// * `qual` - Quality scores string
    /// * `is_first_in_pair` - true for R1, false for R2
    /// * `mate_ref` - Mate reference name ("*" if mate also unmapped)
    /// * `mate_pos` - Mate position (0-based, ignored if mate unmapped)
    /// * `mate_is_reverse` - true if mate is reverse-complemented
    ///
    /// # SAM Flags Set
    /// - PAIRED (sam_flags::PAIRED) - always set
    /// - UNMAPPED (sam_flags::UNMAPPED) - always set
    /// - FIRST_IN_PAIR (sam_flags::FIRST_IN_PAIR) or SECOND_IN_PAIR (sam_flags::SECOND_IN_PAIR) - based on is_first_in_pair
    /// - MATE_REVERSE (sam_flags::MATE_REVERSE) - if mate_is_reverse is true
    ///
    /// # SAM Field Values
    /// - POS: mate position (or 0 if mate unmapped)
    /// - RNAME: mate reference (or "*" if mate unmapped)
    /// - RNEXT: "=" if mate mapped, "*" if unmapped
    /// - PNEXT: mate position + 1 (1-based) if mate mapped, 0 if unmapped
    /// - MAPQ: 0 (unmapped)
    /// - CIGAR: empty (unmapped)
    /// - TLEN: 0 (no template length for unmapped)
    ///
    /// # Example
    /// ```
    /// use ferrous_align::align::Alignment;
    ///
    /// let unmapped = Alignment::create_unmapped(
    ///     "read1".to_string(),
    ///     b"ACGTACGT",
    ///     "IIIIIIII".to_string(),
    ///     true,  // first in pair
    ///     "chr1", // mate reference
    ///     1000,   // mate position
    ///     false   // mate forward
    /// );
    /// assert_eq!(unmapped.mapq, 0);
    /// assert_ne!(unmapped.flag & sam_flags::UNMAPPED, 0); // UNMAPPED flag
    /// ```
    pub fn create_unmapped(
        query_name: String,
        seq: &[u8],
        qual: String,
        is_first_in_pair: bool,
        mate_ref: &str,
        mate_pos: u64,
        mate_is_reverse: bool,
    ) -> Self {
        use sam_flags::*;

        let mut flag = PAIRED | UNMAPPED;
        flag |= if is_first_in_pair {
            FIRST_IN_PAIR
        } else {
            SECOND_IN_PAIR
        };

        let (ref_name, pos, rnext, pnext) = if mate_ref != "*" {
            // Mate is mapped: use mate's position and reference
            (
                mate_ref.to_string(),
                mate_pos,
                "=".to_string(),
                mate_pos + 1,
            )
        } else {
            // Mate is also unmapped
            ("*".to_string(), 0, "*".to_string(), 0)
        };

        if mate_is_reverse {
            flag |= MATE_REVERSE;
        }

        Self {
            query_name,
            flag,
            ref_name,
            ref_id: 0,
            pos,
            mapq: 0,
            score: 0,
            cigar: Vec::new(),
            rnext,
            pnext,
            tlen: 0,
            seq: String::from_utf8_lossy(seq).to_string(),
            qual,
            tags: vec![
                ("AS".to_string(), "i:0".to_string()),
                ("NM".to_string(), "i:0".to_string()),
            ],
            // Internal fields - default for unmapped
            query_start: 0,
            query_end: seq.len() as i32,
            hash: 0,
            seed_coverage: 0,
            frac_rep: 0.0,
        }
    }
}

// ============================================================================
// ALIGNMENT SCORING AND QUALITY ASSESSMENT
// ============================================================================
//
// This section contains functions for:
// - Overlap detection between alignments
// - Chain scoring and filtering
// - MAPQ (mapping quality) calculation
// - Secondary alignment marking
// - Divergence estimation
//
// These functions implement the core scoring logic from C++ bwa-mem2
// (bwamem.cpp, bwamem_pair.cpp)
// ============================================================================

// ----------------------------------------------------------------------------
// Overlap Detection
// ----------------------------------------------------------------------------

/// Check if two alignments overlap significantly on the query sequence
/// Returns true if overlap >= mask_level * min_alignment_length
/// Implements C++ mem_mark_primary_se_core logic (bwamem.cpp:1392-1418)
fn alignments_overlap(a1: &Alignment, a2: &Alignment, mask_level: f32) -> bool {
    // Calculate query bounds
    let a1_qb = a1.query_start;
    let a1_qe = a1.query_end;
    let a2_qb = a2.query_start;
    let a2_qe = a2.query_end;

    // Find overlap region
    let b_max = a1_qb.max(a2_qb);
    let e_min = a1_qe.min(a2_qe);

    if e_min <= b_max {
        return false; // No overlap
    }

    let overlap = e_min - b_max;
    let min_len = (a1_qe - a1_qb).min(a2_qe - a2_qb);

    // Overlap is significant if >= mask_level * min_length
    overlap >= (min_len as f32 * mask_level) as i32
}

// ----------------------------------------------------------------------------
// MAPQ (Mapping Quality) Calculation
// ----------------------------------------------------------------------------

/// Calculate MAPQ (mapping quality) for an alignment
/// Implements C++ mem_approx_mapq_se (bwamem.cpp:1470-1494)
fn calculate_mapq(
    score: i32,
    sub_score: i32,
    seed_coverage: i32,
    sub_count: i32,
    match_score: i32,
    mismatch_penalty: i32,
    frac_rep: f32,
    opt: &MemOpt,
) -> u8 {
    const MEM_MAPQ_COEF: f64 = 30.0;
    const MEM_MAPQ_MAX: u8 = 60;

    // Use sub_score, or min_seed_len * match_score as minimum
    let sub = if sub_score > 0 {
        sub_score
    } else {
        opt.min_seed_len * match_score
    };

    // Return 0 if secondary hit is >= best hit
    if sub >= score {
        return 0;
    }

    // Calculate alignment length (approximate from seed coverage)
    let l = seed_coverage;

    // Calculate sequence identity
    // identity = 1 - (l * a - score) / (a + b) / l
    let identity = 1.0
        - ((l * match_score - score) as f64)
            / ((match_score + mismatch_penalty) as f64)
            / (l as f64);

    if score == 0 {
        return 0;
    }

    let mut mapq: i32;

    if opt.mapq_coef_len > 0.0 {
        let mut tmp_val: f64;
        if (l as f64) < opt.mapq_coef_len as f64 {
            tmp_val = 1.0;
        } else {
            tmp_val = opt.mapq_coef_fac as f64 / (l as f64).ln();
        }
        tmp_val *= identity * identity;
        mapq =
            (6.02 * (score - sub) as f64 / match_score as f64 * tmp_val * tmp_val + 0.499) as i32;
    } else {
        // Traditional MAPQ formula (default when mapQ_coef_len = 0)
        // mapq = 30.0 * (1 - sub/score) * ln(seed_coverage)
        mapq = (MEM_MAPQ_COEF * (1.0 - (sub as f64) / (score as f64)) * (seed_coverage as f64).ln()
            + 0.499) as i32;

        // Apply identity penalty if < 95%
        if identity < 0.95 {
            mapq = (mapq as f64 * identity * identity + 0.499) as i32;
        }
    }

    // Penalty for multiple suboptimal alignments
    // mapq -= ln(sub_n+1) * 4.343
    if sub_count > 0 {
        mapq -= (((sub_count + 1) as f64).ln() * 4.343) as i32;
    }

    // Apply repetitive region penalty (missing in Rust previously)
    // mapq = mapq * (1. - frac_rep)
    mapq = (mapq as f64 * (1.0 - frac_rep as f64) + 0.499) as i32;

    // Cap at max and floor at 0
    if mapq > MEM_MAPQ_MAX as i32 {
        mapq = MEM_MAPQ_MAX as i32;
    }
    if mapq < 0 {
        mapq = 0;
    }

    mapq as u8
}

// ----------------------------------------------------------------------------
// Chain Scoring and Filtering
// ----------------------------------------------------------------------------

/// Calculate chain weight based on seed coverage
/// Implements C++ mem_chain_weight (bwamem.cpp:429-448)
///
/// Weight = minimum of query coverage and reference coverage
/// This accounts for non-overlapping seed lengths in the chain
fn calculate_chain_weight(chain: &Chain, seeds: &[Seed], opt: &MemOpt) -> (i32, i32) {
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

/// Filter chains using drop_ratio and score thresholds
/// Implements C++ mem_chain_flt (bwamem.cpp:506-624)
///
/// Algorithm:
/// 1. Calculate weight for each chain
/// 2. Sort chains by weight (descending)
/// 3. Filter by min_chain_weight
/// 4. Apply drop_ratio: keep chains with weight >= best_weight * drop_ratio
/// 5. Mark overlapping chains as kept=1/2, non-overlapping as kept=3
fn filter_chains(
    chains: &mut Vec<Chain>,
    seeds: &[Seed],
    opt: &MemOpt,
    query_length: i32,
) -> Vec<Chain> {
    if chains.is_empty() {
        return Vec::new();
    }

    // Calculate weights for all chains
    for chain in chains.iter_mut() {
        let (weight, l_rep) = calculate_chain_weight(chain, seeds, opt);
        chain.weight = weight;
        // Calculate frac_rep = l_rep / query_length
        chain.frac_rep = if query_length > 0 {
            l_rep as f32 / query_length as f32
        } else {
            0.0
        };
        chain.kept = 0; // Initially mark as discarded
    }

    // Sort chains by weight (descending)
    chains.sort_by(|a, b| b.weight.cmp(&a.weight));

    // Filter by minimum weight
    let mut kept_chains: Vec<Chain> = Vec::new();

    for i in 0..chains.len() {
        let chain = &chains[i];

        // Skip if below minimum weight
        if chain.weight < opt.min_chain_weight {
            continue;
        }

        // Check overlap with already-kept chains (matching C++ bwamem.cpp:568-589)
        // IMPORTANT: drop_ratio only applies to OVERLAPPING chains, not all chains
        let mut overlaps = false;
        let mut should_discard = false;
        let mut chain_copy = chain.clone();

        for kept_chain in &kept_chains {
            // Check if chains overlap on query
            let qb_max = chain.query_start.max(kept_chain.query_start);
            let qe_min = chain.query_end.min(kept_chain.query_end);

            if qe_min > qb_max {
                // Chains overlap on query
                let overlap = qe_min - qb_max;
                let min_len = (chain.query_end - chain.query_start)
                    .min(kept_chain.query_end - kept_chain.query_start);

                // Check if overlap is significant
                if overlap >= (min_len as f32 * opt.mask_level) as i32 {
                    overlaps = true;
                    chain_copy.kept = 1; // Shadowed by better chain

                    // C++ bwamem.cpp:580-581: Apply drop_ratio ONLY for overlapping chains
                    // Drop if weight < kept_weight * drop_ratio AND difference >= 2 * min_seed_len
                    let weight_threshold = (kept_chain.weight as f32 * opt.drop_ratio) as i32;
                    let weight_diff = kept_chain.weight - chain.weight;

                    if chain.weight < weight_threshold && weight_diff >= (opt.min_seed_len << 1) {
                        log::debug!(
                            "Chain {} dropped: overlaps with kept chain, weight={} < threshold={} (kept_weight={} * drop_ratio={})",
                            i,
                            chain.weight,
                            weight_threshold,
                            kept_chain.weight,
                            opt.drop_ratio
                        );
                        should_discard = true;
                        break;
                    }
                    break;
                }
            }
        }

        // Skip discarded chains
        if should_discard {
            continue;
        }

        // Non-overlapping chains are always kept (C++ line 588: kept = large_ovlp? 2 : 3)
        if !overlaps {
            chain_copy.kept = 3; // Primary chain (no overlap)
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
// Secondary Alignment Marking and XA Tags
// ----------------------------------------------------------------------------

/// Generate XA tag for alternative alignments
/// Implements C++ mem_gen_alt (bwamem_extra.cpp:130-183)
///
/// Algorithm:
/// 1. Find secondary alignments for each primary
/// 2. Filter by score: secondary_score >= primary_score * xa_drop_ratio
/// 3. Limit count: max_xa_hits (5) or max_xa_hits_alt (200)
/// 4. Format as XA:Z:chr1,+100,50M,2;chr2,-200,48M1D2M,3;
///
/// Returns: HashMap<read_name, XA_tag_string>
fn generate_xa_tags(
    alignments: &[Alignment],
    opt: &MemOpt,
) -> std::collections::HashMap<String, String> {
    use std::collections::HashMap;

    let mut xa_tags: HashMap<String, String> = HashMap::new();

    if alignments.is_empty() {
        return xa_tags;
    }

    // Group alignments by query name
    let mut by_read: HashMap<String, Vec<&Alignment>> = HashMap::new();
    for aln in alignments {
        by_read
            .entry(aln.query_name.clone())
            .or_insert_with(Vec::new)
            .push(aln);
    }

    // For each read, generate XA tag from secondary alignments
    for (read_name, read_alns) in by_read.iter() {
        // Find primary alignment (flag & sam_flags::SECONDARY == 0)
        let primary = read_alns.iter().find(|a| a.flag & sam_flags::SECONDARY == 0);

        if primary.is_none() {
            continue; // No primary alignment
        }

        let primary_score = primary.unwrap().score;
        let xa_threshold = (primary_score as f32 * opt.xa_drop_ratio) as i32;

        // Collect secondary alignments that pass score threshold
        let mut secondaries: Vec<&Alignment> = read_alns
            .iter()
            .filter(|a| {
                (a.flag & sam_flags::SECONDARY != 0) && // Is secondary
                (a.score >= xa_threshold) // Score passes threshold
            })
            .cloned()
            .collect();

        if secondaries.is_empty() {
            continue; // No qualifying secondaries
        }

        // Sort secondaries by score (descending) for consistent output
        secondaries.sort_by(|a, b| b.score.cmp(&a.score));

        // Apply hit limit (max_xa_hits or max_xa_hits_alt)
        // TODO: Check if any alignment is to ALT contig for max_xa_hits_alt
        let max_hits = opt.max_xa_hits as usize;
        if secondaries.len() > max_hits {
            secondaries.truncate(max_hits);
        }

        // Format as XA tag: XA:Z:chr1,+100,50M,2;chr2,-200,48M,3;
        let xa_entries: Vec<String> = secondaries.iter().map(|aln| aln.to_xa_entry()).collect();

        if !xa_entries.is_empty() {
            // Return just the value portion (without XA:Z: prefix)
            // to_sam_string() will add the tag name and type
            let xa_tag = format!("Z:{};", xa_entries.join(";"));
            xa_tags.insert(read_name.clone(), xa_tag);
        }
    }

    log::debug!(
        "Generated XA tags for {} reads (from {} total alignments, xa_drop_ratio={}, max_xa_hits={})",
        xa_tags.len(),
        alignments.len(),
        opt.xa_drop_ratio,
        opt.max_xa_hits
    );

    xa_tags
}

///
/// Algorithm:
/// 1. Only generate SA tags for non-secondary alignments (flag & sam_flags::SECONDARY == 0)
/// 2. Include all OTHER non-secondary alignments for the same read
/// 3. Skip if only one non-secondary alignment exists (no supplementary)
///
/// Returns: HashMap<read_name, SA_tag_string>
fn generate_sa_tags(alignments: &[Alignment]) -> std::collections::HashMap<String, String> {
    use std::collections::HashMap;

    let mut all_sa_tags: HashMap<String, String> = HashMap::new();

    if alignments.is_empty() {
        return all_sa_tags;
    }

    // Group alignments by query name
    let mut alignments_by_read: HashMap<String, Vec<&Alignment>> = HashMap::new();
    for aln in alignments {
        alignments_by_read
            .entry(aln.query_name.clone())
            .or_insert_with(Vec::new)
            .push(aln);
    }

    // Generate SA tags for reads with multiple non-secondary alignments
    for (read_name, read_alns) in alignments_by_read.iter() {
        // Collect all non-secondary alignments for this read
        let non_secondary_alns: Vec<&Alignment> = read_alns
            .iter()
            .filter(|a| (a.flag & sam_flags::SECONDARY) == 0) // Exclude secondary alignments
            .cloned()
            .collect();

        // Only generate SA tag if there are 2+ non-secondary alignments (chimeric/split-read)
        if non_secondary_alns.len() < 2 {
            continue;
        }

        // Sort non-secondary alignments for deterministic SA tag output
        let mut sorted_alns = non_secondary_alns.clone();
        sorted_alns.sort_by(|a, b| {
            a.ref_name
                .cmp(&b.ref_name)
                .then_with(|| a.pos.cmp(&b.pos))
                .then_with(|| (a.flag & sam_flags::REVERSE).cmp(&(b.flag & sam_flags::REVERSE)))
        });

        let mut sa_parts: Vec<String> = Vec::new();
        for aln in sorted_alns.iter() {
            // Extract NM tag value (edit distance)
            let nm_value = aln
                .tags
                .iter()
                .find(|(tag_name, _)| tag_name == "NM")
                .map(|(_, val)| val.clone())
                .unwrap_or_else(|| "i:0".to_string())
                .strip_prefix("i:")
                .unwrap_or("0")
                .parse::<i32>()
                .unwrap_or(0);

            // Format: rname,pos,strand,CIGAR,mapQ,NM;
            // Note: pos is 1-based in SAM
            let strand = if (aln.flag & sam_flags::REVERSE) != 0 {
                '-'
            } else {
                '+'
            };
            sa_parts.push(format!(
                "{},{},{},{},{},{}",
                aln.ref_name,
                aln.pos + 1, // SAM is 1-based
                strand,
                aln.cigar_string(),
                aln.mapq,
                nm_value
            ));
        }

        if !sa_parts.is_empty() {
            let sa_tag_value = format!("Z:{};", sa_parts.join(";"));
            all_sa_tags.insert(read_name.clone(), sa_tag_value);
        }
    }

    log::debug!(
        "Generated SA tags for {} reads (chimeric/split-read)",
        all_sa_tags.len()
    );

    all_sa_tags
}

/// Mark secondary alignments and calculate MAPQ values
/// Implements C++ mem_mark_primary_se (bwamem.cpp:1420-1464)
///
/// Algorithm:
/// 1. Sort alignments by score (descending), then by hash
/// 2. For each alignment, check if it overlaps significantly with higher-scoring alignments
/// 3. If overlap >= mask_level, mark as secondary (set sam_flags::SECONDARY flag)
/// 4. Calculate MAPQ based on score difference and overlap count
fn mark_secondary_alignments(alignments: &mut Vec<Alignment>, opt: &MemOpt) {
    if alignments.is_empty() {
        return;
    }

    let mask_level = opt.mask_level;

    // Calculate score gap threshold for sub_count
    // tmp = max(a+b, o_del+e_del, o_ins+e_ins)
    let mut tmp = opt.a + opt.b;
    tmp = tmp.max(opt.o_del + opt.e_del);
    tmp = tmp.max(opt.o_ins + opt.e_ins);

    // Track which alignments are primary (not marked secondary)
    let mut primary_indices: Vec<usize> = Vec::new();
    primary_indices.push(0); // First alignment is always primary

    // Track sub-scores and sub-counts for MAPQ calculation
    let mut sub_scores: Vec<i32> = vec![0; alignments.len()];
    let mut sub_counts: Vec<i32> = vec![0; alignments.len()];

    // Check each alignment against existing primaries for overlap
    for i in 1..alignments.len() {
        let mut is_secondary = false;

        for &j in &primary_indices {
            if alignments_overlap(&alignments[i], &alignments[j], mask_level) {
                // Track sub-score for MAPQ calculation
                if sub_scores[j] == 0 {
                    sub_scores[j] = alignments[i].score;
                }

                // Count close suboptimal hits for MAPQ penalty
                if alignments[j].score - alignments[i].score <= tmp {
                    sub_counts[j] += 1;
                }

                // Mark as secondary
                alignments[i].flag |= sam_flags::SECONDARY;
                is_secondary = true;
                break;
            }
        }

        // If no overlap with any primary, add as new primary
        if !is_secondary {
            primary_indices.push(i);
        }
    }

    // Mark non-overlapping alignments after the first as SUPPLEMENTARY
    // Implements C++ bwa-mem2 logic (bwamem.cpp:1551-1552)
    // First non-overlapping alignment = PRIMARY (no flags)
    // Subsequent non-overlapping alignments = SUPPLEMENTARY (sam_flags::SUPPLEMENTARY)
    for (idx, &i) in primary_indices.iter().enumerate() {
        if idx > 0 {
            // Not the first non-overlapping alignment
            alignments[i].flag |= sam_flags::SUPPLEMENTARY;
            log::debug!(
                "Marked alignment {} as SUPPLEMENTARY ({}:{}, score={})",
                i,
                alignments[i].ref_name,
                alignments[i].pos,
                alignments[i].score
            );
        }
    }

    // Ensure the designated primary alignment does not have the SUPPLEMENTARY flag set.
    // The first alignment in `primary_indices` is considered the primary candidate.
    if !primary_indices.is_empty() {
        let primary_candidate_idx = primary_indices[0];
        // Clear the SUPPLEMENTARY flag (sam_flags::SUPPLEMENTARY) for the primary candidate
        alignments[primary_candidate_idx].flag &= !sam_flags::SUPPLEMENTARY;
        log::debug!(
            "Cleared SUPPLEMENTARY flag for primary candidate alignment {} ({}:{}, score={})",
            primary_candidate_idx,
            alignments[primary_candidate_idx].ref_name,
            alignments[primary_candidate_idx].pos,
            alignments[primary_candidate_idx].score
        );
    }

    // Calculate MAPQ for all alignments
    for i in 0..alignments.len() {
        if alignments[i].flag & sam_flags::SECONDARY == 0 {
            // Primary alignment: calculate MAPQ
            alignments[i].mapq = calculate_mapq(
                alignments[i].score,
                sub_scores[i],
                alignments[i].seed_coverage,
                sub_counts[i],
                opt.a,
                opt.b,
                alignments[i].frac_rep,
                opt,
            );
        } else {
            // Secondary alignment: MAPQ = 0
            alignments[i].mapq = 0;
        }
    }

    // Add XS tags (suboptimal alignment score) to primary alignments
    // XS should only be present if there's a secondary alignment
    for i in 0..alignments.len() {
        if alignments[i].flag & sam_flags::SECONDARY == 0 && sub_scores[i] > 0 {
            // Primary alignment with a suboptimal score
            alignments[i]
                .tags
                .push(("XS".to_string(), format!("i:{}", sub_scores[i])));
        }
    }
}

// ============================================================================
// SMITH-WATERMAN ALIGNMENT EXECUTION
// ============================================================================
//
// This section contains structures and functions for executing Smith-Waterman
// alignment with SIMD optimization and adaptive batch sizing
// ============================================================================

// ----------------------------------------------------------------------------
// Alignment Job Structure and Divergence Estimation
// ----------------------------------------------------------------------------

// Structure to hold alignment job for batching
#[derive(Clone)]
#[cfg_attr(test, derive(Debug))]
pub(crate) struct AlignmentJob {
    #[allow(dead_code)] // Used for tracking but not currently read
    pub seed_idx: usize,
    pub query: Vec<u8>,
    pub target: Vec<u8>,
    pub band_width: i32,
    // C++ STRATEGY: Track query offset for seed-boundary-based alignment
    // Only align seed-covered region, soft-clip the rest like bwa-mem2
    pub query_offset: i32, // Offset of query in full read (for soft-clipping)
    /// Extension direction: LEFT (seed → qb=0) or RIGHT (seed → qe=qlen)
    /// Used for separate left/right extensions matching C++ bwa-mem2
    /// None = legacy single-pass mode (deprecated)
    pub direction: Option<crate::banded_swa::ExtensionDirection>,
    /// Seed length for calculating h0 (initial score = seed_len * match_score)
    /// C++ bwamem.cpp:2232 sets h0 = s->len * opt->a
    pub seed_len: i32,
}

/// Estimate divergence likelihood based on sequence length mismatch
///
/// Returns a score from 0.0 (low divergence) to 1.0 (high divergence)
/// based on the ratio of length mismatch to total length.
///
/// **Heuristic**: Sequences with significant length differences are likely
/// to have insertions/deletions, indicating higher divergence.
fn estimate_divergence_score(query_len: usize, target_len: usize) -> f64 {
    let max_len = query_len.max(target_len);
    let min_len = query_len.min(target_len);

    if max_len == 0 {
        return 0.0;
    }

    // Length mismatch ratio (0.0 = identical length, 1.0 = one sequence is empty)
    let length_mismatch = (max_len - min_len) as f64 / max_len as f64;

    // Scale: 0-10% mismatch → low divergence (0.0-0.3)
    //        10-30% mismatch → medium divergence (0.3-0.7)
    //        30%+ mismatch → high divergence (0.7-1.0)
    (length_mismatch * 2.5).min(1.0)
}

/// Determine optimal batch size based on estimated divergence and SIMD engine
///
/// **Strategy**:
/// - Detect available SIMD engine (SSE2/AVX2/AVX-512)
/// - Low divergence (score < 0.3): Use engine's maximum batch size for efficiency
/// - Medium divergence (0.3-0.7): Use engine's standard batch size
/// - High divergence (> 0.7): Use smaller batches or route to scalar
///
/// **SIMD Engine Batch Sizes**:
/// - SSE2/NEON (128-bit): 16-way parallelism
/// - AVX2 (256-bit): 32-way parallelism
/// - AVX-512 (512-bit): 64-way parallelism
///
/// This reduces batch synchronization penalty for divergent sequences while
/// maximizing SIMD utilization for similar sequences.
fn determine_optimal_batch_size(jobs: &[AlignmentJob]) -> usize {
    use crate::simd::{detect_optimal_simd_engine, get_simd_batch_sizes};

    if jobs.is_empty() {
        return 16; // Default
    }

    // Detect SIMD engine and get optimal batch sizes
    let engine = detect_optimal_simd_engine();
    let (max_batch, standard_batch) = get_simd_batch_sizes(engine);

    // Calculate average divergence score for this batch of jobs
    let total_divergence: f64 = jobs
        .iter()
        .map(|job| estimate_divergence_score(job.query.len(), job.target.len()))
        .sum();

    let avg_divergence = total_divergence / jobs.len() as f64;

    // Adaptive batch sizing based on divergence and SIMD capability
    if avg_divergence < 0.3 {
        // Low divergence: Use engine's maximum batch size for best SIMD utilization
        max_batch
    } else if avg_divergence < 0.7 {
        // Medium divergence: Use engine's standard batch size
        standard_batch
    } else {
        // High divergence: Use smaller batches to reduce synchronization penalty
        // Use half of standard batch, minimum 8
        (standard_batch / 2).max(8)
    }
}

/// Classify jobs as low-divergence or high-divergence for routing
///
/// Returns (low_divergence_jobs, high_divergence_jobs)
fn partition_jobs_by_divergence(jobs: &[AlignmentJob]) -> (Vec<AlignmentJob>, Vec<AlignmentJob>) {
    const DIVERGENCE_THRESHOLD: f64 = 0.7; // Route to scalar if > 0.7

    let mut low_div = Vec::new();
    let mut high_div = Vec::new();

    for job in jobs {
        let div_score = estimate_divergence_score(job.query.len(), job.target.len());
        if div_score > DIVERGENCE_THRESHOLD {
            high_div.push(job.clone());
        } else {
            low_div.push(job.clone());
        }
    }

    (low_div, high_div)
}

/// Execute alignments using batched SIMD (processes up to 16 at a time)
/// Now includes CIGAR generation via hybrid approach
/// NOTE: This function is deprecated - use execute_adaptive_alignments instead
pub(crate) fn execute_batched_alignments(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)> {
    const BATCH_SIZE: usize = 16;
    let mut all_results = vec![(0, Vec::new(), Vec::new(), Vec::new()); jobs.len()];

    // Process jobs in batches of 16
    for batch_start in (0..jobs.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(jobs.len());
        let batch_jobs = &jobs[batch_start..batch_end];

        // Prepare batch data for SIMD dispatch
        // CRITICAL: h0 must be seed_len, not 0 (C++ bwamem.cpp:2232)
        // CRITICAL: Include direction for LEFT/RIGHT extension (fixes insertion detection bug)
        let batch_data: Vec<(
            i32,
            Vec<u8>,
            i32,
            Vec<u8>,
            i32,
            i32,
            Option<crate::banded_swa::ExtensionDirection>,
        )> = batch_jobs
            .iter()
            .map(|job| {
                // For LEFT extension: reverse both query and target (C++ bwamem.cpp:2278)
                let (query, target) =
                    if job.direction == Some(crate::banded_swa::ExtensionDirection::Left) {
                        (
                            job.query.iter().copied().rev().collect(),
                            job.target.iter().copied().rev().collect(),
                        )
                    } else {
                        (job.query.clone(), job.target.clone())
                    };

                (
                    query.len() as i32,
                    query,
                    target.len() as i32,
                    target,
                    job.band_width,
                    job.seed_len, // h0 = seed_len (initial score from existing seed)
                    job.direction,
                )
            })
            .collect();

        // Execute batched alignment with CIGAR generation
        // Use dispatch to automatically route to optimal SIMD width (16/32/64)
        let results = sw_params.simd_banded_swa_dispatch_with_cigar(&batch_data);

        // Extract scores, CIGARs, and aligned sequences from results
        for (i, result) in results.iter().enumerate() {
            if i < batch_jobs.len() {
                all_results[batch_start + i] = (
                    result.score.score,
                    result.cigar.clone(),
                    result.ref_aligned.clone(),
                    result.query_aligned.clone(),
                );
            }
        }
    }

    all_results
}

/// Execute alignments with adaptive strategy selection
///
/// **Hybrid Approach**:
/// 1. Partition jobs into low-divergence and high-divergence based on length mismatch
/// 2. Route high-divergence jobs (>70% length mismatch) to scalar processing
/// 3. Route low-divergence jobs to batched SIMD with adaptive batch sizing
///
/// **Performance Benefits**:
/// - High-divergence sequences avoid SIMD overhead and batch synchronization penalty
/// - Low-divergence sequences use optimal batch sizes for their characteristics
/// - Expected 15-25% improvement over fixed batching strategy
pub(crate) fn execute_adaptive_alignments(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)> {
    if jobs.is_empty() {
        return Vec::new();
    }

    // SIMD routing: Use SIMD for all jobs (length-based heuristic was flawed)
    // The previous divergence-based routing incorrectly compared query vs target lengths,
    // but target is always larger due to alignment margins (~100bp each side).
    // This caused ALL jobs to route to scalar (0% SIMD utilization).
    // SIMD handles variable lengths efficiently with padding, so just use it for everything.
    let (low_div_jobs, high_div_jobs) = (jobs.to_vec(), Vec::new());

    // Calculate average divergence and length statistics for logging
    let avg_divergence: f64 = jobs
        .iter()
        .map(|job| estimate_divergence_score(job.query.len(), job.target.len()))
        .sum::<f64>()
        / jobs.len() as f64;

    let avg_query_len: f64 =
        jobs.iter().map(|job| job.query.len()).sum::<usize>() as f64 / jobs.len() as f64;
    let avg_target_len: f64 =
        jobs.iter().map(|job| job.target.len()).sum::<usize>() as f64 / jobs.len() as f64;

    // Log routing statistics (DEBUG level - too verbose for INFO)
    log::debug!(
        "Adaptive routing: {} total jobs, {} scalar ({:.1}%), {} SIMD ({:.1}%), avg_divergence={:.3}",
        jobs.len(),
        high_div_jobs.len(),
        high_div_jobs.len() as f64 / jobs.len() as f64 * 100.0,
        low_div_jobs.len(),
        low_div_jobs.len() as f64 / jobs.len() as f64 * 100.0,
        avg_divergence
    );

    // Show length statistics to understand why routing fails
    log::debug!(
        "  → avg_query={:.1}bp, avg_target={:.1}bp, length_diff={:.1}%",
        avg_query_len,
        avg_target_len,
        ((avg_query_len - avg_target_len).abs() / avg_query_len.max(avg_target_len) * 100.0)
    );

    // Create result vector with correct size
    let mut all_results = vec![(0, Vec::new(), Vec::new(), Vec::new()); jobs.len()];

    // Process high-divergence jobs with scalar (more efficient for divergent sequences)
    let high_div_results = if !high_div_jobs.is_empty() {
        log::debug!(
            "Processing {} high-divergence jobs with scalar",
            high_div_jobs.len()
        );
        execute_scalar_alignments(sw_params, &high_div_jobs)
    } else {
        Vec::new()
    };

    // Process low-divergence jobs with adaptive batched SIMD
    let low_div_results = if !low_div_jobs.is_empty() {
        let optimal_batch_size = determine_optimal_batch_size(&low_div_jobs);
        log::debug!(
            "Processing {} low-divergence jobs with SIMD (batch_size={})",
            low_div_jobs.len(),
            optimal_batch_size
        );
        execute_batched_alignments_with_size(sw_params, &low_div_jobs, optimal_batch_size)
    } else {
        Vec::new()
    };

    // Since we route everything to SIMD now (high_div_jobs is always empty),
    // just return the SIMD results directly
    if high_div_results.is_empty() {
        // All jobs went to SIMD - return directly (common case now)
        low_div_results
    } else {
        // Mixed routing (should not happen with current logic, but keep for safety)
        let mut low_idx = 0;
        let mut high_idx = 0;

        for (original_idx, job) in jobs.iter().enumerate() {
            let div_score = estimate_divergence_score(job.query.len(), job.target.len());
            let result = if div_score > 0.7 && high_idx < high_div_results.len() {
                // High divergence - get from scalar results
                let res = high_div_results[high_idx].clone();
                high_idx += 1;
                res
            } else {
                // Low divergence - get from SIMD results
                let res = low_div_results[low_idx].clone();
                low_idx += 1;
                res
            };

            all_results[original_idx] = result;
        }
        all_results
    }
}

/// Execute batched alignments with configurable batch size
/// Uses SIMD dispatch to automatically select optimal engine (SSE2/AVX2/AVX-512)
fn execute_batched_alignments_with_size(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
    batch_size: usize,
) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)> {
    let mut all_results = vec![(0, Vec::new(), Vec::new(), Vec::new()); jobs.len()];

    // Process jobs in batches of specified size
    for batch_start in (0..jobs.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(jobs.len());
        let batch_jobs = &jobs[batch_start..batch_end];

        // Prepare batch data
        // CRITICAL: h0 must be seed_len, not 0 (C++ bwamem.cpp:2232)
        // CRITICAL: Include direction for LEFT/RIGHT extension (fixes insertion detection bug)
        let batch_data: Vec<(
            i32,
            Vec<u8>,
            i32,
            Vec<u8>,
            i32,
            i32,
            Option<crate::banded_swa::ExtensionDirection>,
        )> = batch_jobs
            .iter()
            .map(|job| {
                // For LEFT extension: reverse both query and target (C++ bwamem.cpp:2278)
                let (query, target) =
                    if job.direction == Some(crate::banded_swa::ExtensionDirection::Left) {
                        (
                            job.query.iter().copied().rev().collect(),
                            job.target.iter().copied().rev().collect(),
                        )
                    } else {
                        (job.query.clone(), job.target.clone())
                    };

                (
                    query.len() as i32,
                    query,
                    target.len() as i32,
                    target,
                    job.band_width,
                    job.seed_len, // h0 = seed_len (initial score from existing seed)
                    job.direction,
                )
            })
            .collect();

        // Execute batched alignment with CIGAR generation
        // Dispatch automatically selects batch16/32/64 based on detected SIMD engine
        let results = sw_params.simd_banded_swa_dispatch_with_cigar(&batch_data);

        // Extract scores, CIGARs, and aligned sequences from results
        for (i, result) in results.iter().enumerate() {
            if i < batch_jobs.len() {
                all_results[batch_start + i] = (
                    result.score.score,
                    result.cigar.clone(),
                    result.ref_aligned.clone(),
                    result.query_aligned.clone(),
                );

                // Detect pathological CIGARs in SIMD path
                let total_insertions: i32 = result
                    .cigar
                    .iter()
                    .filter(|(op, _)| *op == b'I')
                    .map(|(_, count)| count)
                    .sum();
                let total_deletions: i32 = result
                    .cigar
                    .iter()
                    .filter(|(op, _)| *op == b'D')
                    .map(|(_, count)| count)
                    .sum();

                if total_insertions > 10 || total_deletions > 5 {
                    let job = &batch_jobs[i];
                    // ATOMIC LOG: All data in single statement to avoid multi-threaded interleaving
                    log::debug!(
                        "PATHOLOGICAL_CIGAR_SIMD|idx={}|qlen={}|tlen={}|bw={}|score={}|ins={}|del={}|CIGAR={:?}|QUERY={:?}|TARGET={:?}",
                        batch_start + i,
                        job.query.len(),
                        job.target.len(),
                        job.band_width,
                        result.score.score,
                        total_insertions,
                        total_deletions,
                        result.cigar,
                        job.query,
                        job.target
                    );
                }
            }
        }
    }

    all_results
}

/// Execute alignments using scalar processing (fallback for small batches)
pub(crate) fn execute_scalar_alignments(
    sw_params: &BandedPairWiseSW,
    jobs: &[AlignmentJob],
) -> Vec<(i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>)> {
    jobs.iter()
        .enumerate()
        .map(|(idx, job)| {
            let qlen = job.query.len() as i32;
            let tlen = job.target.len() as i32;

            // Log query sequence being aligned
            let query_str: String = job.query.iter().map(|&b| match b {
                0 => 'A', 1 => 'C', 2 => 'G', 3 => 'T', _ => 'N',
            }).collect();
            log::debug!(
                "SW alignment job {}: qlen={}, tlen={}, bw={}, query_seq={}",
                idx, qlen, tlen, job.band_width, query_str
            );

            // Use directional extension if specified, otherwise use standard SW
            let (score, cigar, ref_aligned, query_aligned) = if let Some(direction) = job.direction {
                // CRITICAL: Calculate h0 from seed length (C++ bwamem.cpp:2232)
                // h0 = seed_length * match_score gives SW algorithm the initial score
                // Without this, SW starts from 0 and finds terrible local alignments
                // For extensions, we need to know we're extending from a high-scoring seed
                let h0 = job.seed_len; // match_score = 1, so h0 = seed_len * 1

                // Use directional extension (LEFT or RIGHT)
                let ext_result = sw_params.scalar_banded_swa_directional(
                    direction,
                    qlen,
                    &job.query,
                    tlen,
                    &job.target,
                    job.band_width,
                    h0,
                );

                log::debug!(
                    "Directional alignment {}: direction={:?}, local_score={}, global_score={}, should_clip={}",
                    idx,
                    direction,
                    ext_result.local_score,
                    ext_result.global_score,
                    ext_result.should_clip
                );

                (
                    ext_result.local_score,
                    ext_result.cigar,
                    ext_result.ref_aligned,
                    ext_result.query_aligned,
                )
            } else {
                // Use standard SW for backward compatibility (legacy tests)
                // CRITICAL: Use h0=seed_len, not 0 (matching production code)
                let h0 = job.seed_len;
                let (score_out, cigar, ref_aligned, query_aligned) = sw_params.scalar_banded_swa(
                    qlen,
                    &job.query,
                    tlen,
                    &job.target,
                    job.band_width,
                    h0,
                );
                (score_out.score, cigar, ref_aligned, query_aligned)
            };

            // Detect pathological CIGARs (excessive insertions/deletions)
            let total_insertions: i32 = cigar.iter()
                .filter(|(op, _)| *op == b'I')
                .map(|(_, count)| count)
                .sum();
            let total_deletions: i32 = cigar.iter()
                .filter(|(op, _)| *op == b'D')
                .map(|(_, count)| count)
                .sum();

            // Log if CIGAR has excessive indels (likely bug)
            if total_insertions > 10 || total_deletions > 5 {
                // ATOMIC LOG: All data in single statement to avoid multi-threaded interleaving
                log::debug!(
                    "PATHOLOGICAL_CIGAR_SCALAR|idx={}|qlen={}|tlen={}|bw={}|score={}|ins={}|del={}|CIGAR={:?}|QUERY={:?}|TARGET={:?}",
                    idx,
                    qlen,
                    tlen,
                    job.band_width,
                    score,
                    total_insertions,
                    total_deletions,
                    cigar,
                    job.query,
                    job.target
                );
            }

            if idx < 3 {
                // Log first 3 alignments for debugging
                log::debug!(
                    "Scalar alignment {}: qlen={}, tlen={}, score={}, CIGAR_len={}, first_op={:?}",
                    idx,
                    qlen,
                    tlen,
                    score,
                    cigar.len(),
                    cigar.first().map(|&(op, len)| (op as char, len))
                );
            }

            (score, cigar, ref_aligned, query_aligned)
        })
        .collect()
}

// ============================================================================
// SEED GENERATION (SMEM EXTRACTION)
// ============================================================================
//
// This section contains the main seed generation pipeline:
// - SMEM (Supermaximal Exact Match) extraction using FM-Index
// - Bidirectional search (forward and reverse complement)
// - Seed extension and filtering
// ============================================================================

pub fn generate_seeds(
    bwa_idx: &BwaIndex,
    pac_data: &[u8], // Pre-loaded PAC data for MD tag generation
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
) -> Vec<Alignment> {
    generate_seeds_with_mode(
        bwa_idx, pac_data, query_name, query_seq, query_qual, true, opt,
    )
}

/// Generate SMEMs for a single strand (forward or reverse complement)
fn generate_smems_for_strand(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_len: usize,
    encoded_query: &[u8],
    is_reverse_complement: bool,
    min_seed_len: i32,
    min_intv: u64,
    all_smems: &mut Vec<SMEM>,
    max_smem_count: &mut usize,
) {
    // OPTIMIZATION: Pre-allocate buffers to avoid repeated allocations
    let mut prev_array_buf: Vec<SMEM> = Vec::with_capacity(query_len);
    let mut curr_array_buf: Vec<SMEM> = Vec::with_capacity(query_len);

    let mut x = 0;
    while x < query_len {
        let a = encoded_query[x];

        if a >= 4 {
            // Skip 'N' bases
            x += 1;
            continue;
        }

        // Initialize SMEM at position x
        let mut smem = SMEM {
            read_id: 0,
            query_start: x as i32,
            query_end: x as i32,
            bwt_interval_start: bwa_idx.bwt.cumulative_count[a as usize],
            bwt_interval_end: bwa_idx.bwt.cumulative_count[(3 - a) as usize],
            interval_size: bwa_idx.bwt.cumulative_count[(a + 1) as usize]
                - bwa_idx.bwt.cumulative_count[a as usize],
            is_reverse_complement,
        };

        if x == 0 && !is_reverse_complement {
            log::debug!(
                "{}: Initial SMEM at x={}: a={}, k={}, l={}, s={}, l2[{}]={}, l2[{}]={}",
                query_name,
                x,
                a,
                smem.bwt_interval_start,
                smem.bwt_interval_end,
                smem.interval_size,
                a,
                bwa_idx.bwt.cumulative_count[a as usize],
                3 - a,
                bwa_idx.bwt.cumulative_count[(3 - a) as usize]
            );
        }

        // Phase 1: Forward extension
        prev_array_buf.clear();
        let mut next_x = x + 1;

        for j in (x + 1)..query_len {
            let a = encoded_query[j];
            next_x = j + 1;

            if a >= 4 {
                if x == 0 && !is_reverse_complement && log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "{}: x={}, forward extension stopped at j={} due to N base",
                        query_name,
                        x,
                        j
                    );
                }
                next_x = j;
                break;
            }

            let new_smem = forward_ext(bwa_idx, smem, a);

            if x == 0 && j <= 12 && !is_reverse_complement {
                log::debug!(
                    "{}: x={}, j={}, a={}, old_smem.interval_size={}, new_smem(k={}, l={}, s={})",
                    query_name,
                    x,
                    j,
                    a,
                    smem.interval_size,
                    new_smem.bwt_interval_start,
                    new_smem.bwt_interval_end,
                    new_smem.interval_size
                );
            }

            if new_smem.interval_size != smem.interval_size {
                if x < 3 && !is_reverse_complement {
                    let s_from_lk = if smem.bwt_interval_end > smem.bwt_interval_start {
                        smem.bwt_interval_end - smem.bwt_interval_start
                    } else {
                        0
                    };
                    log::debug!(
                        "{}: x={}, j={}, pushing smem to prev_array_buf: s={}, l-k={}, match={}",
                        query_name,
                        x,
                        j,
                        smem.interval_size,
                        s_from_lk,
                        smem.interval_size == s_from_lk
                    );
                }
                prev_array_buf.push(smem);
            }

            if new_smem.interval_size < min_intv {
                if x == 0 && !is_reverse_complement && log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "{}: x={}, forward extension stopped at j={} because new_smem.interval_size={} < min_intv={}",
                        query_name,
                        x,
                        j,
                        new_smem.interval_size,
                        min_intv
                    );
                }
                break;
            }

            smem = new_smem;
            smem.query_end = j as i32;
        }

        if smem.interval_size >= min_intv {
            prev_array_buf.push(smem);
        }

        if x < 3 && !is_reverse_complement {
            log::debug!(
                "{}: Position x={}, prev_array_buf.len()={}, smem.interval_size={}, min_intv={}",
                query_name,
                x,
                prev_array_buf.len(),
                smem.interval_size,
                min_intv
            );
        }

        // Phase 2: Backward search
        if !is_reverse_complement {
            log::debug!(
                "{}: [RUST Phase 2] Starting backward search from x={}, prev_array_buf.len()={}",
                query_name,
                x,
                prev_array_buf.len()
            );
        }

        for j in (0..x).rev() {
            let a = encoded_query[j];
            if a >= 4 {
                if !is_reverse_complement {
                    log::debug!(
                        "{}: [RUST Phase 2] Hit 'N' base at j={}, stopping",
                        query_name,
                        j
                    );
                }
                break;
            }

            curr_array_buf.clear();
            let curr_array = &mut curr_array_buf;
            let mut curr_s = None;

            if !is_reverse_complement {
                log::debug!(
                    "{}: [RUST Phase 2] j={}, base={}, prev_array_buf.len()={}",
                    query_name,
                    j,
                    a,
                    prev_array_buf.len()
                );
            }

            for (i, smem) in prev_array_buf.iter().rev().enumerate() {
                let mut new_smem = backward_ext(bwa_idx, *smem, a);
                new_smem.query_start = j as i32;

                if !is_reverse_complement {
                    let old_len = smem.query_end - smem.query_start + 1;
                    let new_len = new_smem.query_end - new_smem.query_start + 1;
                    log::debug!(
                        "{}: [RUST Phase 2] x={}, j={}, i={}: old_smem(m={},n={},len={},k={},l={},s={}), new_smem(m={},n={},len={},k={},l={},s={}), min_intv={}",
                        query_name,
                        x,
                        j,
                        i,
                        smem.query_start,
                        smem.query_end,
                        old_len,
                        smem.bwt_interval_start,
                        smem.bwt_interval_end,
                        smem.interval_size,
                        new_smem.query_start,
                        new_smem.query_end,
                        new_len,
                        new_smem.bwt_interval_start,
                        new_smem.bwt_interval_end,
                        new_smem.interval_size,
                        min_intv
                    );
                }

                if new_smem.interval_size < min_intv
                    && (smem.query_end - smem.query_start + 1) >= min_seed_len
                {
                    if !is_reverse_complement {
                        let s_from_lk = if smem.bwt_interval_end > smem.bwt_interval_start {
                            smem.bwt_interval_end - smem.bwt_interval_start
                        } else {
                            0
                        };
                        let s_matches = smem.interval_size == s_from_lk;
                        log::debug!(
                            "{}: [RUST SMEM OUTPUT] Phase2 line 617: smem(m={},n={},k={},l={},s={}) newSmem.s={} < min_intv={}, l-k={}, s_match={}",
                            query_name,
                            smem.query_start,
                            smem.query_end,
                            smem.bwt_interval_start,
                            smem.bwt_interval_end,
                            smem.interval_size,
                            new_smem.interval_size,
                            min_intv,
                            s_from_lk,
                            s_matches
                        );
                    }
                    all_smems.push(*smem);
                    break;
                }

                if new_smem.interval_size >= min_intv && curr_s != Some(new_smem.interval_size) {
                    curr_s = Some(new_smem.interval_size);
                    curr_array.push(new_smem);
                    if !is_reverse_complement {
                        log::debug!(
                            "{}: [RUST Phase 2] Keeping new_smem (s={} >= min_intv={}), breaking",
                            query_name,
                            new_smem.interval_size,
                            min_intv
                        );
                    }
                    break;
                }

                if !is_reverse_complement {
                    log::debug!(
                        "{}: [RUST Phase 2] Rejecting new_smem (s={} < min_intv={} OR already_seen={})",
                        query_name,
                        new_smem.interval_size,
                        min_intv,
                        curr_s == Some(new_smem.interval_size)
                    );
                }
            }

            for (i, smem) in prev_array_buf.iter().rev().skip(1).enumerate() {
                let mut new_smem = backward_ext(bwa_idx, *smem, a);
                new_smem.query_start = j as i32;

                if !is_reverse_complement {
                    let new_len = new_smem.query_end - new_smem.query_start + 1;
                    log::debug!(
                        "{}: [RUST Phase 2] x={}, j={}, remaining_i={}: smem(m={},n={},s={}), new_smem(m={},n={},len={},s={}), will_push={}",
                        query_name,
                        x,
                        j,
                        i + 1,
                        smem.query_start,
                        smem.query_end,
                        smem.interval_size,
                        new_smem.query_start,
                        new_smem.query_end,
                        new_len,
                        new_smem.interval_size,
                        new_smem.interval_size >= min_intv
                            && curr_s != Some(new_smem.interval_size)
                    );
                }

                if new_smem.interval_size >= min_intv && curr_s != Some(new_smem.interval_size) {
                    curr_s = Some(new_smem.interval_size);
                    curr_array.push(new_smem);
                }
            }

            std::mem::swap(&mut prev_array_buf, &mut curr_array_buf);
            *max_smem_count = (*max_smem_count).max(prev_array_buf.len());

            if !is_reverse_complement {
                log::debug!(
                    "{}: [RUST Phase 2] After j={}, prev_array_buf.len()={}",
                    query_name,
                    j,
                    prev_array_buf.len()
                );
            }

            if prev_array_buf.is_empty() {
                if !is_reverse_complement {
                    log::debug!(
                        "{}: [RUST Phase 2] prev_array_buf empty, breaking at j={}",
                        query_name,
                        j
                    );
                }
                break;
            }
        }

        if !prev_array_buf.is_empty() {
            let smem = prev_array_buf[prev_array_buf.len() - 1];
            let len = smem.query_end - smem.query_start + 1;
            if len >= min_seed_len {
                if !is_reverse_complement {
                    let s_from_lk = if smem.bwt_interval_end > smem.bwt_interval_start {
                        smem.bwt_interval_end - smem.bwt_interval_start
                    } else {
                        0
                    };
                    let s_matches = smem.interval_size == s_from_lk;
                    log::debug!(
                        "{}: [RUST SMEM OUTPUT] Phase2 line 671: smem(m={},n={},k={},l={},s={}), len={}, l-k={}, s_match={}, next_x={}",
                        query_name,
                        smem.query_start,
                        smem.query_end,
                        smem.bwt_interval_start,
                        smem.bwt_interval_end,
                        smem.interval_size,
                        len,
                        s_from_lk,
                        s_matches,
                        next_x
                    );
                }
                all_smems.push(smem);
            } else if !is_reverse_complement {
                log::debug!(
                    "{}: [RUST Phase 2] Rejecting final SMEM: m={}, n={}, len={} < min_seed_len={}, s={}",
                    query_name,
                    smem.query_start,
                    smem.query_end,
                    len,
                    min_seed_len,
                    smem.interval_size
                );
            }
        } else if !is_reverse_complement {
            log::debug!(
                "{}: [RUST Phase 2] No remaining SMEMs at end of backward search for x={}",
                query_name,
                x
            );
        }

        x = next_x;
    }
}

// Internal implementation with option to use batched SIMD
fn generate_seeds_with_mode(
    bwa_idx: &BwaIndex,
    pac_data: &[u8], // Pre-loaded PAC data (loaded once, not per-read!)
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    use_batched_simd: bool,
    _opt: &MemOpt, // TODO: Use opt parameters in Phase 2+
) -> Vec<Alignment> {
    let query_len = query_seq.len();
    if query_len == 0 {
        return Vec::new();
    }

    #[cfg(feature = "debug-logging")]
    let is_debug_read = query_name.contains("1150:14380");

    #[cfg(feature = "debug-logging")]
    if is_debug_read {
        log::debug!("[DEBUG_READ] Generating seeds for: {}", query_name);
        log::debug!("[DEBUG_READ] Query length: {}", query_len);
    }

    // Instantiate BandedPairWiseSW with parameters from MemOpt
    let sw_params = BandedPairWiseSW::new(
        _opt.o_del,      // Gap open deletion penalty
        _opt.e_del,      // Gap extension deletion penalty
        _opt.o_ins,      // Gap open insertion penalty
        _opt.e_ins,      // Gap extension insertion penalty
        _opt.zdrop,      // Z-dropoff
        5,               // end_bonus (reserved for future use)
        _opt.pen_clip5,  // 5' clipping penalty (default=5)
        _opt.pen_clip3,  // 3' clipping penalty (default=5)
        _opt.mat,        // Scoring matrix (generated from -A/-B)
        _opt.a as i8,    // Match score
        -(_opt.b as i8), // Mismatch penalty (negative)
    );

    let mut encoded_query = Vec::with_capacity(query_len);
    let mut encoded_query_rc = Vec::with_capacity(query_len); // Reverse complement
    for &base in query_seq {
        let code = base_to_code(base);
        encoded_query.push(code);
        encoded_query_rc.push(reverse_complement_code(code));
    }
    encoded_query_rc.reverse(); // Reverse the reverse complement to get the sequence in correct order

    let mut all_smems: Vec<SMEM> = Vec::new();

    // --- Generate SMEMs using C++ two-phase algorithm ---
    // C++ reference: FMI_search.cpp getSMEMsOnePosOneThread (lines 496-670)
    // For each starting position x:
    //   Phase 1: Forward extension (collect intermediate SMEMs)
    //   Phase 2: Backward extension (generate final bidirectional SMEMs)

    let min_seed_len = _opt.min_seed_len;
    // CRITICAL FIX: C++ bwa-mem2 uses min_intv=1 during SMEM generation (not max_occ!)
    // See bwamem.cpp:661 - min_intv_ar[l] = 1;
    // The max_occ filter is applied LATER during SMEM filtering, not during generation
    let min_intv = 1u64;

    log::debug!(
        "{}: Starting SMEM generation: min_seed_len={}, min_intv={}, query_len={}",
        query_name,
        min_seed_len,
        min_intv,
        query_len
    );

    let mut max_smem_count = 0usize;

    // Process forward strand
    generate_smems_for_strand(
        bwa_idx,
        query_name,
        query_len,
        &encoded_query,
        false, // is_rev_comp
        min_seed_len,
        min_intv,
        &mut all_smems,
        &mut max_smem_count,
    );

    // Process reverse complement strand
    generate_smems_for_strand(
        bwa_idx,
        query_name,
        query_len,
        &encoded_query_rc,
        true, // is_rev_comp
        min_seed_len,
        min_intv,
        &mut all_smems,
        &mut max_smem_count,
    );

    // eprintln!("all_smems: {:?}", all_smems);

    // --- Filtering SMEMs ---
    let mut unique_filtered_smems: Vec<SMEM> = Vec::new();
    let mut filtered_too_short = 0;
    let mut filtered_too_many_occ = 0;
    let mut duplicates = 0;

    // Sort SMEMs to process unique ones efficiently
    all_smems.sort_by_key(|smem| {
        (
            smem.query_start,
            smem.query_end,
            smem.bwt_interval_start,
            smem.is_reverse_complement,
        )
    });

    let split_len_threshold = (_opt.min_seed_len as f32 * _opt.split_factor) as i32;

    if let Some(mut prev_smem) = all_smems.first().cloned() {
        let seed_len = prev_smem.query_end - prev_smem.query_start + 1;
        let occurrences = prev_smem.interval_size;

        // Standard filter (min_seed_len, max_occ)
        let mut keep_smem = seed_len >= _opt.min_seed_len && occurrences <= _opt.max_occ as u64;

        // Chimeric filter (from BWA-MEM2 bwamem.cpp:685)
        // If SMEM is too short for splitting OR too repetitive, it's NOT considered for chimeric re-seeding
        if seed_len < split_len_threshold || occurrences > _opt.split_width as u64 {
            keep_smem = false; // Mark as not suitable for chimeric processing
        }

        if keep_smem {
            unique_filtered_smems.push(prev_smem);
        } else {
            if seed_len < _opt.min_seed_len {
                filtered_too_short += 1;
            }
            if occurrences > _opt.max_occ as u64 {
                filtered_too_many_occ += 1;
            }
        }

        for i in 1..all_smems.len() {
            let current_smem = all_smems[i];
            if current_smem != prev_smem {
                // Use PartialEq for comparison
                let seed_len = current_smem.query_end - current_smem.query_start + 1;
                let occurrences = current_smem.interval_size;

                // Standard filter (min_seed_len, max_occ)
                let mut keep_smem_current =
                    seed_len >= _opt.min_seed_len && occurrences <= _opt.max_occ as u64;

                // Chimeric filter
                if seed_len < split_len_threshold || occurrences > _opt.split_width as u64 {
                    keep_smem_current = false;
                }

                if keep_smem_current {
                    unique_filtered_smems.push(current_smem);
                } else {
                    if seed_len < _opt.min_seed_len {
                        filtered_too_short += 1;
                    }
                    if occurrences > _opt.max_occ as u64 {
                        filtered_too_many_occ += 1;
                    }
                }
            } else {
                duplicates += 1;
            }
            prev_smem = current_smem;
        }
    }

    if unique_filtered_smems.is_empty() && all_smems.len() > 0 {
        log::debug!(
            "{}: All SMEMs filtered out - too_short={}, too_many_occ={}, duplicates={}, min_len={}, max_occ={}",
            query_name,
            filtered_too_short,
            filtered_too_many_occ,
            duplicates,
            _opt.min_seed_len,
            _opt.max_occ
        );

        // Sample first few SMEMs to see actual values
        for (i, smem) in all_smems.iter().take(5).enumerate() {
            let len = smem.query_end - smem.query_start + 1;
            let occ = smem.bwt_interval_end - smem.bwt_interval_start;
            log::debug!(
                "{}: Sample SMEM {}: len={}, occ={}, m={}, n={}, k={}, l={}",
                query_name,
                i,
                len,
                occ,
                smem.query_start,
                smem.query_end,
                smem.bwt_interval_start,
                smem.bwt_interval_end
            );
        }
    }
    // --- End Filtering SMEMs ---

    log::debug!(
        "{}: Generated {} SMEMs, filtered to {} unique",
        query_name,
        all_smems.len(),
        unique_filtered_smems.len()
    );

    // Convert SMEMs to Seed structs and perform seed extension
    // FIXED: Remove artificial SMEM limit - process ALL seeds like C++ bwa-mem2
    let mut sorted_smems = unique_filtered_smems;
    sorted_smems.sort_by_key(|smem| -(smem.query_end - smem.query_start + 1)); // Sort by length, descending

    let useful_smems = sorted_smems;

    log::debug!(
        "{}: Using {} SMEMs for alignment",
        query_name,
        useful_smems.len()
    );

    let mut seeds = Vec::new();
    let mut alignment_jobs = Vec::new(); // Collect alignment jobs for batching

    // Prepare query segment once - use the FULL query for alignment
    let query_segment_encoded: Vec<u8> = query_seq.iter().map(|&b| base_to_code(b)).collect();

    // Also prepare reverse complement for RC SMEMs
    // CRITICAL FIX: Use reverse_complement_code() to properly handle 'N' bases (code 4)
    // The XOR trick (b ^ 3) breaks for N: 4 ^ 3 = 7 (invalid!)
    let mut query_segment_encoded_rc: Vec<u8> = query_segment_encoded
        .iter()
        .map(|&b| reverse_complement_code(b)) // Properly handles N (4 → 4)
        .collect();
    query_segment_encoded_rc.reverse();

    log::debug!(
        "{}: query_segment_encoded (FWD) first_10={:?}",
        query_name,
        &query_segment_encoded[..10.min(query_segment_encoded.len())]
    );
    log::debug!(
        "{}: query_segment_encoded_rc (RC) first_10={:?}",
        query_name,
        &query_segment_encoded_rc[..10.min(query_segment_encoded_rc.len())]
    );

    for (idx, smem) in useful_smems.iter().enumerate() {
        let smem = *smem;

        // Get SA position and log the reconstruction process
        log::debug!(
            "{}: SMEM {}: BWT interval [k={}, l={}, s={}], query range [m={}, n={}], is_rev_comp={}",
            query_name,
            idx,
            smem.bwt_interval_start,
            smem.bwt_interval_end,
            smem.interval_size,
            smem.query_start,
            smem.query_end,
            smem.is_reverse_complement
        );

        // Try multiple positions in the BWT interval to find which one is correct
        let ref_pos_at_k = get_sa_entry(bwa_idx, smem.bwt_interval_start);
        let ref_pos_at_l_minus_1 = if smem.bwt_interval_end > 0 {
            get_sa_entry(bwa_idx, smem.bwt_interval_end - 1)
        } else {
            ref_pos_at_k
        };
        log::debug!(
            "{}: SMEM {}: SA at k={} -> ref_pos {}, SA at l-1={} -> ref_pos {}",
            query_name,
            idx,
            smem.bwt_interval_start,
            ref_pos_at_k,
            smem.bwt_interval_end - 1,
            ref_pos_at_l_minus_1
        );

        let mut ref_pos = ref_pos_at_k;

        let mut is_rev = smem.is_reverse_complement;

        // CRITICAL FIX: Use correct query orientation based on is_rev_comp flag
        // If SMEM is from RC search, use RC query bases for comparison
        let query_for_smem = if smem.is_reverse_complement {
            &query_segment_encoded_rc
        } else {
            &query_segment_encoded
        };
        let smem_query_bases = &query_for_smem
            [smem.query_start as usize..=(smem.query_end as usize).min(query_for_smem.len() - 1)];
        let smem_len = smem.query_end - smem.query_start + 1;
        log::debug!(
            "{}: SMEM {}: Query bases at [{}..{}] (len={}): {:?}",
            query_name,
            idx,
            smem.query_start,
            smem.query_end,
            smem_len,
            &smem_query_bases[..10.min(smem_query_bases.len())]
        );

        // CRITICAL: Keep seed positions in BIDIRECTIONAL coordinates
        // C++ keeps seeds in bidirectional coords [0, 2*l_pac) and only converts at SAM output
        // - Forward strand: ref_pos in [0, l_pac)
        // - Reverse strand: ref_pos in [l_pac, 2*l_pac)
        // get_reference_segment() handles bidirectional coords automatically

        log::debug!(
            "{}: SMEM {}: Seed ref_pos={} (bidirectional), is_rev={}, l_pac={}",
            query_name,
            idx,
            ref_pos,
            is_rev,
            bwa_idx.bns.packed_sequence_length
        );

        // Diagnostic validation removed - was causing 79,000% performance regression
        // The SMEM validation loop with log::info/warn was being called millions of times

        // Use query position in the coordinate system of the query we're aligning
        // For RC seeds, use smem.query_start directly (RC query coordinates)
        // For forward seeds, use smem.query_start directly (forward query coordinates)
        let query_pos = smem.query_start;

        let seed = Seed {
            query_pos,
            ref_pos, // Keep bidirectional coordinates!
            // CRITICAL: smem.query_end is EXCLUSIVE (see line 187), so len = end - start, NOT +1
            // C++ bwamem uses qend = qbeg + len (exclusive end), matching this calculation
            len: smem.query_end - smem.query_start,
            is_rev,
            interval_size: smem.interval_size, // Propagate interval_size from SMEM to Seed
        };

        // DEBUG: Log each seed creation with SMEM bounds
        log::debug!(
            "[SEED_CREATE] {}: ref_pos={}, qpos={}, len={}, strand={}, SMEM.query=[{}, {}) (query_end is exclusive)",
            query_name,
            seed.ref_pos,
            seed.query_pos,
            seed.len,
            if seed.is_rev { "rev" } else { "fwd" },
            smem.query_start,
            smem.query_end
        );

        // === CHAIN-BASED ALIGNMENT STRATEGY (C++ bwa-mem2) ===
        // Instead of aligning each seed individually, we now:
        // 1. Create all seeds first
        // 2. Chain them together
        // 3. Create alignment jobs using CHAIN bounds (not per-seed bounds)
        // This prevents N-rich regions from being included in DP alignment
        // (Per-seed bounds were too conservative and still included N bases)

        seeds.push(seed);
    }

    log::debug!(
        "{}: Found {} seeds (alignment jobs will be created from chains)",
        query_name,
        seeds.len()
    );

    // === NEW FLOW: Chain seeds FIRST, then create alignment jobs from chains ===
    // This matches C++ bwa-mem2 mem_align1_core() flow exactly

    // --- Seed Chaining ---
    // IMPORTANT: Pass seeds by value (not clone) so chain_seeds() sorts in place.
    // The chain seed indices will then correctly refer to the sorted seeds array.
    // chain_seeds() returns both the chains and the sorted seeds array.
    let (mut chained_results, seeds) = chain_seeds(seeds, _opt);
    log::debug!(
        "{}: Chaining produced {} chains",
        query_name,
        chained_results.len()
    );

    // --- Chain Filtering ---
    // Implements bwa-mem2 mem_chain_flt logic (bwamem.cpp:506-624)
    let filtered_chains = filter_chains(&mut chained_results, &seeds, _opt, query_len as i32);
    log::debug!(
        "{}: Chain filtering kept {} chains (from {} total)",
        query_name,
        filtered_chains.len(),
        chained_results.len()
    );

    // === CREATE ALIGNMENT JOBS FROM FILTERED CHAINS (C++ Strategy) ===
    // Create separate LEFT and RIGHT extension jobs per chain
    // This matches C++ bwa-mem2 separate extension model (bwamem.cpp:2229-2418)
    // CRITICAL: Process ALL seeds per chain, not just one (C++ line 2206)

    // Track left and right job indices for each seed in a chain
    #[derive(Debug, Clone)]
    struct SeedJobMapping {
        seed_idx: usize,              // Index into chain.seeds
        left_job_idx: Option<usize>,  // LEFT extension job index
        right_job_idx: Option<usize>, // RIGHT extension job index
    }

    #[derive(Debug, Clone)]
    struct ChainJobMapping {
        seed_jobs: Vec<SeedJobMapping>, // Multiple seeds per chain
    }

    let mut chain_to_job_map: Vec<ChainJobMapping> = Vec::new();

    for (chain_idx, chain) in filtered_chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            chain_to_job_map.push(ChainJobMapping {
                seed_jobs: Vec::new(),
            });
            continue;
        }

        // CRITICAL: Use correct query orientation based on chain strand
        // For reverse-strand chains: use RC query (matches reference in RC region)
        // For forward-strand chains: use forward query (matches reference in forward region)
        // C++ bwamem.cpp:2116 uses seq_[l].seq, but this is context-dependent
        let full_query = if chain.is_rev {
            &query_segment_encoded_rc
        } else {
            &query_segment_encoded
        };
        let query_len = full_query.len() as i32;

        log::debug!(
            "{}: Chain {}: is_rev={}, query_len={}, full_query[0..10]={:?}",
            query_name,
            chain_idx,
            chain.is_rev,
            query_len,
            full_query.iter().take(10).copied().collect::<Vec<u8>>()
        );

        // Calculate max possible reference span for this chain (C++ bwamem.cpp:2144-2166)
        // This covers all seeds with margins for gaps
        let l_pac = bwa_idx.bns.packed_sequence_length;
        let mut rmax_0 = l_pac << 1; // Start with max possible
        let mut rmax_1 = 0u64; // Start with min possible

        for &seed_idx in &chain.seeds {
            let seed = &seeds[seed_idx];

            // Calculate reference bounds with gap margins
            // b = t->rbeg - (t->qbeg + cal_max_gap(opt, t->qbeg))
            let left_margin = seed.query_pos + cal_max_gap(_opt, seed.query_pos);
            let b = if left_margin as u64 > seed.ref_pos {
                0
            } else {
                seed.ref_pos - left_margin as u64
            };

            // e = t->rbeg + t->len + (remaining_query + cal_max_gap(opt, remaining_query))
            let remaining_query = query_len - seed.query_pos - seed.len;
            let right_margin = remaining_query + cal_max_gap(_opt, remaining_query);
            let e = seed.ref_pos + seed.len as u64 + right_margin as u64;

            // Take min of all b values, max of all e values
            rmax_0 = rmax_0.min(b);
            rmax_1 = rmax_1.max(e);
        }

        // Clamp to valid range
        rmax_0 = rmax_0.max(0);
        rmax_1 = rmax_1.min(l_pac << 1);

        // If span crosses l_pac boundary, clamp to one side (C++ lines 2162-2166)
        if rmax_0 < l_pac && l_pac < rmax_1 {
            let first_seed = &seeds[chain.seeds[0]];
            if first_seed.ref_pos < l_pac {
                rmax_1 = l_pac;
            } else {
                rmax_0 = l_pac;
            }
        }

        log::debug!(
            "{}: Chain {}: rmax=[{}, {}) span={} (l_pac={})",
            query_name,
            chain_idx,
            rmax_0,
            rmax_1,
            rmax_1 - rmax_0,
            l_pac
        );

        // Fetch single large reference buffer covering entire chain
        let rseq = match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
            Ok(seq) => seq,
            Err(e) => {
                log::error!(
                    "{}: Chain {}: Error fetching reference segment: {}",
                    query_name,
                    chain_idx,
                    e
                );
                chain_to_job_map.push(ChainJobMapping {
                    seed_jobs: Vec::new(),
                });
                continue;
            }
        };

        // CRITICAL: Iterate through ALL seeds in REVERSE order (C++ bwamem.cpp:2206)
        // C++: for (int k=c->n-1; k >= 0; k--) { s = &c->seeds[(uint32_t)srt[k]]; ... }
        // Each seed gets its own LEFT/RIGHT extension pair
        let mut seed_job_mappings = Vec::new();

        log::debug!(
            "{}: Chain {}: Processing {} seeds (rmax=[{}, {}) span={}, l_pac={})",
            query_name,
            chain_idx,
            chain.seeds.len(),
            rmax_0,
            rmax_1,
            rmax_1 - rmax_0,
            bwa_idx.bns.packed_sequence_length
        );

        // Show first 10 bases of rseq buffer
        let rseq_first_10: Vec<u8> = rseq.iter().take(10).copied().collect();
        log::debug!(
            "{}: Chain {}: rseq[0..10]={:?}",
            query_name,
            chain_idx,
            rseq_first_10
        );

        // Iterate seeds in reverse order (matching C++)
        for &seed_chain_idx in chain.seeds.iter().rev() {
            let seed = &seeds[seed_chain_idx];

            let seed_query_start = seed.query_pos;
            let seed_query_end = seed.query_pos + seed.len;

            log::debug!(
                "{}: Chain {}: Processing seed {} - query_pos={}, len={}, ref_pos={} (bidirectional)",
                query_name,
                chain_idx,
                seed_chain_idx,
                seed_query_start,
                seed.len,
                seed.ref_pos
            );

            // Verify seed matches: check if query[seed_query_start..seed_query_end] matches rseq[seed_buffer_pos..seed_buffer_pos+seed.len]
            let seed_buffer_pos = (seed.ref_pos - rmax_0) as usize;
            let seed_query_slice: Vec<u8> = full_query
                [seed_query_start as usize..seed_query_end as usize]
                .iter()
                .take(10.min(seed.len as usize))
                .copied()
                .collect();
            let seed_ref_slice: Vec<u8> = if seed_buffer_pos < rseq.len()
                && seed_buffer_pos + seed.len as usize <= rseq.len()
            {
                rseq[seed_buffer_pos..seed_buffer_pos + seed.len as usize]
                    .iter()
                    .take(10.min(seed.len as usize))
                    .copied()
                    .collect()
            } else {
                vec![]
            };
            log::debug!(
                "{}: Chain {}: Seed {}: SEED MATCH CHECK: buffer_pos={}, query_slice[0..10]={:?}, ref_slice[0..10]={:?}",
                query_name,
                chain_idx,
                seed_chain_idx,
                seed_buffer_pos,
                seed_query_slice,
                seed_ref_slice
            );

            // Show seed boundary: last few bases of seed and first few bases after seed
            let seed_end_buffer_pos = seed_buffer_pos + seed.len as usize;
            let boundary_start = seed_end_buffer_pos.saturating_sub(5);
            let boundary_end = (seed_end_buffer_pos + 5).min(rseq.len());
            if boundary_start < rseq.len() {
                let boundary_ref: Vec<u8> = rseq[boundary_start..boundary_end].to_vec();
                log::debug!(
                    "{}: Chain {}: Seed {}: SEED BOUNDARY: rseq[{}..{}]={:?} (pos {} is last seed base, pos {} is first RIGHT base)",
                    query_name,
                    chain_idx,
                    seed_chain_idx,
                    boundary_start,
                    boundary_end,
                    boundary_ref,
                    seed_end_buffer_pos - 1,
                    seed_end_buffer_pos
                );
            }
            let query_boundary_start = seed_query_end.saturating_sub(5) as usize;
            let query_boundary_end = ((seed_query_end + 5).min(query_len)) as usize;
            let boundary_query: Vec<u8> =
                full_query[query_boundary_start..query_boundary_end].to_vec();
            log::debug!(
                "{}: Chain {}: Seed {}: QUERY BOUNDARY: full_query[{}..{}]={:?} (pos {} is last seed base, pos {} is first RIGHT base)",
                query_name,
                chain_idx,
                seed_chain_idx,
                query_boundary_start,
                query_boundary_end,
                boundary_query,
                seed_query_end - 1,
                seed_query_end
            );

            let mut left_job_idx = None;
            let mut right_job_idx = None;

            // --- LEFT Extension: seed start → query position 0 ---
            if seed_query_start > 0 {
                let left_query_len = seed_query_start as usize;
                let left_query: Vec<u8> = full_query[0..left_query_len].to_vec();

                // Index into rseq buffer (C++ bwamem.cpp:2280, 2298)
                // tmp = s->rbeg - rmax[0];
                // for (int64_t i = 0; i < tmp; ++i) rs[i] = rseq[tmp - 1 - i];
                let tmp = (seed.ref_pos - rmax_0) as usize;

                if tmp > 0 && tmp <= rseq.len() {
                    // Extract rseq[0..tmp] for LEFT extension
                    // Note: scalar_banded_swa_directional will reverse this
                    let left_target: Vec<u8> = rseq[0..tmp].to_vec();

                    left_job_idx = Some(alignment_jobs.len());

                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT extension qlen={} tlen={} tmp={}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        left_query_len,
                        left_target.len(),
                        tmp
                    );

                    // DETAILED LEFT DEBUG
                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT query extraction: seed_query_start={}, left_query_len={}, full_query[0..10]={:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        seed_query_start,
                        left_query_len,
                        full_query
                            .iter()
                            .take(10.min(left_query_len))
                            .copied()
                            .collect::<Vec<u8>>()
                    );
                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT target extraction: tmp={}, rseq[0..10]={:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        tmp,
                        rseq.iter().take(10.min(tmp)).copied().collect::<Vec<u8>>()
                    );
                    let left_query_first_10: Vec<u8> =
                        left_query.iter().take(10).copied().collect();
                    let left_target_first_10: Vec<u8> =
                        left_target.iter().take(10).copied().collect();
                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT left_query[0..10]={:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        left_query_first_10
                    );
                    log::debug!(
                        "{}: Chain {}: Seed {}: LEFT left_target[0..10]={:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        left_target_first_10
                    );

                    alignment_jobs.push(AlignmentJob {
                        seed_idx: chain_idx,
                        query: left_query,
                        target: left_target,
                        band_width: _opt.w,
                        query_offset: 0,
                        direction: Some(crate::banded_swa::ExtensionDirection::Left),
                        seed_len: seed.len, // For h0 calculation
                    });
                } else {
                    log::warn!(
                        "{}: Chain {}: Seed {}: Invalid LEFT tmp={} (seed.ref_pos={}, rmax_0={}, rseq.len={})",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        tmp,
                        seed.ref_pos,
                        rmax_0,
                        rseq.len()
                    );
                }
            }

            // --- RIGHT Extension: seed end → query end ---
            if seed_query_end < query_len {
                let right_query_start = seed_query_end as usize;
                let right_query_len = (query_len - seed_query_end) as usize;

                log::debug!(
                    "{}: Chain {}: Seed {}: RIGHT query extraction: seed_query_start={}, seed_query_end={}, right_query_start={}, full_query.len()={}",
                    query_name,
                    chain_idx,
                    seed_chain_idx,
                    seed_query_start,
                    seed_query_end,
                    right_query_start,
                    full_query.len()
                );
                log::debug!(
                    "{}: Chain {}: Seed {}: RIGHT query full_query[{}..{}]={:?}",
                    query_name,
                    chain_idx,
                    seed_chain_idx,
                    right_query_start,
                    right_query_start + 10.min(full_query.len() - right_query_start),
                    full_query[right_query_start..]
                        .iter()
                        .take(10)
                        .copied()
                        .collect::<Vec<u8>>()
                );

                let right_query: Vec<u8> = full_query[right_query_start..].to_vec();

                // Index into rseq buffer (C++ bwamem.cpp:2327, 2354)
                // re = s->rbeg + s->len - rmax[0];
                // sp.len1 = rmax[1] - rmax[0] - re;
                let re = ((seed.ref_pos + seed.len as u64) - rmax_0) as usize;

                if re < rseq.len() {
                    // Extract rseq[re..] for RIGHT extension (forward direction)
                    let right_target: Vec<u8> = rseq[re..].to_vec();

                    right_job_idx = Some(alignment_jobs.len());

                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT extension qlen={} tlen={} buffer_idx={}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        right_query_len,
                        right_target.len(),
                        re
                    );

                    // DETAILED INDEXING DEBUG
                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT indexing: seed.ref_pos={}, seed.len={}, rmax_0={}, re={} (calculation: {} + {} - {} = {})",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        seed.ref_pos,
                        seed.len,
                        rmax_0,
                        re,
                        seed.ref_pos,
                        seed.len,
                        rmax_0,
                        seed.ref_pos + seed.len as u64 - rmax_0
                    );

                    // Show first 10 bases of rseq at re
                    let rseq_at_re: Vec<u8> = rseq[re..].iter().take(10).copied().collect();
                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT buffer rseq[{}..{}]: {:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        re,
                        re + 10.min(rseq.len() - re),
                        rseq_at_re
                    );

                    // Show first 10 bases of right_target
                    let right_target_first_10: Vec<u8> =
                        right_target.iter().take(10).copied().collect();
                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT right_target[0..10]: {:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        right_target_first_10
                    );

                    // Show first 10 bases of right_query
                    let right_query_first_10: Vec<u8> =
                        right_query.iter().take(10).copied().collect();
                    log::debug!(
                        "{}: Chain {}: Seed {}: RIGHT right_query[0..10]: {:?}",
                        query_name,
                        chain_idx,
                        seed_chain_idx,
                        right_query_first_10
                    );

                    alignment_jobs.push(AlignmentJob {
                        seed_idx: chain_idx,
                        query: right_query,
                        target: right_target,
                        band_width: _opt.w,
                        query_offset: seed_query_end,
                        direction: Some(crate::banded_swa::ExtensionDirection::Right),
                        seed_len: seed.len, // For h0 calculation
                    });
                }
            }

            // Store this seed's job mapping
            seed_job_mappings.push(SeedJobMapping {
                seed_idx: seed_chain_idx,
                left_job_idx,
                right_job_idx,
            });
        } // End of seed iteration loop

        // Store all seed job mappings for this chain
        chain_to_job_map.push(ChainJobMapping {
            seed_jobs: seed_job_mappings,
        });
    }

    log::debug!(
        "{}: Created {} alignment jobs from {} filtered chains",
        query_name,
        alignment_jobs.len(),
        filtered_chains.len()
    );

    // --- Execute Alignments (Adaptive Strategy) ---
    // Use adaptive routing strategy that combines:
    // 1. Divergence-based routing (high divergence → scalar, low → SIMD)
    // 2. Adaptive batch sizing (8-32 based on sequence characteristics)
    // 3. Fallback to scalar for small batches (<8 jobs)
    let extended_cigars = if use_batched_simd && alignment_jobs.len() >= 8 {
        // Use adaptive strategy for 8+ jobs
        // This provides optimal performance by routing jobs to the best execution path
        execute_adaptive_alignments(&sw_params, &alignment_jobs)
    } else {
        // Fall back to scalar processing for very small batches
        execute_scalar_alignments(&sw_params, &alignment_jobs)
    };

    // Separate scores, CIGARs, and aligned sequences
    let alignment_scores: Vec<i32> = extended_cigars
        .iter()
        .map(|(score, _, _, _)| *score)
        .collect();
    let alignment_cigars: Vec<Vec<(u8, i32)>> = extended_cigars
        .iter()
        .map(|(_, cigar, _, _)| cigar.clone())
        .collect();
    let ref_aligned_seqs: Vec<Vec<u8>> = extended_cigars
        .iter()
        .map(|(_, _, ref_aligned, _)| ref_aligned.clone())
        .collect();
    let query_aligned_seqs: Vec<Vec<u8>> = extended_cigars
        .into_iter()
        .map(|(_, _, _, query_aligned)| query_aligned)
        .collect();

    // Extract query offsets for soft-clipping (C++ strategy: match mem_reg2aln)
    let query_offsets: Vec<i32> = alignment_jobs.iter().map(|job| job.query_offset).collect();

    log::debug!(
        "{}: Extended {} chains, {} CIGARs produced",
        query_name,
        filtered_chains.len(),
        alignment_cigars.len()
    );

    // === MULTI-ALIGNMENT GENERATION FROM FILTERED CHAINS ===
    // Generate multiple alignment candidates per chain (one per seed)
    // Match C++ bwa-mem2: each seed gets LEFT+SEED+RIGHT, then select best
    let mut alignments = Vec::new();

    for (chain_idx, chain) in filtered_chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            continue;
        }

        // Get all seed job mappings for this chain
        let mapping = &chain_to_job_map[chain_idx];

        // Select appropriate query based on strand
        let full_query = if chain.is_rev {
            &query_segment_encoded_rc
        } else {
            &query_segment_encoded
        };
        let query_len = full_query.len() as i32;

        // Process ALL seeds for this chain, creating alignment candidates
        let mut best_score = 0;
        let mut best_alignment_data: Option<(Vec<(u8, i32)>, i32, u64, i32, i32)> = None;

        log::debug!(
            "{}: Chain {}: Processing {} seed alignments",
            query_name,
            chain_idx,
            mapping.seed_jobs.len()
        );

        for seed_job in &mapping.seed_jobs {
            let seed = &seeds[seed_job.seed_idx];

            // === COMBINE LEFT + SEED + RIGHT EXTENSIONS FOR THIS SEED ===
            let mut combined_cigar = Vec::new();
            let mut combined_score = 0;
            let mut alignment_start_pos = seed.ref_pos;
            let mut query_start_aligned = 0;
            let mut query_end_aligned = query_len;

            // LEFT extension
            if let Some(left_idx) = seed_job.left_job_idx {
                let left_cigar = &alignment_cigars[left_idx];
                let left_score = alignment_scores[left_idx];

                log::debug!(
                    "{}: Chain {}: Seed {}: LEFT extension score={}, CIGAR={:?}",
                    query_name,
                    chain_idx,
                    seed_job.seed_idx,
                    left_score,
                    left_cigar
                );

                // Add left CIGAR operations
                combined_cigar.extend(left_cigar.iter().cloned());
                combined_score += left_score;

                // Update alignment start position (move back by left extension)
                let left_ref_len: i32 = left_cigar
                    .iter()
                    .filter_map(|&(op, len)| match op as char {
                        'M' | 'D' => Some(len),
                        _ => None,
                    })
                    .sum();
                if left_ref_len as u64 <= alignment_start_pos {
                    alignment_start_pos -= left_ref_len as u64;
                } else {
                    alignment_start_pos = 0;
                }
            } else if seed.query_pos > 0 {
                // No left extension job, soft-clip 5' end
                combined_cigar.push((b'S', seed.query_pos));
                query_start_aligned = seed.query_pos;
            }

            // SEED match (perfect match for seed length)
            combined_cigar.push((b'M', seed.len));
            combined_score += seed.len; // Add seed score

            // RIGHT extension
            let seed_end = seed.query_pos + seed.len;
            if let Some(right_idx) = seed_job.right_job_idx {
                let right_cigar = &alignment_cigars[right_idx];
                let right_score = alignment_scores[right_idx];

                log::debug!(
                    "{}: Chain {}: Seed {}: RIGHT extension score={}, CIGAR={:?}",
                    query_name,
                    chain_idx,
                    seed_job.seed_idx,
                    right_score,
                    right_cigar
                );

                // Add right CIGAR operations
                combined_cigar.extend(right_cigar.iter().cloned());
                combined_score += right_score;
            } else if seed_end < query_len {
                // No right extension job, soft-clip 3' end
                combined_cigar.push((b'S', query_len - seed_end));
                query_end_aligned = seed_end;
            }

            // Merge consecutive CIGAR operations (e.g., M+M → M)
            let cigar_for_candidate = merge_cigar_operations(combined_cigar);

            log::debug!(
                "{}: Chain {}: Seed {}: combined_score={} bounds=[{}, {})",
                query_name,
                chain_idx,
                seed_job.seed_idx,
                combined_score,
                query_start_aligned,
                query_end_aligned
            );

            // Store as candidate if better than current best
            if combined_score > best_score {
                best_score = combined_score;
                best_alignment_data = Some((
                    cigar_for_candidate,
                    combined_score,
                    alignment_start_pos,
                    query_start_aligned,
                    query_end_aligned,
                ));
            }
        } // End of seed_job iteration

        // Use the best alignment candidate for this chain
        if let Some((
            mut cigar_for_alignment,
            combined_score,
            alignment_start_pos,
            query_start_aligned,
            query_end_aligned,
        )) = best_alignment_data
        {
            log::debug!(
                "{}: Chain {} (weight={}, kept={}): BEST score={} from {} candidates",
                query_name,
                chain_idx,
                chain.weight,
                chain.kept,
                combined_score,
                mapping.seed_jobs.len()
            );

            // CRITICAL: Validate and correct CIGAR length to match query length
            // Safety check for systematic off-by-one errors in CIGAR generation
            // Only count operations that consume query bases (M, I, S, =, X)
            let cigar_len: i32 = cigar_for_alignment
                .iter()
                .filter_map(|&(op, len)| match op as char {
                    'M' | 'I' | 'S' | '=' | 'X' => Some(len),
                    _ => None, // D, H, N, P don't consume query
                })
                .sum();

            if cigar_len != query_len {
                log::debug!(
                    "{}: Chain {}: CIGAR length mismatch: cigar={}, query={}, diff={} - adjusting last query-consuming operation",
                    query_name,
                    chain_idx,
                    cigar_len,
                    query_len,
                    query_len - cigar_len
                );

                // Adjust the last query-consuming operation (S or M) to match query length
                let diff = query_len - cigar_len;
                for op in cigar_for_alignment.iter_mut().rev() {
                    if op.0 == b'S' || op.0 == b'M' {
                        let old_len = op.1;
                        op.1 += diff;
                        if op.1 > 0 {
                            log::debug!(
                                "{}: Adjusted CIGAR op {} from {} to {}",
                                query_name,
                                op.0 as char,
                                old_len,
                                op.1
                            );
                            break;
                        } else {
                            log::warn!(
                                "{}: CIGAR op {} became negative ({})! Setting to 0 and continuing...",
                                query_name,
                                op.0 as char,
                                op.1
                            );
                            op.1 = 0;
                        }
                    }
                }
            }

            // Use the alignment start position calculated from extensions
            let adjusted_ref_start = alignment_start_pos;

            // Convert global position to chromosome-specific position
            let global_pos = adjusted_ref_start as i64;
            let (pos_f, _is_rev_depos) = bwa_idx.bns.bns_depos(global_pos);
            let rid = bwa_idx.bns.bns_pos2rid(pos_f);

            let (ref_name, ref_id, chr_pos) =
                if rid >= 0 && (rid as usize) < bwa_idx.bns.annotations.len() {
                    let ann = &bwa_idx.bns.annotations[rid as usize];
                    let chr_relative_pos = pos_f - ann.offset as i64;
                    (ann.name.clone(), rid as usize, chr_relative_pos as u64)
                } else {
                    log::warn!(
                        "{}: Invalid reference ID {} for position {}",
                        query_name,
                        rid,
                        global_pos
                    );
                    ("unknown_ref".to_string(), 0, 0)
                };

            // Calculate query bounds from aligned region
            let query_start = query_start_aligned;
            let query_end = query_end_aligned;
            let seed_coverage = chain.weight; // Use chain weight as seed coverage for MAPQ

            // Generate hash for tie-breaking (based on position and strand)
            let hash = hash_64((chr_pos << 1) | (if chain.is_rev { 1 } else { 0 }));

            // Generate MD tag by comparing actual reference and query sequences
            let md_tag = if !pac_data.is_empty() {
                // Calculate reference length from CIGAR (M and D consume reference)
                let ref_len: i32 = cigar_for_alignment
                    .iter()
                    .filter_map(|&(op, len)| match op as char {
                        'M' | 'D' => Some(len),
                        _ => None,
                    })
                    .sum();

                // Extract reference sequence for aligned region
                let ref_start = adjusted_ref_start as i64;
                let ref_end = ref_start + ref_len as i64;
                let ref_aligned = bwa_idx.bns.bns_get_seq(&pac_data, ref_start, ref_end);

                // Extract query sequence for aligned region (already in 2-bit encoding)
                let query_aligned = &full_query[query_start as usize..query_end as usize];

                // Generate MD tag by comparing sequences
                Alignment::generate_md_tag(&ref_aligned, query_aligned, &cigar_for_alignment)
            } else {
                // Fallback if PAC file not available: simple MD tag from CIGAR
                let match_len: i32 = cigar_for_alignment
                    .iter()
                    .filter_map(|&(op, len)| if op == b'M' { Some(len) } else { None })
                    .sum();
                format!("{}", match_len)
            };

            // Calculate exact NM (edit distance) from MD tag and CIGAR
            let nm = Alignment::calculate_exact_nm(&md_tag, &cigar_for_alignment);

            alignments.push(Alignment {
                query_name: query_name.to_string(),
                flag: if chain.is_rev { sam_flags::REVERSE } else { 0 },
                ref_name,
                ref_id,
                pos: chr_pos,
                mapq: 60,              // Will be calculated by mark_secondary_alignments
                score: combined_score, // Use combined score from left + seed + right
                cigar: cigar_for_alignment,
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(query_seq).to_string(),
                qual: query_qual.to_string(),
                tags: vec![
                    ("AS".to_string(), format!("i:{}", combined_score)),
                    ("NM".to_string(), format!("i:{}", nm)),
                    ("MD".to_string(), format!("Z:{}", md_tag)),
                ],
                // Internal fields for alignment selection
                query_start,
                query_end,
                seed_coverage,
                hash,
                frac_rep: chain.frac_rep,
            });
        } else {
            log::warn!(
                "{}: Chain {} has no valid alignment candidates from {} seeds",
                query_name,
                chain_idx,
                mapping.seed_jobs.len()
            );
        }
    } // End of chain iteration

    // Log buffer capacity validation
    if max_smem_count > query_len {
        log::debug!(
            "{}: SMEM buffer grew beyond initial capacity! max_smem_count={} > query_len={} (growth factor: {:.2}x)",
            query_name,
            max_smem_count,
            query_len,
            max_smem_count as f64 / query_len as f64
        );
    } else {
        log::debug!(
            "{}: SMEM buffer stayed within capacity. max_smem_count={} <= query_len={} (utilization: {:.1}%)",
            query_name,
            max_smem_count,
            query_len,
            (max_smem_count as f64 / query_len as f64) * 100.0
        );
    }

    #[cfg(feature = "debug-logging")]
    if is_debug_read {
        log::debug!("[DEBUG_READ] Generated {} SMEM(s)", all_smems.len());
        log::debug!("[DEBUG_READ] Created {} alignment(s)", alignments.len());
        for (i, aln) in alignments.iter().enumerate() {
            log::debug!(
                "[DEBUG_READ] Alignment[{}]: {}:{} MAPQ={} Score={} CIGAR={}",
                i,
                aln.ref_name,
                aln.pos,
                aln.mapq,
                aln.score,
                aln.cigar_string()
            );
        }
    }

    // === ALIGNMENT SELECTION: Sort, Mark Secondary, Calculate MAPQ ===
    // Implements bwa-mem2 mem_mark_primary_se logic (bwamem.cpp:1420-1464)
    if !alignments.is_empty() {
        // Sort alignments by score (descending), then by hash (for tie-breaking)
        // Matches C++ alnreg_hlt comparator in bwamem.cpp:155
        alignments.sort_by(|a, b| {
            match b.score.cmp(&a.score) {
                std::cmp::Ordering::Equal => a.hash.cmp(&b.hash), // Tie-breaker
                other => other,
            }
        });

        // Mark secondary alignments and calculate MAPQ
        // This modifies alignments in-place:
        // - Sets sam_flags::SECONDARY flag for secondary alignments
        // - Calculates proper MAPQ values (0-60) based on score differences
        mark_secondary_alignments(&mut alignments, _opt);

        log::debug!(
            "{}: After alignment selection: {} alignments ({} primary, {} secondary)",
            query_name,
            alignments.len(),
            alignments.iter().filter(|a| a.flag & sam_flags::SECONDARY == 0).count(),
            alignments.iter().filter(|a| a.flag & sam_flags::SECONDARY != 0).count()
        );

        let xa_tags = generate_xa_tags(&alignments, _opt);
        let sa_tags = generate_sa_tags(&alignments);

        for aln in alignments.iter_mut() {
            // Only consider non-secondary alignments for SA/XA tags
            if aln.flag & sam_flags::SECONDARY == 0 {
                if let Some(sa_tag) = sa_tags.get(&aln.query_name) {
                    // This is part of a chimeric read, add SA tag
                    aln.tags.push(("SA".to_string(), sa_tag.clone()));
                    log::debug!(
                        "{}: Added SA tag for read {}",
                        aln.query_name,
                        aln.query_name
                    );
                } else if let Some(xa_tag) = xa_tags.get(&aln.query_name) {
                    // Not a chimeric read, add XA tag if available
                    aln.tags.push(("XA".to_string(), xa_tag.clone()));
                    log::debug!(
                        "{}: Added XA tag with {} alternative alignments",
                        aln.query_name,
                        xa_tag.matches(';').count()
                    );
                }
            }
        }
    } else {
        // === NO ALIGNMENTS FOUND: Create unmapped read (C++ bwa-mem2 behavior) ===
        // When no seeds/chains were generated, we must still output the read as unmapped
        // (SAM flag sam_flags::UNMAPPED) to match C++ bwa-mem2 behavior and avoid silently dropping reads
        log::debug!(
            "{}: No alignments generated (no seeds or all chains filtered), creating unmapped read",
            query_name
        );

        alignments.push(Alignment {
            query_name: query_name.to_string(),
            flag: sam_flags::UNMAPPED,
            ref_name: "*".to_string(),
            ref_id: 0, // 0 for unmapped (doesn't correspond to any real chromosome)
            pos: 0,
            mapq: 0,
            score: 0,
            cigar: Vec::new(), // Empty CIGAR = "*" in SAM format
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::from_utf8_lossy(query_seq).to_string(),
            qual: query_qual.to_string(),
            tags: vec![
                ("AS".to_string(), "i:0".to_string()),
                ("NM".to_string(), "i:0".to_string()),
            ],
            // Internal fields (not used for unmapped reads)
            query_start: 0,
            query_end: 0,
            seed_coverage: 0,
            hash: 0,
            frac_rep: 0.0, // Initial placeholder
        });
    }

    alignments
}

// ============================================================================
// SEED CHAINING
// ============================================================================
//
// This section contains the seed chaining algorithm:
// - Dynamic programming-based chaining
// - Chain scoring and filtering
// - Extension of chains into alignments
// ============================================================================

pub fn chain_seeds(mut seeds: Vec<Seed>, opt: &MemOpt) -> (Vec<Chain>, Vec<Seed>) {
    if seeds.is_empty() {
        return (Vec::new(), seeds);
    }

    // 1. Sort seeds: by query_pos, then by ref_pos
    seeds.sort_by_key(|s| (s.query_pos, s.ref_pos));

    let num_seeds = seeds.len();
    let mut dp = vec![0; num_seeds]; // dp[i] stores the max score of a chain ending at seeds[i]
    let mut prev_seed_idx = vec![None; num_seeds]; // To reconstruct the chain

    // Use max_chain_gap from options for gap penalty calculation
    let max_gap = opt.max_chain_gap;

    // 2. Dynamic Programming
    for i in 0..num_seeds {
        dp[i] = seeds[i].len; // Initialize with the seed's own length as score
        for j in 0..i {
            // Check for compatibility: seed[j] must end before seed[i] starts in both query and reference
            // And they must be on the same strand
            if seeds[j].is_rev == seeds[i].is_rev
                && seeds[j].query_pos + seeds[j].len < seeds[i].query_pos
                && seeds[j].ref_pos + (seeds[j].len as u64) < seeds[i].ref_pos
            {
                // Calculate distances (matching C++ test_and_merge logic)
                let x = seeds[i].query_pos - seeds[j].query_pos; // query distance
                let y = seeds[i].ref_pos as i32 - seeds[j].ref_pos as i32; // reference distance
                let q_gap = x - seeds[j].len; // gap after seed[j] ends
                let r_gap = y - seeds[j].len as i32; // gap after seed[j] ends

                // C++ constraint 1: y >= 0 (new seed downstream on reference)
                if y < 0 {
                    continue;
                }

                // C++ constraint 2: Diagonal band width check (|x - y| <= w)
                // This ensures seeds stay within a diagonal band
                let diagonal_offset = x - y;
                if diagonal_offset.abs() > opt.w {
                    continue; // Seeds too far from diagonal
                }

                // C++ constraint 3 & 4: max_chain_gap check
                if q_gap > max_gap || r_gap > max_gap {
                    continue; // Gap too large
                }

                // Simple gap penalty (average of query and reference gaps)
                let current_gap_penalty = (q_gap + r_gap.abs()) / 2;

                let potential_score = dp[j] + seeds[i].len - current_gap_penalty;

                if potential_score > dp[i] {
                    dp[i] = potential_score;
                    prev_seed_idx[i] = Some(j);
                }
            }
        }
    }

    // 3. Multi-chain extraction via iterative peak finding
    // Algorithm:
    // 1. Find highest-scoring peak in DP array
    // 2. Backtrack to reconstruct chain
    // 3. Mark seeds in chain as "used"
    // 4. Repeat until no more peaks above min_chain_weight
    // This matches bwa-mem2's approach to multi-chain generation

    let mut chains = Vec::new();
    let mut used_seeds = vec![false; num_seeds]; // Track which seeds are already in chains

    // Iteratively extract chains by finding peaks
    loop {
        // Find the highest unused peak
        let mut best_chain_score = opt.min_chain_weight; // Only consider chains above minimum
        let mut best_chain_end_idx: Option<usize> = None;

        for i in 0..num_seeds {
            if !used_seeds[i] && dp[i] >= best_chain_score {
                best_chain_score = dp[i];
                best_chain_end_idx = Some(i);
            }
        }

        // Stop if no more chains above threshold
        if best_chain_end_idx.is_none() {
            break;
        }

        // Backtrack to reconstruct this chain
        let mut current_idx = best_chain_end_idx.unwrap();
        let mut chain_seeds_indices = Vec::new();
        let mut current_seed = &seeds[current_idx];

        let mut query_start = current_seed.query_pos;
        let mut query_end = current_seed.query_pos + current_seed.len;
        let mut ref_start = current_seed.ref_pos;
        let mut ref_end = current_seed.ref_pos + current_seed.len as u64;
        let is_rev = current_seed.is_rev;

        // Backtrack through the chain
        loop {
            chain_seeds_indices.push(current_idx);
            used_seeds[current_idx] = true; // Mark seed as used

            // Get previous seed in chain
            if let Some(prev_idx) = prev_seed_idx[current_idx] {
                current_idx = prev_idx;
                current_seed = &seeds[current_idx];

                // Update chain bounds
                query_start = query_start.min(current_seed.query_pos);
                query_end = query_end.max(current_seed.query_pos + current_seed.len);
                ref_start = ref_start.min(current_seed.ref_pos);
                ref_end = ref_end.max(current_seed.ref_pos + current_seed.len as u64);
            } else {
                break; // Reached start of chain
            }
        }

        chain_seeds_indices.reverse(); // Order from start to end

        log::debug!(
            "Chain extraction: chain_idx={}, num_seeds={}, score={}, query=[{}, {}), ref=[{}, {}), is_rev={}",
            chains.len(),
            chain_seeds_indices.len(),
            best_chain_score,
            query_start,
            query_end,
            ref_start,
            ref_end,
            is_rev
        );

        chains.push(Chain {
            score: best_chain_score,
            seeds: chain_seeds_indices,
            query_start,
            query_end,
            ref_start,
            ref_end,
            is_rev,
            weight: 0,     // Will be calculated by filter_chains()
            kept: 0,       // Will be set by filter_chains()
            frac_rep: 0.0, // Initial placeholder
        });

        // Safety limit: stop after extracting a reasonable number of chains
        // This prevents pathological cases from consuming too much memory
        if chains.len() >= 100 {
            log::debug!(
                "Extracted maximum of 100 chains from {} seeds, stopping",
                num_seeds
            );
            break;
        }
    }

    log::debug!(
        "Chain extraction: {} seeds → {} chains (min_weight={})",
        num_seeds,
        chains.len(),
        opt.min_chain_weight
    );

    (chains, seeds)
}

// ============================================================================
// BWT AND SUFFIX ARRAY HELPER FUNCTIONS
// ============================================================================
//
// This section contains low-level BWT and suffix array access functions
// used during FM-Index search and seed extension
// ============================================================================

// Function to get BWT base from cp_occ format (for loaded indices)
// Returns 0-3 for bases A/C/G/T, or 4 for sentinel
pub fn get_bwt_base_from_cp_occ(cp_occ: &[CpOcc], pos: u64) -> u8 {
    let cp_block = (pos >> CP_SHIFT) as usize;

    // Safety: check bounds
    if cp_block >= cp_occ.len() {
        log::warn!(
            "get_bwt_base_from_cp_occ: cp_block {} >= cp_occ.len() {}",
            cp_block,
            cp_occ.len()
        );
        return 4; // Return sentinel for out-of-bounds
    }

    let offset_in_block = pos & ((1 << CP_SHIFT) - 1);
    let bit_position = 63 - offset_in_block;

    // Check which of the 4 one-hot encoded arrays has a 1 at this position
    for base in 0..4 {
        if (cp_occ[cp_block].bwt_encoding_bits[base] >> bit_position) & 1 == 1 {
            return base as u8;
        }
    }
    4 // Return 4 for sentinel (no bit set means sentinel position)
}

// Function to get the next BWT position from a BWT coordinate
// Returns None if we hit the sentinel (which should not be navigated)
pub fn get_bwt(bwa_idx: &BwaIndex, pos: u64) -> Option<u64> {
    let base = if !bwa_idx.bwt.bwt_data.is_empty() {
        // Index was just built, use raw bwt_data
        bwa_idx.bwt.get_bwt_base(pos)
    } else {
        // Index was loaded from disk, use cp_occ format
        get_bwt_base_from_cp_occ(&bwa_idx.cp_occ, pos)
    };

    // If we hit the sentinel (base == 4), return None
    if base == 4 {
        return None;
    }

    Some(bwa_idx.bwt.cumulative_count[base as usize] + get_occ(bwa_idx, pos as i64, base) as u64)
}

pub fn get_sa_entry(bwa_idx: &BwaIndex, mut pos: u64) -> u64 {
    let original_pos = pos;
    let mut count = 0;
    const MAX_ITERATIONS: u64 = 10000; // Safety limit to prevent infinite loops

    // eprintln!("get_sa_entry: starting with pos={}, sa_intv={}, seq_len={}, cp_occ.len()={}",
    //           original_pos, bwa_idx.bwt.sa_sample_interval, bwa_idx.bwt.seq_len, bwa_idx.cp_occ.len());

    while pos % bwa_idx.bwt.sa_sample_interval as u64 != 0 {
        // Safety check: prevent infinite loops
        if count >= MAX_ITERATIONS {
            log::error!(
                "get_sa_entry exceeded MAX_ITERATIONS ({}) - possible infinite loop!",
                MAX_ITERATIONS
            );
            log::error!(
                "  original_pos={}, current_pos={}, count={}",
                original_pos,
                pos,
                count
            );
            log::error!(
                "  sa_intv={}, seq_len={}",
                bwa_idx.bwt.sa_sample_interval,
                bwa_idx.bwt.seq_len
            );
            return count; // Return what we have so far
        }

        let _old_pos = pos;
        match get_bwt(bwa_idx, pos) {
            Some(new_pos) => {
                pos = new_pos;
                count += 1;
                // if count <= 10 {
                //     eprintln!("  BWT step {}: pos {} -> {} (count={})", count, _old_pos, pos, count);
                // }
            }
            None => {
                // Hit sentinel - return the accumulated count
                // eprintln!("  BWT step {}: pos {} -> SENTINEL (count={})", count + 1, _old_pos, count);
                // eprintln!("get_sa_entry: original_pos={}, hit_sentinel, count={}, result={}",
                //           original_pos, count, count);
                return count;
            }
        }
    }

    let sa_index = (pos / bwa_idx.bwt.sa_sample_interval as u64) as usize;
    let sa_ms_byte = bwa_idx.bwt.sa_high_bytes[sa_index] as u64;
    let sa_ls_word = bwa_idx.bwt.sa_low_words[sa_index] as u64;
    let sa_val = (sa_ms_byte << 32) | sa_ls_word;

    // Handle sentinel: SA values can point to the sentinel position (seq_len)
    // The sentinel represents the end-of-string marker, which wraps to position 0
    // seq_len = (l_pac << 1) + 1 (forward + RC + sentinel)
    // So sentinel position is seq_len - 1 = (l_pac << 1)
    let sentinel_pos = bwa_idx.bns.packed_sequence_length << 1;
    let adjusted_sa_val = if sa_val >= sentinel_pos {
        // SA points to or past sentinel - wrap to beginning (position 0)
        log::debug!(
            "SA value {} is at/past sentinel {} - wrapping to 0",
            sa_val,
            sentinel_pos
        );
        0
    } else {
        sa_val
    };

    let result = adjusted_sa_val + count;

    log::debug!(
        "get_sa_entry: original_pos={}, final_pos={}, count={}, sa_index={}, sa_val={}, adjusted={}, result={}, l_pac={}, sentinel={}",
        original_pos,
        pos,
        count,
        sa_index,
        sa_val,
        adjusted_sa_val,
        result,
        bwa_idx.bns.packed_sequence_length,
        (bwa_idx.bns.packed_sequence_length << 1)
    );
    result
}

#[cfg(test)]
mod tests {
    use super::Alignment;
    use super::sam_flags;
    use crate::align::SMEM;
    use crate::fm_index::{backward_ext, popcount64};
    use crate::index::BwaIndex;
    use std::path::Path;

    #[test]
    fn test_popcount64_neon() {
        // Test the hardware-optimized popcount implementation
        // This ensures our NEON implementation matches the software version

        // Test basic cases
        assert_eq!(popcount64(0), 0);
        assert_eq!(popcount64(1), 1);
        assert_eq!(popcount64(0xFFFFFFFFFFFFFFFF), 64);
        assert_eq!(popcount64(0x8000000000000000), 1);

        // Test various bit patterns
        assert_eq!(popcount64(0b1010101010101010), 8);
        assert_eq!(popcount64(0b11111111), 8);
        assert_eq!(popcount64(0xFF00FF00FF00FF00), 32);
        assert_eq!(popcount64(0x0F0F0F0F0F0F0F0F), 32);

        // Test random patterns that match expected popcount
        assert_eq!(popcount64(0x123456789ABCDEF0), 32);
        assert_eq!(popcount64(0xAAAAAAAAAAAAAAAA), 32); // Alternating bits
        assert_eq!(popcount64(0x5555555555555555), 32); // Alternating bits (complement)
    }

    #[test]
    fn test_backward_ext() {
        let prefix = Path::new("test_data/test_ref.fa");

        // Skip if test data doesn't exist
        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext - could not load index");
                return;
            }
        };

        let smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };
        let new_smem = backward_ext(&bwa_idx, smem, 0); // 0 is 'A'
        assert_ne!(new_smem.interval_size, 0);
    }

    #[test]
    fn test_backward_ext_multiple_bases() {
        // Test backward extension with all four bases
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_multiple_bases - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_multiple_bases - could not load index");
                return;
            }
        };

        // Start with full range
        let initial_smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Test extending with each base
        for base in 0..4 {
            let extended = super::backward_ext(&bwa_idx, initial_smem, base);

            // Extended range should be smaller or equal to initial range
            assert!(
                extended.interval_size <= initial_smem.interval_size,
                "Extended range size {} should be <= initial size {} for base {}",
                extended.interval_size,
                initial_smem.interval_size,
                base
            );

            // k should be within bounds
            assert!(
                extended.bwt_interval_start < bwa_idx.bwt.seq_len,
                "Extended k should be within sequence length"
            );
        }
    }

    #[test]
    fn test_backward_ext_chain() {
        // Test chaining multiple backward extensions (like building a seed)
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_chain - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_chain - could not load index");
                return;
            }
        };

        // Start with full range
        let mut smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Build a seed by extending with ACGT
        let bases = vec![0u8, 1, 2, 3]; // ACGT
        let mut prev_s = smem.interval_size;

        for (i, &base) in bases.iter().enumerate() {
            smem = super::backward_ext(&bwa_idx, smem, base);

            // Range should generally get smaller (or stay same) with each extension
            // (though it could stay the same if the pattern is very common)
            assert!(
                smem.interval_size <= prev_s,
                "After extension {}, range size {} should be <= previous {}",
                i,
                smem.interval_size,
                prev_s
            );

            prev_s = smem.interval_size;

            // If range becomes 0, we can't extend further
            if smem.interval_size == 0 {
                break;
            }
        }
    }

    #[test]
    fn test_backward_ext_zero_range() {
        // Test backward extension when starting with zero range
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_zero_range - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_zero_range - could not load index");
                return;
            }
        };

        let smem = SMEM {
            bwt_interval_start: 0,
            interval_size: 0, // Zero range
            ..Default::default()
        };

        let extended = super::backward_ext(&bwa_idx, smem, 0);

        // Extending a zero range should still give zero range
        assert_eq!(
            extended.interval_size, 0,
            "Extending zero range should give zero range"
        );
    }

    #[test]
    fn test_smem_structure() {
        // Test SMEM structure creation and defaults
        let smem1 = SMEM {
            read_id: 0,
            query_start: 10,
            query_end: 20,
            bwt_interval_start: 5,
            bwt_interval_end: 15,
            interval_size: 10,
            is_reverse_complement: false,
        };

        assert_eq!(smem1.query_start, 10);
        assert_eq!(smem1.query_end, 20);
        assert_eq!(smem1.interval_size, 10);

        // Test default
        let smem2 = SMEM::default();
        assert_eq!(smem2.read_id, 0);
        assert_eq!(smem2.query_start, 0);
        assert_eq!(smem2.query_end, 0);
    }

    // NOTE: Base encoding tests moved to tests/session30_regression_tests.rs
    // This reduces clutter in production code files

    #[test]
    fn test_get_sa_entry_basic() {
        // This test requires an actual index file to be present
        // We'll use a simple test to verify the function doesn't crash
        let prefix = Path::new("test_data/test_ref.fa");

        // Only run if test data exists
        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_basic - could not load index");
                return;
            }
        };

        // Test getting SA entry at position 0 (should return a valid reference position)
        let sa_entry = super::get_sa_entry(&bwa_idx, 0);

        // SA entry should be within the reference sequence length
        assert!(
            sa_entry < bwa_idx.bwt.seq_len,
            "SA entry {} should be less than seq_len {}",
            sa_entry,
            bwa_idx.bwt.seq_len
        );
    }

    #[test]
    fn test_get_sa_entry_sampled_position() {
        // Test getting SA entry at a sampled position (divisible by sa_intv)
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_sampled_position - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_sampled_position - could not load index");
                return;
            }
        };

        // Test at a sampled position (should directly lookup in SA array)
        let sampled_pos = bwa_idx.bwt.sa_sample_interval as u64;
        let sa_entry = super::get_sa_entry(&bwa_idx, sampled_pos);

        assert!(
            sa_entry < bwa_idx.bwt.seq_len,
            "SA entry at sampled position should be within sequence length"
        );
    }

    #[test]
    fn test_get_sa_entry_multiple_positions() {
        // Test getting SA entries for multiple positions
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_multiple_positions - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_multiple_positions - could not load index");
                return;
            }
        };

        // Test several positions
        let test_positions = vec![0u64, 1, 10, 100];

        for pos in test_positions {
            if pos >= bwa_idx.bwt.seq_len {
                continue;
            }

            let sa_entry = super::get_sa_entry(&bwa_idx, pos);

            // All SA entries should be valid (within sequence length)
            assert!(
                sa_entry < bwa_idx.bwt.seq_len,
                "SA entry for pos {} should be within sequence length",
                pos
            );
        }
    }

    #[test]
    fn test_get_sa_entry_consistency() {
        // Test that get_sa_entry returns consistent results
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_consistency - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_consistency - could not load index");
                return;
            }
        };

        let pos = 5u64;
        let sa_entry1 = super::get_sa_entry(&bwa_idx, pos);
        let sa_entry2 = super::get_sa_entry(&bwa_idx, pos);

        // Same position should always return same SA entry
        assert_eq!(
            sa_entry1, sa_entry2,
            "get_sa_entry should return consistent results for the same position"
        );
    }

    #[test]
    fn test_get_bwt_basic() {
        // Test get_bwt function
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_bwt_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_bwt_basic - could not load index");
                return;
            }
        };

        // Test getting BWT at various positions
        for pos in 0..10u64 {
            let bwt_result = super::get_bwt(&bwa_idx, pos);

            // Either we get a valid position or None (sentinel)
            if let Some(new_pos) = bwt_result {
                assert!(
                    new_pos < bwa_idx.bwt.seq_len,
                    "BWT position should be within sequence length"
                );
            }
            // If None, we hit the sentinel - that's ok
        }
    }

    #[test]
    fn test_batched_alignment_infrastructure() {
        // Test that the batched alignment infrastructure works correctly
        use crate::banded_swa::BandedPairWiseSW;

        let sw_params = BandedPairWiseSW::new(
            4,
            2,
            4,
            2,
            100,
            0,
            5,
            5,
            super::DEFAULT_SCORING_MATRIX,
            2,
            -4,
        );

        // Create test alignment jobs
        let query1 = vec![0u8, 1, 2, 3]; // ACGT
        let target1 = vec![0u8, 1, 2, 3]; // ACGT (perfect match)

        let query2 = vec![0u8, 0, 1, 1]; // AACC
        let target2 = vec![0u8, 0, 1, 1]; // AACC (perfect match)

        let jobs = vec![
            super::AlignmentJob {
                seed_idx: 0,
                query: query1.clone(),
                target: target1.clone(),
                band_width: 10,
                query_offset: 0, // Test: align from start
                direction: None, // Legacy test mode
                seed_len: 4,     // Actual sequence length (4bp queries)
            },
            super::AlignmentJob {
                seed_idx: 1,
                query: query2.clone(),
                target: target2.clone(),
                band_width: 10,
                query_offset: 0, // Test: align from start
                direction: None, // Legacy test mode
                seed_len: 4,     // Actual sequence length (4bp queries)
            },
        ];

        // Test scalar execution
        let scalar_results = super::execute_scalar_alignments(&sw_params, &jobs);
        assert_eq!(
            scalar_results.len(),
            2,
            "Should return 2 results for 2 jobs"
        );
        assert!(
            !scalar_results[0].1.is_empty(),
            "First CIGAR should not be empty"
        );
        assert!(
            !scalar_results[1].1.is_empty(),
            "Second CIGAR should not be empty"
        );

        // Test batched execution
        let batched_results = super::execute_batched_alignments(&sw_params, &jobs);
        assert_eq!(
            batched_results.len(),
            2,
            "Should return 2 results for 2 jobs"
        );

        // Results should be identical (CIGARs and scores)
        assert_eq!(
            scalar_results[0].0, batched_results[0].0,
            "Scores should match"
        );
        assert_eq!(
            scalar_results[0].1, batched_results[0].1,
            "CIGARs should match"
        );
        assert_eq!(
            scalar_results[1].0, batched_results[1].0,
            "Scores should match"
        );
        assert_eq!(
            scalar_results[1].1, batched_results[1].1,
            "CIGARs should match"
        );
    }

    #[test]
    fn test_generate_seeds_basic() {
        use crate::mem_opt::MemOpt;
        use std::path::Path;

        let prefix = Path::new("test_data/test_ref.fa");
        if !prefix.exists() {
            eprintln!("Skipping test_generate_seeds_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_generate_seeds_basic - could not load index");
                return;
            }
        };

        let mut opt = MemOpt::default();
        opt.min_seed_len = 10; // Ensure our seed is long enough

        let query_name = "test_query";
        let query_seq = b"ACGTACGTACGT"; // 12bp
        let query_qual = "IIIIIIIIIIII";

        // Test doesn't need real MD tags, use empty pac_data
        let pac_data: &[u8] = &[];
        let alignments =
            super::generate_seeds(&bwa_idx, pac_data, query_name, query_seq, query_qual, &opt);

        assert!(
            !alignments.is_empty(),
            "Expected at least one alignment for a matching query"
        );

        let primary_alignment = alignments.iter().find(|a| a.flag & sam_flags::SECONDARY == 0);
        assert!(primary_alignment.is_some(), "Expected a primary alignment");

        let pa = primary_alignment.unwrap();
        assert_eq!(pa.ref_name, "test_sequence");
        assert!(
            pa.score > 0,
            "Expected a positive score for a good match, got {}",
            pa.score
        );
        assert!(pa.pos < 60, "Position should be within reference length");
        assert_eq!(pa.cigar_string(), "12M", "Expected a perfect match CIGAR");
    }

    #[test]
    fn test_hard_clipping_for_supplementary() {
        // Test that soft clips (S) are converted to hard clips (H) for supplementary alignments
        // Per bwa-mem2 behavior (bwamem.cpp:1585-1586)

        // Create a primary alignment with soft clips
        let primary = Alignment {
            query_name: "read1".to_string(),
            flag: 0, // Primary alignment
            ref_name: "chr1".to_string(),
            ref_id: 0,
            pos: 100,
            mapq: 60,
            score: 100,
            cigar: vec![(b'S', 5), (b'M', 50), (b'S', 10)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: "A".repeat(65),
            qual: "I".repeat(65),
            tags: vec![],
            query_start: 0,
            query_end: 65,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
        };

        // Create a supplementary alignment with soft clips
        let supplementary = Alignment {
            query_name: "read1".to_string(),
            flag: sam_flags::SUPPLEMENTARY,
            ref_name: "chr2".to_string(),
            ref_id: 1,
            pos: 200,
            mapq: 30,
            score: 80,
            cigar: vec![(b'S', 5), (b'M', 50), (b'S', 10)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: "A".repeat(65),
            qual: "I".repeat(65),
            tags: vec![],
            query_start: 0,
            query_end: 65,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
        };

        // Create a secondary alignment with soft clips
        let secondary = Alignment {
            query_name: "read1".to_string(),
            flag: sam_flags::SECONDARY,
            ref_name: "chr3".to_string(),
            ref_id: 2,
            pos: 300,
            mapq: 0,
            score: 70,
            cigar: vec![(b'S', 5), (b'M', 50), (b'S', 10)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: "A".repeat(65),
            qual: "I".repeat(65),
            tags: vec![],
            query_start: 0,
            query_end: 65,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
        };

        // Verify CIGAR strings
        assert_eq!(
            primary.cigar_string(),
            "5S50M10S",
            "Primary alignment should keep soft clips (S)"
        );

        assert_eq!(
            supplementary.cigar_string(),
            "5H50M10H",
            "Supplementary alignment should convert S to H"
        );

        assert_eq!(
            secondary.cigar_string(),
            "5S50M10S",
            "Secondary alignment should keep soft clips (S)"
        );

        println!("✅ Hard clipping test passed!");
        println!("   Primary:       {}", primary.cigar_string());
        println!("   Supplementary: {}", supplementary.cigar_string());
        println!("   Secondary:     {}", secondary.cigar_string());
    }
}
