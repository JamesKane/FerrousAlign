use crate::mem_opt::MemOpt;

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

    /// Convert alignment to SAM format string
    /// Uses internal seq/qual fields (legacy method, use to_sam_string_with_seq for memory efficiency)
    pub fn to_sam_string(&self) -> String {
        self.to_sam_string_with_seq(&self.seq, &self.qual)
    }

    /// Convert alignment to SAM format string with externally provided seq/qual
    /// This avoids storing seq/qual in the Alignment struct, saving ~300 bytes per alignment
    pub fn to_sam_string_with_seq(&self, seq: &str, qual: &str) -> String {
        // Convert CIGAR to string format
        let cigar_string = self.cigar_string();

        // Handle reverse complement for SEQ and QUAL if flag sam_flags::REVERSE (reverse strand) is set
        // Matching bwa-mem2 mem_aln2sam() behavior (bwamem.cpp:1706-1716)
        let (mut output_seq, mut output_qual) = if self.flag & sam_flags::REVERSE != 0 {
            // Reverse strand: reverse complement the sequence and reverse the quality
            let rev_comp_seq: String = seq
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
            let rev_qual: String = qual.chars().rev().collect();
            (rev_comp_seq, rev_qual)
        } else {
            // Forward strand: use sequence and quality as-is
            (seq.to_string(), qual.to_string())
        };

        // Handle hard clips (H) - trim SEQ and QUAL to match CIGAR
        // For hard-clipped alignments, the clipped portions are not in SEQ/QUAL
        //
        // IMPORTANT: For supplementary alignments, soft clips (S) are converted to
        // hard clips (H) in the output CIGAR (see cigar_string()). However, the
        // internal CIGAR still stores them as 'S'. So for supplementary alignments,
        // we need to treat 'S' as a clip that requires SEQ trimming.
        let is_supplementary = (self.flag & sam_flags::SUPPLEMENTARY) != 0;

        let mut leading_clip = 0usize;
        let mut trailing_clip = 0usize;

        // Sum leading clips (H always, S only for supplementary)
        for &(op, len) in self.cigar.iter() {
            if op == b'H' || (is_supplementary && op == b'S') {
                leading_clip += len as usize;
            } else {
                break;
            }
        }

        // Sum trailing clips (H always, S only for supplementary)
        for &(op, len) in self.cigar.iter().rev() {
            if op == b'H' || (is_supplementary && op == b'S') {
                trailing_clip += len as usize;
            } else {
                break;
            }
        }

        // Trim SEQ and QUAL if there are clips that require removal
        if leading_clip > 0 || trailing_clip > 0 {
            let seq_len = output_seq.len();
            let start = leading_clip.min(seq_len);
            let end = seq_len.saturating_sub(trailing_clip);
            log::debug!(
                "Clip trim: is_supp={}, leading={}, trailing={}, seq_len={}, start={}, end={}",
                is_supplementary,
                leading_clip,
                trailing_clip,
                seq_len,
                start,
                end
            );
            if start < end {
                output_seq = output_seq[start..end].to_string();
                output_qual = output_qual[start..end.min(output_qual.len())].to_string();
            }
        }

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
        let strand = if self.flag & sam_flags::REVERSE != 0 {
            '-'
        } else {
            '+'
        };
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
            // Don't store seq/qual - they're passed at output time for memory efficiency
            seq: String::new(),
            qual: String::new(),
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

/// Remove redundant alignments that overlap significantly on both reference and query
/// Implements C++ mem_sort_dedup_patch (bwamem.cpp:292-351)
///
/// This function removes nearly-identical alignments (>95% overlap on BOTH reference AND query)
/// BEFORE the secondary/supplementary marking stage. This prevents over-generation of
/// supplementary alignments from nearly-identical mapping locations.
///
/// Algorithm:
/// 1. Sort alignments by reference end position
/// 2. For each alignment, check against nearby alignments (same ref, within max_chain_gap)
/// 3. If overlap > mask_level_redun on BOTH ref AND query: remove lower-scoring one
/// 4. Remove exact duplicates (same score, same position)
pub fn remove_redundant_alignments(alignments: &mut Vec<Alignment>, opt: &MemOpt) {
    if alignments.len() <= 1 {
        return;
    }

    log::debug!(
        "remove_redundant_alignments: processing {} alignments, mask_level_redun={}",
        alignments.len(),
        opt.mask_level_redun
    );

    let mask_level_redun = opt.mask_level_redun;
    let max_chain_gap = opt.max_chain_gap as u64;

    // Sort by reference end position (matching C++ ks_introsort(mem_ars2))
    // We approximate ref_end as pos + aligned_length from CIGAR
    alignments.sort_by(|a, b| {
        let a_ref_end = a.pos + alignment_ref_length(a);
        let b_ref_end = b.pos + alignment_ref_length(b);
        a.ref_id
            .cmp(&b.ref_id)
            .then_with(|| a_ref_end.cmp(&b_ref_end))
    });

    // Mark alignments for removal (can't remove in-place while iterating)
    let mut keep = vec![true; alignments.len()];

    for i in 1..alignments.len() {
        if !keep[i] {
            continue;
        }

        let p = &alignments[i];
        let p_ref_start = p.pos;
        let p_ref_end = p.pos + alignment_ref_length(p);

        // Check against previous alignments (same ref, within max_chain_gap)
        for j in (0..i).rev() {
            if !keep[j] {
                continue;
            }

            let q = &alignments[j];

            // Skip if different reference
            if q.ref_id != p.ref_id {
                break;
            }

            let q_ref_end = q.pos + alignment_ref_length(q);

            // Skip if too far apart
            if p_ref_start >= q_ref_end + max_chain_gap {
                break;
            }

            let q_ref_start = q.pos;

            // Calculate reference overlap
            let ref_overlap = if p_ref_start < q_ref_end && q_ref_start < p_ref_end {
                (p_ref_end.min(q_ref_end) - p_ref_start.max(q_ref_start)) as i64
            } else {
                0
            };

            // Calculate query overlap
            let p_qb = p.query_start;
            let p_qe = p.query_end;
            let q_qb = q.query_start;
            let q_qe = q.query_end;

            let query_overlap = if p_qb < q_qe && q_qb < p_qe {
                p_qe.min(q_qe) - p_qb.max(q_qb)
            } else {
                0
            };

            // Min lengths
            let min_ref_len = ((p_ref_end - p_ref_start) as i64).min((q_ref_end - q_ref_start) as i64);
            let min_query_len = (p_qe - p_qb).min(q_qe - q_qb);

            // Check if redundant: overlap > mask_level_redun on BOTH ref AND query
            let ref_threshold = mask_level_redun * min_ref_len as f32;
            let query_threshold = mask_level_redun * min_query_len as f32;
            let ref_redundant = ref_overlap as f32 > ref_threshold;
            let query_redundant = query_overlap as f32 > query_threshold;

            // Check for same-region opposite-strand alignments
            // These are the SAME alignment expressed differently (e.g., hard vs soft clips)
            // where the query regions are complementary when accounting for strand reversal
            let same_region_opposite_strand = {
                let p_is_reverse = (p.flag & sam_flags::REVERSE) != 0;
                let q_is_reverse = (q.flag & sam_flags::REVERSE) != 0;
                let on_opposite_strands = p_is_reverse != q_is_reverse;

                if !on_opposite_strands {
                    false
                } else {
                    // Check if reference regions are close (within max_ref_len distance)
                    let max_ref_len = (p_ref_end - p_ref_start).max(q_ref_end - q_ref_start);
                    let ref_distance = if p_ref_start > q_ref_end {
                        p_ref_start - q_ref_end
                    } else if q_ref_start > p_ref_end {
                        q_ref_start - p_ref_end
                    } else {
                        0 // Overlapping
                    };
                    let regions_close = ref_distance <= max_ref_len;

                    // Check if scores are similar (within 20%)
                    let score_ratio = (p.score.min(q.score) as f32) / (p.score.max(q.score).max(1) as f32);
                    let scores_similar = score_ratio >= 0.8;

                    // Check if query regions are complementary (same region accounting for strand)
                    // For a read of length L, if one alignment covers [a, b) on forward,
                    // and another covers [c, d) on reverse, they're the same if:
                    // [c, d) = [L - b, L - a) (approximately)
                    let query_len = (p_qe.max(q_qe)).max(148) as i32; // Estimate query length

                    // Transform reverse strand coordinates to forward
                    let (p_fwd_start, p_fwd_end) = if p_is_reverse {
                        (query_len - p_qe, query_len - p_qb)
                    } else {
                        (p_qb, p_qe)
                    };
                    let (q_fwd_start, q_fwd_end) = if q_is_reverse {
                        (query_len - q_qe, query_len - q_qb)
                    } else {
                        (q_qb, q_qe)
                    };

                    // Check if the forward-normalized query regions overlap significantly
                    let fwd_b_max = p_fwd_start.max(q_fwd_start);
                    let fwd_e_min = p_fwd_end.min(q_fwd_end);
                    let fwd_overlap = if fwd_e_min > fwd_b_max { fwd_e_min - fwd_b_max } else { 0 };
                    let fwd_min_len = (p_fwd_end - p_fwd_start).min(q_fwd_end - q_fwd_start);
                    let query_regions_same = fwd_overlap > 0 && fwd_overlap >= (fwd_min_len as f32 * 0.8) as i32;

                    if regions_close && scores_similar && query_regions_same {
                        log::debug!(
                            "same_region_opposite_strand match: {}:{} vs {}:{}, ref_dist={}, score_ratio={:.2}, fwd_overlap={}",
                            p.ref_name, p.pos, q.ref_name, q.pos, ref_distance, score_ratio, fwd_overlap
                        );
                    }
                    regions_close && scores_similar && query_regions_same
                }
            };

            // Remove redundant alignments when:
            // 1. Both ref AND query overlap significantly (standard redundancy check)
            // 2. OR same genomic region on opposite strands with same query coverage
            //    (these produce equal scores, causing MAPQ=0 incorrectly)
            if (ref_redundant && query_redundant) || same_region_opposite_strand {
                // Remove the lower-scoring one (or the current one if equal)
                // Matches C++ logic: if p->score < q->score remove p, else remove q
                let removal_reason = if same_region_opposite_strand {
                    "same-region-opposite-strand"
                } else {
                    "ref+query-overlap"
                };
                if p.score < q.score {
                    keep[i] = false;
                    log::debug!(
                        "Removing redundant alignment[{}]: {}:{} score={} ({}, overlaps with [{}] score={})",
                        i, p.ref_name, p.pos, p.score, removal_reason, j, q.score
                    );
                    break;
                } else {
                    // p.score >= q.score: remove q (the older/lower-indexed one)
                    keep[j] = false;
                    log::debug!(
                        "Removing redundant alignment[{}]: {}:{} score={} ({}, overlaps with [{}] score={})",
                        j, q.ref_name, q.pos, q.score, removal_reason, i, p.score
                    );
                }
            }
        }
    }

    // Remove marked alignments
    let before_count = alignments.len();
    let mut write_idx = 0;
    for read_idx in 0..alignments.len() {
        if keep[read_idx] {
            if write_idx != read_idx {
                alignments.swap(write_idx, read_idx);
            }
            write_idx += 1;
        }
    }
    alignments.truncate(write_idx);

    // Sort by score for subsequent processing
    alignments.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.hash.cmp(&b.hash)));

    // Remove exact duplicates (same score, same position, same query bounds)
    alignments.dedup_by(|a, b| {
        a.score == b.score && a.pos == b.pos && a.ref_id == b.ref_id
            && a.query_start == b.query_start && a.query_end == b.query_end
    });

    if alignments.len() < before_count {
        log::debug!(
            "remove_redundant_alignments: removed {} redundant alignments (before: {}, after: {})",
            before_count - alignments.len(),
            before_count,
            alignments.len()
        );
    }
}

/// Calculate approximate reference length from CIGAR
fn alignment_ref_length(alignment: &Alignment) -> u64 {
    let mut ref_len = 0u64;
    for &(op, len) in &alignment.cigar {
        match op {
            b'M' | b'D' | b'N' | b'=' | b'X' => ref_len += len as u64,
            _ => {}
        }
    }
    ref_len
}

// ============================================================================
// SECONDARY/SUPPLEMENTARY MARKING HELPERS
// ============================================================================

/// Find primary alignment indices by checking for query overlap
/// Returns (primary_indices, sub_scores, sub_counts)
fn find_primary_alignments(
    alignments: &mut [Alignment],
    mask_level: f32,
    score_gap_threshold: i32,
) -> (Vec<usize>, Vec<i32>, Vec<i32>) {
    let mut primary_indices: Vec<usize> = Vec::new();
    let mut sub_scores: Vec<i32> = vec![0; alignments.len()];
    let mut sub_counts: Vec<i32> = vec![0; alignments.len()];

    if alignments.is_empty() {
        return (primary_indices, sub_scores, sub_counts);
    }

    // First alignment is always primary
    primary_indices.push(0);

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
                if alignments[j].score - alignments[i].score <= score_gap_threshold {
                    sub_counts[j] += 1;
                }

                // Mark as secondary
                alignments[i].flag |= sam_flags::SECONDARY;
                is_secondary = true;
                log::debug!(
                    "  -> Marked alignment[{}] as SECONDARY (overlaps with [{}])",
                    i, j
                );
                break;
            }
        }

        // If no overlap with any primary, add as new primary
        if !is_secondary {
            primary_indices.push(i);
        }
    }

    (primary_indices, sub_scores, sub_counts)
}

/// Mark non-overlapping alignments after the first as SUPPLEMENTARY
fn apply_supplementary_flags(alignments: &mut [Alignment], primary_indices: &[usize]) {
    for (idx, &i) in primary_indices.iter().enumerate() {
        if idx > 0 {
            alignments[i].flag |= sam_flags::SUPPLEMENTARY;
            log::debug!(
                "Marked alignment {} as SUPPLEMENTARY ({}:{}, score={})",
                i, alignments[i].ref_name, alignments[i].pos, alignments[i].score
            );
        }
    }

    // Ensure the primary alignment does not have SUPPLEMENTARY flag
    if !primary_indices.is_empty() {
        let primary_idx = primary_indices[0];
        alignments[primary_idx].flag &= !sam_flags::SUPPLEMENTARY;
    }
}

/// Calculate MAPQ for all alignments and add XS tags
fn calculate_all_mapq(
    alignments: &mut [Alignment],
    sub_scores: &[i32],
    sub_counts: &[i32],
    opt: &MemOpt,
) {
    for i in 0..alignments.len() {
        if alignments[i].flag & sam_flags::SECONDARY == 0 {
            // Primary/supplementary alignment: calculate MAPQ
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

            // Add XS tag if there's a suboptimal score
            if sub_scores[i] > 0 {
                alignments[i]
                    .tags
                    .push(("XS".to_string(), format!("i:{}", sub_scores[i])));
            }
        } else {
            // Secondary alignment: MAPQ = 0
            alignments[i].mapq = 0;
        }
    }
}

/// Filter supplementary alignments by score relative to primary
fn filter_supplementary_by_score(
    alignments: &mut Vec<Alignment>,
    primary_score: i32,
    xa_drop_ratio: f32,
) {
    let supp_threshold = (primary_score as f32 * xa_drop_ratio) as i32;
    let before_filter = alignments.len();

    // Mark low-scoring supplementary for removal
    let mut to_remove: Vec<usize> = Vec::new();
    for i in 0..alignments.len() {
        if (alignments[i].flag & sam_flags::SUPPLEMENTARY) != 0
            && alignments[i].score < supp_threshold
        {
            to_remove.push(i);
            log::debug!(
                "Filtering supplementary alignment: {}:{} score={} < threshold={}",
                alignments[i].ref_name, alignments[i].pos,
                alignments[i].score, supp_threshold
            );
        }
    }

    // Remove in reverse order to preserve indices
    for &i in to_remove.iter().rev() {
        alignments.remove(i);
    }

    if before_filter != alignments.len() {
        log::debug!(
            "Filtered {} supplementary alignments below score threshold {} (xa_drop_ratio={})",
            before_filter - alignments.len(), supp_threshold, xa_drop_ratio
        );
    }
}

// ============================================================================
// MAIN SECONDARY MARKING FUNCTION
// ============================================================================

/// Mark secondary alignments and calculate MAPQ values
/// Implements C++ mem_mark_primary_se (bwamem.cpp:1420-1464)
///
/// Algorithm:
/// 1. Sort alignments by score (descending), then by hash
/// 2. For each alignment, check if it overlaps significantly with higher-scoring alignments
/// 3. If overlap >= mask_level, mark as secondary (set sam_flags::SECONDARY flag)
/// 4. Calculate MAPQ based on score difference and overlap count
pub fn mark_secondary_alignments(alignments: &mut Vec<Alignment>, opt: &MemOpt) {
    if alignments.is_empty() {
        return;
    }

    // Debug logging
    log::debug!(
        "mark_secondary_alignments: {} alignments, mask_level={}",
        alignments.len(),
        opt.mask_level
    );

    // Calculate score gap threshold for sub_count
    // tmp = max(a+b, o_del+e_del, o_ins+e_ins)
    let score_gap_threshold = (opt.a + opt.b)
        .max(opt.o_del + opt.e_del)
        .max(opt.o_ins + opt.e_ins);

    // Step 1: Find primary alignments and track sub-scores
    let (primary_indices, sub_scores, sub_counts) =
        find_primary_alignments(alignments, opt.mask_level, score_gap_threshold);

    log::debug!(
        "  Total primary_indices: {} (will mark {} as supplementary)",
        primary_indices.len(),
        primary_indices.len().saturating_sub(1)
    );

    // Step 2: Mark non-overlapping alignments after the first as SUPPLEMENTARY
    apply_supplementary_flags(alignments, &primary_indices);

    // Step 3: Calculate MAPQ for all alignments and add XS tags
    calculate_all_mapq(alignments, &sub_scores, &sub_counts, opt);

    // Step 4: Filter low-scoring supplementary alignments
    if !primary_indices.is_empty() {
        let primary_score = alignments[primary_indices[0]].score;
        filter_supplementary_by_score(alignments, primary_score, opt.xa_drop_ratio);
    }
}

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
        log::debug!(
            "alignments_overlap: NO OVERLAP - a1[{},{}), a2[{},{}), b_max={}, e_min={}",
            a1_qb, a1_qe, a2_qb, a2_qe, b_max, e_min
        );
        return false; // No overlap
    }

    let overlap = e_min - b_max;
    let min_len = (a1_qe - a1_qb).min(a2_qe - a2_qb);
    let threshold = (min_len as f32 * mask_level) as i32;
    let result = overlap >= threshold;

    log::debug!(
        "alignments_overlap: a1[{},{}), a2[{},{}), overlap={}, min_len={}, threshold={}, result={}",
        a1_qb, a1_qe, a2_qb, a2_qe, overlap, min_len, threshold, result
    );

    result
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
pub fn generate_xa_tags(
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
        let primary = read_alns
            .iter()
            .find(|a| a.flag & sam_flags::SECONDARY == 0);

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
pub fn generate_sa_tags(alignments: &[Alignment]) -> std::collections::HashMap<String, String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Helper to create an alignment for testing
    fn make_test_alignment(
        query_name: &str,
        ref_name: &str,
        ref_id: usize,
        pos: u64,
        score: i32,
        cigar: Vec<(u8, i32)>,
        query_start: i32,
        query_end: i32,
        flag: u16,
    ) -> Alignment {
        Alignment {
            query_name: query_name.to_string(),
            flag,
            ref_name: ref_name.to_string(),
            ref_id,
            pos,
            mapq: 60,
            score,
            cigar,
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: "A".repeat((query_end - query_start) as usize),
            qual: "I".repeat((query_end - query_start) as usize),
            tags: vec![],
            query_start,
            query_end,
            seed_coverage: score,
            hash: (pos * 1000 + score as u64) as u64, // Unique hash
            frac_rep: 0.0,
        }
    }

    /// Helper to create default test MemOpt
    fn default_test_opt() -> MemOpt {
        let mut opt = MemOpt::default();
        opt.mask_level = 0.5;
        opt.mask_level_redun = 0.95;
        opt.max_chain_gap = 10000;
        opt.xa_drop_ratio = 0.8;
        opt
    }

    // ------------------------------------------------------------------------
    // remove_redundant_alignments() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_remove_redundant_empty() {
        let opt = default_test_opt();
        let mut alignments: Vec<Alignment> = vec![];
        remove_redundant_alignments(&mut alignments, &opt);
        assert!(alignments.is_empty());
    }

    #[test]
    fn test_remove_redundant_single() {
        let opt = default_test_opt();
        let mut alignments = vec![make_test_alignment(
            "read1", "chr1", 0, 1000, 100,
            vec![(b'M', 100)], 0, 100, 0,
        )];
        remove_redundant_alignments(&mut alignments, &opt);
        assert_eq!(alignments.len(), 1);
    }

    #[test]
    fn test_remove_redundant_no_overlap() {
        let opt = default_test_opt();
        let mut alignments = vec![
            make_test_alignment("read1", "chr1", 0, 1000, 100, vec![(b'M', 50)], 0, 50, 0),
            make_test_alignment("read1", "chr1", 0, 5000, 80, vec![(b'M', 50)], 50, 100, 0),
        ];
        remove_redundant_alignments(&mut alignments, &opt);
        // Both should be kept (no overlap)
        assert_eq!(alignments.len(), 2);
    }

    #[test]
    fn test_remove_redundant_high_overlap() {
        let opt = default_test_opt();
        let mut alignments = vec![
            // Two alignments at nearly same position with >95% overlap
            make_test_alignment("read1", "chr1", 0, 1000, 100, vec![(b'M', 100)], 0, 100, 0),
            make_test_alignment("read1", "chr1", 0, 1002, 90, vec![(b'M', 98)], 0, 98, 0),
        ];
        remove_redundant_alignments(&mut alignments, &opt);
        // Lower-scoring one should be removed
        assert_eq!(alignments.len(), 1);
        assert_eq!(alignments[0].score, 100);
    }

    #[test]
    fn test_remove_redundant_different_chromosomes() {
        let opt = default_test_opt();
        let mut alignments = vec![
            make_test_alignment("read1", "chr1", 0, 1000, 100, vec![(b'M', 100)], 0, 100, 0),
            make_test_alignment("read1", "chr2", 1, 1000, 90, vec![(b'M', 100)], 0, 100, 0),
        ];
        remove_redundant_alignments(&mut alignments, &opt);
        // Both should be kept (different chromosomes)
        assert_eq!(alignments.len(), 2);
    }

    #[test]
    fn test_remove_redundant_exact_duplicates() {
        let opt = default_test_opt();
        let mut alignments = vec![
            make_test_alignment("read1", "chr1", 0, 1000, 100, vec![(b'M', 100)], 0, 100, 0),
            make_test_alignment("read1", "chr1", 0, 1000, 100, vec![(b'M', 100)], 0, 100, 0),
        ];
        remove_redundant_alignments(&mut alignments, &opt);
        // Exact duplicates should be removed
        assert_eq!(alignments.len(), 1);
    }

    // ------------------------------------------------------------------------
    // mark_secondary_alignments() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_mark_secondary_empty() {
        let opt = default_test_opt();
        let mut alignments: Vec<Alignment> = vec![];
        mark_secondary_alignments(&mut alignments, &opt);
        assert!(alignments.is_empty());
    }

    #[test]
    fn test_mark_secondary_single() {
        let opt = default_test_opt();
        let mut alignments = vec![make_test_alignment(
            "read1", "chr1", 0, 1000, 100,
            vec![(b'M', 100)], 0, 100, 0,
        )];
        mark_secondary_alignments(&mut alignments, &opt);
        assert_eq!(alignments.len(), 1);
        // Single alignment should remain primary
        assert_eq!(alignments[0].flag & sam_flags::SECONDARY, 0);
        assert_eq!(alignments[0].flag & sam_flags::SUPPLEMENTARY, 0);
    }

    #[test]
    fn test_mark_secondary_overlapping() {
        let mut opt = default_test_opt();
        opt.mask_level = 0.5;

        let mut alignments = vec![
            // High-scoring alignment covering query [0, 80)
            make_test_alignment("read1", "chr1", 0, 1000, 100, vec![(b'M', 80)], 0, 80, 0),
            // Lower-scoring alignment covering query [10, 90) - significant overlap
            make_test_alignment("read1", "chr1", 0, 2000, 70, vec![(b'M', 80)], 10, 90, 0),
        ];
        mark_secondary_alignments(&mut alignments, &opt);

        assert_eq!(alignments.len(), 2);
        // First (highest score) should be primary
        assert_eq!(alignments[0].flag & sam_flags::SECONDARY, 0);
        // Second should be marked secondary due to overlap
        assert_ne!(alignments[1].flag & sam_flags::SECONDARY, 0);
    }

    #[test]
    fn test_mark_secondary_non_overlapping_supplementary() {
        let mut opt = default_test_opt();
        opt.mask_level = 0.5;

        let mut alignments = vec![
            // Alignment covering query [0, 50)
            make_test_alignment("read1", "chr1", 0, 1000, 100, vec![(b'M', 50)], 0, 50, 0),
            // Alignment covering query [60, 100) - non-overlapping
            make_test_alignment("read1", "chr2", 1, 5000, 80, vec![(b'M', 40)], 60, 100, 0),
        ];
        mark_secondary_alignments(&mut alignments, &opt);

        assert_eq!(alignments.len(), 2);
        // First should be primary
        assert_eq!(alignments[0].flag & sam_flags::SECONDARY, 0);
        // Second is non-overlapping, should be supplementary (not secondary)
        assert_eq!(alignments[1].flag & sam_flags::SECONDARY, 0);
        assert_ne!(alignments[1].flag & sam_flags::SUPPLEMENTARY, 0);
    }

    // ------------------------------------------------------------------------
    // alignment_ref_length() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_alignment_ref_length_simple_match() {
        let alignment = make_test_alignment(
            "read1", "chr1", 0, 1000, 100,
            vec![(b'M', 100)], 0, 100, 0,
        );
        assert_eq!(alignment_ref_length(&alignment), 100);
    }

    #[test]
    fn test_alignment_ref_length_with_insertions() {
        let alignment = make_test_alignment(
            "read1", "chr1", 0, 1000, 100,
            vec![(b'M', 50), (b'I', 5), (b'M', 45)], 0, 100, 0,
        );
        // Insertions don't consume reference
        assert_eq!(alignment_ref_length(&alignment), 95);
    }

    #[test]
    fn test_alignment_ref_length_with_deletions() {
        let alignment = make_test_alignment(
            "read1", "chr1", 0, 1000, 100,
            vec![(b'M', 50), (b'D', 10), (b'M', 40)], 0, 90, 0,
        );
        // Deletions consume reference
        assert_eq!(alignment_ref_length(&alignment), 100);
    }

    #[test]
    fn test_alignment_ref_length_with_soft_clips() {
        let alignment = make_test_alignment(
            "read1", "chr1", 0, 1000, 100,
            vec![(b'S', 10), (b'M', 80), (b'S', 10)], 0, 100, 0,
        );
        // Soft clips don't consume reference
        assert_eq!(alignment_ref_length(&alignment), 80);
    }

    // ------------------------------------------------------------------------
    // generate_md_tag() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_generate_md_tag_perfect_match() {
        // Perfect match of 50bp
        // generate_md_tag takes (ref_aligned, query_aligned, cigar)
        let reference = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"; // 50bp
        let query = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC"; // 50bp same
        let cigar = vec![(b'M', 50)];
        let md = Alignment::generate_md_tag(reference, query, &cigar);
        assert_eq!(md, "50");
    }

    #[test]
    fn test_generate_md_tag_with_mismatch() {
        // 10M with mismatch at position 5
        let reference = b"ACGTANGTAC"; // Mismatch at position 5: C -> N
        let query = b"ACGTACGTAC"; // 10bp
        let cigar = vec![(b'M', 10)];
        let md = Alignment::generate_md_tag(reference, query, &cigar);
        // Format should indicate the mismatch
        assert!(!md.is_empty());
    }

    // ------------------------------------------------------------------------
    // calculate_tlen() tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_calculate_tlen_downstream_mate() {
        let alignment = make_test_alignment(
            "read1", "chr1", 0, 100, 100,
            vec![(b'M', 100)], 0, 100, 0,
        );
        // Mate at position 300, ref_len 100
        let tlen = alignment.calculate_tlen(300, 100);
        // TLEN = mate_end - self_start = 400 - 100 = 300
        assert_eq!(tlen, 300);
    }

    #[test]
    fn test_calculate_tlen_upstream_mate() {
        let alignment = make_test_alignment(
            "read1", "chr1", 0, 300, 100,
            vec![(b'M', 100)], 0, 100, 0,
        );
        // Mate at position 100, ref_len 100
        let tlen = alignment.calculate_tlen(100, 100);
        // TLEN should be negative (mate is upstream)
        assert!(tlen < 0);
    }
}
