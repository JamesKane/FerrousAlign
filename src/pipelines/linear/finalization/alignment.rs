//! Alignment struct and methods.

use super::sam_flags;

#[derive(Debug, Clone)]
pub struct Alignment {
    pub query_name: String,
    pub flag: u16,
    pub ref_name: String,
    pub ref_id: usize,
    pub pos: u64,
    pub mapq: u8,
    pub score: i32,
    pub cigar: Vec<(u8, i32)>,
    pub rnext: String,
    pub pnext: u64,
    pub tlen: i32,
    pub seq: String,
    pub qual: String,
    pub tags: Vec<(String, String)>,
    pub(crate) query_start: i32,
    pub(crate) query_end: i32,
    pub(crate) seed_coverage: i32,
    pub(crate) hash: u64,
    pub(crate) frac_rep: f32,
    pub(crate) is_alt: bool, // True if alignment to alternate contig/haplotype
}

impl Alignment {
    /// Detect if reference name refers to an alternate contig/haplotype
    /// Matches BWA-MEM2's logic for preferring primary assembly in tie-breaking
    #[inline]
    pub fn is_alternate_contig(ref_name: &str) -> bool {
        // Common patterns in alternate contig names:
        // - Contains "alt" (e.g., chr1_KI270706v1_alt)
        // - Contains "random" (e.g., chr1_KI270706v1_random)
        // - Contains "fix" (e.g., chr1_KI270706v1_fix)
        // - Contains "HLA" (e.g., HLA-A*01:01:01:01)
        // - Contains underscore after chr prefix (common in alternate names)
        ref_name.contains("alt")
            || ref_name.contains("random")
            || ref_name.contains("fix")
            || ref_name.contains("HLA")
            || (ref_name.starts_with("chr") && ref_name.contains('_'))
    }

    /// Get CIGAR string as formatted string (e.g., "50M2I48M")
    pub fn cigar_string(&self) -> String {
        if self.cigar.is_empty() {
            "*".to_string()
        } else {
            let is_supplementary = (self.flag & sam_flags::SUPPLEMENTARY) != 0;
            self.cigar
                .iter()
                .map(|&(op, len)| {
                    let op_char = if is_supplementary && op == b'S' {
                        'H'
                    } else {
                        op as char
                    };
                    format!("{len}{op_char}")
                })
                .collect()
        }
    }

    /// Calculate aligned reference length from CIGAR
    pub fn reference_length(&self) -> i32 {
        self.cigar
            .iter()
            .filter_map(|&(op, len)| match op as char {
                'M' | 'D' | 'N' | '=' | 'X' => Some(len),
                _ => None,
            })
            .sum()
    }

    /// Convert to SAM format string
    pub fn to_sam_string(&self) -> String {
        self.to_sam_string_with_seq(&self.seq, &self.qual)
    }

    /// Convert to SAM format with externally provided seq/qual
    pub fn to_sam_string_with_seq(&self, seq: &str, qual: &str) -> String {
        let cigar_string = self.cigar_string();

        let (mut output_seq, mut output_qual) = if self.flag & sam_flags::REVERSE != 0 {
            let rev_comp_seq: String = seq
                .chars()
                .rev()
                .map(|c| match c {
                    'A' => 'T',
                    'T' => 'A',
                    'C' => 'G',
                    'G' => 'C',
                    'N' => 'N',
                    _ => c,
                })
                .collect();
            let rev_qual: String = qual.chars().rev().collect();
            (rev_comp_seq, rev_qual)
        } else {
            (seq.to_string(), qual.to_string())
        };

        let is_supplementary = (self.flag & sam_flags::SUPPLEMENTARY) != 0;
        let mut leading_clip = 0usize;
        let mut trailing_clip = 0usize;

        for &(op, len) in self.cigar.iter() {
            if op == b'H' || (is_supplementary && op == b'S') {
                leading_clip += len as usize;
            } else {
                break;
            }
        }

        for &(op, len) in self.cigar.iter().rev() {
            if op == b'H' || (is_supplementary && op == b'S') {
                trailing_clip += len as usize;
            } else {
                break;
            }
        }

        if leading_clip > 0 || trailing_clip > 0 {
            let seq_len = output_seq.len();
            let start = leading_clip.min(seq_len);
            let end = seq_len.saturating_sub(trailing_clip);
            if start < end {
                output_seq = output_seq[start..end].to_string();
                output_qual = output_qual[start..end.min(output_qual.len())].to_string();
            }
        }

        let sam_pos = if self.ref_name == "*" {
            0
        } else {
            self.pos + 1
        };
        let mut sam_line = format!(
            "{}	{}	{}	{}	{}	{}	{}	{}	{}	{}	{}",
            self.query_name,
            self.flag,
            self.ref_name,
            sam_pos,
            self.mapq,
            cigar_string,
            self.rnext,
            self.pnext,
            self.tlen,
            output_seq,
            output_qual
        );

        for (tag, value) in &self.tags {
            sam_line.push('\t');
            sam_line.push_str(tag);
            sam_line.push(':');
            sam_line.push_str(value);
        }

        sam_line
    }

    /// Calculate aligned query length from CIGAR
    pub fn query_length(&self) -> i32 {
        self.cigar
            .iter()
            .filter_map(|&(op, len)| match op as char {
                'M' | 'I' | 'S' | '=' | 'X' => Some(len),
                _ => None,
            })
            .sum()
    }

    /// Generate XA tag entry for this alignment
    pub fn to_xa_entry(&self) -> String {
        let strand = if self.flag & sam_flags::REVERSE != 0 {
            '-'
        } else {
            '+'
        };
        let pos = self.pos + 1;
        let cigar = self.cigar_string();
        let nm = self
            .tags
            .iter()
            .find(|(tag_name, _)| tag_name == "NM")
            .and_then(|(_, val)| val.strip_prefix("i:"))
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        format!("{},{}{},{},{}", self.ref_name, strand, pos, cigar, nm)
    }

    /// Set paired-end SAM flags
    #[inline]
    pub fn set_paired_flags(
        &mut self,
        is_first: bool,
        is_proper_pair: bool,
        mate_unmapped: bool,
        mate_reverse: bool,
    ) {
        self.flag |= sam_flags::PAIRED;
        self.flag |= if is_first {
            sam_flags::FIRST_IN_PAIR
        } else {
            sam_flags::SECOND_IN_PAIR
        };
        if is_proper_pair {
            self.flag |= sam_flags::PROPER_PAIR;
        }
        if mate_unmapped {
            self.flag |= sam_flags::MATE_UNMAPPED;
        }
        if mate_reverse {
            self.flag |= sam_flags::MATE_REVERSE;
        }
    }

    /// Calculate template length (TLEN) for paired-end reads
    #[inline]
    pub fn calculate_tlen(&self, mate_pos: u64, mate_ref_len: i32) -> i32 {
        let this_pos = self.pos as i64;
        let mate_pos = mate_pos as i64;
        if this_pos <= mate_pos {
            ((mate_pos - this_pos) + mate_ref_len as i64) as i32
        } else {
            let this_ref_len = self.reference_length();
            -(((this_pos - mate_pos) + this_ref_len as i64) as i32)
        }
    }

    /// Create an unmapped alignment for paired-end reads
    pub fn create_unmapped(
        query_name: String,
        seq: &[u8],
        _qual: String,
        is_first_in_pair: bool,
        mate_ref: &str,
        mate_pos: u64,
        mate_is_reverse: bool,
    ) -> Self {
        let mut flag = sam_flags::PAIRED | sam_flags::UNMAPPED;
        flag |= if is_first_in_pair {
            sam_flags::FIRST_IN_PAIR
        } else {
            sam_flags::SECOND_IN_PAIR
        };

        let (ref_name, pos, rnext, pnext) = if mate_ref != "*" {
            (
                mate_ref.to_string(),
                mate_pos,
                "=".to_string(),
                mate_pos + 1,
            )
        } else {
            ("*".to_string(), 0, "*".to_string(), 0)
        };

        if mate_is_reverse {
            flag |= sam_flags::MATE_REVERSE;
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
            seq: String::new(),
            qual: String::new(),
            tags: vec![
                ("AS".to_string(), "i:0".to_string()),
                ("NM".to_string(), "i:0".to_string()),
            ],
            query_start: 0,
            query_end: seq.len() as i32,
            hash: 0,
            seed_coverage: 0,
            frac_rep: 0.0,
            is_alt: false, // Unmapped reads don't map to alternate contigs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            hash: (pos * 1000 + score as u64),
            frac_rep: 0.0,
            is_alt: false,
        }
    }

    #[test]
    fn test_hard_clipping_for_supplementary() {
        let primary = Alignment {
            query_name: "read1".to_string(),
            flag: 0,
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
            is_alt: false,
        };

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
            is_alt: false,
        };

        assert_eq!(primary.cigar_string(), "5S50M10S");
        assert_eq!(supplementary.cigar_string(), "5H50M10H");
    }

    #[test]
    fn test_calculate_tlen_downstream_mate() {
        let alignment =
            make_test_alignment("read1", "chr1", 0, 100, 100, vec![(b'M', 100)], 0, 100, 0);
        let tlen = alignment.calculate_tlen(300, 100);
        assert_eq!(tlen, 300);
    }

    #[test]
    fn test_calculate_tlen_upstream_mate() {
        let alignment =
            make_test_alignment("read1", "chr1", 0, 300, 100, vec![(b'M', 100)], 0, 100, 0);
        let tlen = alignment.calculate_tlen(100, 100);
        assert!(tlen < 0);
    }
}
