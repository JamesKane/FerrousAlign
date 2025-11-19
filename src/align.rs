// bwa-mem2-rust/src/align.rs

// Import BwaIndex and MemOpt
use crate::banded_swa::BandedPairWiseSW;
use crate::mem::BwaIndex;
use crate::mem_opt::MemOpt;
use crate::utils::hash_64;

// Define a struct to represent a seed
#[derive(Debug, Clone)]
pub struct Seed {
    pub query_pos: i32, // Position in the query
    pub ref_pos: u64,   // Position in the reference
    pub len: i32,       // Length of the seed
    pub is_rev: bool,   // Is it on the reverse strand?
}

// Define a struct to represent a Super Maximal Exact Match (SMEM)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)] // Add PartialEq and Eq for deduplication
pub struct SMEM {
    pub rid: i32, // Read ID (not strictly needed here, but good for consistency with C++)
    pub m: i32,   // Start position in query
    pub n: i32,   // End position in query
    pub k: u64,   // Start of BWT interval (SA coordinate)
    pub l: u64,   // End of BWT interval (SA coordinate)
    pub s: u64,   // Size of BWT interval (l - k)
    pub is_rev_comp: bool, // Added to track if SMEM is from reverse complement
}

// Constants from FMI_search.h

const CP_MASK: u64 = 63;
pub const CP_SHIFT: u64 = 6; // Make CP_SHIFT public

// CP_OCC struct from FMI_search.h
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CpOcc {
    pub cp_count: [i64; 4],
    pub one_hot_bwt_str: [u64; 4],
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

// Global one_hot_mask_array (initialized once)
// Matches C++ bwa-mem2: one_hot_mask_array[i] has the top i bits set
lazy_static::lazy_static! {
    static ref ONE_HOT_MASK_ARRAY: Vec<u64> = {
        let mut array = vec![0u64; 64]; // Size 64 to match C++ (indices 0-63)
        // array[0] is already 0
        let base = 0x8000000000000000u64;
        array[1] = base;  // Explicitly set like C++ does
        for i in 2..64 {
            array[i] = (array[i - 1] >> 1) | base;
        }
        array
    };
}

// Hardware-optimized popcount for 64-bit integers
// Uses NEON on ARM64 and POPCNT on x86_64
#[inline(always)]
fn popcount64(x: u64) -> i64 {
    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON implementation using vcnt (count bits in 8-bit lanes)
        // This is the equivalent of __builtin_popcountl on ARM
        unsafe {
            use std::arch::aarch64::*;

            // Load the 64-bit value into a NEON register
            let vec = vreinterpret_u8_u64(vcreate_u64(x));

            // Count bits in each 8-bit lane (vcnt_u8)
            let cnt = vcnt_u8(vec);

            // Sum all 8 lanes using horizontal add (pairwise additions)
            // vcnt gives us 8 bytes, each with bit count of that byte
            // We need to sum them all to get total popcount
            let sum16 = vpaddl_u8(cnt); // Pairwise add to 4x u16
            let sum32 = vpaddl_u16(sum16); // Pairwise add to 2x u32
            let sum64 = vpaddl_u32(sum32); // Pairwise add to 1x u64

            vget_lane_u64::<0>(sum64) as i64
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // x86_64 POPCNT instruction
        unsafe {
            use std::arch::x86_64::_popcnt64;
            _popcnt64(x as i64) as i64
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        // Fallback for other architectures
        x.count_ones() as i64
    }
}

// Helper function to get occurrences, translating GET_OCC macro
// Now uses hardware-optimized popcount
pub fn get_occ(bwa_idx: &BwaIndex, k: i64, c: u8) -> i64 {
    let cp_shift = CP_SHIFT as i64;
    let cp_mask = CP_MASK as i64;

    let occ_id_k = k >> cp_shift;
    let y_k = k & cp_mask;

    let occ_k = bwa_idx.cp_occ[occ_id_k as usize].cp_count[c as usize];
    let one_hot_bwt_str_c_k = bwa_idx.cp_occ[occ_id_k as usize].one_hot_bwt_str[c as usize];

    let match_mask_k = one_hot_bwt_str_c_k & ONE_HOT_MASK_ARRAY[y_k as usize];
    occ_k + popcount64(match_mask_k)
}

/// Vectorized get_occ for all 4 bases simultaneously
///
/// This function processes all 4 bases (A, C, G, T) in parallel, eliminating
/// the need for 4 sequential get_occ calls. This provides significant speedup
/// by reducing memory access overhead and enabling better CPU pipelining.
///
/// Returns [i64; 4] containing occurrence counts for bases 0, 1, 2, 3 (A, C, G, T)
#[inline(always)]
pub fn get_occ_all_bases(bwa_idx: &BwaIndex, k: i64) -> [i64; 4] {
    let cp_shift = CP_SHIFT as i64;
    let cp_mask = CP_MASK as i64;

    let occ_id_k = k >> cp_shift;
    let y_k = k & cp_mask;

    let cp_occ = &bwa_idx.cp_occ[occ_id_k as usize];
    let mask = ONE_HOT_MASK_ARRAY[y_k as usize];

    // Process all 4 bases in parallel
    let mut result = [0i64; 4];
    for i in 0..4 {
        let match_mask = cp_occ.one_hot_bwt_str[i] & mask;
        result[i] = cp_occ.cp_count[i] + popcount64(match_mask);
    }

    result
}

/// Backward extension matching C++ bwa-mem2 FMI_search::backwardExt()
///
/// CRITICAL: This uses a cumulative sum approach for computing l[] values,
/// NOT the simple l = k + s formula! The l field encodes reverse complement
/// BWT information, so the standard BWT interval invariant s = l - k does NOT hold.
///
/// C++ reference: FMI_search.cpp lines 1025-1052
pub fn backward_ext(bwa_idx: &BwaIndex, mut smem: SMEM, a: u8) -> SMEM {
    let debug_enabled = log::log_enabled!(log::Level::Trace);

    if debug_enabled {
        log::trace!(
            "backward_ext: input smem(k={}, l={}, s={}), a={}",
            smem.k,
            smem.l,
            smem.s,
            a
        );
    }

    let mut k = [0i64; 4];
    let mut l = [0i64; 4];
    let mut s = [0i64; 4];

    // Compute k[] and s[] for all 4 bases (matching C++ lines 1030-1039)
    // OPTIMIZATION: Use vectorized get_occ_all_bases to process all 4 bases at once
    let sp = smem.k as i64;
    let ep = (smem.k + smem.s) as i64;

    let occ_sp = get_occ_all_bases(bwa_idx, sp);
    let occ_ep = get_occ_all_bases(bwa_idx, ep);

    for b in 0..4usize {
        k[b] = bwa_idx.bwt.l2[b] as i64 + occ_sp[b];
        s[b] = occ_ep[b] - occ_sp[b];

        if debug_enabled && b == a as usize {
            log::trace!(
                "backward_ext: base {}: sp={}, ep={}, occ_sp={}, occ_ep={}, k={}, s={}",
                b,
                sp,
                ep,
                occ_sp[b],
                occ_ep[b],
                k[b],
                s[b]
            );
        }
    }

    // Sentinel handling (matching C++ lines 1041-1042)
    let sentinel_offset = if smem.k <= bwa_idx.sentinel_index as u64
        && (smem.k + smem.s) > bwa_idx.sentinel_index as u64
    {
        1i64
    } else {
        0i64
    };

    if debug_enabled {
        log::trace!(
            "backward_ext: sentinel_offset={}, sentinel_index={}",
            sentinel_offset,
            bwa_idx.sentinel_index
        );
    }

    // CRITICAL: Cumulative sum computation for l[] (matching C++ lines 1043-1046)
    // This is NOT l[b] = k[b] + s[b]!
    // Instead: l[3] = smem.l + offset, then l[2] = l[3] + s[3], etc.
    l[3] = smem.l as i64 + sentinel_offset;
    l[2] = l[3] + s[3];
    l[1] = l[2] + s[2];
    l[0] = l[1] + s[1];

    if debug_enabled {
        log::trace!(
            "backward_ext: cumulative l[] = [{}, {}, {}, {}]",
            l[0],
            l[1],
            l[2],
            l[3]
        );
        log::trace!(
            "backward_ext: k[] = [{}, {}, {}, {}]",
            k[0],
            k[1],
            k[2],
            k[3]
        );
        log::trace!(
            "backward_ext: s[] = [{}, {}, {}, {}]",
            s[0],
            s[1],
            s[2],
            s[3]
        );
    }

    // Update SMEM with results for base 'a' (matching C++ lines 1048-1050)
    smem.k = k[a as usize] as u64;
    smem.l = l[a as usize] as u64;
    smem.s = s[a as usize] as u64;

    if debug_enabled {
        log::trace!(
            "backward_ext: output smem(k={}, l={}, s={}) for base {}",
            smem.k,
            smem.l,
            smem.s,
            a
        );
    }

    smem
}

/// Forward extension matching C++ bwa-mem2 pattern
///
/// Forward extension is implemented as:
/// 1. Swap k and l
/// 2. Call backwardExt with complement base (3 - a)
/// 3. Swap k and l back
///
/// C++ reference: FMI_search.cpp lines 546-554
#[inline]
pub fn forward_ext(bwa_idx: &BwaIndex, smem: SMEM, a: u8) -> SMEM {
    // Debug logging for forward extension
    let debug_enabled = log::log_enabled!(log::Level::Trace);

    if debug_enabled {
        log::trace!(
            "forward_ext: input smem(k={}, l={}, s={}), a={}",
            smem.k,
            smem.l,
            smem.s,
            a
        );
    }

    // Step 1: Swap k and l (lines 547-548)
    let mut smem_swapped = smem;
    smem_swapped.k = smem.l;
    smem_swapped.l = smem.k;

    if debug_enabled {
        log::trace!(
            "forward_ext: after swap smem_swapped(k={}, l={})",
            smem_swapped.k,
            smem_swapped.l
        );
    }

    // Step 2: Backward extension with complement base (line 549)
    let mut result = backward_ext(bwa_idx, smem_swapped, 3 - a);

    if debug_enabled {
        log::trace!(
            "forward_ext: after backward_ext result(k={}, l={}, s={})",
            result.k,
            result.l,
            result.s
        );
    }

    // Step 3: Swap k and l back (lines 552-553)
    // NOTE: We swap k and l but KEEP s unchanged (matches C++ behavior)
    // The s value is still valid because it represents interval size
    let k_temp = result.k;
    result.k = result.l;
    result.l = k_temp;

    if debug_enabled {
        log::trace!(
            "forward_ext: after swap back result(k={}, l={}, s={})",
            result.k,
            result.l,
            result.s
        );
    }

    result
}

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
/// let seq = b"ACGT";
/// let encoded = encode_sequence(seq);  // [0, 1, 2, 3]
/// let rc = reverse_complement_sequence(&encoded);  // [3, 2, 1, 0] = TGCA
/// assert_eq!(rc, vec![3, 2, 1, 0]);
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
    pub weight: i32, // Chain weight (seed coverage), calculated by mem_chain_weight
    pub kept: i32,   // Chain status: 0=discarded, 1=shadowed, 2=partial_overlap, 3=primary
}

/// SAM flag bit masks (SAM specification v1.6)
/// Used for setting and querying alignment flags in SAM/BAM format
pub mod sam_flags {
    pub const PAIRED: u16 = 0x1;          // Template having multiple segments in sequencing
    pub const PROPER_PAIR: u16 = 0x2;     // Each segment properly aligned according to the aligner
    pub const UNMAPPED: u16 = 0x4;        // Segment unmapped
    pub const MATE_UNMAPPED: u16 = 0x8;   // Next segment in the template unmapped
    pub const REVERSE: u16 = 0x10;        // SEQ being reverse complemented
    pub const MATE_REVERSE: u16 = 0x20;   // SEQ of the next segment in the template being reverse complemented
    pub const FIRST_IN_PAIR: u16 = 0x40;  // The first segment in the template
    pub const SECOND_IN_PAIR: u16 = 0x80; // The last segment in the template
    pub const SECONDARY: u16 = 0x100;     // Secondary alignment
    pub const QCFAIL: u16 = 0x200;        // Not passing filters, such as platform/vendor quality controls
    pub const DUPLICATE: u16 = 0x400;     // PCR or optical duplicate
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
}

impl Alignment {
    /// Get CIGAR string as a formatted string (e.g., "50M2I48M")
    /// Returns "*" for empty CIGAR (unmapped reads per SAM spec)
    pub fn cigar_string(&self) -> String {
        if self.cigar.is_empty() {
            "*".to_string()
        } else {
            self.cigar
                .iter()
                .map(|&(op, len)| format!("{}{}", len, op as char))
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

        // Handle reverse complement for SEQ and QUAL if flag 0x10 (reverse strand) is set
        // Matching bwa-mem2 mem_aln2sam() behavior (bwamem.cpp:1706-1716)
        let (output_seq, output_qual) = if self.flag & 0x10 != 0 {
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

    /// Generate XA tag entry for this alignment (alternative alignment format)
    /// Format: RNAME,STRAND+POS,CIGAR,NM
    /// Example: chr1,+1000,50M,2
    pub fn to_xa_entry(&self) -> String {
        let strand = if self.flag & 0x10 != 0 { '-' } else { '+' };
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
    /// - Always sets PAIRED (0x1)
    /// - Sets FIRST_IN_PAIR (0x40) or SECOND_IN_PAIR (0x80) based on `is_first`
    /// - Conditionally sets PROPER_PAIR (0x2), MATE_UNMAPPED (0x8), MATE_REVERSE (0x20)
    ///
    /// # Example
    /// ```
    /// use ferrous_align::align::{Alignment, sam_flags};
    ///
    /// let mut alignment = /* ... */;
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
        self.flag |= if is_first { FIRST_IN_PAIR } else { SECOND_IN_PAIR };

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
    /// - PAIRED (0x1) - always set
    /// - UNMAPPED (0x4) - always set
    /// - FIRST_IN_PAIR (0x40) or SECOND_IN_PAIR (0x80) - based on is_first_in_pair
    /// - MATE_REVERSE (0x20) - if mate_is_reverse is true
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
    /// assert_ne!(unmapped.flag & 0x4, 0); // UNMAPPED flag
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
            (mate_ref.to_string(), mate_pos, "=".to_string(), mate_pos + 1)
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
            tags: Vec::new(),
            // Internal fields - default for unmapped
            query_start: 0,
            query_end: seq.len() as i32,
            seed_coverage: 0,
            hash: 0,
        }
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
        return false; // No overlap
    }

    let overlap = e_min - b_max;
    let min_len = (a1_qe - a1_qb).min(a2_qe - a2_qb);

    // Overlap is significant if >= mask_level * min_length
    overlap >= (min_len as f32 * mask_level) as i32
}

/// Calculate MAPQ (mapping quality) for an alignment
/// Implements C++ mem_approx_mapq_se (bwamem.cpp:1470-1494)
fn calculate_mapq(
    score: i32,
    sub_score: i32,
    seed_coverage: i32,
    sub_count: i32,
    match_score: i32,
    mismatch_penalty: i32,
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

    // Traditional MAPQ formula (default when mapQ_coef_len = 0)
    // mapq = 30.0 * (1 - sub/score) * ln(seed_coverage)
    let mut mapq =
        (MEM_MAPQ_COEF * (1.0 - (sub as f64) / (score as f64)) * (seed_coverage as f64).ln()
            + 0.499) as i32;

    // Apply identity penalty if < 95%
    if identity < 0.95 {
        mapq = (mapq as f64 * identity * identity + 0.499) as i32;
    }

    // Penalty for multiple suboptimal alignments
    // mapq -= ln(sub_n+1) * 4.343
    if sub_count > 0 {
        mapq -= (((sub_count + 1) as f64).ln() * 4.343) as i32;
    }

    // Cap at max and floor at 0
    if mapq > MEM_MAPQ_MAX as i32 {
        mapq = MEM_MAPQ_MAX as i32;
    }
    if mapq < 0 {
        mapq = 0;
    }

    mapq as u8
}

/// Calculate chain weight based on seed coverage
/// Implements C++ mem_chain_weight (bwamem.cpp:429-448)
///
/// Weight = minimum of query coverage and reference coverage
/// This accounts for non-overlapping seed lengths in the chain
fn calculate_chain_weight(chain: &Chain, seeds: &[Seed]) -> i32 {
    if chain.seeds.is_empty() {
        return 0;
    }

    // Calculate query coverage (non-overlapping seed lengths on query)
    let mut query_cov = 0;
    let mut last_qe = -1i32;

    for &seed_idx in &chain.seeds {
        let seed = &seeds[seed_idx];
        let qb = seed.query_pos;
        let qe = seed.query_pos + seed.len;

        if qb > last_qe {
            // No overlap with previous seed
            query_cov += seed.len;
        } else if qe > last_qe {
            // Partial overlap
            query_cov += qe - last_qe;
        }
        // else: completely overlapped, no contribution

        last_qe = last_qe.max(qe);
    }

    // Calculate reference coverage (non-overlapping seed lengths on reference)
    let mut ref_cov = 0;
    let mut last_re = 0u64;

    for &seed_idx in &chain.seeds {
        let seed = &seeds[seed_idx];
        let rb = seed.ref_pos;
        let re = seed.ref_pos + seed.len as u64;

        if rb > last_re {
            // No overlap
            ref_cov += seed.len;
        } else if re > last_re {
            // Partial overlap
            ref_cov += (re - last_re) as i32;
        }

        last_re = last_re.max(re);
    }

    // Weight = min(query_cov, ref_cov)
    query_cov.min(ref_cov)
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
fn filter_chains(chains: &mut Vec<Chain>, seeds: &[Seed], opt: &MemOpt) -> Vec<Chain> {
    if chains.is_empty() {
        return Vec::new();
    }

    // Calculate weights for all chains
    for chain in chains.iter_mut() {
        chain.weight = calculate_chain_weight(chain, seeds);
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
        // Find primary alignment (flag & 0x100 == 0)
        let primary = read_alns.iter().find(|a| a.flag & 0x100 == 0);

        if primary.is_none() {
            continue; // No primary alignment
        }

        let primary_score = primary.unwrap().score;
        let xa_threshold = (primary_score as f32 * opt.xa_drop_ratio) as i32;

        // Collect secondary alignments that pass score threshold
        let mut secondaries: Vec<&Alignment> = read_alns
            .iter()
            .filter(|a| {
                (a.flag & 0x100 != 0) && // Is secondary
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

/// Mark secondary alignments and calculate MAPQ values
/// Implements C++ mem_mark_primary_se (bwamem.cpp:1420-1464)
///
/// Algorithm:
/// 1. Sort alignments by score (descending), then by hash
/// 2. For each alignment, check if it overlaps significantly with higher-scoring alignments
/// 3. If overlap >= mask_level, mark as secondary (set 0x100 flag)
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
                alignments[i].flag |= 0x100; // Set secondary flag
                is_secondary = true;
                break;
            }
        }

        // If no overlap with any primary, add as new primary
        if !is_secondary {
            primary_indices.push(i);
        }
    }

    // Calculate MAPQ for all alignments
    for i in 0..alignments.len() {
        if alignments[i].flag & 0x100 == 0 {
            // Primary alignment: calculate MAPQ
            alignments[i].mapq = calculate_mapq(
                alignments[i].score,
                sub_scores[i],
                alignments[i].seed_coverage,
                sub_counts[i],
                opt.a,
                opt.b,
                opt,
            );
        } else {
            // Secondary alignment: MAPQ = 0
            alignments[i].mapq = 0;
        }
    }
}

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
    use crate::simd_abstraction::{SimdEngineType, detect_optimal_simd_engine};

    if jobs.is_empty() {
        return 16; // Default
    }

    // Detect SIMD engine and determine native batch size
    let engine = detect_optimal_simd_engine();
    let (max_batch, standard_batch) = match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => (64, 32), // AVX-512: 64-way max, 32-way standard
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => (32, 16), // AVX2: 32-way max, 16-way standard
        SimdEngineType::Engine128 => (16, 16), // SSE2/NEON: 16-way
    };

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
) -> Vec<(i32, Vec<(u8, i32)>)> {
    const BATCH_SIZE: usize = 16;
    let mut all_results = vec![(0, Vec::new()); jobs.len()];

    // Process jobs in batches of 16
    for batch_start in (0..jobs.len()).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(jobs.len());
        let batch_jobs = &jobs[batch_start..batch_end];

        // Prepare batch data for SIMD dispatch
        let batch_data: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch_jobs
            .iter()
            .map(|job| {
                (
                    job.query.len() as i32,
                    job.query.as_slice(),
                    job.target.len() as i32,
                    job.target.as_slice(),
                    job.band_width,
                    0,
                )
            })
            .collect();

        // Execute batched alignment with CIGAR generation
        // Use dispatch to automatically route to optimal SIMD width (16/32/64)
        let results = sw_params.simd_banded_swa_dispatch_with_cigar(&batch_data);

        // Extract scores and CIGARs from results
        for (i, result) in results.iter().enumerate() {
            if i < batch_jobs.len() {
                all_results[batch_start + i] = (result.score.score, result.cigar.clone());
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
) -> Vec<(i32, Vec<(u8, i32)>)> {
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

    // Log routing statistics (INFO level - shows in default verbosity)
    log::info!(
        "Adaptive routing: {} total jobs, {} scalar ({:.1}%), {} SIMD ({:.1}%), avg_divergence={:.3}",
        jobs.len(),
        high_div_jobs.len(),
        high_div_jobs.len() as f64 / jobs.len() as f64 * 100.0,
        low_div_jobs.len(),
        low_div_jobs.len() as f64 / jobs.len() as f64 * 100.0,
        avg_divergence
    );

    // Show length statistics to understand why routing fails
    log::info!(
        "  → avg_query={:.1}bp, avg_target={:.1}bp, length_diff={:.1}%",
        avg_query_len,
        avg_target_len,
        ((avg_query_len - avg_target_len).abs() / avg_query_len.max(avg_target_len) * 100.0)
    );

    // Create result vector with correct size
    let mut all_results = vec![(0, Vec::new()); jobs.len()];

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
) -> Vec<(i32, Vec<(u8, i32)>)> {
    let mut all_results = vec![(0, Vec::new()); jobs.len()];

    // Process jobs in batches of specified size
    for batch_start in (0..jobs.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(jobs.len());
        let batch_jobs = &jobs[batch_start..batch_end];

        // Prepare batch data
        let batch_data: Vec<(i32, &[u8], i32, &[u8], i32, i32)> = batch_jobs
            .iter()
            .map(|job| {
                (
                    job.query.len() as i32,
                    job.query.as_slice(),
                    job.target.len() as i32,
                    job.target.as_slice(),
                    job.band_width,
                    0,
                )
            })
            .collect();

        // Execute batched alignment with CIGAR generation
        // Dispatch automatically selects batch16/32/64 based on detected SIMD engine
        let results = sw_params.simd_banded_swa_dispatch_with_cigar(&batch_data);

        // Extract scores and CIGARs from results
        for (i, result) in results.iter().enumerate() {
            if i < batch_jobs.len() {
                all_results[batch_start + i] = (result.score.score, result.cigar.clone());

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
                    log::warn!(
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
) -> Vec<(i32, Vec<(u8, i32)>)> {
    jobs.iter()
        .enumerate()
        .map(|(idx, job)| {
            let qlen = job.query.len() as i32;
            let tlen = job.target.len() as i32;
            let (score_out, cigar) = sw_params.scalar_banded_swa(
                qlen,
                &job.query,
                tlen,
                &job.target,
                job.band_width,
                0, // h0
            );

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
                log::warn!(
                    "PATHOLOGICAL_CIGAR_SCALAR|idx={}|qlen={}|tlen={}|bw={}|score={}|ins={}|del={}|CIGAR={:?}|QUERY={:?}|TARGET={:?}",
                    idx,
                    qlen,
                    tlen,
                    job.band_width,
                    score_out.score,
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
                    score_out.score,
                    cigar.len(),
                    cigar.first().map(|&(op, len)| (op as char, len))
                );
            }

            (score_out.score, cigar)
        })
        .collect()
}

pub fn generate_seeds(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
) -> Vec<Alignment> {
    generate_seeds_with_mode(bwa_idx, query_name, query_seq, query_qual, true, opt)
}

/// Generate SMEMs for a single strand (forward or reverse complement)
fn generate_smems_for_strand(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_len: usize,
    encoded_query: &[u8],
    is_rev_comp: bool,
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
            rid: 0,
            m: x as i32,
            n: x as i32,
            k: bwa_idx.bwt.l2[a as usize],
            l: bwa_idx.bwt.l2[(3 - a) as usize],
            s: bwa_idx.bwt.l2[(a + 1) as usize] - bwa_idx.bwt.l2[a as usize],
            is_rev_comp,
        };

        if x == 0 && !is_rev_comp {
            log::debug!(
                "{}: Initial SMEM at x={}: a={}, k={}, l={}, s={}, l2[{}]={}, l2[{}]={}",
                query_name,
                x,
                a,
                smem.k,
                smem.l,
                smem.s,
                a,
                bwa_idx.bwt.l2[a as usize],
                3 - a,
                bwa_idx.bwt.l2[(3 - a) as usize]
            );
        }

        // Phase 1: Forward extension
        prev_array_buf.clear();
        let mut next_x = x + 1;

        for j in (x + 1)..query_len {
            let a = encoded_query[j];
            next_x = j + 1;

            if a >= 4 {
                if x == 0 && !is_rev_comp && log::log_enabled!(log::Level::Debug) {
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

            if x == 0 && j <= 12 && !is_rev_comp {
                log::debug!(
                    "{}: x={}, j={}, a={}, old_smem.s={}, new_smem(k={}, l={}, s={})",
                    query_name,
                    x,
                    j,
                    a,
                    smem.s,
                    new_smem.k,
                    new_smem.l,
                    new_smem.s
                );
            }

            if new_smem.s != smem.s {
                if x < 3 && !is_rev_comp {
                    let s_from_lk = if smem.l > smem.k { smem.l - smem.k } else { 0 };
                    log::debug!(
                        "{}: x={}, j={}, pushing smem to prev_array_buf: s={}, l-k={}, match={}",
                        query_name,
                        x,
                        j,
                        smem.s,
                        s_from_lk,
                        smem.s == s_from_lk
                    );
                }
                prev_array_buf.push(smem);
            }

            if new_smem.s < min_intv {
                if x == 0 && !is_rev_comp && log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "{}: x={}, forward extension stopped at j={} because new_smem.s={} < min_intv={}",
                        query_name,
                        x,
                        j,
                        new_smem.s,
                        min_intv
                    );
                }
                break;
            }

            smem = new_smem;
            smem.n = j as i32;
        }

        if smem.s >= min_intv {
            prev_array_buf.push(smem);
        }

        if x < 3 && !is_rev_comp {
            log::debug!(
                "{}: Position x={}, prev_array_buf.len()={}, smem.s={}, min_intv={}",
                query_name,
                x,
                prev_array_buf.len(),
                smem.s,
                min_intv
            );
        }

        // Phase 2: Backward search
        if !is_rev_comp {
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
                if !is_rev_comp {
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

            if !is_rev_comp {
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
                new_smem.m = j as i32;

                if !is_rev_comp {
                    let old_len = smem.n - smem.m + 1;
                    let new_len = new_smem.n - new_smem.m + 1;
                    log::debug!(
                        "{}: [RUST Phase 2] x={}, j={}, i={}: old_smem(m={},n={},len={},k={},l={},s={}), new_smem(m={},n={},len={},k={},l={},s={}), min_intv={}",
                        query_name,
                        x,
                        j,
                        i,
                        smem.m,
                        smem.n,
                        old_len,
                        smem.k,
                        smem.l,
                        smem.s,
                        new_smem.m,
                        new_smem.n,
                        new_len,
                        new_smem.k,
                        new_smem.l,
                        new_smem.s,
                        min_intv
                    );
                }

                if new_smem.s < min_intv && (smem.n - smem.m + 1) >= min_seed_len {
                    if !is_rev_comp {
                        let s_from_lk = if smem.l > smem.k { smem.l - smem.k } else { 0 };
                        let s_matches = smem.s == s_from_lk;
                        log::debug!(
                            "{}: [RUST SMEM OUTPUT] Phase2 line 617: smem(m={},n={},k={},l={},s={}) newSmem.s={} < min_intv={}, l-k={}, s_match={}",
                            query_name,
                            smem.m,
                            smem.n,
                            smem.k,
                            smem.l,
                            smem.s,
                            new_smem.s,
                            min_intv,
                            s_from_lk,
                            s_matches
                        );
                    }
                    all_smems.push(*smem);
                    break;
                }

                if new_smem.s >= min_intv && curr_s != Some(new_smem.s) {
                    curr_s = Some(new_smem.s);
                    curr_array.push(new_smem);
                    if !is_rev_comp {
                        log::debug!(
                            "{}: [RUST Phase 2] Keeping new_smem (s={} >= min_intv={}), breaking",
                            query_name,
                            new_smem.s,
                            min_intv
                        );
                    }
                    break;
                }

                if !is_rev_comp {
                    log::debug!(
                        "{}: [RUST Phase 2] Rejecting new_smem (s={} < min_intv={} OR already_seen={})",
                        query_name,
                        new_smem.s,
                        min_intv,
                        curr_s == Some(new_smem.s)
                    );
                }
            }

            for (i, smem) in prev_array_buf.iter().rev().skip(1).enumerate() {
                let mut new_smem = backward_ext(bwa_idx, *smem, a);
                new_smem.m = j as i32;

                if !is_rev_comp {
                    let new_len = new_smem.n - new_smem.m + 1;
                    log::debug!(
                        "{}: [RUST Phase 2] x={}, j={}, remaining_i={}: smem(m={},n={},s={}), new_smem(m={},n={},len={},s={}), will_push={}",
                        query_name,
                        x,
                        j,
                        i + 1,
                        smem.m,
                        smem.n,
                        smem.s,
                        new_smem.m,
                        new_smem.n,
                        new_len,
                        new_smem.s,
                        new_smem.s >= min_intv && curr_s != Some(new_smem.s)
                    );
                }

                if new_smem.s >= min_intv && curr_s != Some(new_smem.s) {
                    curr_s = Some(new_smem.s);
                    curr_array.push(new_smem);
                }
            }

            std::mem::swap(&mut prev_array_buf, &mut curr_array_buf);
            *max_smem_count = (*max_smem_count).max(prev_array_buf.len());

            if !is_rev_comp {
                log::debug!(
                    "{}: [RUST Phase 2] After j={}, prev_array_buf.len()={}",
                    query_name,
                    j,
                    prev_array_buf.len()
                );
            }

            if prev_array_buf.is_empty() {
                if !is_rev_comp {
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
            let len = smem.n - smem.m + 1;
            if len >= min_seed_len {
                if !is_rev_comp {
                    let s_from_lk = if smem.l > smem.k { smem.l - smem.k } else { 0 };
                    let s_matches = smem.s == s_from_lk;
                    log::debug!(
                        "{}: [RUST SMEM OUTPUT] Phase2 line 671: smem(m={},n={},k={},l={},s={}), len={}, l-k={}, s_match={}, next_x={}",
                        query_name,
                        smem.m,
                        smem.n,
                        smem.k,
                        smem.l,
                        smem.s,
                        len,
                        s_from_lk,
                        s_matches,
                        next_x
                    );
                }
                all_smems.push(smem);
            } else if !is_rev_comp {
                log::debug!(
                    "{}: [RUST Phase 2] Rejecting final SMEM: m={}, n={}, len={} < min_seed_len={}, s={}",
                    query_name,
                    smem.m,
                    smem.n,
                    len,
                    min_seed_len,
                    smem.s
                );
            }
        } else if !is_rev_comp {
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
        0,               // end_bonus (not configurable in C++)
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
    // TODO: Implement re-seeding (-r split_factor) and split width (-y)
    // Re-seeding splits long MEMs (length > min_seed_len * split_factor) into shorter seeds
    // Split width controls occurrence threshold for splitting (if occurrences <= split_width)
    // See C++ bwamem.cpp:639-695 for full implementation

    // 1. Sort SMEMs
    all_smems.sort_by_key(|smem| (smem.m, smem.n, smem.k, smem.is_rev_comp)); // Include is_rev_comp in sort key

    // 2. Remove duplicates and filter by minimum length and max occurrences
    let mut unique_filtered_smems: Vec<SMEM> = Vec::new();
    let mut filtered_too_short = 0;
    let mut filtered_too_many_occ = 0;
    let mut duplicates = 0;

    if let Some(mut prev_smem) = all_smems.first().cloned() {
        let seed_len = prev_smem.n - prev_smem.m + 1;
        // CRITICAL FIX: Use s field for occurrences, NOT l - k
        // The l field encodes reverse complement BWT info, not interval endpoint
        let occurrences = prev_smem.s;

        // Filter by min_seed_len (-k) and max_occ (-c)
        if seed_len >= _opt.min_seed_len && occurrences <= _opt.max_occ as u64 {
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
                let seed_len = current_smem.n - current_smem.m + 1;
                // CRITICAL FIX: Use s field for occurrences, NOT l - k
                let occurrences = current_smem.s;

                // Filter by min_seed_len (-k) and max_occ (-c)
                if seed_len >= _opt.min_seed_len && occurrences <= _opt.max_occ as u64 {
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
            let len = smem.n - smem.m + 1;
            let occ = smem.l - smem.k;
            log::debug!(
                "{}: Sample SMEM {}: len={}, occ={}, m={}, n={}, k={}, l={}",
                query_name,
                i,
                len,
                occ,
                smem.m,
                smem.n,
                smem.k,
                smem.l
            );
        }
    }
    // eprintln!("unique_filtered_smems: {:?}", unique_filtered_smems);
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
    sorted_smems.sort_by_key(|smem| -(smem.n - smem.m + 1)); // Sort by length, descending

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
            smem.k,
            smem.l,
            smem.s,
            smem.m,
            smem.n,
            smem.is_rev_comp
        );

        // Try multiple positions in the BWT interval to find which one is correct
        let ref_pos_at_k = get_sa_entry(bwa_idx, smem.k);
        let ref_pos_at_l_minus_1 = if smem.l > 0 {
            get_sa_entry(bwa_idx, smem.l - 1)
        } else {
            ref_pos_at_k
        };
        log::debug!(
            "{}: SMEM {}: SA at k={} -> ref_pos {}, SA at l-1={} -> ref_pos {}",
            query_name,
            idx,
            smem.k,
            ref_pos_at_k,
            smem.l - 1,
            ref_pos_at_l_minus_1
        );

        let mut ref_pos = ref_pos_at_k;

        let mut is_rev = smem.is_rev_comp;

        // CRITICAL FIX: Use correct query orientation based on is_rev_comp flag
        // If SMEM is from RC search, use RC query bases for comparison
        let query_for_smem = if smem.is_rev_comp {
            &query_segment_encoded_rc
        } else {
            &query_segment_encoded
        };
        let smem_query_bases =
            &query_for_smem[smem.m as usize..=(smem.n as usize).min(query_for_smem.len() - 1)];
        let smem_len = smem.n - smem.m + 1;
        log::debug!(
            "{}: SMEM {}: Query bases at [{}..{}] (len={}): {:?}",
            query_name,
            idx,
            smem.m,
            smem.n,
            smem_len,
            &smem_query_bases[..10.min(smem_query_bases.len())]
        );

        // Convert positions in reverse complement region to forward strand
        // BWT contains both forward [0, l_pac) and reverse [l_pac, 2*l_pac)
        if ref_pos >= bwa_idx.bns.l_pac {
            ref_pos = (bwa_idx.bns.l_pac << 1) - 1 - ref_pos;
            is_rev = !is_rev; // Flip strand orientation
            log::debug!(
                "{}: SMEM {}: Was in RC region, converted ref_pos to {}, is_rev={}",
                query_name,
                idx,
                ref_pos,
                is_rev
            );
        }

        // Diagnostic validation removed - was causing 79,000% performance regression
        // The SMEM validation loop with log::info/warn was being called millions of times

        // Use query position in the coordinate system of the query we're aligning
        // For RC seeds, use smem.m directly (RC query coordinates)
        // For forward seeds, use smem.m directly (forward query coordinates)
        let query_pos = smem.m;

        let seed = Seed {
            query_pos,
            ref_pos,
            len: smem.n - smem.m + 1,
            is_rev,
        };

        // TRACE: Log each seed to track missing alignments
        log::trace!(
            "[SEED] {}: ref_pos={}, qpos={}, len={}, strand={}",
            query_name,
            seed.ref_pos,
            seed.query_pos,
            seed.len,
            if seed.is_rev { "rev" } else { "fwd" }
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
    let mut chained_results = chain_seeds(seeds.clone(), _opt);
    log::debug!(
        "{}: Chaining produced {} chains",
        query_name,
        chained_results.len()
    );

    // --- Chain Filtering ---
    // Implements bwa-mem2 mem_chain_flt logic (bwamem.cpp:506-624)
    let filtered_chains = filter_chains(&mut chained_results, &seeds, _opt);
    log::debug!(
        "{}: Chain filtering kept {} chains (from {} total)",
        query_name,
        filtered_chains.len(),
        chained_results.len()
    );

    // === CREATE ALIGNMENT JOBS FROM FILTERED CHAINS (C++ Strategy) ===
    // Now create alignment jobs using chain bounds instead of per-seed bounds
    // This matches C++ bwa-mem2 mem_reg2aln() behavior exactly
    let mut chain_to_job_map: Vec<Option<usize>> = Vec::new(); // Map chain_idx → job_idx

    for (chain_idx, chain) in filtered_chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            chain_to_job_map.push(None);
            continue;
        }

        // Use chain bounds (covers all seeds in chain)
        let qb = chain.query_start;
        let qe = chain.query_end;
        let bounded_query_len = (qe - qb) as usize;

        log::debug!(
            "{}: Chain {}: Creating alignment job for bounds=[{}, {}) len={}",
            query_name,
            chain_idx,
            qb,
            qe,
            bounded_query_len
        );

        // Extract bounded query using chain bounds
        let full_query = if chain.is_rev {
            &query_segment_encoded_rc
        } else {
            &query_segment_encoded
        };

        let bounded_query: Vec<u8> = full_query[qb as usize..qe as usize].to_vec();

        // Calculate reference position for bounded query
        // Use first seed's ref_pos and adjust for query offset
        let first_seed_idx = chain.seeds[0];
        let first_seed = &seeds[first_seed_idx];

        let ref_start_for_bounded = if first_seed.query_pos < qb {
            first_seed.ref_pos
        } else {
            let offset = first_seed.query_pos - qb;
            if offset as u64 > first_seed.ref_pos {
                0
            } else {
                first_seed.ref_pos - offset as u64
            }
        };

        // Get reference segment
        let ref_segment_len = (bounded_query_len as u64 + 2 * _opt.w as u64)
            .min(bwa_idx.bns.l_pac - ref_start_for_bounded);

        match bwa_idx
            .bns
            .get_reference_segment(ref_start_for_bounded, ref_segment_len)
        {
            Ok(target_segment) => {
                let job_idx = alignment_jobs.len();
                chain_to_job_map.push(Some(job_idx));

                alignment_jobs.push(AlignmentJob {
                    seed_idx: chain_idx, // Repurpose as chain_idx
                    query: bounded_query,
                    target: target_segment,
                    band_width: _opt.w,
                    query_offset: qb, // Use chain qb as offset
                });
            }
            Err(e) => {
                log::error!(
                    "{}: Chain {}: Error getting reference segment: {}",
                    query_name,
                    chain_idx,
                    e
                );
                chain_to_job_map.push(None);
            }
        }
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

    // Separate scores and CIGARs
    let alignment_scores: Vec<i32> = extended_cigars.iter().map(|(score, _)| *score).collect();
    let alignment_cigars: Vec<Vec<(u8, i32)>> = extended_cigars
        .into_iter()
        .map(|(_, cigar)| cigar)
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
    // Generate one alignment per filtered chain (matching C++ bwa-mem2 behavior)
    // Each chain represents a distinct mapping location for the read
    let mut alignments = Vec::new();

    for (chain_idx, chain) in filtered_chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            continue;
        }

        // Get alignment result for this chain
        let job_idx = match chain_to_job_map[chain_idx] {
            Some(idx) => idx,
            None => {
                log::warn!(
                    "{}: Chain {} has no alignment job, skipping",
                    query_name,
                    chain_idx
                );
                continue; // Skip chains without alignment jobs
            }
        };

        let mut cigar_for_alignment = alignment_cigars[job_idx].clone();
        let score = alignment_scores[job_idx];
        let query_offset = query_offsets[job_idx]; // Same as chain.query_start

        // Use first seed for reference position calculation
        let first_seed_idx = chain.seeds[0];
        let first_seed = &seeds[first_seed_idx];

        log::debug!(
            "{}: Chain {} (weight={}, kept={}): score={} bounds=[{}, {})",
            query_name,
            chain_idx,
            chain.weight,
            chain.kept,
            score,
            chain.query_start,
            chain.query_end
        );

        // === C++ STRATEGY: Add soft-clipping like mem_reg2aln (bwamem.cpp:1783-1796) ===
        // CIGAR from DP only covers bounded region [query_offset, query_offset + bounded_len)
        // Add soft-clips for [0, query_offset) and [query_offset + bounded_len, query_len)

        // Calculate bounded query length from CIGAR (sum of M, I, S operations)
        let bounded_query_len: i32 = cigar_for_alignment
            .iter()
            .filter_map(|&(op, len)| match op as char {
                'M' | 'I' | 'S' => Some(len),
                _ => None,
            })
            .sum();

        let clip5 = query_offset; // Bases before aligned region
        let clip3 = query_len as i32 - query_offset - bounded_query_len; // Bases after aligned region

        log::debug!(
            "{}: Chain {}: Adding soft-clipping: clip5={}, clip3={} (query_offset={}, bounded_len={}, full_len={})",
            query_name,
            chain_idx,
            clip5,
            clip3,
            query_offset,
            bounded_query_len,
            query_len
        );

        // Add 5' soft-clipping (like C++ bwamem.cpp:1789-1791)
        if clip5 > 0 {
            cigar_for_alignment.insert(0, (b'S', clip5));
        }

        // Add 3' soft-clipping (like C++ bwamem.cpp:1793-1795)
        if clip3 > 0 {
            cigar_for_alignment.push((b'S', clip3));
        }

        // Calculate the adjusted reference start position for bounded query alignment
        // Need to adjust based on query_offset since we're aligning a bounded region
        let adjusted_ref_start = if first_seed.query_pos < query_offset {
            // Seed is before our bounded region (shouldn't happen, but handle gracefully)
            first_seed.ref_pos
        } else {
            let offset_within_bounded = first_seed.query_pos - query_offset;
            if offset_within_bounded as u64 > first_seed.ref_pos {
                0
            } else {
                first_seed.ref_pos - offset_within_bounded as u64
            }
        };

        // Convert global position to chromosome-specific position
        let global_pos = adjusted_ref_start as i64;
        let (pos_f, _is_rev_depos) = bwa_idx.bns.bns_depos(global_pos);
        let rid = bwa_idx.bns.bns_pos2rid(pos_f);

        let (ref_name, ref_id, chr_pos) = if rid >= 0 && (rid as usize) < bwa_idx.bns.anns.len() {
            let ann = &bwa_idx.bns.anns[rid as usize];
            let chr_relative_pos = pos_f - ann.offset as i64;
            (ann.name.clone(), rid as usize, chr_relative_pos as u64)
        } else {
            log::warn!(
                "{}: Invalid reference ID {} for position {}",
                query_name,
                rid,
                global_pos
            );
            ("unknown_ref".to_string(), 0, first_seed.ref_pos)
        };

        // Calculate query bounds from chain (more accurate than single seed)
        let query_start = chain.query_start;
        let query_end = chain.query_end;
        let seed_coverage = chain.weight; // Use chain weight as seed coverage for MAPQ

        // Generate hash for tie-breaking (based on position and strand)
        let hash = hash_64((chr_pos << 1) | (if chain.is_rev { 1 } else { 0 }));

        alignments.push(Alignment {
            query_name: query_name.to_string(),
            flag: if chain.is_rev { 0x10 } else { 0 }, // 0x10 for reverse strand
            ref_name,
            ref_id,
            pos: chr_pos,
            mapq: 60, // Will be calculated by mark_secondary_alignments
            score,    // Use alignment score from chain-based DP
            cigar: cigar_for_alignment,
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::from_utf8_lossy(query_seq).to_string(),
            qual: query_qual.to_string(),
            tags: Vec::new(),
            // Internal fields for alignment selection
            query_start,
            query_end,
            seed_coverage,
            hash,
        });
    }

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
        // - Sets 0x100 flag for secondary alignments
        // - Calculates proper MAPQ values (0-60) based on score differences
        mark_secondary_alignments(&mut alignments, _opt);

        log::debug!(
            "{}: After alignment selection: {} alignments ({} primary, {} secondary)",
            query_name,
            alignments.len(),
            alignments.iter().filter(|a| a.flag & 0x100 == 0).count(),
            alignments.iter().filter(|a| a.flag & 0x100 != 0).count()
        );

        // === XA TAG GENERATION ===
        // Generate XA tags for primary alignments with qualifying secondaries
        // Format: XA:Z:chr1,+100,50M,2;chr2,-200,48M,3;
        let xa_tags = generate_xa_tags(&alignments, _opt);

        // Add XA tags to primary alignments
        for aln in alignments.iter_mut() {
            if aln.flag & 0x100 == 0 {
                // This is a primary alignment
                if let Some(xa_tag) = xa_tags.get(&aln.query_name) {
                    // Add XA tag to this alignment's tags
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
        // (SAM flag 0x4) to match C++ bwa-mem2 behavior and avoid silently dropping reads
        log::debug!(
            "{}: No alignments generated (no seeds or all chains filtered), creating unmapped read",
            query_name
        );

        alignments.push(Alignment {
            query_name: query_name.to_string(),
            flag: 0x4, // 0x4 = read unmapped
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
            tags: Vec::new(),
            // Internal fields (not used for unmapped reads)
            query_start: 0,
            query_end: 0,
            seed_coverage: 0,
            hash: 0,
        });
    }

    alignments
}

pub fn chain_seeds(mut seeds: Vec<Seed>, opt: &MemOpt) -> Vec<Chain> {
    if seeds.is_empty() {
        return Vec::new();
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
                // Calculate gap length
                let q_gap = seeds[i].query_pos - (seeds[j].query_pos + seeds[j].len);
                let r_gap =
                    seeds[i].ref_pos as i32 - (seeds[j].ref_pos + seeds[j].len as u64) as i32;

                // Check max_chain_gap constraint (do not chain if gap too large)
                if q_gap.max(r_gap.abs()) > max_gap {
                    continue; // Skip this potential chain connection
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

        chains.push(Chain {
            score: best_chain_score,
            seeds: chain_seeds_indices,
            query_start,
            query_end,
            ref_start,
            ref_end,
            is_rev,
            weight: 0, // Will be calculated by filter_chains()
            kept: 0,   // Will be set by filter_chains()
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

    chains
}
// bwa-mem2-rust/src/align_test.rs

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
        if (cp_occ[cp_block].one_hot_bwt_str[base] >> bit_position) & 1 == 1 {
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

    Some(bwa_idx.bwt.l2[base as usize] + get_occ(bwa_idx, pos as i64, base) as u64)
}

pub fn get_sa_entry(bwa_idx: &BwaIndex, mut pos: u64) -> u64 {
    let original_pos = pos;
    let mut count = 0;
    const MAX_ITERATIONS: u64 = 10000; // Safety limit to prevent infinite loops

    // eprintln!("get_sa_entry: starting with pos={}, sa_intv={}, seq_len={}, cp_occ.len()={}",
    //           original_pos, bwa_idx.bwt.sa_intv, bwa_idx.bwt.seq_len, bwa_idx.cp_occ.len());

    while pos % bwa_idx.bwt.sa_intv as u64 != 0 {
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
                bwa_idx.bwt.sa_intv,
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

    let sa_index = (pos / bwa_idx.bwt.sa_intv as u64) as usize;
    let sa_ms_byte = bwa_idx.bwt.sa_ms_byte[sa_index] as u64;
    let sa_ls_word = bwa_idx.bwt.sa_ls_word[sa_index] as u64;
    let sa_val = (sa_ms_byte << 32) | sa_ls_word;

    // Handle sentinel: SA values can point to the sentinel position (seq_len)
    // The sentinel represents the end-of-string marker, which wraps to position 0
    // seq_len = (l_pac << 1) + 1 (forward + RC + sentinel)
    // So sentinel position is seq_len - 1 = (l_pac << 1)
    let sentinel_pos = bwa_idx.bns.l_pac << 1;
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
        bwa_idx.bns.l_pac,
        (bwa_idx.bns.l_pac << 1)
    );
    result
}

#[cfg(test)]
mod tests {
    use crate::align::{SMEM, backward_ext, popcount64};
    use crate::mem::BwaIndex;
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
            k: 0,
            s: bwa_idx.bwt.seq_len,
            ..Default::default()
        };
        let new_smem = backward_ext(&bwa_idx, smem, 0); // 0 is 'A'
        assert_ne!(new_smem.s, 0);
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
            k: 0,
            s: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Test extending with each base
        for base in 0..4 {
            let extended = super::backward_ext(&bwa_idx, initial_smem, base);

            // Extended range should be smaller or equal to initial range
            assert!(
                extended.s <= initial_smem.s,
                "Extended range size {} should be <= initial size {} for base {}",
                extended.s,
                initial_smem.s,
                base
            );

            // k should be within bounds
            assert!(
                extended.k < bwa_idx.bwt.seq_len,
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
            k: 0,
            s: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Build a seed by extending with ACGT
        let bases = vec![0u8, 1, 2, 3]; // ACGT
        let mut prev_s = smem.s;

        for (i, &base) in bases.iter().enumerate() {
            smem = super::backward_ext(&bwa_idx, smem, base);

            // Range should generally get smaller (or stay same) with each extension
            // (though it could stay the same if the pattern is very common)
            assert!(
                smem.s <= prev_s,
                "After extension {}, range size {} should be <= previous {}",
                i,
                smem.s,
                prev_s
            );

            prev_s = smem.s;

            // If range becomes 0, we can't extend further
            if smem.s == 0 {
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
            k: 0,
            s: 0, // Zero range
            ..Default::default()
        };

        let extended = super::backward_ext(&bwa_idx, smem, 0);

        // Extending a zero range should still give zero range
        assert_eq!(extended.s, 0, "Extending zero range should give zero range");
    }

    #[test]
    fn test_smem_structure() {
        // Test SMEM structure creation and defaults
        let smem1 = SMEM {
            rid: 0,
            m: 10,
            n: 20,
            k: 5,
            l: 15,
            s: 10,
            is_rev_comp: false,
        };

        assert_eq!(smem1.m, 10);
        assert_eq!(smem1.n, 20);
        assert_eq!(smem1.s, 10);

        // Test default
        let smem2 = SMEM::default();
        assert_eq!(smem2.rid, 0);
        assert_eq!(smem2.m, 0);
        assert_eq!(smem2.n, 0);
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
        let sampled_pos = bwa_idx.bwt.sa_intv as u64;
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

        let sw_params =
            BandedPairWiseSW::new(4, 2, 4, 2, 100, 0, super::DEFAULT_SCORING_MATRIX, 2, -4);

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
            },
            super::AlignmentJob {
                seed_idx: 1,
                query: query2.clone(),
                target: target2.clone(),
                band_width: 10,
                query_offset: 0, // Test: align from start
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

        let alignments = super::generate_seeds(&bwa_idx, query_name, query_seq, query_qual, &opt);

        assert!(
            !alignments.is_empty(),
            "Expected at least one alignment for a matching query"
        );

        let primary_alignment = alignments.iter().find(|a| a.flag & 0x100 == 0);
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
}
