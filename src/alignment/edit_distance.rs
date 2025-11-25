//! Edit distance (NM) and MD tag computation - single authoritative implementation
//!
//! This module consolidates all NM/MD calculation to ensure consistent behavior
//! across the alignment pipeline. It replaces:
//! - `Alignment::calculate_edit_distance()` - CIGAR-only approximation
//! - `Alignment::calculate_nm_from_score()` - score-based heuristic
//! - `Alignment::generate_md_tag()` - MD generation
//! - `Alignment::calculate_exact_nm()` - MD-derived NM
//!
//! The single `compute_nm_and_md()` function computes both values in one pass,
//! which is more efficient and ensures consistency.

/// Convert 2-bit encoded base to ASCII character
#[inline(always)]
pub const fn base_to_char(b: u8) -> char {
    match b {
        0 => 'A',
        1 => 'C',
        2 => 'G',
        3 => 'T',
        _ => 'N',
    }
}

/// Compute exact NM (edit distance) and MD tag from aligned sequences and CIGAR.
///
/// This is the single authoritative implementation for NM/MD calculation.
/// Both values are computed in a single pass for efficiency.
///
/// # Arguments
/// * `ref_seq` - Reference sequence (2-bit encoded, only the aligned portion)
/// * `query_seq` - Query sequence (2-bit encoded, only the aligned portion excluding soft clips)
/// * `cigar` - CIGAR operations (soft clips should be present but are handled correctly)
///
/// # Returns
/// * `(nm, md)` - Edit distance and MD tag string
///
/// # NM Calculation
/// NM = mismatches + insertions + deletions
/// - Mismatches: bases where ref != query in M operations
/// - Insertions: sum of I operation lengths
/// - Deletions: sum of D operation lengths
///
/// # MD Tag Format
/// - Numbers: count of matching bases
/// - Letters: mismatching reference base (e.g., "A" means ref was A, query was different)
/// - ^LETTERS: deleted reference bases (e.g., "^AC" means AC was deleted from ref)
/// - Consecutive mismatches are separated by 0: "A0T" not "AT"
///
/// # Example
/// ```ignore
/// let (nm, md) = compute_nm_and_md(&ref_seq, &query_seq, &cigar);
/// // For a perfect 100bp match: nm=0, md="100"
/// // For 50 matches, 1 mismatch (ref=A), 49 matches: nm=1, md="50A49"
/// // For 25 matches, 2bp deletion (AC), 25 matches: nm=2, md="25^AC25"
/// ```
pub fn compute_nm_and_md(ref_seq: &[u8], query_seq: &[u8], cigar: &[(u8, i32)]) -> (i32, String) {
    let mut nm: i32 = 0;
    // Pre-allocate MD string - typical size is cigar.len() * 3
    let mut md = String::with_capacity(cigar.len() * 3 + 8);
    let mut match_count: u32 = 0;

    let mut ri = 0usize; // Reference index
    let mut qi = 0usize; // Query index

    for &(op, len) in cigar {
        let len_usize = len as usize;

        match op {
            b'M' | b'=' | b'X' => {
                // Match/mismatch operations - compare bases
                for _ in 0..len {
                    if ri >= ref_seq.len() || qi >= query_seq.len() {
                        // Bounds check - shouldn't happen with valid input
                        break;
                    }

                    if ref_seq[ri] == query_seq[qi] {
                        // Match
                        match_count += 1;
                    } else {
                        // Mismatch - count for NM, emit for MD
                        nm += 1;
                        // SAM spec: emit match count (even if 0), then mismatch base
                        push_number(&mut md, match_count);
                        match_count = 0;
                        md.push(base_to_char(ref_seq[ri]));
                    }

                    ri += 1;
                    qi += 1;
                }
            }
            b'I' => {
                // Insertion to query - counts for NM, no MD entry
                nm += len;
                qi += len_usize;
            }
            b'D' => {
                // Deletion from reference - counts for NM, emit ^BASES for MD
                nm += len;

                // Emit accumulated matches before deletion
                if match_count > 0 {
                    push_number(&mut md, match_count);
                    match_count = 0;
                }

                // Emit deletion marker and bases
                md.push('^');
                for _ in 0..len {
                    if ri >= ref_seq.len() {
                        break;
                    }
                    md.push(base_to_char(ref_seq[ri]));
                    ri += 1;
                }
            }
            b'N' => {
                // Skipped region (intron) - consume reference, no MD entry
                // Note: N is typically used for RNA-seq spliced alignments
                ri += len_usize;
            }
            b'S' => {
                // Soft clip - the input query_seq should NOT include soft-clipped bases
                // (caller passes query[query_start..query_end] excluding clips)
                // Do NOT advance qi - the aligned portion starts at index 0
            }
            b'H' => {
                // Hard clip - bases not even in the query sequence
                // No index advancement needed
            }
            _ => {
                // Unknown operation - skip
            }
        }
    }

    // Emit final match count
    if match_count > 0 {
        push_number(&mut md, match_count);
    }

    // Handle empty MD tag (shouldn't happen with valid input)
    if md.is_empty() {
        md.push('0');
    }

    (nm, md)
}

/// Compute NM directly from CIGAR and sequences (without generating MD string).
///
/// This is a lightweight version when you only need NM, not MD.
/// Slightly more efficient as it avoids string allocation.
#[inline]
pub fn compute_nm_only(ref_seq: &[u8], query_seq: &[u8], cigar: &[(u8, i32)]) -> i32 {
    let mut nm: i32 = 0;
    let mut ri = 0usize;
    let mut qi = 0usize;

    for &(op, len) in cigar {
        let len_usize = len as usize;

        match op {
            b'M' | b'=' | b'X' => {
                for _ in 0..len {
                    if ri >= ref_seq.len() || qi >= query_seq.len() {
                        break;
                    }
                    if ref_seq[ri] != query_seq[qi] {
                        nm += 1;
                    }
                    ri += 1;
                    qi += 1;
                }
            }
            b'I' => {
                nm += len;
                qi += len_usize;
            }
            b'D' => {
                nm += len;
                ri += len_usize;
            }
            b'N' => {
                ri += len_usize;
            }
            b'S' | b'H' => {
                // Clips don't affect NM
            }
            _ => {}
        }
    }

    nm
}

/// Compute NM from an existing MD tag and CIGAR.
///
/// This is useful when you have the MD tag already and need NM.
/// NM = mismatches (from MD) + insertions (from CIGAR) + deletions (from CIGAR)
///
/// Note: The MD tag encodes mismatches as letters (outside deletion blocks)
/// and deletions as ^LETTERS. We need to be careful not to double-count.
#[inline]
pub fn compute_nm_from_md(md_tag: &str, cigar: &[(u8, i32)]) -> i32 {
    let mut nm: i32 = 0;
    let mut in_deletion = false;

    // Count mismatches from MD tag
    // Mismatches are letters NOT in deletion blocks
    for ch in md_tag.chars() {
        if ch == '^' {
            in_deletion = true;
        } else if ch.is_ascii_digit() {
            in_deletion = false;
        } else if ch.is_ascii_alphabetic() {
            if !in_deletion {
                // This is a mismatch (not a deleted base)
                nm += 1;
            }
            // Deleted bases are counted from CIGAR, not here
        }
    }

    // Count insertions and deletions from CIGAR
    for &(op, len) in cigar {
        match op {
            b'I' | b'D' => {
                nm += len;
            }
            _ => {}
        }
    }

    nm
}

/// Helper to push a number to the MD string
#[inline(always)]
fn push_number(md: &mut String, n: u32) {
    // Fast path for common small numbers
    if n < 10 {
        md.push((b'0' + n as u8) as char);
    } else if n < 100 {
        md.push((b'0' + (n / 10) as u8) as char);
        md.push((b'0' + (n % 10) as u8) as char);
    } else {
        use std::fmt::Write;
        write!(md, "{}", n).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_to_char() {
        assert_eq!(base_to_char(0), 'A');
        assert_eq!(base_to_char(1), 'C');
        assert_eq!(base_to_char(2), 'G');
        assert_eq!(base_to_char(3), 'T');
        assert_eq!(base_to_char(4), 'N');
        assert_eq!(base_to_char(255), 'N');
    }

    #[test]
    fn test_perfect_match() {
        // 10bp perfect match
        let ref_seq = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]; // ACGTACGTAC
        let query_seq = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1]; // ACGTACGTAC
        let cigar = vec![(b'M', 10)];

        let (nm, md) = compute_nm_and_md(&ref_seq, &query_seq, &cigar);
        assert_eq!(nm, 0);
        assert_eq!(md, "10");
    }

    #[test]
    fn test_single_mismatch() {
        // 10bp with 1 mismatch at position 5 (ref=G, query=T)
        let ref_seq = vec![0, 1, 2, 3, 0, 2, 2, 3, 0, 1]; // ACGTAG...
        let query_seq = vec![0, 1, 2, 3, 0, 3, 2, 3, 0, 1]; // ACGTAT...
        let cigar = vec![(b'M', 10)];

        let (nm, md) = compute_nm_and_md(&ref_seq, &query_seq, &cigar);
        assert_eq!(nm, 1);
        assert_eq!(md, "5G4"); // 5 matches, G mismatch, 4 matches
    }

    #[test]
    fn test_consecutive_mismatches() {
        // Consecutive mismatches should be separated by 0
        let ref_seq = vec![0, 2, 3, 1]; // AGTC
        let query_seq = vec![0, 1, 0, 1]; // ACAC
        let cigar = vec![(b'M', 4)];

        let (nm, md) = compute_nm_and_md(&ref_seq, &query_seq, &cigar);
        assert_eq!(nm, 2); // 2 mismatches
        assert_eq!(md, "1G0T1"); // 1 match, G mismatch, 0 matches, T mismatch, 1 match
    }

    #[test]
    fn test_deletion() {
        // 5M2D5M - deletion of 2 bases from reference
        let ref_seq = vec![0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0]; // ref with deleted AC
        let query_seq = vec![0, 1, 2, 3, 0, 0, 1, 2, 3, 0]; // query without those bases
        let cigar = vec![(b'M', 5), (b'D', 2), (b'M', 5)];

        let (nm, md) = compute_nm_and_md(&ref_seq, &query_seq, &cigar);
        assert_eq!(nm, 2); // 2 deletions
        assert_eq!(md, "5^CG5"); // 5 matches, deletion of CG, 5 matches
    }

    #[test]
    fn test_insertion() {
        // 5M2I5M - insertion of 2 bases in query
        let ref_seq = vec![0, 1, 2, 3, 0, 0, 1, 2, 3, 0]; // ref without inserted bases
        let query_seq = vec![0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 0]; // query with inserted CG
        let cigar = vec![(b'M', 5), (b'I', 2), (b'M', 5)];

        let (nm, md) = compute_nm_and_md(&ref_seq, &query_seq, &cigar);
        assert_eq!(nm, 2); // 2 insertions
        assert_eq!(md, "10"); // MD doesn't show insertions, just matches
    }

    #[test]
    fn test_soft_clip() {
        // 3S5M - soft clip doesn't affect NM/MD
        // Note: query_seq should NOT include the soft-clipped bases
        let ref_seq = vec![0, 1, 2, 3, 0]; // ACGTA
        let query_seq = vec![0, 1, 2, 3, 0]; // Aligned portion only (no soft clip bases)
        let cigar = vec![(b'S', 3), (b'M', 5)];

        let (nm, md) = compute_nm_and_md(&ref_seq, &query_seq, &cigar);
        assert_eq!(nm, 0);
        assert_eq!(md, "5");
    }

    #[test]
    fn test_complex_alignment() {
        // 10M1I5M2D10M with 2 mismatches
        // This tests a complex real-world-like alignment
        let ref_seq = vec![
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, // 10 ref bases for first M
            0, 1, 2, 3, 0, // 5 ref bases for second M
            2, 3, // 2 deleted bases
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, // 10 ref bases for third M
        ];
        let query_seq = vec![
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, // 10 query bases for first M
            1, 2, // 2 inserted bases
            0, 1, 2, 3, 0, // 5 query bases for second M
            0, 1, 2, 3, 0, 1, 2, 3, 0, 1, // 10 query bases for third M
        ];
        let cigar = vec![(b'M', 10), (b'I', 2), (b'M', 5), (b'D', 2), (b'M', 10)];

        let (nm, md) = compute_nm_and_md(&ref_seq, &query_seq, &cigar);
        assert_eq!(nm, 4); // 2 insertions + 2 deletions
        assert_eq!(md, "15^GT10");
    }

    #[test]
    fn test_compute_nm_only() {
        let ref_seq = vec![0, 1, 2, 3, 0, 2, 2, 3, 0, 1];
        let query_seq = vec![0, 1, 2, 3, 0, 3, 2, 3, 0, 1];
        let cigar = vec![(b'M', 10)];

        let nm = compute_nm_only(&ref_seq, &query_seq, &cigar);
        assert_eq!(nm, 1);
    }

    #[test]
    fn test_compute_nm_from_md() {
        // MD: "50A49" with CIGAR 100M (1 mismatch, no indels)
        let md = "50A49";
        let cigar = vec![(b'M', 100)];
        assert_eq!(compute_nm_from_md(md, &cigar), 1);

        // MD: "25^AC25" with CIGAR 25M2D25M (0 mismatches, 2 deletions)
        let md = "25^AC25";
        let cigar = vec![(b'M', 25), (b'D', 2), (b'M', 25)];
        assert_eq!(compute_nm_from_md(md, &cigar), 2);

        // MD: "10" with CIGAR 5M2I5M (0 mismatches, 2 insertions)
        let md = "10";
        let cigar = vec![(b'M', 5), (b'I', 2), (b'M', 5)];
        assert_eq!(compute_nm_from_md(md, &cigar), 2);

        // Complex: MD "10A5^GT10" with CIGAR 10M1I5M2D10M
        // 1 mismatch + 1 insertion + 2 deletions = 4
        let md = "10A5^GT10";
        let cigar = vec![(b'M', 10), (b'I', 1), (b'M', 6), (b'D', 2), (b'M', 10)];
        assert_eq!(compute_nm_from_md(md, &cigar), 4);
    }

    #[test]
    fn test_empty_input() {
        let (nm, md) = compute_nm_and_md(&[], &[], &[]);
        assert_eq!(nm, 0);
        assert_eq!(md, "0");
    }
}
