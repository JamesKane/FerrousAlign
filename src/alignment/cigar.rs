//! CIGAR string operations - single authoritative implementation
//!
//! This module consolidates all CIGAR manipulation to ensure consistent behavior
//! across the alignment pipeline. All CIGAR normalization, merging, and utility
//! functions should use this module.

use std::fmt::Write;

/// CIGAR operation type with zero-cost conversion to/from bytes
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CigarOp {
    M = b'M', // Match/mismatch
    I = b'I', // Insertion to reference
    D = b'D', // Deletion from reference
    S = b'S', // Soft clip
    H = b'H', // Hard clip
    N = b'N', // Skipped region (intron)
    X = b'X', // Sequence mismatch
    Eq = b'=', // Sequence match
}

impl CigarOp {
    /// Convert from byte representation
    #[inline(always)]
    pub const fn from_byte(b: u8) -> Option<Self> {
        match b {
            b'M' => Some(Self::M),
            b'I' => Some(Self::I),
            b'D' => Some(Self::D),
            b'S' => Some(Self::S),
            b'H' => Some(Self::H),
            b'N' => Some(Self::N),
            b'X' => Some(Self::X),
            b'=' => Some(Self::Eq),
            _ => None,
        }
    }

    /// Convert to byte representation
    #[inline(always)]
    pub const fn to_byte(self) -> u8 {
        self as u8
    }

    /// Returns true if this operation consumes query bases
    #[inline(always)]
    pub const fn consumes_query(self) -> bool {
        matches!(self, Self::M | Self::I | Self::S | Self::Eq | Self::X)
    }

    /// Returns true if this operation consumes reference bases
    #[inline(always)]
    pub const fn consumes_ref(self) -> bool {
        matches!(self, Self::M | Self::D | Self::N | Self::Eq | Self::X)
    }

    /// Returns true if this operation is a clip (soft or hard)
    #[inline(always)]
    pub const fn is_clip(self) -> bool {
        matches!(self, Self::S | Self::H)
    }
}

/// Check if a byte represents a query-consuming CIGAR operation
#[inline(always)]
pub const fn op_consumes_query(op: u8) -> bool {
    matches!(op, b'M' | b'I' | b'S' | b'=' | b'X')
}

/// Check if a byte represents a reference-consuming CIGAR operation
#[inline(always)]
pub const fn op_consumes_ref(op: u8) -> bool {
    matches!(op, b'M' | b'D' | b'N' | b'=' | b'X')
}

/// Normalize CIGAR in-place by merging adjacent identical operations.
///
/// E.g., `[(M, 10), (M, 5)]` → `[(M, 15)]`
///
/// This is a zero-allocation operation when the CIGAR is already normalized.
#[inline]
pub fn normalize_in_place(cigar: &mut Vec<(u8, i32)>) {
    if cigar.len() <= 1 {
        return;
    }

    let mut write = 0;
    for read in 1..cigar.len() {
        if cigar[read].0 == cigar[write].0 {
            // Same operation - merge lengths
            cigar[write].1 += cigar[read].1;
        } else {
            // Different operation - advance write pointer
            write += 1;
            cigar[write] = cigar[read];
        }
    }
    cigar.truncate(write + 1);
}

/// Normalize CIGAR by merging adjacent identical operations (consuming version).
///
/// E.g., `[(M, 10), (M, 5)]` → `[(M, 15)]`
///
/// Prefer `normalize_in_place` when possible to avoid allocation.
#[inline]
pub fn normalize(cigar: Vec<(u8, i32)>) -> Vec<(u8, i32)> {
    if cigar.len() <= 1 {
        return cigar;
    }

    let mut merged = Vec::with_capacity(cigar.len());
    let mut current_op = cigar[0].0;
    let mut current_len = cigar[0].1;

    for &(op, len) in &cigar[1..] {
        if op == current_op {
            // Same operation, merge
            current_len += len;
        } else {
            // Different operation, push current and start new
            merged.push((current_op, current_len));
            current_op = op;
            current_len = len;
        }
    }

    // Push the last operation
    merged.push((current_op, current_len));
    merged
}

/// Reverse a CIGAR string (used after aligning reversed sequences).
///
/// When we align reversed sequences for left extension, the CIGAR is also reversed.
/// This function reverses it back to the forward orientation.
#[inline]
pub fn reverse(cigar: &[(u8, i32)]) -> Vec<(u8, i32)> {
    cigar.iter().copied().rev().collect()
}

/// Calculate the reference-consuming length from a CIGAR.
///
/// Sums M, D, N, =, X operations (operations that consume reference bases).
#[inline]
pub fn reference_length(cigar: &[(u8, i32)]) -> i32 {
    cigar
        .iter()
        .filter_map(|&(op, len)| {
            if op_consumes_ref(op) {
                Some(len)
            } else {
                None
            }
        })
        .sum()
}

/// Calculate the query-consuming length from a CIGAR.
///
/// Sums M, I, S, =, X operations (operations that consume query bases).
#[inline]
pub fn query_length(cigar: &[(u8, i32)]) -> i32 {
    cigar
        .iter()
        .filter_map(|&(op, len)| {
            if op_consumes_query(op) {
                Some(len)
            } else {
                None
            }
        })
        .sum()
}

/// Convert CIGAR to string representation (e.g., "50M2I48M").
#[inline]
pub fn to_string(cigar: &[(u8, i32)]) -> String {
    if cigar.is_empty() {
        return "*".to_string();
    }

    let mut result = String::with_capacity(cigar.len() * 4);
    for &(op, len) in cigar {
        write!(&mut result, "{}{}", len, op as char).unwrap();
    }
    result
}

/// Write CIGAR to an existing string buffer (avoids allocation).
#[inline]
pub fn write_to_string(cigar: &[(u8, i32)], buf: &mut String) {
    if cigar.is_empty() {
        buf.push('*');
        return;
    }

    for &(op, len) in cigar {
        write!(buf, "{}{}", len, op as char).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cigar_op_from_byte() {
        assert_eq!(CigarOp::from_byte(b'M'), Some(CigarOp::M));
        assert_eq!(CigarOp::from_byte(b'I'), Some(CigarOp::I));
        assert_eq!(CigarOp::from_byte(b'D'), Some(CigarOp::D));
        assert_eq!(CigarOp::from_byte(b'S'), Some(CigarOp::S));
        assert_eq!(CigarOp::from_byte(b'H'), Some(CigarOp::H));
        assert_eq!(CigarOp::from_byte(b'?'), None);
    }

    #[test]
    fn test_op_consumes() {
        assert!(CigarOp::M.consumes_query());
        assert!(CigarOp::M.consumes_ref());
        assert!(CigarOp::I.consumes_query());
        assert!(!CigarOp::I.consumes_ref());
        assert!(!CigarOp::D.consumes_query());
        assert!(CigarOp::D.consumes_ref());
        assert!(CigarOp::S.consumes_query());
        assert!(!CigarOp::S.consumes_ref());
    }

    #[test]
    fn test_normalize_in_place() {
        // Already normalized
        let mut cigar = vec![(b'M', 50)];
        normalize_in_place(&mut cigar);
        assert_eq!(cigar, vec![(b'M', 50)]);

        // Needs merging
        let mut cigar = vec![(b'M', 10), (b'M', 5), (b'I', 2), (b'M', 20)];
        normalize_in_place(&mut cigar);
        assert_eq!(cigar, vec![(b'M', 15), (b'I', 2), (b'M', 20)]);

        // All same op
        let mut cigar = vec![(b'M', 10), (b'M', 20), (b'M', 30)];
        normalize_in_place(&mut cigar);
        assert_eq!(cigar, vec![(b'M', 60)]);

        // Empty
        let mut cigar: Vec<(u8, i32)> = vec![];
        normalize_in_place(&mut cigar);
        assert!(cigar.is_empty());
    }

    #[test]
    fn test_normalize() {
        let cigar = vec![(b'M', 10), (b'M', 5), (b'I', 2), (b'M', 20)];
        let normalized = normalize(cigar);
        assert_eq!(normalized, vec![(b'M', 15), (b'I', 2), (b'M', 20)]);
    }

    #[test]
    fn test_reverse() {
        let cigar = vec![(b'M', 10), (b'I', 2), (b'M', 20)];
        let reversed = reverse(&cigar);
        assert_eq!(reversed, vec![(b'M', 20), (b'I', 2), (b'M', 10)]);
    }

    #[test]
    fn test_reference_length() {
        // 50M2I48M = 50 + 48 = 98 ref bases (I doesn't consume ref)
        let cigar = vec![(b'M', 50), (b'I', 2), (b'M', 48)];
        assert_eq!(reference_length(&cigar), 98);

        // 10M5D10M = 10 + 5 + 10 = 25 ref bases
        let cigar = vec![(b'M', 10), (b'D', 5), (b'M', 10)];
        assert_eq!(reference_length(&cigar), 25);

        // 5S50M5S = 50 ref bases (S doesn't consume ref)
        let cigar = vec![(b'S', 5), (b'M', 50), (b'S', 5)];
        assert_eq!(reference_length(&cigar), 50);
    }

    #[test]
    fn test_query_length() {
        // 50M2I48M = 50 + 2 + 48 = 100 query bases
        let cigar = vec![(b'M', 50), (b'I', 2), (b'M', 48)];
        assert_eq!(query_length(&cigar), 100);

        // 10M5D10M = 10 + 10 = 20 query bases (D doesn't consume query)
        let cigar = vec![(b'M', 10), (b'D', 5), (b'M', 10)];
        assert_eq!(query_length(&cigar), 20);

        // 5S50M5S = 5 + 50 + 5 = 60 query bases
        let cigar = vec![(b'S', 5), (b'M', 50), (b'S', 5)];
        assert_eq!(query_length(&cigar), 60);
    }

    #[test]
    fn test_to_string() {
        let cigar = vec![(b'M', 50), (b'I', 2), (b'M', 48)];
        assert_eq!(to_string(&cigar), "50M2I48M");

        let empty: Vec<(u8, i32)> = vec![];
        assert_eq!(to_string(&empty), "*");
    }
}
