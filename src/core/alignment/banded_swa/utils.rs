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
