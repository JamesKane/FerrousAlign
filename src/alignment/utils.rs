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