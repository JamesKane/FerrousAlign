// Tests for src/align.rs
// Extracted from inline tests to reduce clutter in production code

use ferrous_align::align::{base_to_code, reverse_complement_code};

#[test]
fn test_base_to_code() {
    assert_eq!(base_to_code(b'A'), 0);
    assert_eq!(base_to_code(b'a'), 0);
    assert_eq!(base_to_code(b'C'), 1);
    assert_eq!(base_to_code(b'c'), 1);
    assert_eq!(base_to_code(b'G'), 2);
    assert_eq!(base_to_code(b'g'), 2);
    assert_eq!(base_to_code(b'T'), 3);
    assert_eq!(base_to_code(b't'), 3);
    assert_eq!(base_to_code(b'N'), 4);
    assert_eq!(base_to_code(b'X'), 4); // Other characters
}

#[test]
fn test_reverse_complement_code() {
    assert_eq!(reverse_complement_code(0), 3); // A -> T
    assert_eq!(reverse_complement_code(1), 2); // C -> G
    assert_eq!(reverse_complement_code(2), 1); // G -> C
    assert_eq!(reverse_complement_code(3), 0); // T -> A
    assert_eq!(reverse_complement_code(4), 4); // N -> N
    assert_eq!(reverse_complement_code(5), 4); // Other -> Other
}

#[test]
fn test_reverse_complement_encoding_with_n_bases() {
    // Regression test for Session 30 bug: using XOR trick (b ^ 3) incorrectly encodes N bases
    // For N (code 4): 4 ^ 3 = 7 (INVALID!)
    // This test ensures we use reverse_complement_code() which handles N correctly

    // Test sequence with N bases: "ACGTNACGT"
    let sequence = b"ACGTNACGT";
    let encoded: Vec<u8> = sequence.iter().map(|&b| base_to_code(b)).collect();

    // Expected: [0, 1, 2, 3, 4, 0, 1, 2, 3]
    assert_eq!(encoded, vec![0, 1, 2, 3, 4, 0, 1, 2, 3]);

    // Create reverse complement using reverse_complement_code (CORRECT)
    let rc_encoded: Vec<u8> = encoded
        .iter()
        .map(|&b| reverse_complement_code(b))
        .collect();
    let mut rc_encoded_rev = rc_encoded.clone();
    rc_encoded_rev.reverse();

    // Expected RC: [3, 2, 1, 0, 4, 3, 2, 1, 0] (reversed: [0, 1, 2, 3, 4, 0, 1, 2, 3])
    // Note: Reverse complement of N is N (code 4)
    assert_eq!(rc_encoded, vec![3, 2, 1, 0, 4, 3, 2, 1, 0]);

    // CRITICAL: All codes must be in valid range [0, 4]
    for &code in &rc_encoded {
        assert!(
            code <= 4,
            "Invalid base code {} detected! All codes must be 0-4. Code 7 indicates XOR bug.",
            code
        );
    }

    // Test that XOR trick would fail (for documentation purposes)
    // DON'T USE THIS IN PRODUCTION CODE!
    let bad_rc_with_xor: Vec<u8> = encoded.iter().map(|&b| b ^ 3).collect();
    // For the 'N' base (code 4): 4 ^ 3 = 7 (INVALID!)
    assert_eq!(
        bad_rc_with_xor[4], 7,
        "XOR trick produces invalid code 7 for N"
    );

    // Demonstrate the correct way doesn't produce invalid codes
    assert_ne!(
        rc_encoded[4], 7,
        "reverse_complement_code() correctly handles N"
    );
    assert_eq!(rc_encoded[4], 4, "N reverse complements to N");
}
