// tests/sse_neon_parity.rs
use ferrous_align::core::alignment::banded_swa::{
    OutScore,
    isa_sse_neon::simd_banded_swa_batch16,
};

#[test]
fn test_simd_banded_swa_batch16_basic() {
    // Basic test to ensure the function compiles and runs
    // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
    let query: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
    let target: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
    let batch = vec![(4, &query[..], 4, &target[..], 10, 0)];

    // Default scoring matrix (match=1, mismatch=0)
    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1; // Match score on diagonal
    }

    let results = unsafe {
        simd_banded_swa_batch16(
            &batch, 6,   // o_del
            1,   // e_del
            6,   // o_ins
            1,   // e_ins
            100, // zdrop
            &mat, 5,
        )
    };

    assert_eq!(results.len(), 1);
    // Perfect match should have score >= 4
    assert!(
        results[0].score >= 4,
        "Expected score >= 4 for perfect match, got {}",
        results[0].score
    );
}

#[test]
fn test_simd_banded_swa_batch16_multiple() {
    // Test with multiple alignments in batch
    // Kernel expects 2-bit encoded bases: A=0, C=1, G=2, T=3
    let q1: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
    let t1: [u8; 4] = [0, 1, 2, 3]; // ACGT in 2-bit encoding
    let q2: [u8; 4] = [0, 0, 0, 0]; // AAAA in 2-bit encoding
    let t2: [u8; 4] = [3, 3, 3, 3]; // TTTT in 2-bit encoding

    let batch = vec![
        (4, &q1[..], 4, &t1[..], 10, 0),
        (4, &q2[..], 4, &t2[..], 10, 0),
    ];

    let mut mat = [0i8; 25];
    for i in 0..4 {
        mat[i * 5 + i] = 1; // Match
    }

    let results = unsafe { simd_banded_swa_batch16(&batch, 6, 1, 6, 1, 100, &mat, 5) };

    assert_eq!(results.len(), 2);
    // Perfect match should score higher
    assert!(
        results[0].score >= results[1].score,
        "Perfect match should score >= mismatch: {} vs {}",
        results[0].score,
        results[1].score
    );
}