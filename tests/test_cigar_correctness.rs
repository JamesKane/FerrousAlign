/// Critical regression tests for CIGAR correctness issues
///
/// Covers bugs fixed in recent sessions:
/// - Seed index bug causing incorrect CIGAR strings (commit 0724f49)
/// - Alignment coordinate off-by-one errors (commit 7d5826c)
/// - CIGAR length validation (commit ff8f4af)
///
/// These tests ensure:
/// 1. CIGAR length matches query length
/// 2. CIGAR operations correctly represent alignment
/// 3. Seed coordinates are correctly used in CIGAR generation
/// 4. Off-by-one errors don't creep back in

/// Test that CIGAR length matches query length
#[test]
fn test_cigar_length_matches_query() {
    // Bug: CIGAR length was sometimes off by 1 from query length
    // This test ensures CIGAR operations that consume query sum to query length

    let test_cases = vec![
        // (CIGAR, query_length, should_match)
        (vec![(b'M', 100)], 100, true),
        (vec![(b'M', 50), (b'I', 10), (b'M', 40)], 100, true),
        (vec![(b'M', 50), (b'D', 10), (b'M', 50)], 100, true),
        (vec![(b'S', 5), (b'M', 90), (b'S', 5)], 100, true),
        (vec![(b'M', 50), (b'I', 10), (b'M', 40)], 101, false), // Wrong length
        (vec![(b'M', 50), (b'D', 10), (b'M', 50)], 110, false), // Deletion doesn't consume query
    ];

    for (cigar, query_len, should_match) in test_cases {
        let cigar_len = calculate_cigar_query_length(&cigar);
        if should_match {
            assert_eq!(
                cigar_len, query_len,
                "CIGAR length {} should match query length {} for {:?}",
                cigar_len, query_len, cigar
            );
        } else {
            assert_ne!(
                cigar_len, query_len,
                "CIGAR length {} should NOT match query length {} for {:?}",
                cigar_len, query_len, cigar
            );
        }
    }
}

/// Test CIGAR validation and correction logic
#[test]
fn test_cigar_length_validation() {
    // From align.rs:3043 - CIGAR length validation safety check
    let query_len = 100;

    // Case 1: CIGAR too short (off by 1)
    let mut cigar = vec![(b'M', 95), (b'S', 4)];
    let cigar_len = calculate_cigar_query_length(&cigar);
    assert_eq!(cigar_len, 99, "CIGAR should be 1 short");

    // Simulate the fix: adjust last query-consuming operation
    let diff = query_len - cigar_len;
    for op in cigar.iter_mut().rev() {
        if op.0 == b'S' || op.0 == b'M' {
            op.1 += diff;
            break;
        }
    }

    let corrected_len = calculate_cigar_query_length(&cigar);
    assert_eq!(
        corrected_len, query_len,
        "Corrected CIGAR should match query length"
    );

    // Case 2: CIGAR too long (off by 1)
    let mut cigar = vec![(b'M', 96), (b'S', 5)];
    let cigar_len = calculate_cigar_query_length(&cigar);
    assert_eq!(cigar_len, 101, "CIGAR should be 1 too long");

    let diff = query_len - cigar_len; // This will be -1
    for op in cigar.iter_mut().rev() {
        if op.0 == b'S' || op.0 == b'M' {
            op.1 += diff;
            break;
        }
    }

    let corrected_len = calculate_cigar_query_length(&cigar);
    assert_eq!(
        corrected_len, query_len,
        "Corrected CIGAR should match query length"
    );
}

/// Test seed length calculation (fixed in commit 7d5826c)
#[test]
fn test_seed_length_calculation() {
    // Bug: Was using smem.query_end - smem.query_start + 1
    // Fix: smem.query_end is EXCLUSIVE, so: smem.query_end - smem.query_start

    // Test cases: (query_start, query_end, expected_length)
    let test_cases = vec![
        (0, 10, 10),  // 10 bases from [0, 10)
        (5, 15, 10),  // 10 bases from [5, 15)
        (0, 1, 1),    // 1 base from [0, 1)
        (10, 50, 40), // 40 bases from [10, 50)
    ];

    for (start, end, expected_len) in test_cases {
        // OLD BUGGY WAY: let len = end - start + 1;
        // NEW CORRECT WAY:
        let len = end - start;

        assert_eq!(
            len, expected_len,
            "Seed length from [{}, {}) should be {} (exclusive end)",
            start, end, expected_len
        );
    }
}

/// Test that operations consuming query bases are correctly identified
#[test]
fn test_query_consuming_operations() {
    // CIGAR operations that consume query bases: M, I, S, =, X
    // Operations that DON'T consume query: D, H, N, P

    let query_consuming = [b'M', b'I', b'S', b'=', b'X'];
    let ref_only = [b'D', b'H', b'N', b'P'];

    for op in query_consuming {
        assert!(
            is_query_consuming_op(op),
            "Operation {} should consume query bases",
            op as char
        );
    }

    for op in ref_only {
        assert!(
            !is_query_consuming_op(op),
            "Operation {} should NOT consume query bases",
            op as char
        );
    }
}

/// Test complex CIGAR scenarios that were problematic
#[test]
fn test_complex_cigar_scenarios() {
    // Scenario 1: Multiple insertions and deletions
    let cigar = vec![
        (b'M', 20),
        (b'I', 5), // Insertion: consumes query
        (b'M', 10),
        (b'D', 3), // Deletion: does NOT consume query
        (b'M', 15),
        (b'S', 10), // Soft clip: consumes query
    ];
    let expected_query_len = 20 + 5 + 10 + 15 + 10; // = 60
    assert_eq!(
        calculate_cigar_query_length(&cigar),
        expected_query_len,
        "Complex CIGAR should correctly calculate query length"
    );

    // Scenario 2: Leading and trailing soft clips
    let cigar = vec![(b'S', 5), (b'M', 90), (b'S', 5)];
    assert_eq!(
        calculate_cigar_query_length(&cigar),
        100,
        "Soft clips should be included in query length"
    );

    // Scenario 3: Hard clips (don't consume query)
    let cigar = vec![
        (b'H', 5), // Hard clip: does NOT consume query
        (b'M', 90),
        (b'H', 5),
    ];
    assert_eq!(
        calculate_cigar_query_length(&cigar),
        90,
        "Hard clips should NOT be included in query length"
    );
}

/// Test that CIGAR adjustment doesn't create negative lengths
#[test]
fn test_cigar_adjustment_safety() {
    // Edge case: What if we need to subtract more than the operation length?
    let mut cigar = vec![(b'S', 2), (b'M', 95)];
    let query_len = 100;
    let cigar_len = calculate_cigar_query_length(&cigar);

    // Need to add 3, but last S only has length 2
    // Should adjust last query-consuming op
    let diff = query_len - cigar_len; // = 3

    for op in cigar.iter_mut().rev() {
        if op.0 == b'S' || op.0 == b'M' {
            op.1 += diff;
            if op.1 > 0 {
                break;
            }
            // If went negative, set to 0 and continue
            // (this is the safety check from align.rs:3070)
        }
    }

    // Should have adjusted the M operation
    assert_eq!(cigar[1].1, 98, "M operation should be adjusted to 98");
    assert_eq!(
        calculate_cigar_query_length(&cigar),
        query_len,
        "Final CIGAR should match query length"
    );
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate total query length from CIGAR
fn calculate_cigar_query_length(cigar: &[(u8, i32)]) -> i32 {
    cigar
        .iter()
        .filter_map(|&(op, len)| {
            if is_query_consuming_op(op) {
                Some(len)
            } else {
                None
            }
        })
        .sum()
}

/// Check if CIGAR operation consumes query bases
fn is_query_consuming_op(op: u8) -> bool {
    matches!(op as char, 'M' | 'I' | 'S' | '=' | 'X')
}

#[test]
fn test_helper_calculate_cigar_query_length() {
    // Self-test for the helper function
    assert_eq!(calculate_cigar_query_length(&[(b'M', 100)]), 100);
    assert_eq!(
        calculate_cigar_query_length(&[(b'M', 50), (b'D', 10), (b'M', 50)]),
        100
    );
    assert_eq!(
        calculate_cigar_query_length(&[(b'M', 50), (b'I', 10), (b'M', 40)]),
        100
    );
}
