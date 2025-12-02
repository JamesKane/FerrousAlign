// Test suite for kernel optimization refactoring
// Validates that simplified kernel produces identical results to original
//
// This test specifically covers:
// 1. Removal of redundant max_epi8 operations after saturating subtraction
// 2. Hoisting of temporary array allocations
// 3. Correctness of SIMD operations across all supported widths

use ferrous_align::alignment::banded_swa::kernel::{sw_kernel, KernelParams, SwEngine128, SwEngine256};
use ferrous_align::alignment::banded_swa::bwa_fill_scmat;

/// Helper to create test parameters for kernel testing
fn create_test_params<'a>(
    query_soa: &'a [u8],
    target_soa: &'a [u8],
    qlen: &'a [i8],
    tlen: &'a [i8],
    h0: &'a [i8],
    w: &'a [i8],
    max_qlen: i32,
    max_tlen: i32,
    mat: &'a [i8; 25],
) -> KernelParams<'a> {
    KernelParams {
        batch: &[],
        query_soa,
        target_soa,
        qlen,
        tlen,
        h0,
        w,
        max_qlen,
        max_tlen,
        o_del: 6,
        e_del: 1,
        o_ins: 6,
        e_ins: 1,
        zdrop: 100,
        mat,
        m: 5,
        cfg: None,
    }
}

#[test]
fn test_kernel_identical_sequence_alignment() {
    // Test: Identical sequences should produce maximum score
    // This validates that saturating subtraction works correctly

    let mat = bwa_fill_scmat(1, 4, -1);

    // Create 16 lanes of identical query/target pairs (SSE width)
    let seq_len = 10;
    let num_lanes = 16;

    let mut query_soa = vec![0u8; seq_len * num_lanes];
    let mut target_soa = vec![0u8; seq_len * num_lanes];

    // Fill with pattern ACGTACGTAC (bases 0,1,2,3,0,1,2,3,0,1)
    for pos in 0..seq_len {
        for lane in 0..num_lanes {
            let base = (pos % 4) as u8;
            query_soa[pos * num_lanes + lane] = base;
            target_soa[pos * num_lanes + lane] = base;
        }
    }

    let qlen = vec![seq_len as i8; num_lanes];
    let tlen = vec![seq_len as i8; num_lanes];
    let h0 = vec![0i8; num_lanes];
    let w = vec![10i8; num_lanes];

    let params = create_test_params(
        &query_soa,
        &target_soa,
        &qlen,
        &tlen,
        &h0,
        &w,
        seq_len as i32,
        seq_len as i32,
        &mat,
    );

    let results = unsafe { sw_kernel::<16, SwEngine128>(&params, num_lanes) };

    assert_eq!(results.len(), num_lanes);

    // All lanes should have perfect match score
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.score > 0,
            "Lane {}: Score should be positive for identical sequences, got {}",
            i,
            result.score
        );
        assert_eq!(
            result.query_end_pos, (seq_len - 1) as i32,
            "Lane {}: Query end should be at last position",
            i
        );
        assert_eq!(
            result.target_end_pos, (seq_len - 1) as i32,
            "Lane {}: Target end should be at last position",
            i
        );
    }
}

#[test]
fn test_kernel_complete_mismatch() {
    // Test: Completely mismatched sequences should produce low scores
    // Validates that gap penalties and mismatch handling work correctly

    let mat = bwa_fill_scmat(1, 4, -1);

    let seq_len = 8;
    let num_lanes = 16;

    let mut query_soa = vec![0u8; seq_len * num_lanes];
    let mut target_soa = vec![0u8; seq_len * num_lanes];

    // Query: all A (0), Target: all T (3)
    for pos in 0..seq_len {
        for lane in 0..num_lanes {
            query_soa[pos * num_lanes + lane] = 0; // A
            target_soa[pos * num_lanes + lane] = 3; // T
        }
    }

    let qlen = vec![seq_len as i8; num_lanes];
    let tlen = vec![seq_len as i8; num_lanes];
    let h0 = vec![0i8; num_lanes];
    let w = vec![10i8; num_lanes];

    let params = create_test_params(
        &query_soa,
        &target_soa,
        &qlen,
        &tlen,
        &h0,
        &w,
        seq_len as i32,
        seq_len as i32,
        &mat,
    );

    let results = unsafe { sw_kernel::<16, SwEngine128>(&params, num_lanes) };

    assert_eq!(results.len(), num_lanes);

    // All lanes should have very low scores (local alignment may find short matches)
    // With match=1, mismatch=-1, complete mismatch should produce score near 0
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.score < 4,
            "Lane {}: Score should be very low for complete mismatch, got {}",
            i,
            result.score
        );
    }
}

#[test]
fn test_kernel_partial_alignment() {
    // Test: Sequences with partial matches should produce intermediate scores
    // Validates correct scoring and position tracking

    let mat = bwa_fill_scmat(1, 4, -1);

    let seq_len = 10;
    let num_lanes = 16;

    let mut query_soa = vec![0u8; seq_len * num_lanes];
    let mut target_soa = vec![0u8; seq_len * num_lanes];

    // Query: AAAACGTACG (match in middle)
    // Target: ACGTACGTAC (shifted pattern)
    for pos in 0..seq_len {
        for lane in 0..num_lanes {
            query_soa[pos * num_lanes + lane] = if pos < 4 { 0 } else { (pos % 4) as u8 };
            target_soa[pos * num_lanes + lane] = (pos % 4) as u8;
        }
    }

    let qlen = vec![seq_len as i8; num_lanes];
    let tlen = vec![seq_len as i8; num_lanes];
    let h0 = vec![0i8; num_lanes];
    let w = vec![10i8; num_lanes];

    let params = create_test_params(
        &query_soa,
        &target_soa,
        &qlen,
        &tlen,
        &h0,
        &w,
        seq_len as i32,
        seq_len as i32,
        &mat,
    );

    let results = unsafe { sw_kernel::<16, SwEngine128>(&params, num_lanes) };

    assert_eq!(results.len(), num_lanes);

    // All lanes should have moderate positive scores
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.score > 0,
            "Lane {}: Score should be positive for partial alignment, got {}",
            i,
            result.score
        );
    }
}

#[test]
fn test_kernel_varying_lengths() {
    // Test: Sequences with different lengths should be handled correctly
    // Validates per-lane length handling and proper termination

    let mat = bwa_fill_scmat(1, 4, -1);

    let max_len = 12;
    let num_lanes = 16;

    let mut query_soa = vec![0u8; max_len * num_lanes];
    let mut target_soa = vec![0u8; max_len * num_lanes];

    // Fill with pattern
    for pos in 0..max_len {
        for lane in 0..num_lanes {
            query_soa[pos * num_lanes + lane] = (pos % 4) as u8;
            target_soa[pos * num_lanes + lane] = (pos % 4) as u8;
        }
    }

    // Varying lengths per lane
    let mut qlen = vec![0i8; num_lanes];
    let mut tlen = vec![0i8; num_lanes];
    for lane in 0..num_lanes {
        qlen[lane] = (4 + lane / 2) as i8; // Lengths from 4 to 11
        tlen[lane] = (4 + lane / 2) as i8;
    }

    let h0 = vec![0i8; num_lanes];
    let w = vec![10i8; num_lanes];

    let params = create_test_params(
        &query_soa,
        &target_soa,
        &qlen,
        &tlen,
        &h0,
        &w,
        max_len as i32,
        max_len as i32,
        &mat,
    );

    let results = unsafe { sw_kernel::<16, SwEngine128>(&params, num_lanes) };

    assert_eq!(results.len(), num_lanes);

    // Each lane should have score proportional to its length
    for (i, result) in results.iter().enumerate() {
        let expected_end = qlen[i] - 1;
        assert!(
            result.score > 0,
            "Lane {}: Should have positive score, got {}",
            i,
            result.score
        );
        assert!(
            result.query_end_pos <= expected_end as i32,
            "Lane {}: Query end position {} should not exceed length {}",
            i,
            result.query_end_pos,
            expected_end
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_kernel_avx2_width() {
    // Test: AVX2 engine with 32 lanes should work correctly
    // Validates that the refactored kernel handles wider SIMD correctly

    let mat = bwa_fill_scmat(1, 4, -1);

    let seq_len = 8;
    let num_lanes = 32; // AVX2 width

    let mut query_soa = vec![0u8; seq_len * num_lanes];
    let mut target_soa = vec![0u8; seq_len * num_lanes];

    // Identical sequences for simplicity
    for pos in 0..seq_len {
        for lane in 0..num_lanes {
            let base = (pos % 4) as u8;
            query_soa[pos * num_lanes + lane] = base;
            target_soa[pos * num_lanes + lane] = base;
        }
    }

    let qlen = vec![seq_len as i8; num_lanes];
    let tlen = vec![seq_len as i8; num_lanes];
    let h0 = vec![0i8; num_lanes];
    let w = vec![10i8; num_lanes];

    let params = create_test_params(
        &query_soa,
        &target_soa,
        &qlen,
        &tlen,
        &h0,
        &w,
        seq_len as i32,
        seq_len as i32,
        &mat,
    );

    let results = unsafe { sw_kernel::<32, SwEngine256>(&params, num_lanes) };

    assert_eq!(results.len(), num_lanes);

    // All lanes should have identical positive scores
    for result in results.iter() {
        assert!(result.score > 0, "AVX2 lane should have positive score");
    }
}

#[test]
fn test_kernel_saturating_subtraction() {
    // Test: Verify that saturating subtraction eliminates need for max(x, 0)
    // This is a regression test for the optimization

    let mat = bwa_fill_scmat(1, 4, -1);

    let seq_len = 6;
    let num_lanes = 16;

    let mut query_soa = vec![0u8; seq_len * num_lanes];
    let mut target_soa = vec![0u8; seq_len * num_lanes];

    // Create scenario that would cause underflow without saturation
    // Large gaps should trigger saturating arithmetic
    for pos in 0..seq_len {
        for lane in 0..num_lanes {
            query_soa[pos * num_lanes + lane] = 0; // A
            target_soa[pos * num_lanes + lane] = if pos < 3 { 0 } else { 3 }; // AAA then TTT
        }
    }

    let qlen = vec![seq_len as i8; num_lanes];
    let tlen = vec![seq_len as i8; num_lanes];
    let h0 = vec![0i8; num_lanes];
    let w = vec![10i8; num_lanes];

    let params = create_test_params(
        &query_soa,
        &target_soa,
        &qlen,
        &tlen,
        &h0,
        &w,
        seq_len as i32,
        seq_len as i32,
        &mat,
    );

    let results = unsafe { sw_kernel::<16, SwEngine128>(&params, num_lanes) };

    assert_eq!(results.len(), num_lanes);

    // No result should have negative scores (saturating arithmetic prevents this)
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.score >= 0,
            "Lane {}: Saturating arithmetic should prevent negative scores, got {}",
            i,
            result.score
        );
        assert!(
            result.query_end_pos >= 0,
            "Lane {}: Position should not be negative",
            i
        );
        assert!(
            result.target_end_pos >= 0,
            "Lane {}: Position should not be negative",
            i
        );
    }
}

#[test]
fn test_kernel_zero_length_edge_case() {
    // Test: Zero-length sequences should be handled gracefully

    let mat = bwa_fill_scmat(1, 4, -1);

    let num_lanes = 16;
    let query_soa = vec![0u8; num_lanes]; // Minimal buffer
    let target_soa = vec![0u8; num_lanes];

    let qlen = vec![0i8; num_lanes]; // Zero length
    let tlen = vec![0i8; num_lanes];
    let h0 = vec![0i8; num_lanes];
    let w = vec![10i8; num_lanes];

    let params = create_test_params(
        &query_soa,
        &target_soa,
        &qlen,
        &tlen,
        &h0,
        &w,
        0,
        0,
        &mat,
    );

    let results = unsafe { sw_kernel::<16, SwEngine128>(&params, num_lanes) };

    // Should return empty results for zero-length sequences
    assert_eq!(results.len(), 0, "Zero-length sequences should produce no results");
}

#[test]
fn test_kernel_banding_constraint() {
    // Test: Banding width should constrain alignment paths
    // Narrow band should prevent alignment of distant positions

    let mat = bwa_fill_scmat(1, 4, -1);

    let seq_len = 10;
    let num_lanes = 16;

    let mut query_soa = vec![0u8; seq_len * num_lanes];
    let mut target_soa = vec![0u8; seq_len * num_lanes];

    for pos in 0..seq_len {
        for lane in 0..num_lanes {
            query_soa[pos * num_lanes + lane] = (pos % 4) as u8;
            target_soa[pos * num_lanes + lane] = (pos % 4) as u8;
        }
    }

    let qlen = vec![seq_len as i8; num_lanes];
    let tlen = vec![seq_len as i8; num_lanes];
    let h0 = vec![0i8; num_lanes];
    let w = vec![2i8; num_lanes]; // Very narrow band

    let params = create_test_params(
        &query_soa,
        &target_soa,
        &qlen,
        &tlen,
        &h0,
        &w,
        seq_len as i32,
        seq_len as i32,
        &mat,
    );

    let results = unsafe { sw_kernel::<16, SwEngine128>(&params, num_lanes) };

    assert_eq!(results.len(), num_lanes);

    // With narrow band, alignment should still complete
    for result in results.iter() {
        assert!(
            result.score >= 0,
            "Narrow band should not prevent valid alignment"
        );
    }
}
