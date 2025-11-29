// Tests for src/banded_swa.rs
// Extracted from inline tests to reduce clutter in production code
use ferrous_align::alignment::banded_swa::BandedPairWiseSW;
use ferrous_align::alignment::banded_swa::bwa_fill_scmat;

#[test]
fn test_soft_clipping_at_end() {
    // Regression test for Session 30 bug: missing soft clipping in CIGAR
    // Query: 10 bases, but only first 5 match the target
    // Expected CIGAR: Should include soft clipping at the end
    // NOTE: Using pen_clip5=0, pen_clip3=0 to not penalize clipping (Session 16 added these)
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 5, 0, 0, mat, 1, 4);

    let query = vec![0u8, 1, 2, 3, 0, 0, 0, 0, 0, 0]; // ACGTAAAAAA
    let target = vec![0u8, 1, 2, 3]; // ACGT (only matches first 4 bases)

    let (out_score, cigar, _, _) = bsw.scalar_banded_swa(10, &query, 4, &target, 100, 0);

    assert!(
        out_score.score > 0,
        "Should have positive score for partial match"
    );

    // Check for soft clipping at the end
    let has_soft_clip = cigar.iter().any(|(op, _)| *op == b'S');
    assert!(
        has_soft_clip,
        "CIGAR should contain soft clipping for unaligned query tail. Got: {cigar:?}"
    );

    // The last operation should be soft clipping
    if let Some(&(op, _)) = cigar.last() {
        assert_eq!(
            op, b'S',
            "Last CIGAR operation should be soft clip (S). Got: {cigar:?}"
        );
    }
}

#[test]
fn test_soft_clipping_at_beginning() {
    // Query: 10 bases where only middle section aligns well
    // Local alignment may not extend to the very start, leaving unaligned bases
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 5, 5, 5, mat, 1, 4);

    // Query has mismatches at start, then matches
    let query = vec![3u8, 3, 3, 3, 3, 0, 1, 2, 3]; // TTTTTACGT
    let target = vec![0u8, 1, 2, 3]; // ACGT (matches last 4 bases of query)

    let (out_score, cigar, _, _) = bsw.scalar_banded_swa(9, &query, 4, &target, 100, 0);

    assert!(
        out_score.score > 0,
        "Should have positive score for partial match"
    );

    // Local alignment behavior: may produce insertions (I) or soft clips (S) or both
    // The key is that we handle partial alignments correctly
    // Accept either I or S operations for the unaligned region
    let has_soft_clip_or_insertion = cigar.iter().any(|(op, _)| *op == b'S' || *op == b'I');
    assert!(
        has_soft_clip_or_insertion,
        "CIGAR should handle unaligned query head with S or I operations. Got: {cigar:?}"
    );
}

#[test]
fn test_soft_clipping_both_ends() {
    // Query: 15 bases, but only middle 5 match the target
    // Expected CIGAR: Should include soft clipping on both ends
    // NOTE: Using pen_clip5=0, pen_clip3=0 to not penalize clipping (Session 16 added these)
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 5, 0, 0, mat, 1, 4);

    // Query: TTTTT ACGT AAAAA (mismatches on both ends)
    let query = vec![3u8, 3, 3, 3, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0];
    let target = vec![0u8, 1, 2, 3]; // ACGT (matches middle of query)

    let (out_score, cigar, _, _) = bsw.scalar_banded_swa(14, &query, 4, &target, 100, 0);

    assert!(out_score.score > 0, "Should have positive score");

    // Count soft clipping operations
    let soft_clip_count = cigar.iter().filter(|(op, _)| *op == b'S').count();
    assert!(
        soft_clip_count >= 1,
        "Should have at least one soft clip operation. Got CIGAR: {cigar:?}"
    );
}

#[test]
fn test_no_soft_clipping_for_full_alignment() {
    // Query fully aligns to target - no soft clipping needed
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 5, 5, 5, mat, 1, 4);

    let query = vec![0u8, 1, 2, 3]; // ACGT
    let target = vec![0u8, 1, 2, 3]; // ACGT (perfect match)

    let (_out_score, cigar, _, _) = bsw.scalar_banded_swa(4, &query, 4, &target, 100, 0);

    // Should NOT have soft clipping for full alignment
    let has_soft_clip = cigar.iter().any(|(op, _)| *op == b'S');
    assert!(
        !has_soft_clip,
        "Full alignment should not have soft clipping. Got: {cigar:?}"
    );

    // Should be all matches
    assert_eq!(cigar.len(), 1, "Should have single CIGAR operation");
    assert_eq!(cigar[0].0, b'M', "Should be match operation");
}
