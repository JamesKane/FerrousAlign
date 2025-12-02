// Unit tests for SoAAlignmentResult::merge_all()
//
// Tests the critical merge logic that combines results from parallel processing.
// REGRESSION TEST for thread-safety bug where read_alignment_boundaries were not
// properly offset-adjusted during merge, causing 66% of reads to be lost in multi-threaded mode.

use ferrous_align::pipelines::linear::batch_extension::types::SoAAlignmentResult;

/// Test merging two simple results with correct boundary offset adjustment
#[test]
fn test_merge_all_basic_two_chunks() {
    // Create first chunk result (reads 0-1, 3 alignments total)
    let mut result1 = SoAAlignmentResult::with_capacity(3, 2);

    // Read 0: 2 alignments at indices 0, 1
    result1.query_names.push("read0".to_string());
    result1.flags.push(0);
    result1.ref_names.push("chr1".to_string());
    result1.ref_ids.push(0);
    result1.positions.push(1000);
    result1.mapqs.push(60);
    result1.scores.push(100);
    result1.rnexts.push("=".to_string());
    result1.pnexts.push(1200);
    result1.tlens.push(300);
    result1.seqs.push(b'A');
    result1.quals.push(b'I');
    result1.seq_boundaries.push((0, 1));
    result1.cigar_ops.push(0);
    result1.cigar_lens.push(50);
    result1.cigar_boundaries.push((0, 1));
    result1.tag_names.push("AS".to_string());
    result1.tag_values.push("100".to_string());
    result1.tag_boundaries.push((0, 1));
    result1.query_starts.push(0);
    result1.query_ends.push(50);
    result1.seed_coverages.push(50);
    result1.hashes.push(12345);
    result1.frac_reps.push(0.0);

    result1.query_names.push("read0".to_string());
    result1.flags.push(256); // secondary
    result1.ref_names.push("chr1".to_string());
    result1.ref_ids.push(0);
    result1.positions.push(2000);
    result1.mapqs.push(0);
    result1.scores.push(90);
    result1.rnexts.push("=".to_string());
    result1.pnexts.push(1200);
    result1.tlens.push(300);
    result1.seqs.push(b'C');
    result1.quals.push(b'I');
    result1.seq_boundaries.push((1, 1));
    result1.cigar_ops.push(0);
    result1.cigar_lens.push(50);
    result1.cigar_boundaries.push((1, 1));
    result1.tag_names.push("AS".to_string());
    result1.tag_values.push("90".to_string());
    result1.tag_boundaries.push((1, 1));
    result1.query_starts.push(0);
    result1.query_ends.push(50);
    result1.seed_coverages.push(50);
    result1.hashes.push(12346);
    result1.frac_reps.push(0.0);

    // Read 1: 1 alignment at index 2
    result1.query_names.push("read1".to_string());
    result1.flags.push(0);
    result1.ref_names.push("chr1".to_string());
    result1.ref_ids.push(0);
    result1.positions.push(3000);
    result1.mapqs.push(60);
    result1.scores.push(95);
    result1.rnexts.push("=".to_string());
    result1.pnexts.push(3200);
    result1.tlens.push(300);
    result1.seqs.push(b'G');
    result1.quals.push(b'I');
    result1.seq_boundaries.push((2, 1));
    result1.cigar_ops.push(0);
    result1.cigar_lens.push(50);
    result1.cigar_boundaries.push((2, 1));
    result1.tag_names.push("AS".to_string());
    result1.tag_values.push("95".to_string());
    result1.tag_boundaries.push((2, 1));
    result1.query_starts.push(0);
    result1.query_ends.push(50);
    result1.seed_coverages.push(50);
    result1.hashes.push(12347);
    result1.frac_reps.push(0.0);

    // Set read boundaries for result1
    result1.read_alignment_boundaries.push((0, 2)); // read 0: alignments 0-1
    result1.read_alignment_boundaries.push((2, 1)); // read 1: alignment 2

    // Create second chunk result (reads 2-3, 2 alignments total)
    let mut result2 = SoAAlignmentResult::with_capacity(2, 2);

    // Read 2: 1 alignment at index 0 (within this chunk)
    result2.query_names.push("read2".to_string());
    result2.flags.push(0);
    result2.ref_names.push("chr2".to_string());
    result2.ref_ids.push(1);
    result2.positions.push(4000);
    result2.mapqs.push(60);
    result2.scores.push(105);
    result2.rnexts.push("=".to_string());
    result2.pnexts.push(4200);
    result2.tlens.push(300);
    result2.seqs.push(b'T');
    result2.quals.push(b'I');
    result2.seq_boundaries.push((0, 1));
    result2.cigar_ops.push(0);
    result2.cigar_lens.push(50);
    result2.cigar_boundaries.push((0, 1));
    result2.tag_names.push("AS".to_string());
    result2.tag_values.push("105".to_string());
    result2.tag_boundaries.push((0, 1));
    result2.query_starts.push(0);
    result2.query_ends.push(50);
    result2.seed_coverages.push(50);
    result2.hashes.push(12348);
    result2.frac_reps.push(0.0);

    // Read 3: 1 alignment at index 1 (within this chunk)
    result2.query_names.push("read3".to_string());
    result2.flags.push(0);
    result2.ref_names.push("chr2".to_string());
    result2.ref_ids.push(1);
    result2.positions.push(5000);
    result2.mapqs.push(60);
    result2.scores.push(110);
    result2.rnexts.push("=".to_string());
    result2.pnexts.push(5200);
    result2.tlens.push(300);
    result2.seqs.push(b'A');
    result2.quals.push(b'I');
    result2.seq_boundaries.push((1, 1));
    result2.cigar_ops.push(0);
    result2.cigar_lens.push(50);
    result2.cigar_boundaries.push((1, 1));
    result2.tag_names.push("AS".to_string());
    result2.tag_values.push("110".to_string());
    result2.tag_boundaries.push((1, 1));
    result2.query_starts.push(0);
    result2.query_ends.push(50);
    result2.seed_coverages.push(50);
    result2.hashes.push(12349);
    result2.frac_reps.push(0.0);

    // Set read boundaries for result2 (IMPORTANT: indices are relative to this chunk)
    result2.read_alignment_boundaries.push((0, 1)); // read 2: alignment 0 (in this chunk)
    result2.read_alignment_boundaries.push((1, 1)); // read 3: alignment 1 (in this chunk)

    // Merge results
    let merged = SoAAlignmentResult::merge_all(vec![result1, result2]);

    // Verify total counts
    assert_eq!(merged.num_reads(), 4, "Should have 4 reads total");
    assert_eq!(merged.len(), 5, "Should have 5 alignments total");

    // CRITICAL TEST: Verify read_alignment_boundaries are correctly offset-adjusted
    assert_eq!(merged.read_alignment_boundaries.len(), 4);

    // Read 0 boundaries should be unchanged (0, 2)
    assert_eq!(
        merged.read_alignment_boundaries[0],
        (0, 2),
        "Read 0 should have 2 alignments starting at index 0"
    );

    // Read 1 boundaries should be unchanged (2, 1)
    assert_eq!(
        merged.read_alignment_boundaries[1],
        (2, 1),
        "Read 1 should have 1 alignment at index 2"
    );

    // Read 2 boundaries should be offset-adjusted: (0, 1) -> (3, 1)
    assert_eq!(
        merged.read_alignment_boundaries[2],
        (3, 1),
        "Read 2 should have 1 alignment at index 3 (was 0 in chunk, +3 offset)"
    );

    // Read 3 boundaries should be offset-adjusted: (1, 1) -> (4, 1)
    assert_eq!(
        merged.read_alignment_boundaries[3],
        (4, 1),
        "Read 3 should have 1 alignment at index 4 (was 1 in chunk, +3 offset)"
    );

    // Verify alignment data is accessible via boundaries
    let (start, count) = merged.read_alignment_boundaries[2];
    assert_eq!(merged.query_names[start], "read2");
    assert_eq!(count, 1);

    let (start, count) = merged.read_alignment_boundaries[3];
    assert_eq!(merged.query_names[start], "read3");
    assert_eq!(count, 1);
}

/// Test merging with empty results
#[test]
fn test_merge_all_with_empty_chunks() {
    let result1 = SoAAlignmentResult::new();
    let mut result2 = SoAAlignmentResult::with_capacity(1, 1);

    result2.query_names.push("read0".to_string());
    result2.flags.push(0);
    result2.ref_names.push("chr1".to_string());
    result2.ref_ids.push(0);
    result2.positions.push(1000);
    result2.mapqs.push(60);
    result2.scores.push(100);
    result2.rnexts.push("=".to_string());
    result2.pnexts.push(1200);
    result2.tlens.push(300);
    result2.seqs.push(b'A');
    result2.quals.push(b'I');
    result2.seq_boundaries.push((0, 1));
    result2.cigar_ops.push(0);
    result2.cigar_lens.push(50);
    result2.cigar_boundaries.push((0, 1));
    result2.tag_names.push("AS".to_string());
    result2.tag_values.push("100".to_string());
    result2.tag_boundaries.push((0, 1));
    result2.query_starts.push(0);
    result2.query_ends.push(50);
    result2.seed_coverages.push(50);
    result2.hashes.push(12345);
    result2.frac_reps.push(0.0);
    result2.read_alignment_boundaries.push((0, 1));

    let merged = SoAAlignmentResult::merge_all(vec![result1, result2]);

    assert_eq!(merged.num_reads(), 1);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged.read_alignment_boundaries[0], (0, 1));
}

/// Test merging three chunks to verify cumulative offset adjustment
#[test]
fn test_merge_all_three_chunks() {
    // Chunk 1: 2 alignments
    let mut result1 = SoAAlignmentResult::with_capacity(2, 1);
    for i in 0..2 {
        result1.query_names.push(format!("read0"));
        result1.flags.push(if i == 0 { 0 } else { 256 });
        result1.ref_names.push("chr1".to_string());
        result1.ref_ids.push(0);
        result1.positions.push(1000 + i as u64 * 100);
        result1.mapqs.push(60);
        result1.scores.push(100);
        result1.rnexts.push("=".to_string());
        result1.pnexts.push(1200);
        result1.tlens.push(300);
        result1.seqs.push(b'A');
        result1.quals.push(b'I');
        result1.seq_boundaries.push((i, 1));
        result1.cigar_ops.push(0);
        result1.cigar_lens.push(50);
        result1.cigar_boundaries.push((i, 1));
        result1.tag_names.push("AS".to_string());
        result1.tag_values.push("100".to_string());
        result1.tag_boundaries.push((i, 1));
        result1.query_starts.push(0);
        result1.query_ends.push(50);
        result1.seed_coverages.push(50);
        result1.hashes.push(12345);
        result1.frac_reps.push(0.0);
    }
    result1.read_alignment_boundaries.push((0, 2));

    // Chunk 2: 3 alignments
    let mut result2 = SoAAlignmentResult::with_capacity(3, 1);
    for i in 0..3 {
        result2.query_names.push(format!("read1"));
        result2.flags.push(if i == 0 { 0 } else { 256 });
        result2.ref_names.push("chr1".to_string());
        result2.ref_ids.push(0);
        result2.positions.push(2000 + i as u64 * 100);
        result2.mapqs.push(60);
        result2.scores.push(100);
        result2.rnexts.push("=".to_string());
        result2.pnexts.push(2200);
        result2.tlens.push(300);
        result2.seqs.push(b'C');
        result2.quals.push(b'I');
        result2.seq_boundaries.push((i, 1));
        result2.cigar_ops.push(0);
        result2.cigar_lens.push(50);
        result2.cigar_boundaries.push((i, 1));
        result2.tag_names.push("AS".to_string());
        result2.tag_values.push("100".to_string());
        result2.tag_boundaries.push((i, 1));
        result2.query_starts.push(0);
        result2.query_ends.push(50);
        result2.seed_coverages.push(50);
        result2.hashes.push(12346);
        result2.frac_reps.push(0.0);
    }
    result2.read_alignment_boundaries.push((0, 3)); // Starts at 0 in this chunk

    // Chunk 3: 1 alignment
    let mut result3 = SoAAlignmentResult::with_capacity(1, 1);
    result3.query_names.push(format!("read2"));
    result3.flags.push(0);
    result3.ref_names.push("chr1".to_string());
    result3.ref_ids.push(0);
    result3.positions.push(3000);
    result3.mapqs.push(60);
    result3.scores.push(100);
    result3.rnexts.push("=".to_string());
    result3.pnexts.push(3200);
    result3.tlens.push(300);
    result3.seqs.push(b'G');
    result3.quals.push(b'I');
    result3.seq_boundaries.push((0, 1));
    result3.cigar_ops.push(0);
    result3.cigar_lens.push(50);
    result3.cigar_boundaries.push((0, 1));
    result3.tag_names.push("AS".to_string());
    result3.tag_values.push("100".to_string());
    result3.tag_boundaries.push((0, 1));
    result3.query_starts.push(0);
    result3.query_ends.push(50);
    result3.seed_coverages.push(50);
    result3.hashes.push(12347);
    result3.frac_reps.push(0.0);
    result3.read_alignment_boundaries.push((0, 1)); // Starts at 0 in this chunk

    let merged = SoAAlignmentResult::merge_all(vec![result1, result2, result3]);

    assert_eq!(merged.num_reads(), 3);
    assert_eq!(merged.len(), 6); // 2 + 3 + 1

    // Verify cumulative offset adjustment
    assert_eq!(
        merged.read_alignment_boundaries[0],
        (0, 2),
        "Read 0: alignments 0-1"
    );
    assert_eq!(
        merged.read_alignment_boundaries[1],
        (2, 3),
        "Read 1: alignments 2-4 (was 0-2, +2 offset)"
    );
    assert_eq!(
        merged.read_alignment_boundaries[2],
        (5, 1),
        "Read 2: alignment 5 (was 0, +5 offset)"
    );
}

/// Test that reads with zero alignments are handled correctly
#[test]
fn test_merge_all_with_zero_alignment_reads() {
    let mut result1 = SoAAlignmentResult::with_capacity(1, 2);

    // Read 0: 1 alignment
    result1.query_names.push("read0".to_string());
    result1.flags.push(0);
    result1.ref_names.push("chr1".to_string());
    result1.ref_ids.push(0);
    result1.positions.push(1000);
    result1.mapqs.push(60);
    result1.scores.push(100);
    result1.rnexts.push("=".to_string());
    result1.pnexts.push(1200);
    result1.tlens.push(300);
    result1.seqs.push(b'A');
    result1.quals.push(b'I');
    result1.seq_boundaries.push((0, 1));
    result1.cigar_ops.push(0);
    result1.cigar_lens.push(50);
    result1.cigar_boundaries.push((0, 1));
    result1.tag_names.push("AS".to_string());
    result1.tag_values.push("100".to_string());
    result1.tag_boundaries.push((0, 1));
    result1.query_starts.push(0);
    result1.query_ends.push(50);
    result1.seed_coverages.push(50);
    result1.hashes.push(12345);
    result1.frac_reps.push(0.0);

    result1.read_alignment_boundaries.push((0, 1)); // Read 0: 1 alignment
    result1.read_alignment_boundaries.push((1, 0)); // Read 1: 0 alignments (unmapped)

    let mut result2 = SoAAlignmentResult::with_capacity(1, 1);

    // Read 2: 1 alignment
    result2.query_names.push("read2".to_string());
    result2.flags.push(0);
    result2.ref_names.push("chr1".to_string());
    result2.ref_ids.push(0);
    result2.positions.push(2000);
    result2.mapqs.push(60);
    result2.scores.push(100);
    result2.rnexts.push("=".to_string());
    result2.pnexts.push(2200);
    result2.tlens.push(300);
    result2.seqs.push(b'C');
    result2.quals.push(b'I');
    result2.seq_boundaries.push((0, 1));
    result2.cigar_ops.push(0);
    result2.cigar_lens.push(50);
    result2.cigar_boundaries.push((0, 1));
    result2.tag_names.push("AS".to_string());
    result2.tag_values.push("100".to_string());
    result2.tag_boundaries.push((0, 1));
    result2.query_starts.push(0);
    result2.query_ends.push(50);
    result2.seed_coverages.push(50);
    result2.hashes.push(12346);
    result2.frac_reps.push(0.0);

    result2.read_alignment_boundaries.push((0, 1));

    let merged = SoAAlignmentResult::merge_all(vec![result1, result2]);

    assert_eq!(merged.num_reads(), 3);
    assert_eq!(merged.len(), 2); // Only 2 alignments total

    assert_eq!(
        merged.read_alignment_boundaries[0],
        (0, 1),
        "Read 0: 1 alignment"
    );
    assert_eq!(
        merged.read_alignment_boundaries[1],
        (1, 0),
        "Read 1: 0 alignments (unmapped)"
    );
    assert_eq!(
        merged.read_alignment_boundaries[2],
        (1, 1),
        "Read 2: 1 alignment at index 1 (was 0, +1 offset)"
    );
}

/// Regression test for the exact bug that caused 66% mapping loss
#[test]
fn test_merge_all_regression_thread_safety_bug() {
    // Simulate the exact scenario: multi-threaded processing with 2 parallel chunks
    // Without the fix, chunk 2's read boundaries would reference wrong indices

    let mut chunk1 = SoAAlignmentResult::with_capacity(150, 100);
    for read_idx in 0..100 {
        // Give each read 1-2 alignments
        let num_aligns = if read_idx % 3 == 0 { 2 } else { 1 };
        let start_idx = chunk1.query_names.len();

        for _ in 0..num_aligns {
            chunk1.query_names.push(format!("read{}", read_idx));
            chunk1.flags.push(0);
            chunk1.ref_names.push("chr1".to_string());
            chunk1.ref_ids.push(0);
            chunk1.positions.push(1000);
            chunk1.mapqs.push(60);
            chunk1.scores.push(100);
            chunk1.rnexts.push("=".to_string());
            chunk1.pnexts.push(1200);
            chunk1.tlens.push(300);
            chunk1.seqs.push(b'A');
            chunk1.quals.push(b'I');
            chunk1.seq_boundaries.push((chunk1.seqs.len() - 1, 1));
            chunk1.cigar_ops.push(0);
            chunk1.cigar_lens.push(50);
            chunk1
                .cigar_boundaries
                .push((chunk1.cigar_ops.len() - 1, 1));
            chunk1.tag_names.push("AS".to_string());
            chunk1.tag_values.push("100".to_string());
            chunk1.tag_boundaries.push((chunk1.tag_names.len() - 1, 1));
            chunk1.query_starts.push(0);
            chunk1.query_ends.push(50);
            chunk1.seed_coverages.push(50);
            chunk1.hashes.push(12345);
            chunk1.frac_reps.push(0.0);
        }

        chunk1
            .read_alignment_boundaries
            .push((start_idx, num_aligns));
    }

    let chunk1_total_alignments = chunk1.len();

    let mut chunk2 = SoAAlignmentResult::with_capacity(140, 100);
    for read_idx in 100..200 {
        let num_aligns = if read_idx % 3 == 0 { 2 } else { 1 };
        let start_idx = chunk2.query_names.len(); // Relative to this chunk!

        for _ in 0..num_aligns {
            chunk2.query_names.push(format!("read{}", read_idx));
            chunk2.flags.push(0);
            chunk2.ref_names.push("chr1".to_string());
            chunk2.ref_ids.push(0);
            chunk2.positions.push(2000);
            chunk2.mapqs.push(60);
            chunk2.scores.push(100);
            chunk2.rnexts.push("=".to_string());
            chunk2.pnexts.push(2200);
            chunk2.tlens.push(300);
            chunk2.seqs.push(b'C');
            chunk2.quals.push(b'I');
            chunk2.seq_boundaries.push((chunk2.seqs.len() - 1, 1));
            chunk2.cigar_ops.push(0);
            chunk2.cigar_lens.push(50);
            chunk2
                .cigar_boundaries
                .push((chunk2.cigar_ops.len() - 1, 1));
            chunk2.tag_names.push("AS".to_string());
            chunk2.tag_values.push("100".to_string());
            chunk2.tag_boundaries.push((chunk2.tag_names.len() - 1, 1));
            chunk2.query_starts.push(0);
            chunk2.query_ends.push(50);
            chunk2.seed_coverages.push(50);
            chunk2.hashes.push(12346);
            chunk2.frac_reps.push(0.0);
        }

        chunk2
            .read_alignment_boundaries
            .push((start_idx, num_aligns));
    }

    let merged = SoAAlignmentResult::merge_all(vec![chunk1, chunk2]);

    assert_eq!(merged.num_reads(), 200);

    // CRITICAL: Verify that chunk2's read boundaries were offset-adjusted
    for read_idx in 100..200 {
        let (start, count) = merged.read_alignment_boundaries[read_idx];

        // All indices should be >= chunk1_total_alignments
        assert!(
            start >= chunk1_total_alignments,
            "Read {} boundary start {} should be >= {} (chunk1 size)",
            read_idx,
            start,
            chunk1_total_alignments
        );

        // Verify we can access the alignment data
        assert!(
            start + count <= merged.len(),
            "Read {} boundary [{}, {}) exceeds merged alignment count {}",
            read_idx,
            start,
            start + count,
            merged.len()
        );

        // Verify the alignment actually belongs to this read
        if count > 0 {
            let query_name = &merged.query_names[start];
            assert_eq!(
                query_name,
                &format!("read{}", read_idx),
                "Read {} alignment at index {} has wrong query name: {}",
                read_idx,
                start,
                query_name
            );
        }
    }
}
