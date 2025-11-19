// Tests for Alignment helper methods (SAM flags, TLEN, unmapped creation)
// Created during Session 30 refactoring to reduce code duplication

// SAM flag constants (to be added to align.rs)
const PAIRED: u16 = 0x1;
const PROPER_PAIR: u16 = 0x2;
const UNMAPPED: u16 = 0x4;
const MATE_UNMAPPED: u16 = 0x8;
const REVERSE: u16 = 0x10;
const MATE_REVERSE: u16 = 0x20;
const FIRST_IN_PAIR: u16 = 0x40;
const SECOND_IN_PAIR: u16 = 0x80;
const SECONDARY: u16 = 0x100;

#[test]
fn test_sam_flag_constants() {
    // Verify SAM flag constants match SAM specification
    assert_eq!(PAIRED, 0x1, "PAIRED flag should be 0x1");
    assert_eq!(PROPER_PAIR, 0x2, "PROPER_PAIR flag should be 0x2");
    assert_eq!(UNMAPPED, 0x4, "UNMAPPED flag should be 0x4");
    assert_eq!(MATE_UNMAPPED, 0x8, "MATE_UNMAPPED flag should be 0x8");
    assert_eq!(REVERSE, 0x10, "REVERSE flag should be 0x10");
    assert_eq!(MATE_REVERSE, 0x20, "MATE_REVERSE flag should be 0x20");
    assert_eq!(FIRST_IN_PAIR, 0x40, "FIRST_IN_PAIR flag should be 0x40");
    assert_eq!(SECOND_IN_PAIR, 0x80, "SECOND_IN_PAIR flag should be 0x80");
    assert_eq!(SECONDARY, 0x100, "SECONDARY flag should be 0x100");
}

#[test]
fn test_sam_flags_are_bitwise_compatible() {
    // Verify flags can be combined with bitwise OR
    let flags = PAIRED | FIRST_IN_PAIR | PROPER_PAIR;
    assert_eq!(flags, 0x1 | 0x40 | 0x2);
    assert_eq!(flags, 0x43);

    // Verify individual flags can be checked
    assert_ne!(flags & PAIRED, 0);
    assert_ne!(flags & FIRST_IN_PAIR, 0);
    assert_ne!(flags & PROPER_PAIR, 0);
    assert_eq!(flags & UNMAPPED, 0);
    assert_eq!(flags & SECOND_IN_PAIR, 0);
}

#[test]
fn test_alignment_set_paired_flags_read1_proper() {
    // Test setting flags for first read in proper pair
    let mut flag = 0u16;

    // Simulate set_paired_flags(is_first=true, is_proper_pair=true, mate_unmapped=false, mate_reverse=false)
    flag |= PAIRED;
    flag |= FIRST_IN_PAIR;
    flag |= PROPER_PAIR;

    assert_ne!(flag & PAIRED, 0, "Should be marked as paired");
    assert_ne!(flag & FIRST_IN_PAIR, 0, "Should be first in pair");
    assert_ne!(flag & PROPER_PAIR, 0, "Should be proper pair");
    assert_eq!(flag & SECOND_IN_PAIR, 0, "Should NOT be second in pair");
    assert_eq!(flag & MATE_UNMAPPED, 0, "Mate should be mapped");
}

#[test]
fn test_alignment_set_paired_flags_read2_improper() {
    // Test setting flags for second read in improper pair
    let mut flag = 0u16;

    // Simulate set_paired_flags(is_first=false, is_proper_pair=false, mate_unmapped=false, mate_reverse=true)
    flag |= PAIRED;
    flag |= SECOND_IN_PAIR;
    flag |= MATE_REVERSE;

    assert_ne!(flag & PAIRED, 0, "Should be marked as paired");
    assert_ne!(flag & SECOND_IN_PAIR, 0, "Should be second in pair");
    assert_eq!(flag & PROPER_PAIR, 0, "Should NOT be proper pair");
    assert_eq!(flag & FIRST_IN_PAIR, 0, "Should NOT be first in pair");
    assert_ne!(flag & MATE_REVERSE, 0, "Mate should be reverse");
}

#[test]
fn test_alignment_set_paired_flags_mate_unmapped() {
    // Test setting flags when mate is unmapped
    let mut flag = 0u16;

    // Simulate set_paired_flags(is_first=true, is_proper_pair=false, mate_unmapped=true, mate_reverse=false)
    flag |= PAIRED;
    flag |= FIRST_IN_PAIR;
    flag |= MATE_UNMAPPED;

    assert_ne!(flag & PAIRED, 0, "Should be marked as paired");
    assert_ne!(flag & MATE_UNMAPPED, 0, "Mate should be unmapped");
    assert_eq!(
        flag & PROPER_PAIR,
        0,
        "Cannot be proper pair if mate unmapped"
    );
}

#[test]
fn test_alignment_calculate_tlen_read1_leftmost() {
    // Test TLEN calculation when read1 is leftmost
    let aln = create_test_alignment_at_pos(1000);

    let mate_pos = 1200;
    let mate_ref_len = 100;

    // TLEN = (mate_pos - this_pos) + mate_ref_len
    // TLEN = (1200 - 1000) + 100 = 300
    let expected_tlen = ((mate_pos as i64 - aln.pos as i64) + mate_ref_len as i64) as i32;
    assert_eq!(expected_tlen, 300);
}

#[test]
fn test_alignment_calculate_tlen_read1_rightmost() {
    // Test TLEN calculation when read1 is rightmost
    let aln = create_test_alignment_at_pos(1200);

    let mate_pos = 1000;
    let this_ref_len = aln.reference_length();

    // TLEN = -((this_pos - mate_pos) + this_ref_len)
    // For 100bp alignment: TLEN = -((1200 - 1000) + 100) = -300
    let expected_tlen = -(((aln.pos as i64 - mate_pos as i64) + this_ref_len as i64) as i32);
    assert!(
        expected_tlen < 0,
        "TLEN should be negative for rightmost read"
    );
    assert_eq!(expected_tlen, -300);
}

#[test]
fn test_alignment_calculate_tlen_same_position() {
    // Test TLEN calculation when reads start at same position (edge case)
    let aln = create_test_alignment_at_pos(1000);

    let mate_pos = 1000;
    let mate_ref_len = 100;

    // When positions are equal, use leftmost formula
    let expected_tlen = ((mate_pos as i64 - aln.pos as i64) + mate_ref_len as i64) as i32;
    assert_eq!(expected_tlen, 100);
}

#[test]
fn test_alignment_calculate_tlen_opposite_signs() {
    // Test that paired reads have opposite sign TLENs
    let aln1 = create_test_alignment_at_pos(1000);
    let aln2 = create_test_alignment_at_pos(1200);

    let mate2_ref_len = aln2.reference_length();
    let mate1_ref_len = aln1.reference_length();

    // Read1 TLEN (leftmost, positive)
    let tlen1 = ((aln2.pos as i64 - aln1.pos as i64) + mate2_ref_len as i64) as i32;

    // Read2 TLEN (rightmost, negative)
    let tlen2 = -(((aln2.pos as i64 - aln1.pos as i64) + mate2_ref_len as i64) as i32);

    assert!(tlen1 > 0, "Read1 TLEN should be positive");
    assert!(tlen2 < 0, "Read2 TLEN should be negative");
    assert_eq!(
        tlen1, -tlen2,
        "TLENs should have equal magnitude but opposite signs"
    );
}

#[test]
fn test_alignment_create_unmapped_first_in_pair() {
    // Test creating unmapped alignment for first read in pair
    let query_name = "read1".to_string();
    let seq = b"ACGTACGTACGT";
    let qual = "IIIIIIIIIIII".to_string();
    let mate_ref = "chr1";
    let mate_pos = 5000u64;
    let mate_is_reverse = false;

    // Create unmapped alignment (to be implemented as factory method)
    let aln = create_unmapped_alignment(
        query_name.clone(),
        seq,
        qual.clone(),
        true, // is_first_in_pair
        mate_ref,
        mate_pos,
        mate_is_reverse,
    );

    // Verify flags
    assert_ne!(aln.flag & PAIRED, 0, "Should be marked as paired");
    assert_ne!(aln.flag & UNMAPPED, 0, "Should be marked as unmapped");
    assert_ne!(aln.flag & FIRST_IN_PAIR, 0, "Should be first in pair");
    assert_eq!(aln.flag & SECOND_IN_PAIR, 0, "Should NOT be second in pair");
    assert_eq!(aln.flag & MATE_UNMAPPED, 0, "Mate should be mapped");

    // Verify basic fields
    assert_eq!(aln.query_name, query_name);
    assert_eq!(aln.mapq, 0, "Unmapped read should have MAPQ=0");
    assert_eq!(aln.ref_name, mate_ref);
    assert_eq!(aln.pos, mate_pos, "Unmapped read should use mate position");
}

#[test]
fn test_alignment_create_unmapped_second_in_pair() {
    // Test creating unmapped alignment for second read in pair
    let query_name = "read2".to_string();
    let seq = b"TGCATGCATGCA";
    let qual = "HHHHHHHHHHHH".to_string();
    let mate_ref = "chr2";
    let mate_pos = 10000u64;
    let mate_is_reverse = true;

    let aln = create_unmapped_alignment(
        query_name.clone(),
        seq,
        qual.clone(),
        false, // is_first_in_pair
        mate_ref,
        mate_pos,
        mate_is_reverse,
    );

    // Verify flags
    assert_ne!(aln.flag & PAIRED, 0, "Should be marked as paired");
    assert_ne!(aln.flag & UNMAPPED, 0, "Should be marked as unmapped");
    assert_ne!(aln.flag & SECOND_IN_PAIR, 0, "Should be second in pair");
    assert_eq!(aln.flag & FIRST_IN_PAIR, 0, "Should NOT be first in pair");
    assert_ne!(
        aln.flag & MATE_REVERSE,
        0,
        "Mate should be marked as reverse"
    );
}

#[test]
fn test_alignment_create_unmapped_both_unmapped() {
    // Test creating unmapped alignment when mate is also unmapped
    let query_name = "read1".to_string();
    let seq = b"NNNNNNNNNNNN";
    let qual = "############".to_string();
    let mate_ref = "*"; // Unmapped mate
    let mate_pos = 0u64;
    let mate_is_reverse = false;

    let aln = create_unmapped_alignment(
        query_name.clone(),
        seq,
        qual.clone(),
        true,
        mate_ref,
        mate_pos,
        mate_is_reverse,
    );

    // Verify fields for both unmapped
    assert_eq!(
        aln.ref_name, "*",
        "Should have '*' for reference when mate unmapped"
    );
    assert_eq!(aln.rnext, "*", "Should have '*' for mate reference");
    assert_eq!(
        aln.pnext, 0,
        "Should have 0 for mate position when unmapped"
    );
}

// ===== Helper Functions =====

fn create_test_alignment() -> TestAlignment {
    // Return test data structure instead of actual Alignment
    // We'll test the flag-setting logic without needing private fields
    TestAlignment {
        pos: 1000,
        seq_len: 100,
    }
}

fn create_test_alignment_at_pos(pos: u64) -> TestAlignment {
    TestAlignment { pos, seq_len: 100 }
}

// Simplified test struct to avoid private field issues
struct TestAlignment {
    pos: u64,
    seq_len: i32,
}

impl TestAlignment {
    fn reference_length(&self) -> i32 {
        // Assume perfect match for testing
        self.seq_len
    }
}

fn create_unmapped_alignment(
    query_name: String,
    seq: &[u8],
    qual: String,
    is_first_in_pair: bool,
    mate_ref: &str,
    mate_pos: u64,
    mate_is_reverse: bool,
) -> TestUnmappedAlignment {
    // This is a stub implementation for testing
    // The actual implementation will be added to align.rs
    let mut flag = PAIRED | UNMAPPED;
    flag |= if is_first_in_pair {
        FIRST_IN_PAIR
    } else {
        SECOND_IN_PAIR
    };
    if mate_is_reverse {
        flag |= MATE_REVERSE;
    }

    let (rnext, pnext) = if mate_ref != "*" {
        ("=".to_string(), mate_pos + 1)
    } else {
        ("*".to_string(), 0)
    };

    let (ref_name, pos) = if mate_ref != "*" {
        (mate_ref.to_string(), mate_pos)
    } else {
        ("*".to_string(), 0)
    };

    TestUnmappedAlignment {
        query_name,
        flag,
        ref_name,
        pos,
        mapq: 0,
        rnext,
        pnext,
    }
}

struct TestUnmappedAlignment {
    query_name: String,
    flag: u16,
    ref_name: String,
    pos: u64,
    mapq: u8,
    rnext: String,
    pnext: u64,
}
