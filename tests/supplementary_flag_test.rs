/// Integration test for supplementary alignment flag handling
///
/// This test verifies that the pipeline correctly marks supplementary alignments
/// and does not create flag conflicts between SECONDARY (256) and SUPPLEMENTARY (2048).
///
/// Background: Supplementary alignments represent non-overlapping alternative alignments
/// for chimeric/split reads. They should have flag 2048 set and flag 256 clear.
/// The pairing logic must not overwrite SUPPLEMENTARY flags with SECONDARY flags.
///
/// TODO: Long-term integration test strategy
/// - Create portable test data fixtures in tests/data/
/// - Use small synthetic references and reads
/// - Avoid hard-coded paths to user-specific data
/// - Consider using golden file comparison for regression testing
use std::path::Path;
use std::process::Command;

#[test]
fn test_supplementary_flags_no_conflict() {
    // Skip if test data is missing
    let ref_path =
        "/home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna";
    let r1_path = "/tmp/test_100_R1.fq";
    let r2_path = "/tmp/test_100_R2.fq";

    if !Path::new(ref_path).exists() || !Path::new(r1_path).exists() || !Path::new(r2_path).exists()
    {
        eprintln!("Skipping test: required test data not found");
        return;
    }

    // Run ferrous-align
    let output = Command::new("./target/release/ferrous-align")
        .args(&["mem", "-t", "1", ref_path, r1_path, r2_path])
        .output()
        .expect("Failed to run ferrous-align");

    assert!(
        output.status.success(),
        "ferrous-align failed: {:?}",
        output.status
    );

    let sam_output = String::from_utf8_lossy(&output.stdout);

    // Parse SAM output and check flags
    let mut supplementary_count = 0;
    let mut secondary_count = 0;
    let mut conflict_count = 0;

    for line in sam_output.lines() {
        if line.starts_with('@') {
            continue; // Skip header
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 {
            continue;
        }

        let flag: u16 = fields[1].parse().unwrap_or(0);

        const SECONDARY: u16 = 256;
        const SUPPLEMENTARY: u16 = 2048;

        let is_secondary = (flag & SECONDARY) != 0;
        let is_supplementary = (flag & SUPPLEMENTARY) != 0;

        if is_supplementary {
            supplementary_count += 1;
        }
        if is_secondary {
            secondary_count += 1;
        }
        if is_secondary && is_supplementary {
            conflict_count += 1;
            eprintln!(
                "Flag conflict detected: flag={} (SECONDARY+SUPPLEMENTARY)",
                flag
            );
        }
    }

    println!("Supplementary alignments: {}", supplementary_count);
    println!("Secondary alignments: {}", secondary_count);
    println!(
        "Flag conflicts (SECONDARY+SUPPLEMENTARY): {}",
        conflict_count
    );

    // Verify no flag conflicts
    assert_eq!(
        conflict_count, 0,
        "Found {} alignments with both SECONDARY and SUPPLEMENTARY flags",
        conflict_count
    );

    // Verify we have at least some supplementary alignments in the test data
    // (100 read pairs from HG002 WGS typically has 1-3 chimeric reads)
    assert!(
        supplementary_count >= 1,
        "Expected at least 1 supplementary alignment, found {}",
        supplementary_count
    );
}

#[test]
fn test_supplementary_flags_in_batch() {
    // Test with larger batch to ensure flag handling is correct across batch boundaries
    let ref_path =
        "/home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna";
    let r1_path = "/tmp/test_1k_R1.fq";
    let r2_path = "/tmp/test_1k_R2.fq";

    if !Path::new(ref_path).exists() || !Path::new(r1_path).exists() || !Path::new(r2_path).exists()
    {
        eprintln!("Skipping test: required test data not found");
        return;
    }

    // Run ferrous-align with multiple threads to test batch processing
    let output = Command::new("./target/release/ferrous-align")
        .args(&["mem", "-t", "4", ref_path, r1_path, r2_path])
        .output()
        .expect("Failed to run ferrous-align");

    assert!(
        output.status.success(),
        "ferrous-align failed: {:?}",
        output.status
    );

    let sam_output = String::from_utf8_lossy(&output.stdout);

    // Check for flag conflicts
    let mut conflict_count = 0;

    for line in sam_output.lines() {
        if line.starts_with('@') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 {
            continue;
        }

        let flag: u16 = fields[1].parse().unwrap_or(0);

        const SECONDARY: u16 = 256;
        const SUPPLEMENTARY: u16 = 2048;

        if (flag & SECONDARY) != 0 && (flag & SUPPLEMENTARY) != 0 {
            conflict_count += 1;
        }
    }

    assert_eq!(
        conflict_count, 0,
        "Found {} flag conflicts in batch processing",
        conflict_count
    );
}

#[test]
fn test_supplementary_flag_structure() {
    // Unit test for flag bit patterns
    const PAIRED: u16 = 1;
    const REVERSE: u16 = 16;
    const SECONDARY: u16 = 256;
    const SUPPLEMENTARY: u16 = 2048;

    // Test that SUPPLEMENTARY and SECONDARY are distinct bits
    assert_eq!(
        SUPPLEMENTARY & SECONDARY,
        0,
        "SUPPLEMENTARY and SECONDARY flags should be distinct bits"
    );

    // Test typical flag combinations
    let primary_reverse = PAIRED | REVERSE;
    assert_eq!(primary_reverse & SUPPLEMENTARY, 0);
    assert_eq!(primary_reverse & SECONDARY, 0);

    let supplementary_reverse = PAIRED | REVERSE | SUPPLEMENTARY;
    assert_ne!(supplementary_reverse & SUPPLEMENTARY, 0);
    assert_eq!(
        supplementary_reverse & SECONDARY,
        0,
        "Supplementary alignment should not have SECONDARY flag"
    );

    let secondary_reverse = PAIRED | REVERSE | SECONDARY;
    assert_eq!(
        secondary_reverse & SUPPLEMENTARY,
        0,
        "Secondary alignment should not have SUPPLEMENTARY flag"
    );
    assert_ne!(secondary_reverse & SECONDARY, 0);
}
