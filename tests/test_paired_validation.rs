//! Integration tests for paired-end validation
//!
//! These tests verify that FerrousAlign correctly detects and reports
//! mismatched paired-end FASTQ files, preventing silent data corruption.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

/// Helper to create a test FASTQ file with N reads
fn create_test_fastq(path: &PathBuf, num_reads: usize, read_prefix: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    for i in 0..num_reads {
        writeln!(file, "@{}_{}", read_prefix, i)?;
        writeln!(file, "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT")?; // 40bp read
        writeln!(file, "+")?;
        writeln!(file, "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")?; // Q40 quality
    }

    Ok(())
}

/// Helper to get the ferrous-align binary path
fn get_ferrous_binary() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test binary name
    path.pop(); // Remove 'deps' directory

    // In debug builds, binary is in target/debug
    // In release builds, binary is in target/release
    if path.ends_with("debug") || path.ends_with("release") {
        path.push("ferrous-align");
    } else {
        // Fallback: try release build
        path = PathBuf::from("target/release/ferrous-align");
    }

    path
}

/// Test 1: Normal paired-end files with equal read counts (should pass)
#[test]
#[ignore] // Requires reference index to be built
fn test_paired_equal_counts_pass() {
    let temp_dir = TempDir::new().unwrap();
    let r1_path = temp_dir.path().join("R1.fq");
    let r2_path = temp_dir.path().join("R2.fq");
    let out_path = temp_dir.path().join("out.sam");

    // Create matching paired files (100 reads each)
    create_test_fastq(&r1_path, 100, "read").unwrap();
    create_test_fastq(&r2_path, 100, "read").unwrap();

    // Run ferrous-align
    let output = Command::new(get_ferrous_binary())
        .args(&[
            "mem",
            "test_data/test_ref.fa",
            r1_path.to_str().unwrap(),
            r2_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ferrous-align");

    // Should succeed (exit code 0)
    assert!(
        output.status.success(),
        "Expected success with equal read counts. Stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Output file should be created
    assert!(out_path.exists(), "Output SAM file should be created");
}

/// Test 2: R2 has fewer reads than R1 (should fail with clear error)
#[test]
fn test_paired_r2_truncated_fail() {
    let temp_dir = TempDir::new().unwrap();
    let r1_path = temp_dir.path().join("R1.fq");
    let r2_path = temp_dir.path().join("R2_truncated.fq");
    let out_path = temp_dir.path().join("out.sam");

    // Create mismatched files
    create_test_fastq(&r1_path, 1000, "read").unwrap(); // R1: 1000 reads
    create_test_fastq(&r2_path, 998, "read").unwrap(); // R2: 998 reads (2 missing)

    // Build a minimal test reference if it doesn't exist
    let ref_path = temp_dir.path().join("test_ref.fa");
    let mut ref_file = File::create(&ref_path).unwrap();
    writeln!(ref_file, ">chr1").unwrap();
    writeln!(
        ref_file,
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    )
    .unwrap();

    // Try to run ferrous-align
    let output = Command::new(get_ferrous_binary())
        .args(&[
            "mem",
            ref_path.to_str().unwrap(),
            r1_path.to_str().unwrap(),
            r2_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ferrous-align");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should detect the mismatch
    assert!(
        stderr.contains("read count mismatch") || stderr.contains("R1=") && stderr.contains("R2="),
        "Expected error message about read count mismatch. Stderr: {}",
        stderr
    );

    // Should mention 1000 and 998
    assert!(
        stderr.contains("1000") || stderr.contains("998"),
        "Expected specific read counts in error. Stderr: {}",
        stderr
    );
}

/// Test 3: R1 has fewer reads than R2 (should fail with clear error)
#[test]
fn test_paired_r1_truncated_fail() {
    let temp_dir = TempDir::new().unwrap();
    let r1_path = temp_dir.path().join("R1_truncated.fq");
    let r2_path = temp_dir.path().join("R2.fq");
    let out_path = temp_dir.path().join("out.sam");

    // Create mismatched files
    create_test_fastq(&r1_path, 500, "read").unwrap(); // R1: 500 reads
    create_test_fastq(&r2_path, 502, "read").unwrap(); // R2: 502 reads (2 extra)

    // Build a minimal test reference
    let ref_path = temp_dir.path().join("test_ref.fa");
    let mut ref_file = File::create(&ref_path).unwrap();
    writeln!(ref_file, ">chr1").unwrap();
    writeln!(
        ref_file,
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    )
    .unwrap();

    // Try to run ferrous-align
    let output = Command::new(get_ferrous_binary())
        .args(&[
            "mem",
            ref_path.to_str().unwrap(),
            r1_path.to_str().unwrap(),
            r2_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ferrous-align");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should detect the mismatch
    assert!(
        stderr.contains("read count mismatch") || stderr.contains("R1=") && stderr.contains("R2="),
        "Expected error message about read count mismatch. Stderr: {}",
        stderr
    );
}

/// Test 4: EOF desynchronization (R1 ends early, R2 continues)
#[test]
fn test_paired_eof_desync_fail() {
    let temp_dir = TempDir::new().unwrap();
    let r1_path = temp_dir.path().join("R1_short.fq");
    let r2_path = temp_dir.path().join("R2_long.fq");
    let out_path = temp_dir.path().join("out.sam");

    // Create files where R1 is much shorter
    // With batch_size=500, this will trigger EOF desync
    create_test_fastq(&r1_path, 1000, "read").unwrap(); // R1: 1000 reads
    create_test_fastq(&r2_path, 1500, "read").unwrap(); // R2: 1500 reads

    // Build a minimal test reference
    let ref_path = temp_dir.path().join("test_ref.fa");
    let mut ref_file = File::create(&ref_path).unwrap();
    writeln!(ref_file, ">chr1").unwrap();
    writeln!(
        ref_file,
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    )
    .unwrap();

    // Try to run ferrous-align
    let output = Command::new(get_ferrous_binary())
        .args(&[
            "mem",
            ref_path.to_str().unwrap(),
            r1_path.to_str().unwrap(),
            r2_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ferrous-align");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should detect EOF mismatch
    assert!(
        stderr.contains("file ended") || stderr.contains("remaining"),
        "Expected error message about file EOF mismatch. Stderr: {}",
        stderr
    );
}

/// Test 5: Both files empty (should handle gracefully)
#[test]
fn test_paired_both_empty_graceful() {
    let temp_dir = TempDir::new().unwrap();
    let r1_path = temp_dir.path().join("R1_empty.fq");
    let r2_path = temp_dir.path().join("R2_empty.fq");
    let out_path = temp_dir.path().join("out.sam");

    // Create empty files
    File::create(&r1_path).unwrap();
    File::create(&r2_path).unwrap();

    // Build a minimal test reference
    let ref_path = temp_dir.path().join("test_ref.fa");
    let mut ref_file = File::create(&ref_path).unwrap();
    writeln!(ref_file, ">chr1").unwrap();
    writeln!(
        ref_file,
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    )
    .unwrap();

    // Try to run ferrous-align
    let output = Command::new(get_ferrous_binary())
        .args(&[
            "mem",
            ref_path.to_str().unwrap(),
            r1_path.to_str().unwrap(),
            r2_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ferrous-align");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should warn about empty files (not error)
    assert!(
        stderr.contains("No data") || stderr.contains("empty"),
        "Expected warning about empty input. Stderr: {}",
        stderr
    );
}

/// Test 6: Large batch size boundary case
/// Tests validation across batch boundaries
#[test]
#[ignore] // Requires reference index and is slow
fn test_paired_batch_boundary_mismatch() {
    let temp_dir = TempDir::new().unwrap();
    let r1_path = temp_dir.path().join("R1_boundary.fq");
    let r2_path = temp_dir.path().join("R2_boundary.fq");
    let out_path = temp_dir.path().join("out.sam");

    // Create files that will span multiple batches
    // Default batch_size is 500K, use 1M reads to test
    create_test_fastq(&r1_path, 1_000_000, "read").unwrap();
    create_test_fastq(&r2_path, 999_998, "read").unwrap(); // 2 reads short

    // Run ferrous-align
    let output = Command::new(get_ferrous_binary())
        .args(&[
            "mem",
            "test_data/test_ref.fa",
            r1_path.to_str().unwrap(),
            r2_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run ferrous-align");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should detect mismatch even across batch boundaries
    assert!(
        stderr.contains("read count mismatch"),
        "Expected batch size mismatch detection. Stderr: {}",
        stderr
    );
}
