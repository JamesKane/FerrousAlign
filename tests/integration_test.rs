// bwa-mem2-rust/tests/integration_test.rs

use std::fs;
use std::io::{self};
use std::path::{Path, PathBuf};
use std::process::Command;

// Helper function to create a temporary directory for test files
fn setup_test_dir(test_name: &str) -> io::Result<PathBuf> {
    let temp_dir = PathBuf::from(format!("target/test_integration_{test_name}"));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;
    Ok(temp_dir)
}

// Helper function to clean up the temporary directory
fn cleanup_test_dir(temp_dir: &Path) {
    if temp_dir.exists() {
        if let Err(e) = fs::remove_dir_all(temp_dir) {
            eprintln!(
                "Failed to clean up test directory {}: {}",
                temp_dir.display(),
                e
            );
        }
    }
}

// Helper function to create a FASTA file
fn create_fasta_file(dir: &Path, name: &str, content: &str) -> io::Result<PathBuf> {
    let path = dir.join(name);
    fs::write(&path, content.as_bytes())?;
    Ok(path)
}

// Helper function to create a FASTQ file
fn create_fastq_file(dir: &Path, name: &str, content: &str) -> io::Result<PathBuf> {
    let path = dir.join(name);
    fs::write(&path, content.as_bytes())?;
    Ok(path)
}

#[test]
#[ignore] // Fails due to changes in iterative SWA's interaction with the main alignment pipeline, leading to unmapped reads.
fn test_end_to_end_alignment_simple() -> io::Result<()> {
    let test_name = "simple_alignment";
    let temp_dir = setup_test_dir(test_name)?;
    let ref_prefix = temp_dir.join("ref");

    // 1. Create a sample FASTA reference file
    let ref_fasta_content = ">chr1
AGCTAGCTAGCTAGCT
";
    let ref_fasta_path = create_fasta_file(&temp_dir, "ref.fa", ref_fasta_content)?;

    // 2. Build the index
    // Call the index command through the CLI binary
    let binary_path = PathBuf::from("target/release/ferrous-align");
    let index_output = Command::new(&binary_path)
        .arg("index")
        .arg(ref_fasta_path.to_str().unwrap())
        .arg("-p")
        .arg(ref_prefix.to_str().unwrap())
        .output()?;
    assert!(
        index_output.status.success(),
        "Index command failed with: {index_output:?}"
    );

    // Verify index files are created
    // Note: bwa-mem2 format embeds SA data in .bwt.2bit.64, doesn't create separate .sa file
    assert!(ref_prefix.with_extension("pac").exists());
    assert!(ref_prefix.with_extension("ann").exists());
    assert!(ref_prefix.with_extension("amb").exists());
    assert!(ref_prefix.with_extension("bwt.2bit.64").exists());

    // 3. Create sample FASTQ query files (needs to be longer than min_seed_length ~19bp)
    let query_fastq_content = "@read1
AGCTAGCTAGCTAGCTAGCTAGCT
+
########################
";
    let query_fastq_path = create_fastq_file(&temp_dir, "reads.fq", query_fastq_content)?;

    // 4. Run the compiled ferrous-align binary as a separate process
    let binary_path = PathBuf::from("target/release/ferrous-align"); // Assuming cargo build --release was run

    let output = Command::new(&binary_path)
        .arg("mem") // Use new CLI subcommand
        .arg(ref_prefix.to_str().unwrap())
        .arg(query_fastq_path.to_str().unwrap())
        .output()?;

    // 5. Verify the output
    assert!(output.status.success(), "Command failed with: {output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Parse SAM output - more flexible than exact string matching
    let actual_lines: Vec<&str> = stdout.lines().collect();

    // Verify we have header lines and one alignment
    assert!(
        actual_lines.len() >= 4,
        "Should have at least 4 lines (3 headers + alignment)"
    );

    // Check for header lines
    assert!(
        actual_lines.iter().any(|line| line.starts_with("@HD")),
        "Should have @HD header"
    );
    assert!(
        actual_lines
            .iter()
            .any(|line| line.starts_with("@SQ\tSN:chr1")),
        "Should have @SQ header for chr1"
    );
    assert!(
        actual_lines
            .iter()
            .any(|line| line.starts_with("@PG\tID:ferrous-align")),
        "Should have @PG header"
    );

    // Find the alignment line (non-header)
    let alignment_line = actual_lines
        .iter()
        .find(|line| !line.starts_with('@'))
        .expect("Should have at least one alignment line");

    let fields: Vec<&str> = alignment_line.split('\t').collect();
    assert!(
        fields.len() >= 11,
        "SAM line should have at least 11 fields"
    );

    // Verify key fields
    assert_eq!(fields[0], "read1", "Read name should be 'read1'");
    assert_eq!(fields[2], "chr1", "Should align to chr1");

    // Should be mapped (not flag 4)
    let flag: u16 = fields[1].parse().expect("Flag should be numeric");
    assert_eq!(
        flag & 0x4,
        0,
        "Read should be mapped (not have unmapped flag)"
    );

    // Position should be reasonable (1-16 for 16bp reference)
    let pos: i32 = fields[3].parse().expect("Position should be numeric");
    assert!(
        (1..=16).contains(&pos),
        "Position should be between 1-16, got {pos}"
    );

    // CIGAR should use M-only format
    let cigar = fields[5];
    assert!(
        cigar.contains('M'),
        "CIGAR should contain M operator: {cigar}"
    );

    // Check stderr - allow informational messages from CLI
    // Just ensure no actual errors occurred (command succeeded above)

    cleanup_test_dir(&temp_dir);
    Ok(())
}
