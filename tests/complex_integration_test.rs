// bwa-mem2-rust/tests/complex_integration_test.rs

use std::fs;
use std::io::{self};
use std::path::{Path, PathBuf};
use std::process::Command;

// Helper function to create a temporary directory for test files
fn setup_test_dir(test_name: &str) -> io::Result<PathBuf> {
    let temp_dir = PathBuf::from(format!("target/test_integration_{}", test_name));
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
fn test_alignment_with_snp() -> io::Result<()> {
    let test_name = "alignment_with_snp";
    let temp_dir = setup_test_dir(test_name)?;
    let ref_prefix = temp_dir.join("ref");

    // 1. Create a sample FASTA reference file
    let ref_fasta_content = ">chr1
AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
";
    let ref_fasta_path = create_fasta_file(&temp_dir, "ref.fa", ref_fasta_content)?;

    // 2. Build the index
    ferrous_align::bwa_index::bwa_index(&ref_fasta_path, &ref_prefix)?;

    // Verify index files are created
    // Note: bwa-mem2 format embeds SA data in .bwt.2bit.64, doesn't create separate .sa file
    assert!(ref_prefix.with_extension("pac").exists());
    assert!(ref_prefix.with_extension("ann").exists());
    assert!(ref_prefix.with_extension("amb").exists());
    assert!(ref_prefix.with_extension("bwt.2bit.64").exists());

    // 3. Create sample FASTQ query files with a SNP
    let query_fastq_content = "@read1
AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCA
+
################################
"; // Last base is A instead of T
    let query_fastq_path = create_fastq_file(&temp_dir, "reads.fq", query_fastq_content)?;

    // 4. Run the compiled ferrous-align binary as a separate process
    let binary_path = PathBuf::from("target/debug/ferrous-align");

    let output = Command::new(&binary_path)
        .arg("mem") // Use new CLI subcommand
        .arg(ref_prefix.to_str().unwrap())
        .arg(query_fastq_path.to_str().unwrap())
        .output()?;

    // 5. Verify the output
    assert!(
        output.status.success(),
        "Command failed with: {:?}\nStdout: {}\nStderr: {}",
        output,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Expected SAM output (simplified) - CIGAR should reflect the SNP (31M1X)
    let expected_sam_output_lines = vec![
        "@HD\tVN:1.0\tSO:unsorted",
        "@SQ\tSN:chr1\tLN:32",
        "read1\t0\tchr1\t1\t60\t31M1X\t*\t0\t0\tAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCA\t################################",
    ];
    let expected_sam_output = expected_sam_output_lines.join("\n");

    // Compare line by line, ignoring potential differences in whitespace or order of header lines
    let actual_lines: Vec<&str> = stdout.lines().collect();
    let expected_lines: Vec<&str> = expected_sam_output.lines().collect();

    // Basic check for number of lines and presence of key elements
    assert_eq!(
        actual_lines.len(),
        expected_lines.len(),
        "Mismatch in number of SAM output lines.\nExpected:\n{}\nActual:\n{}",
        expected_sam_output,
        stdout
    );

    for (i, expected_line) in expected_lines.iter().enumerate() {
        assert!(
            actual_lines[i].contains(expected_line),
            "Mismatch on line {}.\nExpected to contain: '{}'\nActual line: '{}'",
            i,
            expected_line,
            actual_lines[i]
        );
    }

    // Check stderr - allow informational messages from CLI
    // Just ensure no actual errors occurred (command succeeded above)

    cleanup_test_dir(&temp_dir);
    Ok(())
}
