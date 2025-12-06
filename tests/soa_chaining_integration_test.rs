// tests/soa_chaining_integration_test.rs

use ferrous_align::core::io::soa_readers::SoaFastqReader;
use ferrous_align::index::index::BwaIndex;
use ferrous_align::pipelines::linear::chaining::{chain_seeds_batch, filter_chains_batch};
use ferrous_align::pipelines::linear::mem_opt::MemOpt;
use ferrous_align::pipelines::linear::seeding::find_seeds_batch;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

// Helper function to create a temporary directory for test files
fn setup_test_dir(test_name: &str) -> io::Result<PathBuf> {
    let temp_dir = PathBuf::from(format!("target/test_soa_chaining_{test_name}"));
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
fn test_soa_seeding_and_chaining_simple() -> io::Result<()> {
    let test_name = "soa_simple_alignment";
    let temp_dir = setup_test_dir(test_name)?;
    let ref_prefix = temp_dir.join("ref");

    // 1. Create a sample FASTA reference file
    // Use non-palindromic sequences to avoid ambiguous strand matching
    // GATTACAGG... is NOT its own reverse complement, ensuring strand specificity
    let ref_fasta_content = ">chr1
GATTACAGGATTACAGGATTACAGGATTACAGGATTACAG
>chr2
TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
";
    let ref_fasta_path = create_fasta_file(&temp_dir, "ref.fa", ref_fasta_content)?;

    // 2. Build the index
    // Call the index command through the CLI binary
    let binary_path = PathBuf::from("target/release/ferrous-align");
    if !binary_path.exists() {
        eprintln!("Binary not found at {binary_path:?}, attempting to build...");
        let build_output = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .output()?;
        if !build_output.status.success() {
            panic!("Failed to build project: {build_output:?}");
        }
    }

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

    // Load the index
    let bwa_idx = BwaIndex::bwa_idx_load(&ref_prefix)?;

    // 3. Create sample FASTQ query files
    // read1 matches first 24bp of chr1 (forward strand)
    // read2 matches full chr1 (forward strand, but ends at 39 not 40 - last base different)
    let query_fastq_content = "@read1
GATTACAGGATTACAGGATTACAG
+
########################
@read2
GATTACAGGATTACAGGATTACAGGATTACAGGATTACAG
+
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
";
    let query_fastq_path = create_fastq_file(&temp_dir, "reads.fq", query_fastq_content)?;

    // 4. Read the FASTQ file into an SoAReadBatch
    let mut reader = SoaFastqReader::new(query_fastq_path.to_str().unwrap())?;
    let soa_read_batch = reader.read_batch(512)?;

    // 5. Initialize MemOpt
    let opt = MemOpt::default();

    // 6. Call find_seeds_batch
    let (soa_seed_batch, _, _) = find_seeds_batch(&bwa_idx, &soa_read_batch, &opt);

    // Assert that seeds were generated
    assert!(
        !soa_seed_batch.query_pos.is_empty(),
        "SoASeedBatch should not be empty"
    );
    assert_eq!(
        soa_seed_batch.read_seed_boundaries.len(),
        2,
        "Should have seed boundaries for 2 reads"
    );

    // 7. Call chain_seeds_batch
    let l_pac = bwa_idx.bns.packed_sequence_length;
    let mut soa_chain_batch = chain_seeds_batch(&soa_seed_batch, &opt, l_pac);

    // Assert that chains were generated
    assert!(
        !soa_chain_batch.score.is_empty(),
        "SoAChainBatch should not be empty"
    );
    assert_eq!(
        soa_chain_batch.read_chain_boundaries.len(),
        2,
        "Should have chain boundaries for 2 reads"
    );

    // 8. Call filter_chains_batch
    let query_lengths: Vec<i32> = soa_read_batch
        .read_boundaries
        .iter()
        .map(|(_, len)| *len as i32)
        .collect();
    filter_chains_batch(
        &mut soa_chain_batch,
        &soa_seed_batch,
        &opt,
        &query_lengths,
        &soa_read_batch.names,
    );

    // Assert filtering results
    // For read1, expect 1 primary chain
    let (read1_chain_start, read1_num_chains) = soa_chain_batch.read_chain_boundaries[0];
    let mut read1_primary_chains = 0;
    for i in 0..read1_num_chains {
        if soa_chain_batch.kept[read1_chain_start + i] == 3 {
            read1_primary_chains += 1;
        }
    }
    assert_eq!(
        read1_primary_chains, 1,
        "Read 1 should have 1 primary chain"
    );

    // For read2, expect 1 primary chain
    let (read2_chain_start, read2_num_chains) = soa_chain_batch.read_chain_boundaries[1];
    let mut read2_primary_chains = 0;
    for i in 0..read2_num_chains {
        if soa_chain_batch.kept[read2_chain_start + i] == 3 {
            read2_primary_chains += 1;
        }
    }
    assert_eq!(
        read2_primary_chains, 1,
        "Read 2 should have 1 primary chain"
    );

    // Check specific chain properties for read1 (GATTACAGGATTACAGGATTACAG - 24bp)
    // Expected to align to chr1 forward strand (len 40)
    // Note: Due to min_seed_len=19 (default), the chain covers [0, 19) of the query
    let first_chain_idx = soa_chain_batch.read_chain_boundaries[0].0;
    assert_eq!(
        soa_chain_batch.rid[first_chain_idx], 0,
        "First chain should be on chr1 (rid 0)"
    );
    assert!(
        !soa_chain_batch.is_rev[first_chain_idx],
        "First chain should be forward strand (non-palindromic sequence)"
    );
    assert_eq!(soa_chain_batch.query_start[first_chain_idx], 0);
    // Chain covers minimum seed length (19bp), not full read length
    assert!(
        soa_chain_batch.query_end[first_chain_idx] >= 19,
        "Chain should cover at least min_seed_len (19bp), got {}",
        soa_chain_batch.query_end[first_chain_idx]
    );
    assert!(
        soa_chain_batch.score[first_chain_idx] >= 19,
        "Score should be at least 19, got {}",
        soa_chain_batch.score[first_chain_idx]
    );

    // Check specific chain properties for read2 (full chr1: GATTACAGG... - 40bp)
    // Expected to align to chr1 forward strand (len 40)
    let second_chain_idx = soa_chain_batch.read_chain_boundaries[1].0;
    assert_eq!(
        soa_chain_batch.rid[second_chain_idx], 0,
        "Second chain should be on chr1 (rid 0)"
    );
    assert!(
        !soa_chain_batch.is_rev[second_chain_idx],
        "Second chain should be forward strand (non-palindromic sequence)"
    );
    assert_eq!(soa_chain_batch.query_start[second_chain_idx], 0);
    // For a 40bp read matching chr1 exactly, expect coverage of most/all of the read
    assert!(
        soa_chain_batch.query_end[second_chain_idx] >= 19,
        "Chain should cover at least min_seed_len (19bp), got {}",
        soa_chain_batch.query_end[second_chain_idx]
    );
    assert!(
        soa_chain_batch.score[second_chain_idx] >= 19,
        "Score should be at least 19, got {}",
        soa_chain_batch.score[second_chain_idx]
    );

    cleanup_test_dir(&temp_dir);
    Ok(())
}

#[test]
fn test_end_to_end_soa_pipeline() -> io::Result<()> {
    let test_name = "end_to_end_soa";
    let temp_dir = setup_test_dir(test_name)?;
    let ref_prefix = temp_dir.join("ref");

    // 1. Create a sample FASTA reference file
    let ref_fasta_content = ">chr1
GATTACAGGATTACAGGATTACAGGATTACAGGATTACAG
>chr2
TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
";
    let ref_fasta_path = create_fasta_file(&temp_dir, "ref.fa", ref_fasta_content)?;

    // 2. Build the index
    let binary_path = PathBuf::from("target/release/ferrous-align");
    if !binary_path.exists() {
        eprintln!("Binary not found at {binary_path:?}, attempting to build...");
        let build_output = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .output()?;
        if !build_output.status.success() {
            panic!("Failed to build project: {build_output:?}");
        }
    }

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

    // 3. Create sample FASTQ query files (seq and qual must have same length)
    let query_fastq_content = "@read1
GATTACAGGATTACAGGATTACAG
+
########################
@read2
GATTACAGGATTACAGGATTACAGGATTACAGGATTACAG
+
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
";
    let query_fastq_path = create_fastq_file(&temp_dir, "reads.fq", query_fastq_content)?;

    // 4. Run alignment using the end-to-end SoA pipeline
    let output_sam_path = temp_dir.join("output.sam");
    let stderr_log_path = temp_dir.join("stderr.log");
    let align_output = Command::new(&binary_path)
        .arg("mem")
        .arg("-v")
        .arg("3") // Set verbosity to INFO level
        .arg(ref_prefix.to_str().unwrap())
        .arg(query_fastq_path.to_str().unwrap())
        .env("FERROUS_SOA_PIPELINE", "1") // Enable SoA pipeline
        .stdout(std::process::Stdio::from(std::fs::File::create(
            &output_sam_path,
        )?))
        .stderr(std::process::Stdio::from(std::fs::File::create(
            &stderr_log_path,
        )?))
        .output()?;

    // Read stderr log for debugging
    let stderr_contents = fs::read_to_string(&stderr_log_path)?;

    // 5. Verify alignment succeeded
    assert!(
        align_output.status.success(),
        "Alignment command failed: {:?}\nStderr: {}",
        align_output.status,
        stderr_contents
    );

    // 6. Verify output SAM file exists and is non-empty
    assert!(output_sam_path.exists(), "Output SAM file not created");

    let output_contents = fs::read_to_string(&output_sam_path)?;
    assert!(!output_contents.is_empty(), "Output SAM file is empty");

    // 7. Verify SAM headers are present
    assert!(
        output_contents.contains("@HD"),
        "SAM file missing @HD header"
    );
    assert!(
        output_contents.contains("@SQ"),
        "SAM file missing @SQ header"
    );
    assert!(
        output_contents.contains("@PG"),
        "SAM file missing @PG header"
    );

    // 8. Verify alignment records are present
    let alignment_lines: Vec<&str> = output_contents
        .lines()
        .filter(|line| !line.starts_with('@'))
        .collect();

    assert!(
        alignment_lines.len() >= 2,
        "Expected at least 2 alignment records, found {}",
        alignment_lines.len()
    );

    // 9. Verify log messages indicate SoA pipeline was used
    // SingleEndOrchestrator uses the SoA pipeline
    assert!(
        stderr_contents.contains("SingleEndOrchestrator")
            || stderr_contents.contains("SOA_PIPELINE")
            || stderr_contents.contains("Using end-to-end SoA pipeline"),
        "SoA pipeline was not used. Stderr: {stderr_contents}"
    );

    cleanup_test_dir(&temp_dir);
    Ok(())
}
