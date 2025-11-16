/// Synthetic test to verify SMEM generation and SA position mapping
/// This test creates a simple reference, indexes it, and verifies that SMEMs point to correct positions

use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    println!("Creating synthetic reference...");

    // Create a simple reference with a known repeating pattern
    // Pattern: "AAAACCCCGGGGTTTT" repeated 100 times = 1600 bases
    let mut reference = String::from(">test_chr\n");
    let pattern = "AAAACCCCGGGGTTTT";
    for _ in 0..100 {
        reference.push_str(pattern);
    }
    reference.push('\n');

    // Write reference to file
    let ref_path = "/tmp/test_synthetic.fasta";
    fs::write(ref_path, &reference).expect("Failed to write reference");
    println!("Reference written to {}", ref_path);
    println!("Pattern: {} (repeated 100 times)", pattern);
    println!("Total length: {} bases", pattern.len() * 100);

    // Index the reference
    println!("\nIndexing reference...");
    let index_output = Command::new("./target/release/ferrous-align")
        .args(&["index", ref_path])
        .output()
        .expect("Failed to run indexer");

    if !index_output.status.success() {
        eprintln!("Indexing failed:");
        eprintln!("{}", String::from_utf8_lossy(&index_output.stderr));
        return;
    }
    println!("Index created successfully");

    // Create a query that exactly matches part of the reference
    // Query: "CCCCGGGGTTTTAAAA" (20 bases starting from position 4 in the pattern)
    let query_seq = "CCCCGGGGTTTTAAAA";
    let query = format!(">test_query\n{}\n", query_seq);
    let query_path = "/tmp/test_synthetic_query.fasta";
    fs::write(query_path, &query).expect("Failed to write query");
    println!("\nQuery: {}", query_seq);
    println!("This should match at positions: 4, 20, 36, 52, ... (every 16 bases)");

    // Run alignment with verbose debug output
    println!("\nRunning alignment with debug output...");
    let align_output = Command::new("./target/release/ferrous-align")
        .args(&["mem", "-T", "0", "-v", "4", ref_path, query_path])
        .output()
        .expect("Failed to run aligner");

    let stderr = String::from_utf8_lossy(&align_output.stderr);

    // Print ALL stderr for debugging
    println!("\n=== Full STDERR Output ===");
    println!("{}", stderr);

    // Look for SMEM information
    println!("\n=== Filtered SMEM Information ===");
    for line in stderr.lines() {
        if line.contains("SMEM") || line.contains("PERFECT MATCH") || line.contains("NO PERFECT MATCH") {
            println!("{}", line);
        }
    }

    println!("\n=== SAM Output ===");
    let stdout = String::from_utf8_lossy(&align_output.stdout);
    for line in stdout.lines() {
        if !line.starts_with('@') {
            println!("{}", line);
        }
    }

    // Cleanup
    println!("\nCleaning up temp files...");
    let _ = fs::remove_file(ref_path);
    let _ = fs::remove_file(query_path);
    let _ = fs::remove_file(format!("{}.amb", ref_path));
    let _ = fs::remove_file(format!("{}.ann", ref_path));
    let _ = fs::remove_file(format!("{}.bwt.2bit.64", ref_path));
    let _ = fs::remove_file(format!("{}.pac", ref_path));
    let _ = fs::remove_file(format!("{}.sa", ref_path));
    println!("Done!");
}
