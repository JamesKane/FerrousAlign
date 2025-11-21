use anyhow::Result;
// bwa-mem2-rust/src/mem.rs
//
// Main entry point for BWA-MEM alignment
// Orchestrates single-end and paired-end read processing

use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;
use crate::paired_end::process_paired_end;
use crate::single_end::process_single_end;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

pub fn main_mem(
    idx_prefix: &Path,
    query_files: &Vec<String>,
    output: Option<&Path>,
    opt: &MemOpt,
) -> Result<()> {
    // Detect and display SIMD capabilities
use crate::simd::{detect_optimal_simd_engine, simd_engine_description};
    let simd_engine = detect_optimal_simd_engine();
    let simd_desc = simd_engine_description(simd_engine);
    log::info!("Using SIMD engine: {}", simd_desc);

    // Load the BWA index
    let bwa_idx = BwaIndex::bwa_idx_load(idx_prefix).map_err(|e| anyhow::anyhow!("Error loading BWA index: {}", e))?;

    // Determine output writer
    let mut writer: Box<dyn Write> = match output {
        Some(file_name) => Box::new(File::create(file_name).map_err(|e| anyhow::anyhow!("Error creating output file {}: {}", file_name.display(), e))?),
        None => Box::new(io::stdout()),
    };

    // Write SAM header
    writeln!(writer, "@HD\tVN:1.0\tSO:unsorted").map_err(|e| anyhow::anyhow!("Error writing SAM header: {}", e))?;

    // Write @SQ lines for reference sequences
    for ann in &bwa_idx.bns.annotations {
        writeln!(writer, "@SQ\tSN:{}\tLN:{}", ann.name, ann.sequence_length).map_err(|e| anyhow::anyhow!("Error writing SAM header: {}", e))?;
    }

    // Write @PG (program) line
    // Get program name and version from Cargo.toml at compile time
    // This allows the program name to automatically update if we rename the project
    // (e.g., to "FerrousAlign" or another name) without code changes
    const PKG_NAME: &str = env!("CARGO_PKG_NAME");
    const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

    writeln!(
        writer,
        "@PG\tID:{}\tPN:{}\tVN:{}\tCL:{}",
        PKG_NAME,
        PKG_NAME,
        PKG_VERSION,
        std::env::args().collect::<Vec<_>>().join(" ")
    ).map_err(|e| anyhow::anyhow!("Error writing @PG header: {}", e))?;

    // Write read group header if provided (-R option)
    if let Some(ref rg_line) = opt.read_group {
        // Ensure line starts with @RG
        let formatted_rg = if rg_line.starts_with("@RG") {
            rg_line.clone()
        } else {
            format!("@RG\t{}", rg_line)
        };

        writeln!(writer, "{}", formatted_rg).map_err(|e| anyhow::anyhow!("Error writing read group header: {}", e))?;
    }

    // Write custom header lines if provided (-H option)
    for header_line in &opt.header_lines {
        writeln!(writer, "{}", header_line).map_err(|e| anyhow::anyhow!("Error writing custom header: {}", e))?;
    }

    // Detect paired-end mode: if exactly 2 query files provided
    if query_files.len() == 2 {
        // Paired-end mode
        process_paired_end(&bwa_idx, &query_files[0], &query_files[1], &mut writer, opt);
    } else {
        // Single-end mode (original behavior)
        process_single_end(&bwa_idx, query_files, &mut writer, opt);
    }
    Ok(())
}
