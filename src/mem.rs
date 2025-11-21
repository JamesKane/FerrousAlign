use crate::index::BwaIndex;
use crate::mem_opt::{MemCliOptions, MemOpt};
use crate::paired_end::process_paired_end;
use crate::single_end::process_single_end;
use anyhow::Result;
use std::fs::File;
use std::io::{self, Write}; // Added import for BwaIndex

pub fn main_mem(opts: &MemCliOptions) -> Result<()> {
    // Detect and display SIMD capabilities
    use crate::simd::{detect_optimal_simd_engine, simd_engine_description};
    let simd_engine = detect_optimal_simd_engine();
    let simd_desc = simd_engine_description(simd_engine);
    log::info!("Using SIMD engine: {}", simd_desc);

    // Extract common options
    let verbosity = opts.verbosity;

    // Initialize MemOpt from MemCliOptions
    let mut opt = MemOpt::default();

    // Algorithm options
    opt.min_seed_len = opts.min_seed_len;
    opt.w = opts.band_width;
    opt.zdrop = opts.off_diagonal_dropoff;
    opt.split_factor = opts.reseed_factor;
    opt.max_mem_intv = opts.seed_occurrence_3rd;
    opt.max_occ = opts.max_occurrences;
    opt.drop_ratio = opts.drop_chain_fraction;
    opt.min_chain_weight = opts.min_chain_weight;
    opt.max_matesw = opts.max_mate_rescues;

    // Scoring options
    opt.update_scoring(opts.match_score, opts.mismatch_penalty);
    opt.pen_unpaired = opts.unpaired_penalty;

    // Parse gap penalties
    let (o_del, o_ins) =
        MemOpt::parse_gap_penalties(&opts.gap_open).map_err(|e| anyhow::anyhow!("{}", e))?;
    opt.o_del = o_del;
    opt.o_ins = o_ins;

    let (e_del, e_ins) =
        MemOpt::parse_gap_penalties(&opts.gap_extend).map_err(|e| anyhow::anyhow!("{}", e))?;
    opt.e_del = e_del;
    opt.e_ins = e_ins;

    // Parse clipping penalties
    let (pen_clip5, pen_clip3) = MemOpt::parse_clip_penalties(&opts.clipping_penalty)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    opt.pen_clip5 = pen_clip5;
    opt.pen_clip3 = pen_clip3;

    // Output options
    opt.t = opts.min_score;
    let (max_xa_hits, max_xa_hits_alt) = crate::mem_opt::parse_xa_hits(&opts.max_xa_hits) // Corrected path
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    opt.max_xa_hits = max_xa_hits;
    opt.max_xa_hits_alt = max_xa_hits_alt;

    // Chunk size
    if let Some(cs) = opts.chunk_size {
        opt.chunk_size = cs;
    }

    // Set mapq_coef_fac after mapq_coef_len is finalized
    opt.mapq_coef_fac = ((opt.mapq_coef_len as f64).ln()) as i32;

    // Configure rayon thread pool
    // Match C++ bwa-mem2 validation: n_threads = n_threads > 1 ? n_threads : 1
    // Default to number of CPU cores if not specified
    let mut num_threads = opts.threads.unwrap_or_else(|| num_cpus::get());

    // Sanity checks (matching C++ fastmap.cpp:674 and :810)
    if num_threads < 1 {
        log::warn!("Invalid thread count {}, using 1 thread", num_threads);
        num_threads = 1;
    }

    // Reasonable upper bound to prevent accidental resource exhaustion
    let max_threads = num_cpus::get() * 2;
    if num_threads > max_threads {
        log::warn!(
            "Thread count {} exceeds recommended maximum {}, capping at {}",
            num_threads,
            max_threads,
            max_threads
        );
        num_threads = max_threads;
    }

    opt.n_threads = num_threads as i32;

    match rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
    {
        Ok(_) => {
            log::debug!(
                "Successfully built global Rayon thread pool with {} threads",
                num_threads
            );
        }
        Err(e) => {
            log::warn!(
                "Failed to configure thread pool: {} (may already be initialized)",
                e
            );
        }
    }

    // Verify actual thread pool size
    let actual_threads = rayon::current_num_threads();
    if actual_threads != num_threads {
        log::warn!(
            "Rayon thread pool has {} threads but requested {}",
            actual_threads,
            num_threads
        );
    } else {
        log::debug!(
            "Rayon thread pool verified: {} threads active",
            actual_threads
        );
    }

    // Output formatting options (Phase 6)
    // Set read group if provided
    opt.read_group = opts.read_group.clone();

    // Parse custom header lines if provided
    if let Some(ref header_str) = opts.header {
        opt.header_lines =
            MemOpt::parse_header_input(header_str).map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    // Phase 7: Advanced options
    // Parse insert size override if provided
    if let Some(ref insert_str) = opts.insert_size {
        opt.insert_size_override =
            Some(MemOpt::parse_insert_size(insert_str).map_err(|e| anyhow::anyhow!("{}", e))?);
    }

    // Set verbosity level
    opt.verbosity = verbosity;

    // Advanced flags
    opt.smart_pairing = opts.smart_pairing;
    opt.treat_alt_as_primary = opts.treat_alt_as_primary;
    opt.smallest_coord_primary = opts.smallest_coord_primary;
    opt.output_all_alignments = opts.output_all;

    // Load the BWA index
    let bwa_idx = BwaIndex::bwa_idx_load(&opts.index)
        .map_err(|e| anyhow::anyhow!("Error loading BWA index: {}", e))?;

    // Determine output writer
    let mut writer: Box<dyn Write> = match opts.output {
        Some(ref file_name) => Box::new(File::create(file_name).map_err(|e| {
            anyhow::anyhow!("Error creating output file {}: {}", file_name.display(), e)
        })?),
        None => Box::new(io::stdout()),
    };

    // Write SAM header
    writeln!(writer, "@HD\tVN:1.0\tSO:unsorted")
        .map_err(|e| anyhow::anyhow!("Error writing SAM header: {}", e))?;

    // Write @SQ lines for reference sequences
    for ann in &bwa_idx.bns.annotations {
        writeln!(writer, "@SQ\tSN:{}\tLN:{}", ann.name, ann.sequence_length)
            .map_err(|e| anyhow::anyhow!("Error writing SAM header: {}", e))?;
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
    )
    .map_err(|e| anyhow::anyhow!("Error writing @PG header: {}", e))?;

    // Write read group header if provided (-R option)
    if let Some(ref rg_line) = opt.read_group {
        // Ensure line starts with @RG
        let formatted_rg = if rg_line.starts_with("@RG") {
            rg_line.clone()
        } else {
            format!("@RG\t{}", rg_line)
        };

        writeln!(writer, "{}", formatted_rg)
            .map_err(|e| anyhow::anyhow!("Error writing read group header: {}", e))?;
    }

    // Write custom header lines if provided (-H option)
    for header_line in &opt.header_lines {
        writeln!(writer, "{}", header_line)
            .map_err(|e| anyhow::anyhow!("Error writing custom header: {}", e))?;
    }

    // Detect paired-end mode: if exactly 2 query files provided
    if opts.reads.len() == 2 {
        // Paired-end mode
        process_paired_end(
            &bwa_idx,
            opts.reads[0]
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid read1 file path"))?,
            opts.reads[1]
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid read2 file path"))?,
            &mut writer,
            &opt,
        );
    } else {
        // Single-end mode (original behavior)
        let read_files_str: Vec<String> = opts
            .reads
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        process_single_end(&bwa_idx, &read_files_str, &mut writer, &opt);
    }
    Ok(())
}
