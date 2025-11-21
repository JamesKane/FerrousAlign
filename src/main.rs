use clap::{Parser, Subcommand};
use std::path::PathBuf;

use ferrous_align::{bwa_index, mem, mem_opt::MemOpt};

#[derive(Parser)]
#[command(name = "ferrous-align")]
#[command(about = "FerrousAlign - Burrows-Wheeler Aligner for DNA sequences (Rust implementation)", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build FM-index for the reference genome
    Index {
        /// Input FASTA file
        #[arg(value_name = "REF.FA")]
        fasta: PathBuf,

        /// Prefix for index files (default: same as FASTA)
        #[arg(short = 'p', long, value_name = "PREFIX")]
        prefix: Option<PathBuf>,
    },

    /// Align reads to reference genome
    Mem {
        /// Index prefix (built with 'index' command)
        #[arg(value_name = "INDEX")]
        index: PathBuf,

        /// Input FASTQ file(s) - single file for single-end, two files for paired-end
        #[arg(value_name = "READS.FQ", required = true)]
        reads: Vec<PathBuf>,

        // ===== Algorithm Options =====
        /// Minimum seed length
        #[arg(short = 'k', long, value_name = "INT", default_value = "19")]
        min_seed_len: i32,

        /// Band width for banded alignment
        #[arg(short = 'w', long, value_name = "INT", default_value = "100")]
        band_width: i32,

        /// Off-diagonal X-dropoff
        #[arg(short = 'd', long, value_name = "INT", default_value = "100")]
        off_diagonal_dropoff: i32,

        /// Look for internal seeds inside a seed longer than {-k} * FLOAT
        #[arg(short = 'r', long, value_name = "FLOAT", default_value = "1.5")]
        reseed_factor: f32,

        /// Seed occurrence for the 3rd round seeding
        #[arg(short = 'y', long, value_name = "INT", default_value = "20")]
        seed_occurrence_3rd: u64,

        /// Skip seeds with more than INT occurrences
        #[arg(short = 'c', long, value_name = "INT", default_value = "500")]
        max_occurrences: i32,

        /// Drop chains shorter than FLOAT fraction of the longest overlapping chain
        #[arg(short = 'D', long, value_name = "FLOAT", default_value = "0.50")]
        drop_chain_fraction: f32,

        /// Discard a chain if seeded bases shorter than INT
        #[arg(short = 'W', long, value_name = "INT", default_value = "0")]
        min_chain_weight: i32,

        /// Perform at most INT rounds of mate rescues for each read
        #[arg(short = 'm', long, value_name = "INT", default_value = "50")]
        max_mate_rescues: i32,

        /// Skip mate rescue
        #[arg(short = 'S', long)]
        skip_mate_rescue: bool,

        /// Skip pairing; mate rescue performed unless -S also in use
        #[arg(short = 'P', long)]
        skip_pairing: bool,

        // ===== Scoring Options =====
        /// Score for a sequence match, which scales options -TdBOELU unless overridden
        #[arg(short = 'A', long, value_name = "INT", default_value = "1")]
        match_score: i32,

        /// Penalty for a mismatch
        #[arg(short = 'B', long, value_name = "INT", default_value = "4")]
        mismatch_penalty: i32,

        /// Gap open penalties for deletions and insertions [6,6]
        #[arg(short = 'O', long, value_name = "INT[,INT]", default_value = "6,6")]
        gap_open: String,

        /// Gap extension penalty; a gap of size k cost '{-O} + {-E}*k' [1,1]
        #[arg(short = 'E', long, value_name = "INT[,INT]", default_value = "1,1")]
        gap_extend: String,

        /// Penalty for 5'- and 3'-end clipping [5,5]
        #[arg(short = 'L', long, value_name = "INT[,INT]", default_value = "5,5")]
        clipping_penalty: String,

        /// Penalty for an unpaired read pair
        #[arg(short = 'U', long, value_name = "INT", default_value = "17")]
        unpaired_penalty: i32,

        // ===== Input/Output Options =====
        /// Output SAM file (default: stdout)
        #[arg(short = 'o', long, value_name = "FILE")]
        output: Option<String>,

        /// Read group header line such as '@RG\tID:foo\tSM:bar'
        #[arg(short = 'R', long, value_name = "STR")]
        read_group: Option<String>,

        /// Insert STR to header if it starts with @; or insert lines in FILE
        #[arg(short = 'H', long, value_name = "STR")]
        header: Option<String>,

        /// Treat ALT contigs as part of the primary assembly (i.e. ignore <idxbase>.alt file)
        #[arg(short = 'j', long)]
        treat_alt_as_primary: bool,

        /// For split alignment, take the alignment with the smallest coordinate as primary
        #[arg(short = '5', long)]
        smallest_coord_primary: bool,

        /// Don't modify mapQ of supplementary alignments
        #[arg(short = 'q', long)]
        no_modify_mapq: bool,

        /// Process INT input bases in each batch regardless of nThreads (for reproducibility)
        #[arg(short = 'K', long, value_name = "INT")]
        chunk_size: Option<i64>,

        /// Verbose level: 1=error, 2=warning, 3=message, 4+=debugging
        #[arg(short = 'v', long, value_name = "INT", default_value = "3")]
        verbosity: i32,

        /// Minimum score to output
        #[arg(short = 'T', long, value_name = "INT", default_value = "30")]
        min_score: i32,

        /// If there are <INT hits with score >80% of the max score, output all in XA [5,200]
        /// Note: bwa-mem2 uses -h, but we use --max-xa-hits to avoid conflict with --help
        #[arg(long, value_name = "INT[,INT]", default_value = "5,200")]
        max_xa_hits: String,

        /// Output all alignments for SE or unpaired PE
        #[arg(short = 'a', long)]
        output_all: bool,

        /// Append FASTA/FASTQ comment to SAM output
        #[arg(short = 'C', long)]
        append_comment: bool,

        /// Output the reference FASTA header in the XR tag
        #[arg(short = 'V', long)]
        output_ref_header: bool,

        /// Use soft clipping for supplementary alignments
        #[arg(short = 'Y', long)]
        soft_clip_supplementary: bool,

        /// Mark shorter split hits as secondary
        #[arg(short = 'M', long)]
        mark_secondary: bool,

        /// Smart pairing (ignoring in2.fq)
        #[arg(short = 'p', long)]
        smart_pairing: bool,

        /// Specify the mean, standard deviation (10% of the mean if absent), max
        /// (4 sigma from the mean if absent) and min of the insert size distribution.
        /// FR orientation only. [inferred]
        #[arg(short = 'I', long, value_name = "FLOAT[,FLOAT[,INT[,INT]]]")]
        insert_size: Option<String>,

        // ===== Processing Options =====
        /// Number of threads (default: all available cores)
        #[arg(short = 't', long, value_name = "INT")]
        threads: Option<usize>,
    },
}

/// Parse XA hits string "INT" or "INT,INT"
fn parse_xa_hits(s: &str) -> Result<(i32, i32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    match parts.len() {
        1 => {
            let val = parts[0]
                .parse::<i32>()
                .map_err(|_| format!("Invalid XA hits value: {}", s))?;
            Ok((val, 200)) // Default alt hits to 200
        }
        2 => {
            let primary = parts[0]
                .parse::<i32>()
                .map_err(|_| format!("Invalid primary XA hits: {}", parts[0]))?;
            let alt = parts[1]
                .parse::<i32>()
                .map_err(|_| format!("Invalid alt XA hits: {}", parts[1]))?;
            Ok((primary, alt))
        }
        _ => Err(format!("XA hits must be INT or INT,INT: {}", s)),
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { fasta, prefix } => {
            // Initialize logger with info level for index command
            env_logger::Builder::from_default_env()
                .filter_level(log::LevelFilter::Info)
                .format_timestamp(None)
                .format_target(false)
                .init();

            // Use FASTA filename as prefix if not specified
            let idx_prefix = prefix.unwrap_or_else(|| fasta.clone());

            log::info!("Building index for reference: {}", fasta.display());
            log::info!("Index prefix: {}", idx_prefix.display());

            if let Err(e) = bwa_index::bwa_index(&fasta, &idx_prefix) {
                log::error!("Index building failed: {}", e);
                std::process::exit(1);
            }

            log::info!("Index building completed successfully");
        }

        Commands::Mem {
            index,
            reads,
            // Algorithm options
            min_seed_len,
            band_width,
            off_diagonal_dropoff,
            reseed_factor,
            seed_occurrence_3rd,
            max_occurrences,
            drop_chain_fraction,
            min_chain_weight,
            max_mate_rescues,
            skip_mate_rescue,
            skip_pairing,
            // Scoring options
            match_score,
            mismatch_penalty,
            gap_open,
            gap_extend,
            clipping_penalty,
            unpaired_penalty,
            // I/O options
            output,
            read_group,
            header,
            treat_alt_as_primary,
            smallest_coord_primary,
            no_modify_mapq,
            chunk_size,
            verbosity,
            min_score,
            max_xa_hits,
            output_all,
            append_comment,
            output_ref_header,
            soft_clip_supplementary,
            mark_secondary,
            smart_pairing,
            insert_size,
            // Processing
            threads,
        } => {
            // Initialize logger based on verbosity level
            // Map bwa-mem2 verbosity (1=error, 2=warning, 3=message, 4=debug, 5+=trace)
            // to Rust log levels
            let log_level = match verbosity {
                v if v <= 1 => log::LevelFilter::Error,
                2 => log::LevelFilter::Warn,
                3 => log::LevelFilter::Info,
                4 => log::LevelFilter::Debug,
                _ => log::LevelFilter::Trace, // 5+ = trace
            };

            env_logger::Builder::from_default_env()
                .filter_level(log_level)
                .format_timestamp(None) // Don't show timestamps
                .format_target(false) // Don't show module names
                .init();

            log::info!("Aligning reads to index: {}", index.display());

            if reads.is_empty() {
                log::error!("No read files specified");
                std::process::exit(1);
            }

            if reads.len() > 2 {
                log::error!(
                    "Maximum 2 read files allowed (paired-end), got {} files:",
                    reads.len()
                );
                for (i, r) in reads.iter().enumerate() {
                    log::error!("  File {}: {}", i + 1, r.display());
                }
                std::process::exit(1);
            }

            // Build MemOpt from command-line arguments
            let mut opt = MemOpt::default();

            // Algorithm options
            opt.min_seed_len = min_seed_len;
            opt.w = band_width;
            opt.zdrop = off_diagonal_dropoff;
            opt.split_factor = reseed_factor;
            opt.max_mem_intv = seed_occurrence_3rd;
            opt.max_occ = max_occurrences;
            opt.drop_ratio = drop_chain_fraction;
            opt.min_chain_weight = min_chain_weight;
            opt.max_matesw = max_mate_rescues;

            // Scoring options
            opt.update_scoring(match_score, mismatch_penalty);
            opt.pen_unpaired = unpaired_penalty;

            // Parse gap penalties
            match MemOpt::parse_gap_penalties(&gap_open) {
                Ok((del, ins)) => {
                    opt.o_del = del;
                    opt.o_ins = ins;
                }
                Err(e) => {
                    log::error!("{}", e);
                    std::process::exit(1);
                }
            }

            match MemOpt::parse_gap_penalties(&gap_extend) {
                Ok((del, ins)) => {
                    opt.e_del = del;
                    opt.e_ins = ins;
                }
                Err(e) => {
                    log::error!("{}", e);
                    std::process::exit(1);
                }
            }

            // Parse clipping penalties
            match MemOpt::parse_clip_penalties(&clipping_penalty) {
                Ok((clip5, clip3)) => {
                    opt.pen_clip5 = clip5;
                    opt.pen_clip3 = clip3;
                }
                Err(e) => {
                    log::error!("{}", e);
                    std::process::exit(1);
                }
            }

            // Output options
            opt.t = min_score;

            match parse_xa_hits(&max_xa_hits) {
                Ok((primary, alt)) => {
                    opt.max_xa_hits = primary;
                    opt.max_xa_hits_alt = alt;
                }
                Err(e) => {
                    log::error!("{}", e);
                    std::process::exit(1);
                }
            }

            // Chunk size
            if let Some(cs) = chunk_size {
                opt.chunk_size = cs;
            }

            // Set mapq_coef_fac after mapq_coef_len is finalized
            opt.mapq_coef_fac = ((opt.mapq_coef_len as f64).ln()) as i32;

            // Configure rayon thread pool
            // Match C++ bwa-mem2 validation: n_threads = n_threads > 1 ? n_threads : 1
            // Default to number of CPU cores if not specified
            let mut num_threads = threads.unwrap_or_else(|| num_cpus::get());

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

            let thread_word = if num_threads == 1 {
                "thread"
            } else {
                "threads"
            };
            log::info!("Using {} {}", num_threads, thread_word);

            // Print key parameter values if verbosity >= 3
            if verbosity >= 3 {
                log::info!("Algorithm parameters:");
                log::info!("  Min seed length: {}", opt.min_seed_len);
                log::info!("  Band width: {}", opt.w);
                log::info!("  X-dropoff: {}", opt.zdrop);
                log::info!("  Max seed occurrences: {}", opt.max_occ);
                log::info!("  Re-seed factor: {}", opt.split_factor);
                log::info!("Scoring parameters:");
                log::info!("  Match: {}, Mismatch: {}", opt.a, opt.b);
                log::info!(
                    "  Gap open: ({},{}), Gap extend: ({},{})",
                    opt.o_del,
                    opt.o_ins,
                    opt.e_del,
                    opt.e_ins
                );
                log::info!(
                    "  Clipping: ({},{}), Unpaired: {}",
                    opt.pen_clip5,
                    opt.pen_clip3,
                    opt.pen_unpaired
                );
            }

            // Phase 6: Output and formatting options
            // Set read group if provided
            opt.read_group = read_group.clone();

            // Parse custom header lines if provided
            if let Some(ref header_str) = header {
                match MemOpt::parse_header_input(header_str) {
                    Ok(lines) => opt.header_lines = lines,
                    Err(e) => {
                        log::error!("{}", e);
                        std::process::exit(1);
                    }
                }
            }

            // Phase 7: Advanced options
            // Parse insert size override if provided
            if let Some(ref insert_str) = insert_size {
                match MemOpt::parse_insert_size(insert_str) {
                    Ok(is_override) => {
                        if verbosity >= 3 {
                            log::info!(
                                "Insert size override: mean={:.1}, std={:.1}, max={}, min={}",
                                is_override.mean,
                                is_override.stddev,
                                is_override.max,
                                is_override.min
                            );
                        }
                        opt.insert_size_override = Some(is_override);
                    }
                    Err(e) => {
                        log::error!("{}", e);
                        std::process::exit(1);
                    }
                }
            }

            // Set verbosity level
            opt.verbosity = verbosity;

            // Advanced flags
            opt.smart_pairing = smart_pairing;
            opt.treat_alt_as_primary = treat_alt_as_primary;
            opt.smallest_coord_primary = smallest_coord_primary;
            opt.output_all_alignments = output_all;

            // Store flags as options for later use (not yet implemented)
            // These will be used when we implement the corresponding features
            let _skip_mate_rescue = skip_mate_rescue;
            let _skip_pairing = skip_pairing;
            let _no_modify_mapq = no_modify_mapq;
            let _append_comment = append_comment;
            let _output_ref_header = output_ref_header;
            let _soft_clip_supplementary = soft_clip_supplementary;
            let _mark_secondary = mark_secondary;

            let read_files: Vec<String> = reads
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect();

            mem::main_mem(&index, &read_files, output.as_ref(), &opt);
        }
    }
}
