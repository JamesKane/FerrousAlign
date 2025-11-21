use clap::{Parser, Subcommand};
use std::path::PathBuf;

use ferrous_align::{bwa_index, mem, mem_opt::MemCliOptions};

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
        #[clap(flatten)]
        options: MemCliOptions,
    },
}

fn init_logging(verbosity: u8) {
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
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Index { fasta, prefix } => {
            init_logging(3); // 3 corresponds to Info level

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

        Commands::Mem { options } => {
            init_logging(options.verbosity as u8);

            log::info!("Aligning reads to index: {}", options.index.display());

            if options.reads.is_empty() {
                log::error!("No read files specified");
                std::process::exit(1);
            }

            if options.reads.len() > 2 {
                log::error!(
                    "Maximum 2 read files allowed (paired-end), got {} files:",
                    options.reads.len()
                );
                for (i, r) in options.reads.iter().enumerate() {
                    log::error!("  File {}: {}", i + 1, r.display());
                }
                std::process::exit(1);
            }

            // The mem::main_mem function now directly accepts MemCliOptions
            mem::main_mem(&options).expect("FerrousAlign encountered a fatal error.");
        }
    }
}
