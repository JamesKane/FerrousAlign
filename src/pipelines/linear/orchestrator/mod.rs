//! Pipeline orchestration layer
//!
//! Orchestrators coordinate the execution of pipeline stages, handling:
//! - Batch loading and iteration
//! - Stage sequencing
//! - AoS/SoA representation transitions (for paired-end)
//! - Statistics aggregation
//! - Output writing
//!
//! # Architecture
//!
//! The pipeline uses a **hybrid AoS/SoA architecture** discovered during v0.7.0:
//! - **Single-end**: Pure SoA pipeline (no conversions)
//! - **Paired-end**: Hybrid with explicit transitions
//!
//! ```text
//! Paired-End Flow:
//! [SoA] Alignment → [AoS] Pairing → [SoA] Mate Rescue → [AoS] Output
//! ```
//!
//! The pairing stage MUST use AoS to maintain per-read alignment boundaries.
//! Pure SoA pairing causes 96% duplicate reads due to index corruption.
//!
//! See `documents/Pipeline_Restructure_v0.8_Plan.md` for details.

use super::stages::StageError;
use std::fmt;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

// ============================================================================
// PIPELINE MODE
// ============================================================================

/// Pipeline execution mode.
///
/// Determines whether to run single-end or paired-end processing,
/// which affects the data flow and representation transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    /// Single-end reads: pure SoA pipeline, no conversions
    SingleEnd,

    /// Paired-end reads: hybrid AoS/SoA with explicit transitions
    PairedEnd,
}

impl fmt::Display for PipelineMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineMode::SingleEnd => write!(f, "single-end"),
            PipelineMode::PairedEnd => write!(f, "paired-end"),
        }
    }
}

// ============================================================================
// PIPELINE STATISTICS
// ============================================================================

/// Aggregate statistics from pipeline execution.
///
/// Collected during processing and returned upon completion.
/// Used for logging, benchmarking, and throughput analysis.
#[derive(Debug, Clone, Default)]
pub struct PipelineStatistics {
    /// Total number of reads processed
    pub total_reads: usize,

    /// Total number of bases processed
    pub total_bases: usize,

    /// Total number of alignments generated
    pub total_alignments: usize,

    /// Number of batches processed
    pub batches_processed: usize,

    /// Wall clock time in seconds
    pub wall_time_secs: f64,

    /// CPU time in seconds (user + system)
    pub cpu_time_secs: f64,

    /// Number of reads that mapped
    pub mapped_reads: usize,

    /// Number of properly paired reads (paired-end only)
    pub properly_paired: usize,
}

impl PipelineStatistics {
    /// Create a new empty statistics tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate throughput in megabases per second.
    pub fn throughput_mbases_per_sec(&self) -> f64 {
        if self.wall_time_secs > 0.0 {
            (self.total_bases as f64 / 1_000_000.0) / self.wall_time_secs
        } else {
            0.0
        }
    }

    /// Calculate reads per second.
    pub fn reads_per_second(&self) -> f64 {
        if self.wall_time_secs > 0.0 {
            self.total_reads as f64 / self.wall_time_secs
        } else {
            0.0
        }
    }

    /// Calculate mapping rate as a percentage.
    pub fn mapping_rate(&self) -> f64 {
        if self.total_reads > 0 {
            (self.mapped_reads as f64 / self.total_reads as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Merge statistics from another instance (for parallel processing).
    pub fn merge(&mut self, other: &PipelineStatistics) {
        self.total_reads += other.total_reads;
        self.total_bases += other.total_bases;
        self.total_alignments += other.total_alignments;
        self.batches_processed += other.batches_processed;
        self.mapped_reads += other.mapped_reads;
        self.properly_paired += other.properly_paired;
        // Time fields are not merged - they're set at the orchestrator level
    }
}

impl fmt::Display for PipelineStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Processed {} reads ({:.2} Mbases) in {:.2}s ({:.2} reads/sec, {:.2} Mbases/sec)",
            self.total_reads,
            self.total_bases as f64 / 1_000_000.0,
            self.wall_time_secs,
            self.reads_per_second(),
            self.throughput_mbases_per_sec()
        )
    }
}

// ============================================================================
// ORCHESTRATOR ERROR
// ============================================================================

/// Errors that can occur during pipeline orchestration.
#[derive(Debug)]
pub enum OrchestratorError {
    /// Error from a pipeline stage
    Stage(StageError),

    /// I/O error (file access, writing)
    Io(std::io::Error),

    /// Invalid input files or configuration
    InvalidInput(String),

    /// Paired-end file mismatch (different read counts)
    PairedEndMismatch { r1_count: usize, r2_count: usize },
}

impl fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrchestratorError::Stage(err) => write!(f, "Stage error: {}", err),
            OrchestratorError::Io(err) => write!(f, "I/O error: {}", err),
            OrchestratorError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            OrchestratorError::PairedEndMismatch { r1_count, r2_count } => {
                write!(
                    f,
                    "Paired-end file mismatch: R1 has {} reads, R2 has {} reads",
                    r1_count, r2_count
                )
            }
        }
    }
}

impl std::error::Error for OrchestratorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OrchestratorError::Stage(err) => Some(err),
            OrchestratorError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<StageError> for OrchestratorError {
    fn from(err: StageError) -> Self {
        OrchestratorError::Stage(err)
    }
}

impl From<std::io::Error> for OrchestratorError {
    fn from(err: std::io::Error) -> Self {
        OrchestratorError::Io(err)
    }
}

// ============================================================================
// PIPELINE ORCHESTRATOR TRAIT
// ============================================================================

/// High-level pipeline orchestrator trait.
///
/// Orchestrators coordinate the full alignment pipeline, from reading input files
/// through all processing stages to writing output. Different orchestrators
/// handle single-end vs paired-end processing with their respective data flows.
///
/// # Implementations
///
/// - `SingleEndOrchestrator`: Pure SoA pipeline for unpaired reads
/// - `PairedEndOrchestrator`: Hybrid AoS/SoA pipeline for paired reads
///
/// # Example
///
/// ```ignore
/// let mut orchestrator = SingleEndOrchestrator::new(&index, &options, &compute_ctx);
/// let stats = orchestrator.run(&[input_file], &mut output)?;
/// println!("{}", stats);
/// ```
pub trait PipelineOrchestrator {
    /// Run the complete alignment pipeline.
    ///
    /// # Arguments
    /// * `input_files` - Input FASTQ file paths (1 for single-end, 2 for paired-end)
    /// * `output` - Writer for SAM output
    ///
    /// # Returns
    /// Statistics about the pipeline execution, or an error if processing fails.
    fn run(
        &mut self,
        input_files: &[PathBuf],
        output: &mut dyn Write,
    ) -> Result<PipelineStatistics, OrchestratorError>;

    /// Get the pipeline mode (single-end or paired-end).
    fn mode(&self) -> PipelineMode;
}

// ============================================================================
// TIMING UTILITIES
// ============================================================================

/// A simple timer for measuring pipeline execution time.
pub struct PipelineTimer {
    start: Instant,
}

impl PipelineTimer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Get elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Stop the timer and return elapsed time in seconds.
    pub fn stop(&self) -> f64 {
        self.elapsed_secs()
    }
}

// ============================================================================
// SUBMODULES
// ============================================================================

pub mod conversions;
pub mod paired_end;
pub mod single_end;

// Re-export orchestrator implementations
pub use paired_end::PairedEndOrchestrator;
pub use single_end::SingleEndOrchestrator;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_mode_display() {
        assert_eq!(format!("{}", PipelineMode::SingleEnd), "single-end");
        assert_eq!(format!("{}", PipelineMode::PairedEnd), "paired-end");
    }

    #[test]
    fn test_pipeline_statistics_new() {
        let stats = PipelineStatistics::new();
        assert_eq!(stats.total_reads, 0);
        assert_eq!(stats.total_bases, 0);
    }

    #[test]
    fn test_pipeline_statistics_throughput() {
        let mut stats = PipelineStatistics::new();
        stats.total_bases = 1_000_000; // 1 Mbase
        stats.wall_time_secs = 2.0;

        assert!((stats.throughput_mbases_per_sec() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_statistics_merge() {
        let mut stats1 = PipelineStatistics::new();
        stats1.total_reads = 100;
        stats1.total_bases = 15000;
        stats1.mapped_reads = 95;

        let mut stats2 = PipelineStatistics::new();
        stats2.total_reads = 200;
        stats2.total_bases = 30000;
        stats2.mapped_reads = 190;

        stats1.merge(&stats2);

        assert_eq!(stats1.total_reads, 300);
        assert_eq!(stats1.total_bases, 45000);
        assert_eq!(stats1.mapped_reads, 285);
    }

    #[test]
    fn test_pipeline_timer() {
        let timer = PipelineTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= 0.01);
    }

    #[test]
    fn test_orchestrator_error_display() {
        let err = OrchestratorError::InvalidInput("test".to_string());
        assert_eq!(format!("{}", err), "Invalid input: test");

        let err = OrchestratorError::PairedEndMismatch {
            r1_count: 100,
            r2_count: 99,
        };
        assert!(format!("{}", err).contains("100"));
        assert!(format!("{}", err).contains("99"));
    }
}
