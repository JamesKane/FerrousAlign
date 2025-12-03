//! Linear alignment pipeline (BWA-MEM style).
//!
//! This pipeline implements read alignment to linear reference genomes using
//! the BWA-MEM algorithm: seeding via FM-Index, chaining, and extension.
//!
//! # Module Organization (v0.8.0)
//!
//! The pipeline is organized into stages coordinated by orchestrators:
//!
//! - `orchestrator/` - Pipeline coordination
//!   - `SingleEndOrchestrator` - Pure SoA pipeline for unpaired reads
//!   - `PairedEndOrchestrator` - Hybrid AoS/SoA for paired reads
//!
//! - `stages/` - Individual pipeline stages
//!   - `SeedingStage` - SMEM extraction via FM-Index
//!   - `ChainingStage` - Seed chaining via DP
//!   - `ExtensionStage` - Smith-Waterman alignment
//!   - `FinalizationStage` - CIGAR, MD tags, filtering
//!
//! # Entry Point
//!
//! The main entry point is `mem::main_mem()` which dispatches to the
//! appropriate orchestrator based on input file count.

// === Core algorithm modules ===
pub mod batch_extension; // Cross-read batched extension (SIMD kernels)
pub mod chaining; // Seed chaining algorithms
pub mod coordinates; // Coordinate transforms
pub mod finalization; // Alignment finalization (CIGAR, MD, MAPQ)
pub mod index; // FM-Index, BWT, suffix array
pub mod mem; // Main entry point
pub mod mem_opt; // Alignment options
pub mod paired; // Paired-end support (insert size, mate rescue, pairing)
pub mod pipeline; // Legacy pipeline types
pub mod region; // Reference region handling
pub mod seeding; // SMEM extraction

// === Stage-based architecture (v0.8.0) ===
pub mod orchestrator; // Pipeline orchestrators
pub mod stages; // Pipeline stage implementations
