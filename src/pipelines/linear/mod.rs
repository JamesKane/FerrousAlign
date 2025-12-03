//! Linear alignment pipeline (BWA-MEM style).
//!
//! This pipeline implements read alignment to linear reference genomes using
//! the BWA-MEM algorithm: seeding via FM-Index, chaining, and extension.
//!
//! # Module Organization (v0.8.0 restructure in progress)
//!
//! New modular structure being introduced:
//! - `orchestrator/` - Pipeline coordination (main loop abstraction)
//! - `stages/` - Individual pipeline stages (seeding, chaining, extension, finalization)
//! - `modes/` - Mode-specific logic (single-end, paired-end)
//!
//! See `documents/Pipeline_Restructure_v0.8_Plan.md` for details.

// === Existing modules (will be gradually refactored) ===
pub mod batch_extension; // Cross-read batched extension (depends on linear types)
pub mod chaining;
pub mod coordinates;
pub mod finalization;
pub mod index;
pub mod mem;
pub mod mem_opt;
pub mod paired;
pub mod pipeline;
pub mod region;
pub mod seeding;
pub mod single_end;

// === New modular structure (v0.8.0) ===
// These modules are being built incrementally. They will be wired in
// once all stages are implemented and tested.
pub mod modes;
pub mod orchestrator;
pub mod stages;
