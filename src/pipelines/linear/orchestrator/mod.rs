//! Pipeline orchestration layer
//!
//! Orchestrators coordinate the execution of pipeline stages, handling:
//! - Batch loading and iteration
//! - Stage sequencing
//! - AoS/SoA representation transitions (for paired-end)
//! - Statistics aggregation
//! - Output writing
//!
//! # Architecture Note
//!
//! The paired-end pipeline requires a **hybrid AoS/SoA architecture**:
//! - SoA for compute-heavy stages (alignment, mate rescue) - SIMD benefits
//! - AoS for logic-heavy stages (pairing, output) - correctness requirement
//!
//! See `documents/Pipeline_Restructure_v0.8_Plan.md` for details.

// Submodules will be added in Phase 1
// pub mod single_end;
// pub mod paired_end;
// pub mod statistics;
// pub mod conversions;
