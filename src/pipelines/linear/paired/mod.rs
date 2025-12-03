//! Paired-end processing support modules.
//!
//! These modules provide the core algorithms for paired-end alignment:
//! - Insert size estimation and statistics
//! - Mate rescue (Smith-Waterman on unmapped mates)
//! - Pair scoring and selection (AoS - required for index correctness)
//!
//! The main orchestration is now in `orchestrator::PairedEndOrchestrator`.

pub mod insert_size;
pub mod mate_rescue;
pub mod pairing_aos; // AoS pairing - REQUIRED for per-read index correctness
