//! Linear alignment pipeline (BWA-MEM style).
//!
//! This pipeline implements read alignment to linear reference genomes using
//! the BWA-MEM algorithm: seeding via FM-Index, chaining, and extension.

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
