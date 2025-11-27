//! Alignment pipelines for different reference structures.
//!
//! Each pipeline implements a complete alignment workflow:
//! - `linear`: BWA-MEM style alignment to linear reference genomes
//! - `graph`: (Future) Pangenome graph alignment

pub mod linear;
pub mod graph;
