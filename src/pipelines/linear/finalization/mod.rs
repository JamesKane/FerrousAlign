//! Alignment finalization module.
//!
//! This module handles the final stages of alignment processing:
//! - MAPQ calculation
//! - Secondary/supplementary marking
//! - XA/SA tag generation
//! - Redundant alignment removal
//!
//! ## Module Organization
//!
//! - `sam_flags` - SAM flag bit constants
//! - `alignment` - Alignment struct and methods
//! - `redundancy` - Redundant alignment removal
//! - `secondary` - Secondary/supplementary marking and MAPQ
//! - `tags` - XA/SA tag generation

pub mod sam_flags;
mod alignment;
mod redundancy;
mod secondary;
mod tags;

// Re-export public types and functions
pub use alignment::Alignment;
pub use redundancy::{remove_redundant_alignments, alignment_ref_length};
pub use secondary::mark_secondary_alignments;
pub use tags::{generate_xa_tags, generate_sa_tags};

// Re-export is_alternate_contig as a free function for use in mate rescue
pub fn is_alternate_contig(ref_name: &str) -> bool {
    Alignment::is_alternate_contig(ref_name)
}
