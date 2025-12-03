//! Seeding module for SMEM extraction.
//!
//! This module implements Super Maximal Exact Match (SMEM) extraction using
//! bidirectional FM-index search. SMEMs are the foundation of the BWA-MEM
//! alignment algorithm.
//!
//! # Module Organization
//!
//! - `types` - Core data structures (`Seed`, `SMEM`, `SoASeedBatch`)
//! - `smem` - SMEM generation algorithms (bidirectional search)
//! - `bwt` - BWT and suffix array helper functions
//! - `collection` - Batch seed collection (`find_seeds_batch`)
//!
//! # Algorithm Overview
//!
//! The seeding process consists of three passes:
//!
//! 1. **Initial SMEM generation**: Bidirectional FM-index search to find
//!    all maximal exact matches
//!
//! 2. **Re-seeding**: For long unique matches, re-seed from the middle
//!    to find split alignments (chimeric reads)
//!
//! 3. **Forward-only seeding** (optional): Additional seeding pass for
//!    reads with many mismatches

mod bwt;
mod collection;
mod smem;
mod types;

// Re-export types
pub use types::{Seed, SoASeedBatch, SoAEncodedQueryBatch, SMEM};

// Re-export SoAReadBatch from core::io
pub use crate::core::io::soa_readers::SoAReadBatch;

// Re-export BWT helper functions
pub use bwt::{get_bwt, get_bwt_base_from_cp_occ, get_sa_entries, get_sa_entry};

// Re-export SMEM generation functions
pub use smem::{forward_only_seed_strategy, generate_smems_for_strand, generate_smems_from_position};

// Re-export collection functions
pub use collection::find_seeds_batch;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::linear::index::fm_index::backward_ext;
    use crate::pipelines::linear::index::index::BwaIndex;
    use std::path::Path;

    #[test]
    fn test_backward_ext() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext - could not load index");
                return;
            }
        };

        let smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };
        let new_smem = backward_ext(&bwa_idx, smem, 0);
        assert_ne!(new_smem.interval_size, 0);
    }

    #[test]
    fn test_backward_ext_multiple_bases() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_multiple_bases - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_multiple_bases - could not load index");
                return;
            }
        };

        let initial_smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        for base in 0..4 {
            let extended = backward_ext(&bwa_idx, initial_smem, base);

            assert!(
                extended.interval_size <= initial_smem.interval_size,
                "Extended range size {} should be <= initial size {} for base {}",
                extended.interval_size,
                initial_smem.interval_size,
                base
            );

            assert!(
                extended.bwt_interval_start < bwa_idx.bwt.seq_len,
                "Extended k should be within sequence length"
            );
        }
    }

    #[test]
    fn test_backward_ext_chain() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_chain - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_chain - could not load index");
                return;
            }
        };

        let mut smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        let bases = [0u8, 1, 2, 3];
        let mut prev_s = smem.interval_size;

        for (i, &base) in bases.iter().enumerate() {
            smem = backward_ext(&bwa_idx, smem, base);

            assert!(
                smem.interval_size <= prev_s,
                "After extension {}, range size {} should be <= previous {}",
                i,
                smem.interval_size,
                prev_s
            );

            prev_s = smem.interval_size;

            if smem.interval_size == 0 {
                break;
            }
        }
    }

    #[test]
    fn test_backward_ext_zero_range() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_zero_range - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_zero_range - could not load index");
                return;
            }
        };

        let smem = SMEM {
            bwt_interval_start: 0,
            interval_size: 0,
            ..Default::default()
        };

        let extended = backward_ext(&bwa_idx, smem, 0);

        assert_eq!(
            extended.interval_size, 0,
            "Extending zero range should give zero range"
        );
    }

    #[test]
    fn test_get_sa_entry_multiple_positions() {
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_multiple_positions - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_multiple_positions - could not load index");
                return;
            }
        };

        let test_positions = vec![0u64, 1, 10, 100];

        for pos in test_positions {
            if pos >= bwa_idx.bwt.seq_len {
                continue;
            }

            let sa_entry = get_sa_entry(&bwa_idx, pos);

            assert!(
                sa_entry < bwa_idx.bwt.seq_len,
                "SA entry for pos {pos} should be within sequence length"
            );
        }
    }
}
