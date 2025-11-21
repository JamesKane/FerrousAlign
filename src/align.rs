// bwa-mem2-rust/src/align.rs

// Import BwaIndex and MemOpt
use crate::banded_swa::{BandedPairWiseSW, merge_cigar_operations};
use crate::fm_index::{CP_SHIFT, CpOcc, backward_ext, forward_ext, get_occ};
use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;
use crate::utils::hash_64;













// ============================================================================
// ALIGNMENT SCORING AND QUALITY ASSESSMENT
// ============================================================================
//
// This section contains functions for:
// - Overlap detection between alignments
// - Chain scoring and filtering
// - MAPQ (mapping quality) calculation
// - Secondary alignment marking
// - Divergence estimation
//
// These functions implement the core scoring logic from C++ bwa-mem2
// (bwamem.cpp, bwamem_pair.cpp)
// ============================================================================

// ----------------------------------------------------------------------------
// Overlap Detection
// ----------------------------------------------------------------------------








// ============================================================================
// SMITH-WATERMAN ALIGNMENT EXECUTION
// ============================================================================
//
// This section contains structures and functions for executing Smith-Waterman
// alignment with SIMD optimization and adaptive batch sizing
// ============================================================================



#[cfg(test)]
mod tests {
    use super::Alignment;
    use super::sam_flags;
    use crate::align::SMEM;
    use crate::fm_index::{backward_ext, popcount64};
    use crate::index::BwaIndex;
    use std::path::Path;

    #[test]
    fn test_popcount64_neon() {
        // Test the hardware-optimized popcount implementation
        // This ensures our NEON implementation matches the software version

        // Test basic cases
        assert_eq!(popcount64(0), 0);
        assert_eq!(popcount64(1), 1);
        assert_eq!(popcount64(0xFFFFFFFFFFFFFFFF), 64);
        assert_eq!(popcount64(0x8000000000000000), 1);

        // Test various bit patterns
        assert_eq!(popcount64(0b1010101010101010), 8);
        assert_eq!(popcount64(0b11111111), 8);
        assert_eq!(popcount64(0xFF00FF00FF00FF00), 32);
        assert_eq!(popcount64(0x0F0F0F0F0F0F0F0F), 32);

        // Test random patterns that match expected popcount
        assert_eq!(popcount64(0x123456789ABCDEF0), 32);
        assert_eq!(popcount64(0xAAAAAAAAAAAAAAAA), 32); // Alternating bits
        assert_eq!(popcount64(0x5555555555555555), 32); // Alternating bits (complement)
    }

    #[test]
    fn test_backward_ext() {
        let prefix = Path::new("test_data/test_ref.fa");

        // Skip if test data doesn't exist
        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
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
        let new_smem = backward_ext(&bwa_idx, smem, 0); // 0 is 'A'
        assert_ne!(new_smem.interval_size, 0);
    }

    #[test]
    fn test_backward_ext_multiple_bases() {
        // Test backward extension with all four bases
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_multiple_bases - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_multiple_bases - could not load index");
                return;
            }
        };

        // Start with full range
        let initial_smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Test extending with each base
        for base in 0..4 {
            let extended = super::backward_ext(&bwa_idx, initial_smem, base);

            // Extended range should be smaller or equal to initial range
            assert!(
                extended.interval_size <= initial_smem.interval_size,
                "Extended range size {} should be <= initial size {} for base {}",
                extended.interval_size,
                initial_smem.interval_size,
                base
            );

            // k should be within bounds
            assert!(
                extended.bwt_interval_start < bwa_idx.bwt.seq_len,
                "Extended k should be within sequence length"
            );
        }
    }

    #[test]
    fn test_backward_ext_chain() {
        // Test chaining multiple backward extensions (like building a seed)
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_chain - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_chain - could not load index");
                return;
            }
        };

        // Start with full range
        let mut smem = SMEM {
            bwt_interval_start: 0,
            interval_size: bwa_idx.bwt.seq_len,
            ..Default::default()
        };

        // Build a seed by extending with ACGT
        let bases = vec![0u8, 1, 2, 3]; // ACGT
        let mut prev_s = smem.interval_size;

        for (i, &base) in bases.iter().enumerate() {
            smem = super::backward_ext(&bwa_idx, smem, base);

            // Range should generally get smaller (or stay same) with each extension
            // (though it could stay the same if the pattern is very common)
            assert!(
                smem.interval_size <= prev_s,
                "After extension {}, range size {} should be <= previous {}",
                i,
                smem.interval_size,
                prev_s
            );

            prev_s = smem.interval_size;

            // If range becomes 0, we can't extend further
            if smem.interval_size == 0 {
                break;
            }
        }
    }

    #[test]
    fn test_backward_ext_zero_range() {
        // Test backward extension when starting with zero range
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_backward_ext_zero_range - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_backward_ext_zero_range - could not load index");
                return;
            }
        };

        let smem = SMEM {
            bwt_interval_start: 0,
            interval_size: 0, // Zero range
            ..Default::default()
        };

        let extended = super::backward_ext(&bwa_idx, smem, 0);

        // Extending a zero range should still give zero range
        assert_eq!(
            extended.interval_size, 0,
            "Extending zero range should give zero range"
        );
    }

    #[test]
    fn test_smem_structure() {
        // Test SMEM structure creation and defaults
        let smem1 = SMEM {
            read_id: 0,
            query_start: 10,
            query_end: 20,
            bwt_interval_start: 5,
            bwt_interval_end: 15,
            interval_size: 10,
            is_reverse_complement: false,
        };

        assert_eq!(smem1.query_start, 10);
        assert_eq!(smem1.query_end, 20);
        assert_eq!(smem1.interval_size, 10);

        // Test default
        let smem2 = SMEM::default();
        assert_eq!(smem2.read_id, 0);
        assert_eq!(smem2.query_start, 0);
        assert_eq!(smem2.query_end, 0);
    }

    // NOTE: Base encoding tests moved to tests/session30_regression_tests.rs
    // This reduces clutter in production code files

    #[test]
    fn test_get_sa_entry_basic() {
        // This test requires an actual index file to be present
        // We'll use a simple test to verify the function doesn't crash
        let prefix = Path::new("test_data/test_ref.fa");

        // Only run if test data exists
        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_basic - could not load index");
                return;
            }
        };

        // Test getting SA entry at position 0 (should return a valid reference position)
        let sa_entry = super::get_sa_entry(&bwa_idx, 0);

        // SA entry should be within the reference sequence length
        assert!(
            sa_entry < bwa_idx.bwt.seq_len,
            "SA entry {} should be less than seq_len {}",
            sa_entry,
            bwa_idx.bwt.seq_len
        );
    }

    #[test]
    fn test_get_sa_entry_sampled_position() {
        // Test getting SA entry at a sampled position (divisible by sa_intv)
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_sampled_position - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_sampled_position - could not load index");
                return;
            }
        };

        // Test at a sampled position (should directly lookup in SA array)
        let sampled_pos = bwa_idx.bwt.sa_sample_interval as u64;
        let sa_entry = super::get_sa_entry(&bwa_idx, sampled_pos);

        assert!(
            sa_entry < bwa_idx.bwt.seq_len,
            "SA entry at sampled position should be within sequence length"
        );
    }

    #[test]
    fn test_get_sa_entry_multiple_positions() {
        // Test getting SA entries for multiple positions
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_multiple_positions - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_multiple_positions - could not load index");
                return;
            }
        };

        // Test several positions
        let test_positions = vec![0u64, 1, 10, 100];

        for pos in test_positions {
            if pos >= bwa_idx.bwt.seq_len {
                continue;
            }

            let sa_entry = super::get_sa_entry(&bwa_idx, pos);

            // All SA entries should be valid (within sequence length)
            assert!(
                sa_entry < bwa_idx.bwt.seq_len,
                "SA entry for pos {} should be within sequence length",
                pos
            );
        }
    }

    #[test]
    fn test_get_sa_entry_consistency() {
        // Test that get_sa_entry returns consistent results
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_sa_entry_consistency - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_sa_entry_consistency - could not load index");
                return;
            }
        };

        let pos = 5u64;
        let sa_entry1 = super::get_sa_entry(&bwa_idx, pos);
        let sa_entry2 = super::get_sa_entry(&bwa_idx, pos);

        // Same position should always return same SA entry
        assert_eq!(
            sa_entry1, sa_entry2,
            "get_sa_entry should return consistent results for the same position"
        );
    }

    #[test]
    fn test_get_bwt_basic() {
        // Test get_bwt function
        let prefix = Path::new("test_data/test_ref.fa");

        if !prefix.exists() {
            eprintln!("Skipping test_get_bwt_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_get_bwt_basic - could not load index");
                return;
            }
        };

        // Test getting BWT at various positions
        for pos in 0..10u64 {
            let bwt_result = super::get_bwt(&bwa_idx, pos);

            // Either we get a valid position or None (sentinel)
            if let Some(new_pos) = bwt_result {
                assert!(
                    new_pos < bwa_idx.bwt.seq_len,
                    "BWT position should be within sequence length"
                );
            }
            // If None, we hit the sentinel - that's ok
        }
    }

    #[test]
    fn test_batched_alignment_infrastructure() {
        // Test that the batched alignment infrastructure works correctly
        use crate::banded_swa::BandedPairWiseSW;

        let sw_params = BandedPairWiseSW::new(
            4,
            2,
            4,
            2,
            100,
            0,
            5,
            5,
            super::DEFAULT_SCORING_MATRIX,
            2,
            -4,
        );

        // Create test alignment jobs
        let query1 = vec![0u8, 1, 2, 3]; // ACGT
        let target1 = vec![0u8, 1, 2, 3]; // ACGT (perfect match)

        let query2 = vec![0u8, 0, 1, 1]; // AACC
        let target2 = vec![0u8, 0, 1, 1]; // AACC (perfect match)

        let jobs = vec![
            super::AlignmentJob {
                seed_idx: 0,
                query: query1.clone(),
                target: target1.clone(),
                band_width: 10,
                query_offset: 0, // Test: align from start
                direction: None, // Legacy test mode
                seed_len: 4,     // Actual sequence length (4bp queries)
            },
            super::AlignmentJob {
                seed_idx: 1,
                query: query2.clone(),
                target: target2.clone(),
                band_width: 10,
                query_offset: 0, // Test: align from start
                direction: None, // Legacy test mode
                seed_len: 4,     // Actual sequence length (4bp queries)
            },
        ];

        // Test scalar execution
        let scalar_results = super::execute_scalar_alignments(&sw_params, &jobs);
        assert_eq!(
            scalar_results.len(),
            2,
            "Should return 2 results for 2 jobs"
        );
        assert!(
            !scalar_results[0].1.is_empty(),
            "First CIGAR should not be empty"
        );
        assert!(
            !scalar_results[1].1.is_empty(),
            "Second CIGAR should not be empty"
        );

        // Test batched execution
        let batched_results = super::execute_batched_alignments(&sw_params, &jobs);
        assert_eq!(
            batched_results.len(),
            2,
            "Should return 2 results for 2 jobs"
        );

        // Results should be identical (CIGARs and scores)
        assert_eq!(
            scalar_results[0].0, batched_results[0].0,
            "Scores should match"
        );
        assert_eq!(
            scalar_results[0].1, batched_results[0].1,
            "CIGARs should match"
        );
        assert_eq!(
            scalar_results[1].0, batched_results[1].0,
            "Scores should match"
        );
        assert_eq!(
            scalar_results[1].1, batched_results[1].1,
            "CIGARs should match"
        );
    }

    #[test]
    fn test_generate_seeds_basic() {
        use crate::mem_opt::MemOpt;
        use std::path::Path;

        let prefix = Path::new("test_data/test_ref.fa");
        if !prefix.exists() {
            eprintln!("Skipping test_generate_seeds_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_generate_seeds_basic - could not load index");
                return;
            }
        };

        let mut opt = MemOpt::default();
        opt.min_seed_len = 10; // Ensure our seed is long enough

        let query_name = "test_query";
        let query_seq = b"ACGTACGTACGT"; // 12bp
        let query_qual = "IIIIIIIIIIII";

        // Test doesn't need real MD tags, use empty pac_data
        let pac_data: &[u8] = &[];
        let alignments =
            super::generate_seeds(&bwa_idx, pac_data, query_name, query_seq, query_qual, &opt);

        assert!(
            !alignments.is_empty(),
            "Expected at least one alignment for a matching query"
        );

        let primary_alignment = alignments
            .iter()
            .find(|a| a.flag & sam_flags::SECONDARY == 0);
        assert!(primary_alignment.is_some(), "Expected a primary alignment");

        let pa = primary_alignment.unwrap();
        assert_eq!(pa.ref_name, "test_sequence");
        assert!(
            pa.score > 0,
            "Expected a positive score for a good match, got {}",
            pa.score
        );
        assert!(pa.pos < 60, "Position should be within reference length");
        assert_eq!(pa.cigar_string(), "12M", "Expected a perfect match CIGAR");
    }

    #[test]
    fn test_hard_clipping_for_supplementary() {
        // Test that soft clips (S) are converted to hard clips (H) for supplementary alignments
        // Per bwa-mem2 behavior (bwamem.cpp:1585-1586)

        // Create a primary alignment with soft clips
        let primary = Alignment {
            query_name: "read1".to_string(),
            flag: 0, // Primary alignment
            ref_name: "chr1".to_string(),
            ref_id: 0,
            pos: 100,
            mapq: 60,
            score: 100,
            cigar: vec![(b'S', 5), (b'M', 50), (b'S', 10)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: "A".repeat(65),
            qual: "I".repeat(65),
            tags: vec![],
            query_start: 0,
            query_end: 65,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
        };

        // Create a supplementary alignment with soft clips
        let supplementary = Alignment {
            query_name: "read1".to_string(),
            flag: sam_flags::SUPPLEMENTARY,
            ref_name: "chr2".to_string(),
            ref_id: 1,
            pos: 200,
            mapq: 30,
            score: 80,
            cigar: vec![(b'S', 5), (b'M', 50), (b'S', 10)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: "A".repeat(65),
            qual: "I".repeat(65),
            tags: vec![],
            query_start: 0,
            query_end: 65,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
        };

        // Create a secondary alignment with soft clips
        let secondary = Alignment {
            query_name: "read1".to_string(),
            flag: sam_flags::SECONDARY,
            ref_name: "chr3".to_string(),
            ref_id: 2,
            pos: 300,
            mapq: 0,
            score: 70,
            cigar: vec![(b'S', 5), (b'M', 50), (b'S', 10)],
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: "A".repeat(65),
            qual: "I".repeat(65),
            tags: vec![],
            query_start: 0,
            query_end: 65,
            seed_coverage: 50,
            hash: 0,
            frac_rep: 0.0,
        };

        // Verify CIGAR strings
        assert_eq!(
            primary.cigar_string(),
            "5S50M10S",
            "Primary alignment should keep soft clips (S)"
        );

        assert_eq!(
            supplementary.cigar_string(),
            "5H50M10H",
            "Supplementary alignment should convert S to H"
        );

        assert_eq!(
            secondary.cigar_string(),
            "5S50M10S",
            "Secondary alignment should keep soft clips (S)"
        );

        println!("✅ Hard clipping test passed!");
        println!("   Primary:       {}", primary.cigar_string());
        println!("   Supplementary: {}", supplementary.cigar_string());
        println!("   Secondary:     {}", secondary.cigar_string());
    }
}
