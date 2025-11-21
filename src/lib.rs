// Enable unstable features for AVX-512 support (requires nightly Rust)
#![cfg_attr(feature = "avx512", feature(stdarch_x86_avx512))]
#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]

pub mod alignment;
pub mod bntseq;
pub mod bwa_index;
pub mod bwt;
pub mod defaults;
pub mod fastq_reader; // FASTQ reader using bio::io::fastq (used for query reads)
pub mod fm_index; // FM-Index operations (BWT search, occurrence counting)
pub mod index; // Index management (BwaIndex loading/dumping)
pub mod insert_size; // Insert size statistics
pub mod kseq; // Used for FASTA reference reading during index building
pub mod mate_rescue; // Mate rescue using Smith-Waterman
pub mod mem;
pub mod mem_opt;
pub mod paired_end; // Paired-end read processing
pub mod pairing; // Paired-end alignment scoring
pub mod simd;
pub mod simd_abstraction;
pub mod single_end; // Single-end read processing
pub mod utils;

// Test modules
#[cfg(test)]
#[path = "fastq_reader_test.rs"]
mod fastq_reader_test;

// Note: SAIS implementation removed - we use the `bio` crate's suffix array construction instead
