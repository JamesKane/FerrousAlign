pub mod bwt;
pub mod bntseq;
pub mod kseq;  // Used for FASTA reference reading during index building
pub mod fastq_reader;  // FASTQ reader using bio::io::fastq (used for query reads)
pub mod align;
pub mod mem;
pub mod mem_opt;
pub mod banded_swa;
pub mod utils;
pub mod bwa_index;
pub mod simd_abstraction;

// AVX2-specific SIMD implementations (x86_64 only)
#[cfg(target_arch = "x86_64")]
pub mod banded_swa_avx2;

// AVX-512-specific SIMD implementations (x86_64 only)
#[cfg(target_arch = "x86_64")]
pub mod banded_swa_avx512;

// Test modules
#[cfg(test)]
#[path = "fastq_reader_test.rs"]
mod fastq_reader_test;

// Note: SAIS implementation removed - we use the `bio` crate's suffix array construction instead