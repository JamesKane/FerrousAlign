//! Banded Smith–Waterman: shared kernel + thin ISA wrappers + dispatch

pub mod shared;      // pad_batch, soa_transform (no‑op when pre‑SoA), pack_outscores
pub mod kernel;      // SwSimd trait + sw_kernel + KernelParams
mod engines;         // SwEngine128/256/512 adapters (no public API)
pub mod scalar;
pub mod types;
pub mod scoring;
pub mod utils;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod isa_sse_neon;
#[cfg(target_arch = "x86_64")]
pub mod isa_avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod isa_avx512_int8;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod isa_avx512_int16;

pub mod dispatch;

// Re‑exports (keep external API stable)
pub use dispatch::*;
pub use kernel::KernelParams;
pub use types::*;
pub use scoring::bwa_fill_scmat;
pub use utils::{reverse_cigar, reverse_sequence};