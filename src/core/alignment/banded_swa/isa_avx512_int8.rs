//! AVX‑512 int8 thin wrapper (64 lanes)
#![cfg(target_arch = "x86_64")]

use crate::core::alignment::banded_swa::OutScore;
use crate::generate_swa_entry;
use crate::generate_swa_entry_soa;
use super::engines::SwEngine512; // Updated path

generate_swa_entry!(
    name = simd_banded_swa_batch64,
    width = 64,
    engine = SwEngine512,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw,avx512f",
);

generate_swa_entry_soa!(
    name = simd_banded_swa_batch64_soa,
    width = 64,
    engine = SwEngine512,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw",
);