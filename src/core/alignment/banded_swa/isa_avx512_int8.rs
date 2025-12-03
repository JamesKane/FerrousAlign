//! AVX‑512 int8 thin wrapper (64 lanes)
#![cfg(target_arch = "x86_64")]

use super::engines::SwEngine512;
use crate::generate_swa_entry_soa;

generate_swa_entry_soa!(
    name = simd_banded_swa_batch64_soa,
    width = 64,
    engine = SwEngine512,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw",
);
