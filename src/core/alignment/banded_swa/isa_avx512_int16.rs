//! AVX‑512 int16 path (placeholder/manual for now)
// Keep minimal until an i16 shared kernel lands

#![cfg(target_arch = "x86_64")]

use crate::alignment::banded_swa::engines16::SwEngine512_16;
use crate::generate_swa_entry_i16_soa;

generate_swa_entry_i16_soa!(
    name = simd_banded_swa_batch32_int16_soa,
    width = 32,
    engine = SwEngine512_16,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw,avx512f",
);
