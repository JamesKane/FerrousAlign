use super::engines::SwEngine128;
use super::engines16::SwEngine128_16;
use crate::generate_swa_entry_i16;
use crate::generate_swa_entry_i16_soa;
use crate::generate_swa_entry_i16_soa_with_ws;
use crate::generate_swa_entry_soa;
use crate::generate_swa_entry_soa_with_ws;

use crate::core::alignment::banded_swa::OutScore;

#[cfg(target_arch = "x86_64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "sse2",
);

#[cfg(target_arch = "x86_64")]
generate_swa_entry_soa_with_ws!(
    name = simd_banded_swa_batch16_soa_with_ws,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "sse2",
);

generate_swa_entry_i16!(
    name = simd_banded_swa_batch8_int16,
    width = 8,
    engine = SwEngine128_16,
    cfg = cfg(any(target_arch = "x86_64", target_arch = "aarch64")),
    target_feature = "",
);

generate_swa_entry_i16_soa!(
    name = simd_banded_swa_batch8_int16_soa,
    width = 8,
    engine = SwEngine128_16,
    cfg = cfg(any(target_arch = "x86_64", target_arch = "aarch64")),
    target_feature = "",
);

generate_swa_entry_i16_soa_with_ws!(
    name = simd_banded_swa_batch8_int16_soa_with_ws,
    width = 8,
    engine = SwEngine128_16,
    cfg = cfg(any(target_arch = "x86_64", target_arch = "aarch64")),
    target_feature = "",
);

#[cfg(target_arch = "aarch64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "aarch64"),
    target_feature = "neon",
);

#[cfg(target_arch = "aarch64")]
generate_swa_entry_soa_with_ws!(
    name = simd_banded_swa_batch16_soa_with_ws,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "aarch64"),
    target_feature = "neon",
);
