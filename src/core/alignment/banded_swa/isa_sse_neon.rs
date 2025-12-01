

use crate::core::alignment::banded_swa::OutScore;
use super::engines::SwEngine128;
use crate::core::alignment::banded_swa::KernelParams;
use crate::core::alignment::banded_swa::kernel::sw_kernel;

use super::engines16::SwEngine128_16;
use crate::generate_swa_entry_i16;
use crate::generate_swa_entry_i16_soa;


use super::shared::{pad_batch, soa_transform}; // Updated path



/// SSE/NEON-optimized banded Smith-Waterman for batches of up to 16 alignments
/// Processes 16 alignments in parallel (baseline SIMD for all platforms)
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn simd_banded_swa_batch16(
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    zdrop: i32,
    mat: &[i8; 25],
    m: i32,
) -> Vec<OutScore> {
    const W: usize = 16;
    const MAX_SEQ_LEN: usize = 128; // keep i8 limits aligned with AVX2

    let (qlen, tlen, h0, w_arr, max_qlen, max_tlen, padded) = pad_batch::<W>(batch);
    let (query_soa, target_soa) = soa_transform::<W, MAX_SEQ_LEN>(&padded);

    let params = KernelParams {
        batch,
        query_soa: &query_soa,
        target_soa: &target_soa,
        qlen: &qlen,
        tlen: &tlen,
        h0: &h0,
        w: &w_arr,
        max_qlen,
        max_tlen,
        o_del,
        e_del,
        o_ins,
        e_ins,
        zdrop,
        mat,
        m,
    };

    sw_kernel::<W, SwEngine128>(&params)
}



use crate::generate_swa_entry_soa; // This macro is exported at crate root by shared module

#[cfg(target_arch = "x86_64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
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

#[cfg(target_arch = "aarch64")]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch16_soa,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(target_arch = "aarch64"),
    target_feature = "neon",
);