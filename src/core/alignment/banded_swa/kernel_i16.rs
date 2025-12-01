use crate::alignment::banded_swa::OutScore;

// kernel_i16.rs
pub trait SwSimd16: Copy {
    type V16: Copy;
    const LANES: usize; // 8 (SSE/NEON), 16 (AVX2), 32 (AVX-512)
    unsafe fn setzero_epi16() -> Self::V16;
    unsafe fn set1_epi16(x: i16) -> Self::V16;
    unsafe fn loadu_epi16(ptr: *const i16) -> Self::V16;
    unsafe fn storeu_epi16(ptr: *mut i16, v: Self::V16);
    unsafe fn adds_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn subs_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn max_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn min_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn cmpeq_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn cmpgt_epi16(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn and_si128(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn or_si128(a: Self::V16, b: Self::V16) -> Self::V16;
    unsafe fn andnot_si128(a: Self::V16, b: Self::V16) -> Self::V16; // ~a & b
}

pub struct KernelParams16<'a> {
    pub batch: &'a [(i32,&'a [u8],i32,&'a [u8],i32,i32)],
    pub query_soa: &'a [u8],
    pub target_soa: &'a [u8],
    pub qlen: &'a [i8], pub tlen: &'a [i8],
    pub h0:   &'a [i16], pub w: &'a [i8],
    pub max_qlen: i32, pub max_tlen: i32,
    pub o_del: i32, pub e_del: i32, pub o_ins: i32, pub e_ins: i32,
    pub zdrop: i32,
    pub mat: &'a [i8; 25], pub m: i32,
}

#[inline]
pub unsafe fn sw_kernel_i16<const W: usize, E: SwSimd16>(p: &KernelParams16<'_>) -> Vec<OutScore> {
    // Copy the i8 kernel’s structure:
    // - i16 DP buffers (H/E/F)
    // - compare-and-blend scoring from 2-bit bases → i16 scores (match/mismatch)
    // - gap updates with adds/subs_epi16, clamp to 0 via max with zero
    // - per-lane band/length masks identical to i8 path; mask on store
    // - strict max update (cmpgt only) with positions
    // - z-drop row scan (i16); terminate lanes accordingly
    
    return vec![];
}