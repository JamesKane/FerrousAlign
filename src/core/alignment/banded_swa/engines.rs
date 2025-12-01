use crate::core::compute::simd_abstraction as simd;
use super::kernel::SwSimd;
use crate::core::compute::simd_abstraction::SimdEngine;

/// SSE/NEON 128-bit engine adapter (16 lanes of i8)
#[derive(Copy, Clone)]
pub struct SwEngine128;

impl SwSimd for SwEngine128 {
    type V8 = <simd::SimdEngine128 as simd::SimdEngine>::Vec8;
    const LANES: usize = 16;

    #[inline(always)]
    unsafe fn setzero_epi8() -> Self::V8 {
        unsafe { simd::SimdEngine128::setzero_epi8() }
    }
    #[inline(always)]
    unsafe fn set1_epi8(x: i8) -> Self::V8 {
        unsafe { simd::SimdEngine128::set1_epi8(x) }
    }
    #[inline(always)]
    unsafe fn loadu_epi8(ptr: *const i8) -> Self::V8 {
        unsafe { simd::SimdEngine128::loadu_si128(ptr as *const _) }
    }
    #[inline(always)]
    unsafe fn storeu_epi8(ptr: *mut i8, v: Self::V8) {
        unsafe { simd::SimdEngine128::storeu_si128(ptr as *mut _, v) }
    }
    #[inline(always)]
    unsafe fn adds_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::adds_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn subs_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::subs_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn max_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::max_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn min_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::min_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::cmpeq_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::cmpgt_epi8(b, a) }
    }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::blendv_epi8(a, b, mask) }
    }
    #[inline(always)]
    unsafe fn and_si128(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::and_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn or_si128(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::or_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpgt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::cmpgt_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn min_epu8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine128::min_epu8(a, b) }
    }
}


/// AVX2 256-bit engine adapter (32 lanes of i8)
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
pub struct SwEngine256;

#[cfg(target_arch = "x86_64")]
impl SwSimd for SwEngine256 {
    type V8 = <simd::SimdEngine256 as simd::SimdEngine>::Vec8;
    const LANES: usize = 32;

    #[inline(always)]
    unsafe fn setzero_epi8() -> Self::V8 {
        unsafe { simd::SimdEngine256::setzero_epi8() }
    }
    #[inline(always)]
    unsafe fn set1_epi8(x: i8) -> Self::V8 {
        unsafe { simd::SimdEngine256::set1_epi8(x) }
    }
    #[inline(always)]
    unsafe fn loadu_epi8(ptr: *const i8) -> Self::V8 {
        unsafe { simd::SimdEngine256::loadu_si128(ptr as *const _) }
    }
    #[inline(always)]
    unsafe fn storeu_epi8(ptr: *mut i8, v: Self::V8) {
        unsafe { simd::SimdEngine256::storeu_si128(ptr as *mut _, v) }
    }
    #[inline(always)]
    unsafe fn adds_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::adds_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn subs_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::subs_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn max_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::max_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn min_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::min_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::cmpeq_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::cmpgt_epi8(b, a) }
    }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::blendv_epi8(a, b, mask) }
    }
    #[inline(always)]
    unsafe fn and_si128(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::and_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn or_si128(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::or_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpgt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::cmpgt_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn min_epu8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine256::min_epu8(a, b) }
    }
}

/// AVX-512 512-bit engine adapter (64 lanes of i8)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Copy, Clone)]
pub struct SwEngine512;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl SwSimd for SwEngine512 {
    type V8 = <simd::SimdEngine512 as simd::SimdEngine>::Vec8;
    const LANES: usize = 64;

    #[inline(always)]
    unsafe fn setzero_epi8() -> Self::V8 {
        unsafe { simd::SimdEngine512::setzero_epi8() }
    }
    #[inline(always)]
    unsafe fn set1_epi8(x: i8) -> Self::V8 {
        unsafe { simd::SimdEngine512::set1_epi8(x) }
    }
    #[inline(always)]
    unsafe fn loadu_epi8(ptr: *const i8) -> Self::V8 {
        unsafe { simd::SimdEngine512::loadu_si128(ptr as *const _) }
    }
    #[inline(always)]
    unsafe fn storeu_epi8(ptr: *mut i8, v: Self::V8) {
        unsafe { simd::SimdEngine512::storeu_si128(ptr as *mut _, v) }
    }
    #[inline(always)]
    unsafe fn adds_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::adds_epi8(a, b) }
    }
    #[inline(always)]
    unsafe fn subs_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::subs_epi8(a, b) }
    }    #[inline(always)]
    unsafe fn max_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::max_epi8(a, b) }
    }    #[inline(always)]
    unsafe fn min_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::min_epi8(a, b) }
    }    #[inline(always)]
    unsafe fn cmpeq_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::cmpeq_epi8(a, b) }
    }    #[inline(always)]
    unsafe fn cmplt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::cmpgt_epi8(b, a) }
    }    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V8, b: Self::V8, mask: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::blendv_epi8(a, b, mask) }
    }    #[inline(always)]
    unsafe fn and_si128(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::and_si128(a, b) }
    }    #[inline(always)]
    unsafe fn or_si128(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::or_si128(a, b) }
    }    #[inline(always)]
    unsafe fn cmpgt_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::cmpgt_epi8(a, b) }
    }    #[inline(always)]
    unsafe fn min_epu8(a: Self::V8, b: Self::V8) -> Self::V8 {
        unsafe { simd::SimdEngine512::min_epu8(a, b) }
    }
}
