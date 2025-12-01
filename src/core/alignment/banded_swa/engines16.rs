use crate::alignment::banded_swa::kernel_i16::SwSimd16;
use crate::core::compute::simd_abstraction::SimdEngine;

// engines.rs (or engines16.rs)
#[derive(Copy, Clone)]
pub struct SwEngine128_16; // 8 lanes of i16 (SSE/NEON)
#[derive(Copy, Clone)]
pub struct SwEngine256_16; // 16 lanes of i16 (AVX2)

impl SwSimd16 for SwEngine128_16 {
    type V16 = <crate::core::compute::simd_abstraction::SimdEngine128 as crate::core::compute::simd_abstraction::SimdEngine>::Vec16; // 128-bit vector used for i16 ops
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn setzero_epi16() -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::setzero_epi16() }
    }
    #[inline(always)]
    unsafe fn set1_epi16(x: i16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::set1_epi16(x) }
    }
    #[inline(always)]
    unsafe fn loadu_epi16(ptr: *const i16) -> Self::V16 {
        unsafe {
            crate::core::compute::simd_abstraction::SimdEngine128::loadu_si128_16(ptr as *const _)
        }
    }
    #[inline(always)]
    unsafe fn storeu_epi16(ptr: *mut i16, v: Self::V16) {
        unsafe {
            crate::core::compute::simd_abstraction::SimdEngine128::storeu_si128_16(ptr as *mut _, v)
        }
    }
    #[inline(always)]
    unsafe fn adds_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::adds_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn subs_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::subs_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn max_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::max_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn min_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::min_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpeq_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::cmpeq_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpgt_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::cmpgt_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn and_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::and_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn or_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::or_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn andnot_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine128::andnot_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V16, b: Self::V16, mask: Self::V16) -> Self::V16 {
        unsafe {
            std::mem::transmute(
                crate::core::compute::simd_abstraction::SimdEngine128::blendv_epi8(
                    std::mem::transmute(a),
                    std::mem::transmute(b),
                    std::mem::transmute(mask),
                ),
            )
        }
    }
}

impl SwSimd16 for SwEngine256_16 {
    type V16 = <crate::core::compute::simd_abstraction::SimdEngine256 as crate::core::compute::simd_abstraction::SimdEngine>::Vec16; // 256-bit vector used for i16 ops
    const LANES: usize = 16;

    #[inline(always)]
    unsafe fn setzero_epi16() -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::setzero_epi16() }
    }
    #[inline(always)]
    unsafe fn set1_epi16(x: i16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::set1_epi16(x) }
    }
    #[inline(always)]
    unsafe fn loadu_epi16(ptr: *const i16) -> Self::V16 {
        unsafe {
            crate::core::compute::simd_abstraction::SimdEngine256::loadu_si128_16(ptr as *const _)
        }
    }
    #[inline(always)]
    unsafe fn storeu_epi16(ptr: *mut i16, v: Self::V16) {
        unsafe {
            crate::core::compute::simd_abstraction::SimdEngine256::storeu_si128_16(ptr as *mut _, v)
        }
    }
    #[inline(always)]
    unsafe fn adds_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::adds_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn subs_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::subs_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn max_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::max_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn min_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::min_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpeq_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::cmpeq_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpgt_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::cmpgt_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn and_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::and_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn or_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::or_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn andnot_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine256::andnot_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V16, b: Self::V16, mask: Self::V16) -> Self::V16 {
        unsafe {
            std::mem::transmute(
                crate::core::compute::simd_abstraction::SimdEngine256::blendv_epi8(
                    std::mem::transmute(a),
                    std::mem::transmute(b),
                    std::mem::transmute(mask),
                ),
            )
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[derive(Copy, Clone)]
pub struct SwEngine512_16; // 32 lanes of i16 (AVX-512)

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl SwSimd16 for SwEngine512_16 {
    type V16 = <crate::core::compute::simd_abstraction::SimdEngine512 as crate::core::compute::simd_abstraction::SimdEngine>::Vec16; // 512-bit vector used for i16 ops
    const LANES: usize = 32;
    #[inline(always)]
    unsafe fn setzero_epi16() -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::setzero_epi16() }
    }
    #[inline(always)]
    unsafe fn set1_epi16(x: i16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::set1_epi16(x) }
    }
    #[inline(always)]
    unsafe fn loadu_epi16(ptr: *const i16) -> Self::V16 {
        unsafe {
            crate::core::compute::simd_abstraction::SimdEngine512::loadu_si128_16(ptr as *const _)
        }
    }
    #[inline(always)]
    unsafe fn storeu_epi16(ptr: *mut i16, v: Self::V16) {
        unsafe {
            crate::core::compute::simd_abstraction::SimdEngine512::storeu_si128_16(ptr as *mut _, v)
        }
    }
    #[inline(always)]
    unsafe fn adds_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::adds_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn subs_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::subs_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn max_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::max_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn min_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::min_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpeq_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::cmpeq_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn cmpgt_epi16(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::cmpgt_epi16(a, b) }
    }
    #[inline(always)]
    unsafe fn and_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::and_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn or_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::or_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn andnot_si128(a: Self::V16, b: Self::V16) -> Self::V16 {
        unsafe { crate::core::compute::simd_abstraction::SimdEngine512::andnot_si128(a, b) }
    }
    #[inline(always)]
    unsafe fn blendv_epi8(a: Self::V16, b: Self::V16, mask: Self::V16) -> Self::V16 {
        unsafe {
            std::mem::transmute(
                crate::core::compute::simd_abstraction::SimdEngine512::blendv_epi8(
                    std::mem::transmute(a),
                    std::mem::transmute(b),
                    std::mem::transmute(mask),
                ),
            )
        }
    }
}
