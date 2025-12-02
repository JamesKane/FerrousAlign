// tests/intrinsics.rs

#[cfg(all(test, target_arch = "aarch64"))]
mod aarch64_tests {
    use ferrous_align::compute::simd_abstraction::SimdEngine;
    use ferrous_align::compute::simd_abstraction::engine128::SimdEngine128;
    use ferrous_align::compute::simd_abstraction::portable_intrinsics::{
        _mm_add_epi16, _mm_blendv_epi8, _mm_cmpgt_epi16, _mm_max_epi16, _mm_set1_epi16,
        _mm_setzero_si128, _mm_slli_si128_var, _mm_srli_si128_var, _mm_sub_epi16,
    };
    use ferrous_align::compute::simd_abstraction::types::__m128i;
    use ferrous_align::{mm_alignr_epi8, mm_srli_si128};
    use std::arch::aarch64;

    #[test]
    fn test_mm_alignr_epi8_lt16() {
        unsafe {
            let a_data: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let b_data: [i8; 16] = [
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr() as *const u8));

            // Shift right by 4 bytes.
            let result = mm_alignr_epi8!(a, b, 4);

            let mut actual: [i8; 16] = [0; 16];
            aarch64::vst1q_s8(actual.as_mut_ptr(), result.as_s8());

            // Expected result for _mm_alignr_epi8(a, b, 4)
            // This is the last 12 bytes of b, followed by the first 4 bytes of a.
            let expected: [i8; 16] = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3];
            assert_eq!(actual, expected, "test_mm_alignr_epi8_lt16");
        }
    }

    #[test]
    fn test_mm_srli_si128() {
        unsafe {
            let a_data: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));

            let result = _mm_srli_si128_var(a, 4);

            let mut actual: [i8; 16] = [0; 16];
            aarch64::vst1q_s8(actual.as_mut_ptr(), result.as_s8());

            let expected: [i8; 16] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0];
            assert_eq!(actual, expected, "test_mm_srli_si128");
        }
    }

    #[test]
    fn test_mm_slli_si128() {
        unsafe {
            let a_data: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));

            let result = _mm_slli_si128_var(a, 4);

            let mut actual: [i8; 16] = [0; 16];
            aarch64::vst1q_s8(actual.as_mut_ptr(), result.as_s8());

            let expected: [i8; 16] = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
            assert_eq!(actual, expected, "test_mm_slli_si128");
        }
    }

    #[test]
    fn test_mm_max_epi16() {
        unsafe {
            let a_data: [i16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let b_data: [i16; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr() as *const u8));

            let result = _mm_max_epi16(a, b);

            let mut actual: [i16; 8] = [0; 8];
            aarch64::vst1q_s16(actual.as_mut_ptr(), result.as_s16());

            let expected: [i16; 8] = [7, 6, 5, 4, 4, 5, 6, 7];
            assert_eq!(actual, expected, "test_mm_max_epi16");
        }
    }

    #[test]
    fn test_mm_add_epi16() {
        unsafe {
            let a_data: [i16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let b_data: [i16; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr() as *const u8));

            let result = _mm_add_epi16(a, b);

            let mut actual: [i16; 8] = [0; 8];
            aarch64::vst1q_s16(actual.as_mut_ptr(), result.as_s16());

            let expected: [i16; 8] = [7, 7, 7, 7, 7, 7, 7, 7];
            assert_eq!(actual, expected, "test_mm_add_epi16");
        }
    }

    #[test]
    fn test_mm_sub_epi16() {
        unsafe {
            let a_data: [i16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let b_data: [i16; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr() as *const u8));

            let result = _mm_sub_epi16(a, b);

            let mut actual: [i16; 8] = [0; 8];
            aarch64::vst1q_s16(actual.as_mut_ptr(), result.as_s16());

            let expected: [i16; 8] = [-7, -5, -3, -1, 1, 3, 5, 7];
            assert_eq!(actual, expected, "test_mm_sub_epi16");
        }
    }

    #[test]
    fn test_mm_cmpgt_epi16() {
        unsafe {
            let a_data: [i16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let b_data: [i16; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr() as *const u8));

            let result = _mm_cmpgt_epi16(a, b);

            let mut actual: [i16; 8] = [0; 8];
            aarch64::vst1q_s16(actual.as_mut_ptr(), result.as_s16());

            let expected: [i16; 8] = [0, 0, 0, 0, -1, -1, -1, -1];
            assert_eq!(actual, expected, "test_mm_cmpgt_epi16");
        }
    }

    #[test]
    fn test_mm_blendv_epi8() {
        unsafe {
            let a_data: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let b_data: [i8; 16] = [
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ];
            let mask_data: [i8; 16] = [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr() as *const u8));
            let mask = __m128i(aarch64::vld1q_u8(mask_data.as_ptr() as *const u8));

            let result = _mm_blendv_epi8(a, b, mask);

            let mut actual: [i8; 16] = [0; 16];
            aarch64::vst1q_s8(actual.as_mut_ptr(), result.as_s8());

            let expected: [i8; 16] = [16, 1, 18, 3, 20, 5, 22, 7, 24, 9, 26, 11, 28, 13, 30, 15];
            assert_eq!(actual, expected, "test_mm_blendv_epi8");
        }
    }

    #[test]
    fn test_movemask_epi8() {
        unsafe {
            let a_data: [i8; 16] = [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));

            let result = SimdEngine128::movemask_epi8(a);

            let expected = 0b0101010101010101;
            assert_eq!(result, expected, "test_movemask_epi8");
        }
    }

    #[test]
    fn test_shuffle_epi8() {
        unsafe {
            let a_data: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let b_data: [i8; 16] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr() as *const u8));

            let result = SimdEngine128::shuffle_epi8(a, b);

            let mut actual: [i8; 16] = [0; 16];
            aarch64::vst1q_s8(actual.as_mut_ptr(), result.as_s8());

            let expected: [i8; 16] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
            assert_eq!(actual, expected, "test_shuffle_epi8");
        }
    }

    #[test]
    fn test_shuffle_epi8_zeroing() {
        unsafe {
            let a_data: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let b_data: [i8; 16] = [-1, 14, -1, 12, -1, 10, -1, 8, -1, 6, -1, 4, -1, 2, -1, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr() as *const u8));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr() as *const u8));

            let result = SimdEngine128::shuffle_epi8(a, b);

            let mut actual: [i8; 16] = [0; 16];
            aarch64::vst1q_s8(actual.as_mut_ptr(), result.as_s8());

            let expected: [i8; 16] = [0, 14, 0, 12, 0, 10, 0, 8, 0, 6, 0, 4, 0, 2, 0, 0];
            assert_eq!(actual, expected, "test_shuffle_epi8_zeroing");
        }
    }

    #[test]
    fn test_cmpgt_epu8() {
        unsafe {
            // Test unsigned comparison: 200 > 100 (true), 50 > 100 (false)
            let a_data: [u8; 16] = [200, 50, 255, 0, 128, 127, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7];
            let b_data: [u8; 16] = [100, 100, 254, 1, 127, 128, 99, 101, 0, 0, 0, 0, 0, 0, 0, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr()));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr()));

            let result = SimdEngine128::cmpgt_epu8(a, b);

            let mut actual: [u8; 16] = [0; 16];
            aarch64::vst1q_u8(actual.as_mut_ptr(), result.0);

            // Expected: 0xFF where a > b (unsigned), 0x00 otherwise
            // 200>100=T, 50>100=F, 255>254=T, 0>1=F, 128>127=T, 127>128=F, 100>99=T, 100>101=F
            // 0>0=F, 1>0=T, 2>0=T, 3>0=T, 4>0=T, 5>0=T, 6>0=T, 7>0=T
            let expected: [u8; 16] = [
                0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            ];
            assert_eq!(actual, expected, "test_cmpgt_epu8");
        }
    }

    #[test]
    fn test_cmpge_epu8() {
        unsafe {
            // Test unsigned greater-or-equal
            let a_data: [u8; 16] = [200, 100, 255, 1, 128, 128, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7];
            let b_data: [u8; 16] = [100, 100, 254, 1, 127, 128, 99, 101, 0, 0, 0, 0, 0, 0, 0, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr()));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr()));

            let result = SimdEngine128::cmpge_epu8(a, b);

            let mut actual: [u8; 16] = [0; 16];
            aarch64::vst1q_u8(actual.as_mut_ptr(), result.0);

            // Expected: 0xFF where a >= b (unsigned), 0x00 otherwise
            // 200>=100=T, 100>=100=T, 255>=254=T, 1>=1=T, 128>=127=T, 128>=128=T, 100>=99=T, 100>=101=F
            // 0>=0=T, 1>=0=T, 2>=0=T, 3>=0=T, 4>=0=T, 5>=0=T, 6>=0=T, 7>=0=T
            let expected: [u8; 16] = [
                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                0xFF, 0xFF,
            ];
            assert_eq!(actual, expected, "test_cmpge_epu8");
        }
    }

    #[test]
    fn test_adds_epu8() {
        unsafe {
            // Test saturating unsigned add
            let a_data: [u8; 16] = [200, 100, 255, 0, 128, 127, 100, 200, 0, 1, 2, 3, 4, 5, 6, 7];
            let b_data: [u8; 16] = [100, 100, 10, 0, 127, 128, 99, 100, 0, 0, 0, 0, 0, 0, 0, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr()));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr()));

            let result = SimdEngine128::adds_epu8(a, b);

            let mut actual: [u8; 16] = [0; 16];
            aarch64::vst1q_u8(actual.as_mut_ptr(), result.0);

            // Expected: saturating add (clamps at 255)
            // 200+100=255(sat), 100+100=200, 255+10=255(sat), 0+0=0, 128+127=255, 127+128=255, 100+99=199, 200+100=255(sat)
            // 0+0=0, 1+0=1, 2+0=2, etc.
            let expected: [u8; 16] = [255, 200, 255, 0, 255, 255, 199, 255, 0, 1, 2, 3, 4, 5, 6, 7];
            assert_eq!(actual, expected, "test_adds_epu8");
        }
    }

    #[test]
    fn test_subs_epu8() {
        unsafe {
            // Test saturating unsigned subtract
            let a_data: [u8; 16] = [
                200, 100, 255, 0, 128, 127, 100, 50, 10, 10, 10, 10, 10, 10, 10, 10,
            ];
            let b_data: [u8; 16] = [
                100, 150, 10, 5, 127, 128, 99, 100, 0, 5, 10, 15, 20, 25, 30, 35,
            ];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr()));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr()));

            let result = SimdEngine128::subs_epu8(a, b);

            let mut actual: [u8; 16] = [0; 16];
            aarch64::vst1q_u8(actual.as_mut_ptr(), result.0);

            // Expected: saturating subtract (clamps at 0)
            // 200-100=100, 100-150=0(sat), 255-10=245, 0-5=0(sat), 128-127=1, 127-128=0(sat), 100-99=1, 50-100=0(sat)
            // 10-0=10, 10-5=5, 10-10=0, 10-15=0(sat), 10-20=0(sat), 10-25=0(sat), 10-30=0(sat), 10-35=0(sat)
            let expected: [u8; 16] = [100, 0, 245, 0, 1, 0, 1, 0, 10, 5, 0, 0, 0, 0, 0, 0];
            assert_eq!(actual, expected, "test_subs_epu8");
        }
    }

    #[test]
    fn test_max_epu8() {
        unsafe {
            let a_data: [u8; 16] = [200, 50, 255, 0, 128, 127, 100, 100, 0, 1, 2, 3, 4, 5, 6, 7];
            let b_data: [u8; 16] = [100, 100, 254, 1, 127, 128, 99, 101, 0, 0, 0, 0, 0, 0, 0, 0];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr()));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr()));

            let result = SimdEngine128::max_epu8(a, b);

            let mut actual: [u8; 16] = [0; 16];
            aarch64::vst1q_u8(actual.as_mut_ptr(), result.0);

            let expected: [u8; 16] = [200, 100, 255, 1, 128, 128, 100, 101, 0, 1, 2, 3, 4, 5, 6, 7];
            assert_eq!(actual, expected, "test_max_epu8");
        }
    }

    #[test]
    fn test_unpacklo_epi8() {
        unsafe {
            let a_data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let b_data: [u8; 16] = [
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr()));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr()));

            let result = SimdEngine128::unpacklo_epi8(a, b);

            let mut actual: [u8; 16] = [0; 16];
            aarch64::vst1q_u8(actual.as_mut_ptr(), result.0);

            // SSE _mm_unpacklo_epi8: interleaves low 8 bytes of a and b
            // Result: [a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7]
            let expected: [u8; 16] = [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23];
            assert_eq!(actual, expected, "test_unpacklo_epi8");
        }
    }

    #[test]
    fn test_unpackhi_epi8() {
        unsafe {
            let a_data: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            let b_data: [u8; 16] = [
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            ];
            let a = __m128i(aarch64::vld1q_u8(a_data.as_ptr()));
            let b = __m128i(aarch64::vld1q_u8(b_data.as_ptr()));

            let result = SimdEngine128::unpackhi_epi8(a, b);

            let mut actual: [u8; 16] = [0; 16];
            aarch64::vst1q_u8(actual.as_mut_ptr(), result.0);

            // SSE _mm_unpackhi_epi8: interleaves high 8 bytes of a and b
            // Result: [a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15]
            let expected: [u8; 16] = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31];
            assert_eq!(actual, expected, "test_unpackhi_epi8");
        }
    }
}
