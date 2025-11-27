//! Unit tests for the SIMD abstraction layer.

#[cfg(test)]
mod tests {
    use super::super::{SimdEngine, SimdEngine128}; // Adjust use path as necessary

    /// Test basic arithmetic operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_basic_ops() {
        unsafe {
            // Test set1 and add
            let a = SimdEngine128::set1_epi8(5);
            let b = SimdEngine128::set1_epi8(3);
            let sum = SimdEngine128::add_epi8(a, b);

            // Extract and verify (8 is expected since 5 + 3 = 8)
            let mut result = [0u8; 16];
            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, sum);
            assert_eq!(result[0], 8);
            assert_eq!(result[15], 8);
        }
    }

    /// Test saturating arithmetic for SimdEngine128
    #[test]
    fn test_simd_engine_128_saturating_ops() {
        unsafe {
            // Test saturating add (unsigned)
            let a = SimdEngine128::set1_epi8(250_u8 as i8);
            let b = SimdEngine128::set1_epi8(10_u8 as i8);
            let sum = SimdEngine128::adds_epu8(a, b);

            let mut result = [0u8; 16];
            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, sum);
            // 250 + 10 = 260, but should saturate to 255
            assert_eq!(result[0], 255);

            // Test saturating subtract (unsigned)
            let a = SimdEngine128::set1_epi8(10_u8 as i8);
            let b = SimdEngine128::set1_epi8(20_u8 as i8);
            let diff = SimdEngine128::subs_epu8(a, b);

            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, diff);
            // 10 - 20 would be negative, but should saturate to 0
            assert_eq!(result[0], 0);
        }
    }

    /// Test max/min operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_max_min() {
        unsafe {
            let a = SimdEngine128::set1_epi8(10_u8 as i8);
            let b = SimdEngine128::set1_epi8(20_u8 as i8);

            let max_val = SimdEngine128::max_epu8(a, b);
            let min_val = SimdEngine128::min_epu8(a, b);

            let mut max_result = [0u8; 16];
            let mut min_result = [0u8; 16];
            SimdEngine128::storeu_si128(max_result.as_mut_ptr() as *mut _, max_val);
            SimdEngine128::storeu_si128(min_result.as_mut_ptr() as *mut _, min_val);

            assert_eq!(max_result[0], 20);
            assert_eq!(min_result[0], 10);
        }
    }

    /// Test comparison operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_compare() {
        unsafe {
            let a = SimdEngine128::set1_epi8(42);
            let b = SimdEngine128::set1_epi8(42);
            let c = SimdEngine128::set1_epi8(10);

            // Test equality
            let eq_mask = SimdEngine128::cmpeq_epi8(a, b);
            let mut eq_result = [0u8; 16];
            SimdEngine128::storeu_si128(eq_result.as_mut_ptr() as *mut _, eq_mask);
            // All bits should be set (0xFF) for equal values
            assert_eq!(eq_result[0], 0xFF);

            // Test greater than (signed comparison)
            let gt_mask = SimdEngine128::cmpgt_epi8(a, c);
            let mut gt_result = [0u8; 16];
            SimdEngine128::storeu_si128(gt_result.as_mut_ptr() as *mut _, gt_mask);
            // 42 > 10, so mask should be 0xFF
            assert_eq!(gt_result[0], 0xFF);
        }
    }

    /// Test bitwise operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_bitwise() {
        unsafe {
            let a = SimdEngine128::set1_epi8(0b11110000_u8 as i8);
            let b = SimdEngine128::set1_epi8(0b10101010_u8 as i8);

            // Test AND
            let and_val = SimdEngine128::and_si128(a, b);
            let mut and_result = [0u8; 16];
            SimdEngine128::storeu_si128(and_result.as_mut_ptr() as *mut _, and_val);
            assert_eq!(and_result[0], 0b10100000);

            // Test OR
            let or_val = SimdEngine128::or_si128(a, b);
            let mut or_result = [0u8; 16];
            SimdEngine128::storeu_si128(or_result.as_mut_ptr() as *mut _, or_val);
            assert_eq!(or_result[0], 0b11111010);

            // Test AND-NOT (a & ~b)
            let andnot_val = SimdEngine128::andnot_si128(b, a);
            let mut andnot_result = [0u8; 16];
            SimdEngine128::storeu_si128(andnot_result.as_mut_ptr() as *mut _, andnot_val);
            assert_eq!(andnot_result[0], 0b01010000);
        }
    }

    /// Test memory load/store operations for SimdEngine128
    #[test]
    fn test_simd_engine_128_memory_ops() {
        unsafe {
            let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

            // Test unaligned load
            let vec = SimdEngine128::loadu_si128(data.as_ptr() as *const _);

            // Test unaligned store
            let mut result = [0u8; 16];
            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, vec);

            assert_eq!(data, result);
        }
    }

    /// Test zero vector creation
    #[test]
    fn test_simd_engine_128_zero() {
        unsafe {
            let zero_vec = SimdEngine128::setzero_epi8();
            let mut result = [0xFFu8; 16];
            SimdEngine128::storeu_si128(result.as_mut_ptr() as *mut _, zero_vec);

            assert_eq!(result, [0u8; 16]);
        }
    }

    /// Test SimdEngine256 basic operations (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_engine_256_basic_ops() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test - CPU does not support AVX2");
            return;
        }

        use super::super::SimdEngine256;
        unsafe {
            // Test set1 and add
            let a = SimdEngine256::set1_epi8(5);
            let b = SimdEngine256::set1_epi8(3);
            let sum = SimdEngine256::add_epi8(a, b);

            // Extract and verify (32 lanes for AVX2)
            let mut result = [0u8; 32];
            SimdEngine256::storeu_si128(result.as_mut_ptr() as *mut _, sum);
            assert_eq!(result[0], 8);
            assert_eq!(result[15], 8);
            assert_eq!(result[31], 8);
        }
    }

    /// Test SimdEngine256 max/min operations (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_engine_256_max_min() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test - CPU does not support AVX2");
            return;
        }
        use super::super::SimdEngine256;
        unsafe {
            let a = SimdEngine256::set1_epi8(10_u8 as i8);
            let b = SimdEngine256::set1_epi8(20_u8 as i8);

            let max_val = SimdEngine256::max_epu8(a, b);
            let min_val = SimdEngine256::min_epu8(a, b);

            let mut max_result = [0u8; 32];
            let mut min_result = [0u8; 32];
            SimdEngine256::storeu_si128(max_result.as_mut_ptr() as *mut _, max_val);
            SimdEngine256::storeu_si128(min_result.as_mut_ptr() as *mut _, min_val);

            assert_eq!(max_result[0], 20);
            assert_eq!(max_result[31], 20);
            assert_eq!(min_result[0], 10);
            assert_eq!(min_result[31], 10);
        }
    }

    /// Test SimdEngine512 basic operations (x86_64 with avx512 feature only)
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_simd_engine_512_basic_ops() {
        if !is_x86_feature_detected!("avx512bw") {
            eprintln!("Skipping AVX-512 test - CPU does not support AVX-512BW");
            return;
        }
        use super::super::SimdEngine512;
        unsafe {
            // Test set1 and add
            let a = SimdEngine512::set1_epi8(5);
            let b = SimdEngine512::set1_epi8(3);
            let sum = SimdEngine512::add_epi8(a, b);

            // Extract and verify (64 lanes for AVX-512)
            let mut result = [0u8; 64];
            SimdEngine512::storeu_si128(result.as_mut_ptr() as *mut _, sum);
            assert_eq!(result[0], 8);
            assert_eq!(result[31], 8);
            assert_eq!(result[63], 8);
        }
    }

    /// Test that all engines can handle the same data pattern correctly
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_cross_engine_consistency() {
        // Test data pattern
        let test_pattern = [5u8; 16];

        // Test with SSE (always available)
        let mut sse_result = [0u8; 16];
        unsafe {
            let vec = SimdEngine128::loadu_si128(test_pattern.as_ptr() as *const _);
            let doubled = SimdEngine128::add_epi8(vec, vec);
            SimdEngine128::storeu_si128(sse_result.as_mut_ptr() as *mut _, doubled);
        }

        // All values should be 10 (5 + 5)
        for i in 0..16 {
            assert_eq!(sse_result[i], 10, "SSE lane {} incorrect", i);
        }

        // Test with AVX2 if available
        if is_x86_feature_detected!("avx2") {
            use super::super::SimdEngine256;
            let mut avx2_test_pattern = [5u8; 32];
            let mut avx2_result = [0u8; 32];

            unsafe {
                let vec = SimdEngine256::loadu_si128(avx2_test_pattern.as_ptr() as *const _);
                let doubled = SimdEngine256::add_epi8(vec, vec);
                SimdEngine256::storeu_si128(avx2_result.as_mut_ptr() as *mut _, doubled);
            }

            // All values should be 10 (5 + 5)
            for i in 0..32 {
                assert_eq!(avx2_result[i], 10, "AVX2 lane {} incorrect", i);
            }
        }

        // Test with AVX-512 if available and feature enabled
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512bw") {
                use super::super::SimdEngine512;
                let mut avx512_test_pattern = [5u8; 64];
                let mut avx512_result = [0u8; 64];

                unsafe {
                    let vec = SimdEngine512::loadu_si128(avx512_test_pattern.as_ptr() as *const _);
                    let doubled = SimdEngine512::add_epi8(vec, vec);
                    SimdEngine512::storeu_si128(avx512_result.as_mut_ptr() as *mut _, doubled);
                }

                // All values should be 10 (5 + 5)
                for i in 0..64 {
                    assert_eq!(avx512_result[i], 10, "AVX-512 lane {} incorrect", i);
                }
            }
        }
    }
}
