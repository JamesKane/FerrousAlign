#![cfg(all(test, target_arch = "aarch64"))]

use super::SimdEngine;
use super::engine128::SimdEngine128;
use super::types::__m128i;

#[test]
fn test_neon_movemask_epi8() {
    unsafe {
        // Test case 1: Alternating sign bits
        let values = [-1i8, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0];
        let vec = __m128i::from_slice(&values);
        let mask = SimdEngine128::movemask_epi8(vec);
        // High bit is set for negative numbers.
        // The mask should have a 1 for each negative number.
        // Binary: 0101010101010101
        assert_eq!(mask, 0x5555);

        // Test case 2: All negative
        let values = [-1i8; 16];
        let vec = __m128i::from_slice(&values);
        let mask = SimdEngine128::movemask_epi8(vec);
        assert_eq!(mask, 0xFFFF);

        // Test case 3: All positive
        let values = [1i8; 16];
        let vec = __m128i::from_slice(&values);
        let mask = SimdEngine128::movemask_epi8(vec);
        assert_eq!(mask, 0x0000);

        // Test case 4: Single bit set
        let mut values = [0i8; 16];
        values[15] = -1;
        let vec = __m128i::from_slice(&values);
        let mask = SimdEngine128::movemask_epi8(vec);
        assert_eq!(mask, 1 << 15);
    }
}

#[test]
fn test_neon_shuffle_epi8() {
    unsafe {
        let source_values = [0i8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let source_vec = __m128i::from_slice(&source_values);

        // Test case 1: Identity shuffle
        let control_values = [0i8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let control_vec = __m128i::from_slice(&control_values);
        let result_vec = SimdEngine128::shuffle_epi8(source_vec, control_vec);
        let mut result_values = [0i8; 16];
        result_vec.copy_to_slice(&mut result_values);
        assert_eq!(result_values, source_values);

        // Test case 2: Reverse shuffle
        let control_values = [15i8, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let control_vec = __m128i::from_slice(&control_values);
        let result_vec = SimdEngine128::shuffle_epi8(source_vec, control_vec);
        let mut result_values = [0i8; 16];
        result_vec.copy_to_slice(&mut result_values);
        let expected_values = [15i8, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        assert_eq!(result_values, expected_values);

        // Test case 3: Zeroing lanes (high bit set in control)
        let control_values = [0i8, 1, 2, -1, 4, 5, 6, -128, 8, 9, 10, 11, 12, 13, 14, 15];
        let control_vec = __m128i::from_slice(&control_values);
        let result_vec = SimdEngine128::shuffle_epi8(source_vec, control_vec);
        let mut result_values = [0i8; 16];
        result_vec.copy_to_slice(&mut result_values);
        let expected_values = [0i8, 1, 2, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15];
        assert_eq!(result_values, expected_values);

        // Test case 4: Broadcast first element
        let control_values = [0i8; 16];
        let control_vec = __m128i::from_slice(&control_values);
        let result_vec = SimdEngine128::shuffle_epi8(source_vec, control_vec);
        let mut result_values = [0i8; 16];
        result_vec.copy_to_slice(&mut result_values);
        let expected_values = [0i8; 16];
        assert_eq!(result_values, expected_values);
    }
}
