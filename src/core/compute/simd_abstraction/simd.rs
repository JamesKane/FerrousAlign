//! Module for runtime SIMD engine detection and management.
//!
//! This module provides functionality to detect the optimal SIMD engine
//! available on the CPU at runtime and to retrieve associated metadata
//! like human-readable descriptions and batch sizes.

/// Available SIMD engine types based on CPU capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdEngineType {
    /// 128-bit SIMD (SSE/NEON) - always available
    Engine128,
    /// 256-bit SIMD (AVX2) - x86_64 only
    #[cfg(target_arch = "x86_64")]
    Engine256,
    /// 512-bit SIMD (AVX-512) - x86_64 only (requires avx512 feature flag)
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    Engine512,
}

/// Detects the optimal SIMD engine based on CPU features
///
/// Environment variable overrides for testing/debugging (x86_64 only):
/// - `FERROUS_ALIGN_FORCE_SSE=1`: Force SSE/128-bit engine (skip AVX2/AVX-512)
/// - `FERROUS_ALIGN_FORCE_AVX2=1`: Force AVX2/256-bit engine (skip AVX-512)
pub fn detect_optimal_simd_engine() -> SimdEngineType {
    #[cfg(target_arch = "x86_64")]
    {
        // Check for environment variable overrides (useful for testing)
        if std::env::var("FERROUS_ALIGN_FORCE_SSE").map(|v| v == "1").unwrap_or(false) {
            log::info!("FERROUS_ALIGN_FORCE_SSE=1: Using SSE (128-bit) engine");
            return SimdEngineType::Engine128;
        }

        #[cfg(feature = "avx512")]
        let force_avx2 = std::env::var("FERROUS_ALIGN_FORCE_AVX2").map(|v| v == "1").unwrap_or(false);

        // Check for AVX-512 support (only if feature is enabled)
        // AVX-512BW (Byte/Word) is required for 8-bit/16-bit operations
        #[cfg(feature = "avx512")]
        {
            if !force_avx2 && is_x86_feature_detected!("avx512bw") {
                return SimdEngineType::Engine512;
            }
            if force_avx2 {
                log::info!("FERROUS_ALIGN_FORCE_AVX2=1: Using AVX2 (256-bit) engine");
            }
        }

        // Check for AVX2 support
        if is_x86_feature_detected!("avx2") {
            return SimdEngineType::Engine256;
        }

        // Fallback to SSE (always available on x86_64)
        SimdEngineType::Engine128
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // ARM/aarch64 always uses 128-bit NEON
        SimdEngineType::Engine128
    }
}

/// Returns a human-readable description of the SIMD engine
pub fn simd_engine_description(engine: SimdEngineType) -> &'static str {
    match engine {
        SimdEngineType::Engine128 => {
            #[cfg(target_arch = "x86_64")]
            {
                "SSE (128-bit, 8-way parallelism)"
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                "NEON (128-bit, 8-way parallelism)"
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => "AVX2 (256-bit, 32-way parallelism)",
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => "AVX-512 (512-bit, 64-way parallelism)",
    }
}

/// Returns the optimal batch sizes for Smith-Waterman alignment for a given SIMD engine
///
/// Returns (max_batch_size, standard_batch_size):
/// - max_batch_size: Maximum parallelism for low-divergence sequences
/// - standard_batch_size: Standard batch size for medium-divergence sequences
///
/// **Batch Sizes by Engine**:
/// - SSE2/NEON (128-bit): 16-way parallelism (max=16, standard=16)
/// - AVX2 (256-bit): 32-way parallelism (max=32, standard=16)
/// - AVX-512 (512-bit): 64-way parallelism (max=64, standard=32)
pub fn get_simd_batch_sizes(engine: SimdEngineType) -> (usize, usize) {
    match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => (64, 32), // AVX-512: 64-way max, 32-way standard
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => (32, 16), // AVX2: 32-way max, 16-way standard
        SimdEngineType::Engine128 => (16, 16), // SSE2/NEON: 16-way
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test runtime SIMD engine detection
    #[test]
    fn test_simd_engine_detection() {
        let engine = detect_optimal_simd_engine();
        let description = simd_engine_description(engine);

        println!("Detected SIMD engine: {:?}", engine);
        println!("Description: {}", description);

        // Verify that we got a valid engine type
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512bw") {
                #[cfg(feature = "avx512")]
                assert_eq!(engine, SimdEngineType::Engine512);
            } else if is_x86_feature_detected!("avx2") {
                assert_eq!(engine, SimdEngineType::Engine256);
            } else {
                assert_eq!(engine, SimdEngineType::Engine128);
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            assert_eq!(engine, SimdEngineType::Engine128);
        }
    }
}
