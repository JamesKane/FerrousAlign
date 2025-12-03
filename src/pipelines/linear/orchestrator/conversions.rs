//! AoS/SoA conversion utilities for hybrid pipeline
//!
//! The paired-end pipeline requires transitioning between data representations:
//! - **SoA** (Structure-of-Arrays) for compute-heavy stages (SIMD alignment, mate rescue)
//! - **AoS** (Array-of-Structures) for logic-heavy stages (pairing, output)
//!
//! # Why Hybrid?
//!
//! During v0.7.0 development, we discovered that pure SoA pairing causes 96%
//! duplicate reads due to index boundary corruption. The SoA layout flattens
//! all alignments into contiguous arrays, losing per-read boundaries that
//! pairing logic requires for correctness.
//!
//! # Conversion Points
//!
//! ```text
//! [SoA] Finalization → [AoS] Pairing → [SoA] Mate Rescue → [AoS] Output
//!                    ↑               ↑                    ↑
//!              soa_to_aos      aos_to_soa           soa_to_aos
//! ```
//!
//! # Performance
//!
//! Conversion overhead is approximately 2% of total pipeline time, which is
//! acceptable given the correctness guarantee.

// ============================================================================
// MARKER TRAITS
// ============================================================================

/// Marker trait for SoA (Structure-of-Arrays) data layouts.
///
/// Types implementing this trait store data in columnar format,
/// optimized for SIMD vectorization across multiple elements.
pub trait SoALayout {}

/// Marker trait for AoS (Array-of-Structures) data layouts.
///
/// Types implementing this trait store data in row format,
/// with all fields of a single element grouped together.
pub trait AoSLayout {}

// ============================================================================
// CONVERSION TRAITS
// ============================================================================

/// Convert from any layout to SoA.
///
/// Implement this trait to enable conversion from AoS or other layouts
/// into SoA format for SIMD-optimized processing.
pub trait IntoSoA<T: SoALayout> {
    /// Convert self into SoA layout.
    fn into_soa(self) -> T;
}

/// Convert from any layout to AoS.
///
/// Implement this trait to enable conversion from SoA or other layouts
/// into AoS format for logic-heavy processing.
pub trait IntoAoS<T: AoSLayout> {
    /// Convert self into AoS layout.
    fn into_aos(self) -> T;
}

// ============================================================================
// CONVERSION FUNCTIONS (stubs for now)
// ============================================================================
//
// These functions will wrap existing conversion logic once we wire up the
// concrete types. For now, they serve as documentation of the API.

// The actual implementations will be added when we integrate with:
// - SoAAlignmentResult (from core/io/sam_output.rs)
// - Alignment (from finalization.rs)
// - Per-read alignment vectors for pairing

// Example future implementations:
//
// /// Convert SoA alignment results to per-read AoS vectors for pairing.
// ///
// /// This is required because pairing logic needs per-read indexing that
// /// SoA's flattened structure cannot provide correctly.
// pub fn soa_to_aos_for_pairing(
//     soa_r1: &SoAAlignmentResult,
//     soa_r2: &SoAAlignmentResult,
// ) -> (Vec<Vec<Alignment>>, Vec<Vec<Alignment>>) {
//     // Extract per-read alignments from SoA structure
//     todo!("Wrap existing conversion logic")
// }
//
// /// Convert paired AoS alignments back to SoA for mate rescue.
// pub fn aos_to_soa_for_rescue(
//     aos_r1: &[Vec<Alignment>],
//     aos_r2: &[Vec<Alignment>],
// ) -> (SoAAlignmentResult, SoAAlignmentResult) {
//     todo!("Wrap existing conversion logic")
// }
//
// /// Convert SoA results to flat AoS vector for output.
// pub fn soa_to_aos_for_output(soa: &SoAAlignmentResult) -> Vec<Alignment> {
//     todo!("Wrap existing conversion logic")
// }

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test marker trait implementations
    struct DummySoA;
    impl SoALayout for DummySoA {}

    struct DummyAoS;
    impl AoSLayout for DummyAoS {}

    impl IntoSoA<DummySoA> for DummyAoS {
        fn into_soa(self) -> DummySoA {
            DummySoA
        }
    }

    impl IntoAoS<DummyAoS> for DummySoA {
        fn into_aos(self) -> DummyAoS {
            DummyAoS
        }
    }

    #[test]
    fn test_conversion_traits_compile() {
        let aos = DummyAoS;
        let soa: DummySoA = aos.into_soa();
        let _aos_again: DummyAoS = soa.into_aos();
    }
}
