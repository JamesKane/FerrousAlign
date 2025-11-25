//! # Heterogeneous Compute Abstraction Layer
//!
//! This module provides the abstraction layer for routing alignment computations
//! to different hardware backends. It is the primary integration point for adding
//! new compute accelerators to FerrousAlign.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    HETEROGENEOUS COMPUTE ENTRY POINT                    │
//! │                                                                         │
//! │  This module defines the compute backend abstraction that allows        │
//! │  plugging in different hardware accelerators:                           │
//! │                                                                         │
//! │  • CPU SIMD (SSE/AVX2/AVX-512/NEON) - Current default                  │
//! │  • GPU (Metal/CUDA/ROCm) - Future integration point (NO-OP)            │
//! │  • NPU (ANE/Hexagon/OpenVINO) - Future integration point (NO-OP)       │
//! │                                                                         │
//! │  To add a new backend:                                                  │
//! │  1. Add variant to ComputeBackend enum                                  │
//! │  2. Implement detection in detect_optimal_backend()                     │
//! │  3. Add routing case in extension.rs::execute_with_backend()            │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod encoding;

use crate::simd::SimdEngineType;

// ============================================================================
// HETEROGENEOUS COMPUTE BACKEND ENUM
// ============================================================================
//
// This enum is the primary dispatch point for routing alignment work to
// different hardware accelerators. When adding a new accelerator:
//
// 1. Add a new variant below (e.g., `MetalGpu(MetalContext)`)
// 2. Update detect_optimal_backend() to detect the new hardware
// 3. Update extension.rs to route jobs to the new backend
// 4. Implement the backend-specific alignment kernel
//
// CURRENT STATUS: GPU and NPU variants are NO-OPs that fall back to CPU SIMD.
// They exist as documented integration points for future implementation.
// ============================================================================

/// Compute backend for alignment operations.
///
/// # Heterogeneous Compute Integration Point
///
/// This enum defines all available compute backends. To integrate a new
/// accelerator (GPU, NPU, etc.), add a variant here and implement the
/// corresponding execution path in `alignment/extension.rs`.
///
/// ## Current Backends
///
/// - `CpuSimd`: CPU-based SIMD acceleration (SSE2/AVX2/AVX-512/NEON)
///
/// ## Future Backends (NO-OP Integration Points)
///
/// - `Gpu`: GPU acceleration via Metal (macOS), CUDA (NVIDIA), or ROCm (AMD)
///   **Currently falls back to CpuSimd**
/// - `Npu`: Neural Processing Unit for ML-based seed filtering
///   **Currently falls back to CpuSimd**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    // ========================================================================
    // CPU SIMD BACKEND (ACTIVE)
    // ========================================================================
    /// CPU-based SIMD acceleration.
    ///
    /// Uses vectorized Smith-Waterman with automatic SIMD width selection:
    /// - SSE2/NEON (128-bit): 16-way parallelism
    /// - AVX2 (256-bit): 32-way parallelism
    /// - AVX-512 (512-bit): 64-way parallelism
    ///
    /// This is the default and always-available backend.
    CpuSimd(SimdEngineType),

    // ========================================================================
    // GPU BACKEND (FUTURE - CURRENTLY NO-OP)
    // ========================================================================
    /// GPU acceleration for Smith-Waterman alignment.
    ///
    /// # CURRENT STATUS: NO-OP
    ///
    /// This variant exists as an integration point. When selected, it
    /// currently falls back to CpuSimd. To implement:
    ///
    /// 1. Add GPU feature flag to Cargo.toml
    /// 2. Implement GPU kernel in `compute/gpu.rs`
    /// 3. Update `effective_backend()` to use GPU when available
    ///
    /// # Integration Notes
    ///
    /// When implementing GPU support, consider:
    /// - Batch size threshold: GPU dispatch overhead ~20-50μs, need batch >= 1024
    /// - Memory transfer: Use zero-copy where possible (Metal shared memory)
    /// - Fallback: Return to CpuSimd for small batches
    Gpu,

    // ========================================================================
    // NPU BACKEND (FUTURE - CURRENTLY NO-OP)
    // ========================================================================
    /// Neural Processing Unit for seed pre-filtering.
    ///
    /// # CURRENT STATUS: NO-OP
    ///
    /// This variant exists as an integration point. When selected, it
    /// currently falls back to CpuSimd. To implement:
    ///
    /// 1. Add NPU feature flag to Cargo.toml
    /// 2. Implement NPU inference in `compute/npu.rs`
    /// 3. Train seed viability classifier model
    /// 4. Update `effective_backend()` to use NPU when available
    ///
    /// # Integration Notes
    ///
    /// The NPU backend uses machine learning to filter seeds before the
    /// expensive Smith-Waterman step. This requires:
    ///
    /// 1. ONE-HOT encoding instead of classic 2-bit encoding
    ///    (see `encoding` submodule for EncodingStrategy trait)
    /// 2. Pre-trained model for seed viability classification
    /// 3. Fallback to CPU for sequences NPU cannot process
    Npu,
}

impl ComputeBackend {
    /// Returns the effective backend to use for computation.
    ///
    /// # Heterogeneous Compute Integration Point
    ///
    /// This method resolves the requested backend to an actually-usable one.
    /// GPU and NPU currently fall back to CPU SIMD since they are not yet
    /// implemented.
    ///
    /// When implementing a new backend, update this method to return the
    /// requested backend when hardware is available.
    pub fn effective_backend(&self) -> ComputeBackend {
        match self {
            // CPU SIMD is always available
            ComputeBackend::CpuSimd(engine) => ComputeBackend::CpuSimd(*engine),

            // ================================================================
            // GPU: NO-OP - Falls back to CPU SIMD
            // ================================================================
            // TODO: When GPU is implemented, check for hardware availability
            // and return ComputeBackend::Gpu if available.
            ComputeBackend::Gpu => {
                log::debug!("GPU backend requested but not implemented, falling back to CPU SIMD");
                ComputeBackend::CpuSimd(crate::simd::detect_optimal_simd_engine())
            }

            // ================================================================
            // NPU: NO-OP - Falls back to CPU SIMD
            // ================================================================
            // TODO: When NPU is implemented, check for hardware availability
            // and return ComputeBackend::Npu if available.
            ComputeBackend::Npu => {
                log::debug!("NPU backend requested but not implemented, falling back to CPU SIMD");
                ComputeBackend::CpuSimd(crate::simd::detect_optimal_simd_engine())
            }
        }
    }

    /// Check if this backend uses CPU SIMD (including fallback cases).
    pub fn is_cpu_simd(&self) -> bool {
        matches!(self.effective_backend(), ComputeBackend::CpuSimd(_))
    }

    /// Get the SIMD engine type for CPU backends.
    ///
    /// Returns the SIMD engine if this backend resolves to CPU SIMD,
    /// or None if it's a non-CPU backend (future implementation).
    pub fn simd_engine(&self) -> Option<SimdEngineType> {
        match self.effective_backend() {
            ComputeBackend::CpuSimd(engine) => Some(engine),
            _ => None,
        }
    }
}

// ============================================================================
// BACKEND DETECTION
// ============================================================================

/// Detect the optimal compute backend for the current system.
///
/// # Heterogeneous Compute Integration Point
///
/// This function probes available hardware and returns the best compute
/// backend. Currently returns CPU SIMD since GPU/NPU are not implemented.
///
/// ## Future Detection Priority
///
/// When GPU/NPU are implemented, detection priority will be:
/// 1. GPU (if available and batch size appropriate)
/// 2. NPU (if available and enabled for seed filtering)
/// 3. CPU SIMD (always available fallback)
///
/// ## Adding New Backend Detection
///
/// ```rust,ignore
/// // Example: Add Metal GPU detection
/// #[cfg(all(target_os = "macos", feature = "metal"))]
/// if metal::Device::system_default().is_some() {
///     return ComputeBackend::Gpu;
/// }
/// ```
pub fn detect_optimal_backend() -> ComputeBackend {
    // ========================================================================
    // GPU DETECTION (FUTURE - PLACEHOLDER)
    // ========================================================================
    // When implementing, add hardware detection here:
    //
    // #[cfg(feature = "metal")]
    // if is_metal_available() {
    //     log::info!("Metal GPU detected");
    //     return ComputeBackend::Gpu;
    // }

    // ========================================================================
    // NPU DETECTION (FUTURE - PLACEHOLDER)
    // ========================================================================
    // When implementing, add hardware detection here:
    //
    // #[cfg(feature = "npu")]
    // if is_ane_available() {
    //     log::info!("Apple Neural Engine detected");
    //     return ComputeBackend::Npu;
    // }

    // ========================================================================
    // CPU SIMD DETECTION (DEFAULT - ALWAYS AVAILABLE)
    // ========================================================================
    let simd_engine = crate::simd::detect_optimal_simd_engine();
    ComputeBackend::CpuSimd(simd_engine)
}

/// Returns a human-readable description of the compute backend.
pub fn backend_description(backend: ComputeBackend) -> &'static str {
    match backend {
        ComputeBackend::CpuSimd(engine) => crate::simd::simd_engine_description(engine),
        ComputeBackend::Gpu => "GPU (not implemented - using CPU SIMD fallback)",
        ComputeBackend::Npu => "NPU (not implemented - using CPU SIMD fallback)",
    }
}

// ============================================================================
// COMPUTE CONTEXT
// ============================================================================

/// Runtime context for compute operations.
///
/// # Heterogeneous Compute Integration Point
///
/// This struct holds runtime state needed for compute dispatch:
/// - Selected backend
/// - Batch size thresholds for backend switching
/// - Configuration flags for optional features
///
/// ## Future Expansion
///
/// When GPU/NPU backends are implemented, this context will hold:
/// - GPU device handles and command queues
/// - NPU model references and session state
/// - Memory pools for zero-copy transfers
#[derive(Debug, Clone)]
pub struct ComputeContext {
    /// Selected compute backend
    pub backend: ComputeBackend,

    /// Minimum batch size to use GPU (if available)
    /// Below this threshold, falls back to CPU SIMD
    pub gpu_batch_threshold: usize,

    /// Enable NPU seed pre-filtering (if available)
    pub npu_prefilter_enabled: bool,
}

impl Default for ComputeContext {
    fn default() -> Self {
        Self {
            backend: detect_optimal_backend(),
            gpu_batch_threshold: 1024, // GPU overhead ~20-50μs, need batch >= 1024
            npu_prefilter_enabled: false, // Disabled by default until implemented
        }
    }
}

impl ComputeContext {
    /// Create a new compute context with auto-detected backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a compute context with explicit backend selection.
    pub fn with_backend(backend: ComputeBackend) -> Self {
        Self {
            backend,
            ..Default::default()
        }
    }

    /// Get the effective backend (resolving NO-OPs to actual implementations).
    pub fn effective_backend(&self) -> ComputeBackend {
        self.backend.effective_backend()
    }
}

// ============================================================================
// UNIT TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_optimal_backend() {
        let backend = detect_optimal_backend();
        // Should always return CPU SIMD (GPU/NPU not implemented)
        assert!(matches!(backend, ComputeBackend::CpuSimd(_)));
        let desc = backend_description(backend);
        assert!(!desc.is_empty());
        println!("Detected backend: {} ({:?})", desc, backend);
    }

    #[test]
    fn test_gpu_fallback_to_cpu() {
        let backend = ComputeBackend::Gpu;
        let effective = backend.effective_backend();
        // GPU should fall back to CPU SIMD
        assert!(matches!(effective, ComputeBackend::CpuSimd(_)));
    }

    #[test]
    fn test_npu_fallback_to_cpu() {
        let backend = ComputeBackend::Npu;
        let effective = backend.effective_backend();
        // NPU should fall back to CPU SIMD
        assert!(matches!(effective, ComputeBackend::CpuSimd(_)));
    }

    #[test]
    fn test_compute_context_default() {
        let ctx = ComputeContext::default();
        // Default should use CPU SIMD
        assert!(ctx.effective_backend().is_cpu_simd());
    }

    #[test]
    fn test_simd_engine_extraction() {
        let backend = ComputeBackend::CpuSimd(SimdEngineType::Engine128);
        assert_eq!(backend.simd_engine(), Some(SimdEngineType::Engine128));

        // GPU/NPU should also return SIMD engine via fallback
        let gpu = ComputeBackend::Gpu;
        assert!(gpu.simd_engine().is_some());
    }
}
