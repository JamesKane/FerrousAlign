# ARM SVE and SME Support Roadmap

## Overview

This document outlines the plan for adding ARM Scalable Vector Extension (SVE) and Scalable Matrix Extension (SME) support to FerrousAlign. This is targeted for **post-1.x** releases, as the primary development platform (Apple Silicon) does not benefit from these extensions for the core Smith-Waterman algorithm.

## Executive Summary

| Extension | Target Hardware | Viability for SW | Priority | Timeline |
|-----------|----------------|------------------|----------|----------|
| **ARM SVE (256-bit)** | AWS Graviton 3+, Neoverse N2/V1 | Excellent | P1 | Post-1.0 |
| **ARM SVE (512-bit)** | Neoverse V1+, Fujitsu A64FX | Good | P2 | Post-1.0 |
| **ARM SME** | Apple M4+, Neoverse V2+ | Poor (auxiliary only) | P3 | Post-1.x |

## Platform Analysis

### Apple Silicon (M1-M4)

| Chip | NEON | SVE | SME | Best Option for SW |
|------|------|-----|-----|-------------------|
| M1/M2/M3 | 128-bit | No | No | NEON |
| M4+ | 128-bit | Streaming only (SSVE) | SME2 (512-bit tiles) | NEON |

**Key Finding**: Apple M4 implements SME2 but does **not** implement regular SVE - only the streaming subset (SSVE) required for matrix operations. This means:
- Entering streaming mode (`smstart`) has overhead
- SSVE is optimized for outer products, not sequential DP
- NEON remains the best option for Smith-Waterman on Apple Silicon

### Non-Apple ARM (Cloud/HPC)

| Platform | SVE Width | SVE2 | SME | Notes |
|----------|-----------|------|-----|-------|
| AWS Graviton 3 | 256-bit | Yes | No | Full SVE support |
| AWS Graviton 4 | 256-bit | Yes | No | Full SVE support |
| Neoverse N2 | 256-bit | Yes | No | Arm Cortex reference |
| Neoverse V1 | 256-bit | Yes | No | HPC-focused |
| Neoverse V2 | 128-bit | Yes | SME | Reduced SVE width |
| Fujitsu A64FX | 512-bit | No | No | Fugaku supercomputer |

## Why SVE Benefits Smith-Waterman (on non-Apple ARM)

### Current NEON Implementation
- Fixed 128-bit vectors (16 lanes for int8, 8 lanes for int16)
- Processes 16 alignments in parallel per batch
- Well-optimized, validated on Apple Silicon

### SVE Advantages
1. **Wider vectors**: 256-bit = 32 lanes (2x throughput potential)
2. **Predication**: Native mask registers eliminate branch mispredicts
3. **Gather/scatter**: Efficient irregular memory access
4. **Vector-length agnostic code**: Single implementation scales across widths

### Why SME Does NOT Benefit Smith-Waterman

SME is designed for matrix outer products (GEMM-style):
```
ZA[i,j] += X[i] * Y[j]  // All pairs simultaneously
```

Smith-Waterman has sequential dependencies:
```
H[i,j] = max(H[i-1,j-1] + score, H[i-1,j] - gap, H[i,j-1] - gap)
```

The diagonal wavefront dependency pattern does not map to outer products.

## High-Level Design

### Phase 1: SVE Foundation

#### 1.1 Feature Flags

```toml
# Cargo.toml
[features]
sve = []       # ARM SVE (requires nightly for full support)
sve2 = ["sve"] # ARM SVE2 (implies sve)
```

#### 1.2 Engine Type Extension

```rust
// src/core/compute/simd_abstraction/simd.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdEngineType {
    Engine128,                              // SSE2/NEON (baseline)
    #[cfg(target_arch = "x86_64")]
    Engine256,                              // AVX2
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    Engine512,                              // AVX-512
    #[cfg(all(target_arch = "aarch64", feature = "sve"))]
    EngineSVE256,                           // SVE 256-bit
    #[cfg(all(target_arch = "aarch64", feature = "sve"))]
    EngineSVE512,                           // SVE 512-bit (A64FX)
}
```

#### 1.3 Runtime Detection

```rust
// src/core/compute/simd_abstraction/simd.rs
pub fn detect_optimal_simd_engine() -> SimdEngineType {
    #[cfg(all(target_arch = "aarch64", feature = "sve"))]
    {
        if let Some(vl) = detect_sve_vector_length() {
            match vl {
                512 => return SimdEngineType::EngineSVE512,
                256 => return SimdEngineType::EngineSVE256,
                _ => {} // Fall through to NEON
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return SimdEngineType::Engine512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdEngineType::Engine256;
        }
    }

    SimdEngineType::Engine128 // NEON or SSE2
}

#[cfg(all(target_arch = "aarch64", feature = "sve"))]
fn detect_sve_vector_length() -> Option<usize> {
    // Use HWCAP or direct SVE instruction to query VL
    // Returns None if SVE not available
    unsafe {
        // svcntb() returns vector length in bytes
        let vl_bytes: usize;
        asm!("cntb {}", out(reg) vl_bytes);
        if vl_bytes >= 32 { Some(vl_bytes * 8) } else { None }
    }
}
```

### Phase 2: SimdEngine Trait Implementation

#### 2.1 Fixed-Width Approach (Recommended)

Rather than variable-width SVE, implement fixed 256-bit and 512-bit engines:

```rust
// src/core/compute/simd_abstraction/engine_sve256.rs
#[cfg(all(target_arch = "aarch64", feature = "sve"))]
pub struct SimdEngineSVE256;

#[cfg(all(target_arch = "aarch64", feature = "sve"))]
impl SimdEngine for SimdEngineSVE256 {
    const WIDTH_8: usize = 32;   // 256 bits / 8 bits
    const WIDTH_16: usize = 16;  // 256 bits / 16 bits

    type Vec8 = svuint8_t;
    type Vec16 = svint16_t;

    #[inline]
    unsafe fn setzero_epi8() -> Self::Vec8 {
        svdup_n_u8(0)
    }

    #[inline]
    unsafe fn set1_epi8(a: i8) -> Self::Vec8 {
        svdup_n_s8(a)
    }

    #[inline]
    unsafe fn add_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        // SVE requires predicate; use all-true for full vector
        svadd_s16_x(svptrue_b16(), a, b)
    }

    #[inline]
    unsafe fn max_epi16(a: Self::Vec16, b: Self::Vec16) -> Self::Vec16 {
        svmax_s16_x(svptrue_b16(), a, b)
    }

    // ... remaining 60+ operations
}
```

#### 2.2 Predicate Handling

SVE operations require predicates. For full-vector operations (no masking):

```rust
// Helper: all-true predicate for element width
#[inline]
unsafe fn ptrue_b8() -> svbool_t { svptrue_b8() }
#[inline]
unsafe fn ptrue_b16() -> svbool_t { svptrue_b16() }

// Wrapped operations use implicit all-true predicate
#[inline]
unsafe fn sve_add_epi8(a: svint8_t, b: svint8_t) -> svint8_t {
    svadd_s8_x(ptrue_b8(), a, b)
}
```

### Phase 3: Alignment Kernels

#### 3.1 Banded Smith-Waterman (SVE 256-bit)

```rust
// src/core/alignment/banded_swa/isa_sve256.rs
#[cfg(all(target_arch = "aarch64", feature = "sve"))]
pub struct SwEngineSVE256;

#[cfg(all(target_arch = "aarch64", feature = "sve"))]
impl SwSimd for SwEngineSVE256 {
    type V8 = <SimdEngineSVE256 as SimdEngine>::Vec8;
    type V16 = <SimdEngineSVE256 as SimdEngine>::Vec16;
    const LANES: usize = 32;

    // Delegate to trait methods
    unsafe fn adds_epi8(a: Self::V8, b: Self::V8) -> Self::V8 {
        SimdEngineSVE256::adds_epi8(a, b)
    }
    // ...
}

// Generate SoA entry point via macro
#[cfg(all(target_arch = "aarch64", feature = "sve"))]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch32_sve_soa,
    width = 32,
    engine = SwEngineSVE256,
    cfg = cfg(all(target_arch = "aarch64", feature = "sve")),
    target_feature = "sve",
);
```

#### 3.2 Dispatch Integration

```rust
// src/core/alignment/banded_swa/dispatch.rs
pub fn simd_banded_swa_dispatch_soa<const W: usize>(...) -> Vec<OutScore> {
    match W {
        64 => {
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            { isa_avx512_int8::simd_banded_swa_batch64_soa(...) }
        }
        32 => {
            #[cfg(all(target_arch = "aarch64", feature = "sve"))]
            { isa_sve256::simd_banded_swa_batch32_sve_soa(...) }

            #[cfg(target_arch = "x86_64")]
            { isa_avx2::simd_banded_swa_batch32_soa(...) }
        }
        16 => {
            isa_sse_neon::simd_banded_swa_batch16_soa(...)
        }
        _ => unreachable!()
    }
}
```

#### 3.3 KSW Horizontal Batching

```rust
// src/core/alignment/kswv_sve.rs
#[cfg(all(target_arch = "aarch64", feature = "sve"))]
pub unsafe fn batch_ksw_align_sve256(
    jobs: &[AlignJob],
    workspace: &mut AlignmentWorkspace,
    config: &KernelConfig,
) -> usize {
    // Process 32 alignments in parallel using SVE 256-bit
    // Similar structure to kswv_avx2.rs
}
```

### Phase 4: SME Support (Limited Scope)

SME support is **optional** and limited to auxiliary operations where outer products apply:

#### 4.1 Potential Use Cases

| Use Case | Viability | Notes |
|----------|-----------|-------|
| Scoring matrix batch lookup | Medium | 64x64 base-pair scores |
| Seed embedding similarity | High | If using learned models |
| Post-alignment statistics | Low | Rarely bottleneck |

#### 4.2 Architecture (If Implemented)

```rust
// src/core/compute/sme.rs (future)
#[cfg(all(target_arch = "aarch64", feature = "sme"))]
pub struct SmeContext {
    tile_size: usize,  // 64x64 bytes on M4
}

#[cfg(all(target_arch = "aarch64", feature = "sme"))]
impl SmeContext {
    pub fn enter_streaming_mode(&self) {
        unsafe { asm!("smstart"); }
    }

    pub fn exit_streaming_mode(&self) {
        unsafe { asm!("smstop"); }
    }

    /// Batch score lookup: score[i,j] = matrix[query[i]][ref[j]]
    pub unsafe fn batch_score_lookup(
        &self,
        query: &[u8],    // Up to 64 bases
        reference: &[u8], // Up to 64 bases
        matrix: &[[i8; 4]; 4],
    ) -> [[i8; 64]; 64] {
        // Use outer product semantics
        // ZA[i,j] = matrix[query[i]][ref[j]]
    }
}
```

## File Structure

```
src/core/compute/simd_abstraction/
├── mod.rs                    # SimdEngine trait
├── simd.rs                   # Detection logic (update)
├── types.rs                  # Type bindings
├── engine128.rs              # SSE/NEON
├── engine256.rs              # AVX2
├── engine512.rs              # AVX-512
├── engine_sve256.rs          # NEW: SVE 256-bit
├── engine_sve512.rs          # NEW: SVE 512-bit
├── portable_intrinsics.rs    # x86/NEON helpers
├── portable_intrinsics_sve.rs # NEW: SVE helpers
└── tests_sve.rs              # NEW: SVE tests

src/core/alignment/banded_swa/
├── isa_sse_neon.rs           # 128-bit kernel
├── isa_avx2.rs               # 256-bit kernel (x86)
├── isa_avx512_int8.rs        # 512-bit kernel (x86)
├── isa_sve256.rs             # NEW: 256-bit kernel (ARM)
├── isa_sve512.rs             # NEW: 512-bit kernel (ARM)
└── dispatch.rs               # Update for SVE routing

src/core/alignment/
├── kswv_sse_neon.rs          # 128-bit horizontal
├── kswv_avx2.rs              # 256-bit horizontal (x86)
├── kswv_avx512.rs            # 512-bit horizontal (x86)
├── kswv_sve256.rs            # NEW: 256-bit horizontal (ARM)
└── kswv_sve512.rs            # NEW: 512-bit horizontal (ARM)

src/core/compute/
└── sme.rs                    # NEW: SME context (future, optional)
```

## Implementation Phases

### Phase 1: Foundation (Est. 2 weeks)
- [ ] Add `sve` and `sve2` feature flags
- [ ] Implement `detect_sve_vector_length()`
- [ ] Create `SimdEngineSVE256` with all trait operations
- [ ] Add `portable_intrinsics_sve.rs` for variable-immediate shims
- [ ] Unit tests for SVE engine

### Phase 2: Kernels (Est. 2 weeks)
- [ ] Implement `isa_sve256.rs` for banded Smith-Waterman
- [ ] Implement `kswv_sve256.rs` for horizontal batching
- [ ] Update dispatch logic in `banded_swa/dispatch.rs`
- [ ] Update dispatch logic in `batch_extension/dispatch.rs`
- [ ] Integration tests comparing SVE vs NEON outputs

### Phase 3: Validation (Est. 1 week)
- [ ] Test on AWS Graviton 3 instance
- [ ] Benchmark throughput vs NEON baseline
- [ ] Verify GATK parity maintained
- [ ] Profile for cache/memory bottlenecks

### Phase 4: SVE 512-bit (Optional, Est. 1 week)
- [ ] Create `SimdEngineSVE512` for A64FX
- [ ] Implement `isa_sve512.rs` and `kswv_sve512.rs`
- [ ] Test on Fugaku or emulator

### Phase 5: SME Exploration (Future, Post-1.x)
- [ ] Evaluate specific use cases where outer products apply
- [ ] Implement SME context with streaming mode management
- [ ] Benchmark overhead of mode switching
- [ ] Determine if any auxiliary operations benefit

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Unstable Rust intrinsics | High | Feature-gate like AVX-512; use nightly |
| No test hardware | Medium | Use QEMU SVE emulation; AWS Graviton |
| Predicate overhead | Medium | Profile on real hardware; tune kernels |
| SVE width varies | Low | Fixed-width engines (256/512) |

## Testing Strategy

### Unit Tests
- All 70+ `SimdEngine` operations verified against NEON reference
- Randomized input tests for arithmetic correctness
- Edge cases: zero vectors, saturation, overflow

### Integration Tests
- Compare SVE kernel output to NEON on identical inputs
- Golden read set alignment with diff checking
- Paired-end insert size and proper pairing metrics

### Benchmarks
- Throughput (reads/sec) on Graviton 3
- Memory bandwidth utilization
- Comparison vs NEON baseline (expect 1.8-2.2x)

## References

- [ARM SVE Introduction](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction)
- [ARM SME Introduction](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction)
- [Apple M4 SME Exploration](https://github.com/tzakharko/m4-sme-exploration)
- [Linux Kernel SME Documentation](https://docs.kernel.org/arch/arm64/sme.html)
- [corsix/amx - Apple AMX Documentation](https://github.com/corsix/amx)

## Conclusion

ARM SVE provides a clear path to 2x throughput on non-Apple ARM platforms (Graviton, Neoverse). However, Apple Silicon users (M1-M4) will not benefit from SVE/SME for Smith-Waterman due to:

1. M4's lack of regular SVE (only streaming mode)
2. SME's outer-product model not matching SW's sequential DP pattern
3. NEON's 128-bit being optimal for the algorithm on Apple hardware

This positions SVE support as valuable for cloud/HPC ARM deployments but **not a priority for Apple Silicon users**, hence the post-1.x timeline.
