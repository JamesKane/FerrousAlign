# RISC-V Vector Extension (RVV) Support Roadmap

## Overview

This document outlines the design for adding experimental RISC-V Vector Extension (RVV 1.0) support to FerrousAlign. RVV provides scalable vector processing similar to ARM SVE, with hardware now available on consumer boards (Banana Pi BPI-F3, Lichee Pi 3A) and SoCs (SpacemiT K1, SiFive Intelligence series).

**Status**: Experimental (post-1.x)

**Target Hardware**: SpacemiT K1 (256-bit RVV), SiFive X280/X390 (512-bit RVV)

## Executive Summary

| Aspect | Specification |
|--------|---------------|
| **Extension** | RISC-V Vector Extension 1.0 (RVV 1.0) |
| **Vector Widths** | 128-bit (min), 256-bit (K1), 512-bit (SiFive) |
| **Rust Support** | Nightly only (`riscv_ext_intrinsics` feature) |
| **Expected Speedup** | 2x over scalar (256-bit), 4x (512-bit) |
| **Priority** | P3 - Experimental (after SVE) |
| **Timeline** | Post-1.x |

## Platform Analysis

### Available Hardware (2024-2025)

| Platform | SoC | RVV Version | Vector Width | RAM | Price |
|----------|-----|-------------|--------------|-----|-------|
| **Orange Pi RV2** | Ky X1 (K1 variant) | RVV 1.0 | 256-bit | 2-8 GB | **$30-65** |
| Banana Pi BPI-F3 | SpacemiT K1 | RVV 1.0 | 256-bit | 4-16 GB | ~$80 |
| Lichee Pi 3A | SpacemiT K1 | RVV 1.0 | 256-bit | 8-16 GB | ~$100 |
| Milk-V Jupiter | SpacemiT K1 | RVV 1.0 | 256-bit | 8-16 GB | ~$90 |
| MUSE Book Laptop | SpacemiT K1 | RVV 1.0 | 256-bit | 8-16 GB | ~$300 |
| SiFive HiFive Premier P550 | SiFive P550 | RVV 1.0 | 256-bit | 8-16 GB | Dev boards |
| SiFive Intelligence X280 | Custom | RVV 1.0 | 512-bit | Varies | Evaluation |

### Orange Pi RV2 (Development Target)

The [Orange Pi RV2](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-RV2.html) is our primary development board due to low cost and availability:

- **SoC**: Ky X1 (SpacemiT K1 variant with SpacemiT X60 cores)
- **Cores**: 8× SpacemiT X60 @ 1.6 GHz (throttled from 2.0 GHz for passive cooling)
- **Vector**: RVV 1.0, 256-bit VLEN
- **AI**: 2 TOPS NPU
- **Memory**: 2/4/8 GB LPDDR4X
- **Storage**: eMMC + 2× M.2 NVMe (PCIe 2.0 x2)
- **Network**: 2× Gigabit Ethernet, WiFi 5, BT 5.0
- **OS**: Ubuntu 24.04 (official support)
- **Price**: $30 (2GB) to $65 (8GB)

Performance claim: "Single-core integer performance 130% of Cortex-A55 at same frequency, 80% power consumption"

**Memory Constraints**: The 2GB and 4GB variants are memory-constrained for genomics workloads. The 8GB variant (~$65) is recommended for development and testing with small reference genomes (bacteria, viruses). Human genome alignment requires more RAM than available.

### SpacemiT K1 Details

The [SpacemiT K1](https://www.spacemit.com/en/) is the most accessible RVV 1.0 platform:

- **Cores**: 8× SpacemiT X60 (RISC-V RVA22 profile)
- **Clock**: Up to 2.0 GHz (22nm process)
- **Vector**: RVV 1.0, 256-bit VLEN
- **AI**: 2 TOPS NPU
- **Memory**: Up to 16 GB LPDDR4

Key claim: "2× SIMD parallel processing capability compared to NEON" due to 256-bit vectors.

### Comparison with Other SIMD

| ISA | Width | Lanes (i8) | Lanes (i16) | Scalable |
|-----|-------|------------|-------------|----------|
| SSE/NEON | 128-bit | 16 | 8 | No |
| AVX2 | 256-bit | 32 | 16 | No |
| AVX-512 | 512-bit | 64 | 32 | No |
| ARM SVE | 128-2048 bit | 16-256 | 8-128 | Yes |
| **RVV (K1)** | **256-bit** | **32** | **16** | **Yes** |
| **RVV (X280)** | **512-bit** | **64** | **32** | **Yes** |

## RVV Architecture

### Key Concepts

Unlike fixed-width SIMD (SSE, AVX), RVV uses **vector-length agnostic (VLA)** programming:

```
┌─────────────────────────────────────────────────────────────┐
│                    RVV Register File                         │
├─────────────────────────────────────────────────────────────┤
│  v0-v31: 32 vector registers (VLEN bits each)               │
│  VLEN: Implementation-defined (128, 256, 512, 1024, 2048)   │
│                                                              │
│  vtype: Vector type configuration                            │
│    - SEW: Selected Element Width (8, 16, 32, 64 bits)       │
│    - LMUL: Length Multiplier (1/8 to 8)                     │
│                                                              │
│  vl: Vector length (active elements)                        │
│  vstart: Starting element for operations                    │
└─────────────────────────────────────────────────────────────┘
```

### VLA Programming Model

```c
// Traditional fixed-width (AVX2):
for (int i = 0; i < n; i += 32) {
    __m256i a = _mm256_load_si256(&arr[i]);
    __m256i b = _mm256_add_epi8(a, ones);
    _mm256_store_si256(&arr[i], b);
}

// RVV vector-length agnostic:
size_t vl;
for (size_t i = 0; i < n; i += vl) {
    vl = vsetvl_e8m1(n - i);           // Set VL for remaining elements
    vint8m1_t a = vle8_v_i8m1(&arr[i], vl);  // Load VL elements
    vint8m1_t b = vadd_vv_i8m1(a, ones, vl); // Add VL elements
    vse8_v_i8m1(&arr[i], b, vl);             // Store VL elements
}
```

### Predication (Masking)

RVV uses `v0` as a mask register for predicated operations:

```c
// Masked maximum: only update where mask is true
vint16m1_t result = vmax_vv_i16m1_m(mask, h_prev, e_val, vl);
```

## Rust Integration Challenges

### Current State (2024)

RISC-V vector intrinsics in Rust face significant challenges:

1. **Nightly-only**: Behind `#![feature(riscv_ext_intrinsics)]`
2. **Scalable types**: RVV registers have runtime-determined size
3. **No stable ABI**: Vector types can't cross FFI boundaries reliably
4. **Limited tooling**: Cross-compilation requires custom target JSON

From [Rust tracking issue #114544](https://github.com/rust-lang/rust/issues/114544):
> "Dealing with the Vector extension is very difficult due to dynamically sized registers"

### Workaround Strategies

#### Strategy A: Fixed VLEN Assumption (Recommended for Now)

Assume a specific VLEN and fail gracefully if hardware differs:

```rust
// Assume 256-bit VLEN (SpacemiT K1)
#[cfg(all(target_arch = "riscv64", feature = "rvv"))]
pub const ASSUMED_VLEN: usize = 256;
pub const LANES_I8: usize = ASSUMED_VLEN / 8;   // 32
pub const LANES_I16: usize = ASSUMED_VLEN / 16; // 16

/// Verify hardware matches assumed VLEN at runtime.
pub fn verify_rvv_vlen() -> Result<(), &'static str> {
    let actual_vlen = unsafe { query_vlen_bits() };
    if actual_vlen < ASSUMED_VLEN {
        Err("Hardware VLEN smaller than assumed")
    } else {
        Ok(())
    }
}
```

#### Strategy B: Inline Assembly

Use inline assembly for critical kernels:

```rust
#[cfg(all(target_arch = "riscv64", feature = "rvv"))]
unsafe fn rvv_add_i16(a: *const i16, b: *const i16, c: *mut i16, n: usize) {
    let mut i = 0usize;
    let mut vl: usize;

    asm!(
        "1:",
        "vsetvli {vl}, {n}, e16, m1, ta, ma",  // Set VL for i16
        "vle16.v v8, ({a})",                    // Load a
        "vle16.v v16, ({b})",                   // Load b
        "vadd.vv v24, v8, v16",                 // c = a + b
        "vse16.v v24, ({c})",                   // Store c
        "add {a}, {a}, {vl}",                   // Advance pointers
        "add {a}, {a}, {vl}",
        "add {b}, {b}, {vl}",
        "add {b}, {b}, {vl}",
        "add {c}, {c}, {vl}",
        "add {c}, {c}, {vl}",
        "sub {n}, {n}, {vl}",                   // Decrement count
        "bnez {n}, 1b",                         // Loop if more
        vl = out(reg) vl,
        a = inout(reg) a => _,
        b = inout(reg) b => _,
        c = inout(reg) c => _,
        n = inout(reg) n => _,
        out("v8") _,
        out("v16") _,
        out("v24") _,
    );
}
```

#### Strategy C: C FFI Bridge

Write RVV kernels in C and call via FFI:

```c
// rvv_kernels.c (compiled with riscv64-unknown-linux-gnu-gcc -march=rv64gcv)
#include <riscv_vector.h>

void rvv_sw_kernel_i16(
    const int16_t* query_profile,
    const int16_t* h_row,
    int16_t* h_out,
    size_t n,
    int16_t gap_open,
    int16_t gap_extend
) {
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m1(n - i);
        vint16m1_t h = __riscv_vle16_v_i16m1(&h_row[i], vl);
        vint16m1_t p = __riscv_vle16_v_i16m1(&query_profile[i], vl);
        // ... Smith-Waterman DP ...
        __riscv_vse16_v_i16m1(&h_out[i], h, vl);
    }
}
```

```rust
// Rust FFI binding
#[cfg(all(target_arch = "riscv64", feature = "rvv"))]
extern "C" {
    fn rvv_sw_kernel_i16(
        query_profile: *const i16,
        h_row: *const i16,
        h_out: *mut i16,
        n: usize,
        gap_open: i16,
        gap_extend: i16,
    );
}
```

## High-Level Design

### Feature Flags

```toml
# Cargo.toml
[features]
# RISC-V Vector Extension (experimental, requires nightly)
rvv = []
rvv256 = ["rvv"]  # Assume 256-bit VLEN (SpacemiT K1)
rvv512 = ["rvv"]  # Assume 512-bit VLEN (SiFive X280)
```

### SimdEngineType Extension

```rust
// src/core/compute/simd_abstraction/simd.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdEngineType {
    Engine128,                              // SSE/NEON
    #[cfg(target_arch = "x86_64")]
    Engine256,                              // AVX2
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    Engine512,                              // AVX-512
    #[cfg(all(target_arch = "aarch64", feature = "sve"))]
    EngineSVE256,                           // ARM SVE 256-bit
    #[cfg(all(target_arch = "riscv64", feature = "rvv256"))]
    EngineRVV256,                           // RISC-V RVV 256-bit
    #[cfg(all(target_arch = "riscv64", feature = "rvv512"))]
    EngineRVV512,                           // RISC-V RVV 512-bit
}
```

### Runtime Detection

```rust
// src/core/compute/simd_abstraction/simd.rs

pub fn detect_optimal_simd_engine() -> SimdEngineType {
    #[cfg(all(target_arch = "riscv64", feature = "rvv"))]
    {
        if let Some(vlen) = detect_rvv_vlen() {
            match vlen {
                512.. => {
                    #[cfg(feature = "rvv512")]
                    return SimdEngineType::EngineRVV512;
                }
                256..=511 => {
                    #[cfg(feature = "rvv256")]
                    return SimdEngineType::EngineRVV256;
                }
                _ => {} // Fall through to scalar
            }
        }
    }

    // ... existing x86_64, aarch64 detection ...

    SimdEngineType::Engine128
}

#[cfg(all(target_arch = "riscv64", feature = "rvv"))]
fn detect_rvv_vlen() -> Option<usize> {
    // Query VLEN using vsetvl instruction
    unsafe {
        let vl: usize;
        // vsetvli with e8, m1 returns VLEN/8
        asm!(
            "vsetvli {}, zero, e8, m1, ta, ma",
            out(reg) vl,
            options(nomem, nostack)
        );
        if vl > 0 { Some(vl * 8) } else { None }
    }
}
```

### Engine Implementation

```rust
// src/core/compute/simd_abstraction/engine_rvv256.rs

#[cfg(all(target_arch = "riscv64", feature = "rvv256"))]
pub struct SimdEngineRVV256;

#[cfg(all(target_arch = "riscv64", feature = "rvv256"))]
impl SimdEngineRVV256 {
    pub const WIDTH_8: usize = 32;   // 256 bits / 8 bits
    pub const WIDTH_16: usize = 16;  // 256 bits / 16 bits

    /// Add i16 elements
    #[inline]
    pub unsafe fn add_epi16(a: *const i16, b: *const i16, c: *mut i16, n: usize) {
        // Using inline assembly until Rust intrinsics stabilize
        let mut remaining = n;
        let mut pa = a;
        let mut pb = b;
        let mut pc = c;

        while remaining > 0 {
            let vl: usize;
            asm!(
                "vsetvli {vl}, {n}, e16, m1, ta, ma",
                "vle16.v v8, ({a})",
                "vle16.v v16, ({b})",
                "vadd.vv v24, v8, v16",
                "vse16.v v24, ({c})",
                vl = out(reg) vl,
                a = in(reg) pa,
                b = in(reg) pb,
                c = in(reg) pc,
                n = in(reg) remaining,
                out("v8") _,
                out("v16") _,
                out("v24") _,
            );
            pa = pa.add(vl);
            pb = pb.add(vl);
            pc = pc.add(vl);
            remaining -= vl;
        }
    }

    /// Saturating add i8 elements
    #[inline]
    pub unsafe fn adds_epi8(a: *const i8, b: *const i8, c: *mut i8, n: usize) {
        let mut remaining = n;
        let mut pa = a;
        let mut pb = b;
        let mut pc = c;

        while remaining > 0 {
            let vl: usize;
            asm!(
                "vsetvli {vl}, {n}, e8, m1, ta, ma",
                "vle8.v v8, ({a})",
                "vle8.v v16, ({b})",
                "vsadd.vv v24, v8, v16",  // Saturating add
                "vse8.v v24, ({c})",
                vl = out(reg) vl,
                a = in(reg) pa,
                b = in(reg) pb,
                c = in(reg) pc,
                n = in(reg) remaining,
                out("v8") _,
                out("v16") _,
                out("v24") _,
            );
            pa = pa.add(vl);
            pb = pb.add(vl);
            pc = pc.add(vl);
            remaining -= vl;
        }
    }

    /// Maximum of i16 elements
    #[inline]
    pub unsafe fn max_epi16(a: *const i16, b: *const i16, c: *mut i16, n: usize) {
        let mut remaining = n;
        let mut pa = a;
        let mut pb = b;
        let mut pc = c;

        while remaining > 0 {
            let vl: usize;
            asm!(
                "vsetvli {vl}, {n}, e16, m1, ta, ma",
                "vle16.v v8, ({a})",
                "vle16.v v16, ({b})",
                "vmax.vv v24, v8, v16",
                "vse16.v v24, ({c})",
                vl = out(reg) vl,
                a = in(reg) pa,
                b = in(reg) pb,
                c = in(reg) pc,
                n = in(reg) remaining,
                out("v8") _,
                out("v16") _,
                out("v24") _,
            );
            pa = pa.add(vl);
            pb = pb.add(vl);
            pc = pc.add(vl);
            remaining -= vl;
        }
    }
}
```

### Smith-Waterman Kernel

```rust
// src/core/alignment/banded_swa/isa_rvv256.rs

#[cfg(all(target_arch = "riscv64", feature = "rvv256"))]
pub struct SwEngineRVV256;

#[cfg(all(target_arch = "riscv64", feature = "rvv256"))]
impl SwEngineRVV256 {
    const LANES: usize = 32;  // 256-bit / 8-bit

    /// Banded Smith-Waterman using RVV 256-bit vectors.
    /// Processes 32 alignments in parallel.
    pub unsafe fn sw_kernel_batch32(
        soa: &SwSoA,
        config: &KernelConfig,
        workspace: &mut AlignmentWorkspace,
    ) -> Vec<OutScore> {
        let lanes = Self::LANES;
        assert!(soa.lanes <= lanes);

        // Allocate DP rows (H, E, F)
        let qmax = soa.max_qlen as usize;
        let h_row = workspace.get_h_row_mut();
        let e_row = workspace.get_e_row_mut();

        // Initialize scoring parameters
        let gap_o = config.gaps.o_del as i8;
        let gap_e = config.gaps.e_del as i8;

        // Main DP loop (simplified)
        for j in 0..soa.max_tlen as usize {
            for i in 0..qmax {
                // Load previous H values
                // Compute: H[i,j] = max(H[i-1,j-1] + score, E[i,j], F[i,j])
                // Using RVV vector operations on 32 lanes
                // ...
            }
        }

        // Extract results
        extract_scores(h_row, soa.lanes)
    }
}

// Generate entry point macro
#[cfg(all(target_arch = "riscv64", feature = "rvv256"))]
generate_swa_entry_soa!(
    name = simd_banded_swa_batch32_rvv_soa,
    width = 32,
    engine = SwEngineRVV256,
    cfg = cfg(all(target_arch = "riscv64", feature = "rvv256")),
);
```

### Dispatch Integration

```rust
// src/core/alignment/banded_swa/dispatch.rs

pub fn simd_banded_swa_dispatch_soa<const W: usize>(...) -> Vec<OutScore> {
    match W {
        64 => {
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            { isa_avx512_int8::simd_banded_swa_batch64_soa(...) }
        }
        32 => {
            // RVV 256-bit (32 lanes)
            #[cfg(all(target_arch = "riscv64", feature = "rvv256"))]
            { isa_rvv256::simd_banded_swa_batch32_rvv_soa(...) }

            // ARM SVE 256-bit (32 lanes)
            #[cfg(all(target_arch = "aarch64", feature = "sve"))]
            { isa_sve256::simd_banded_swa_batch32_sve_soa(...) }

            // x86_64 AVX2 (32 lanes)
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

## File Structure

```
src/core/compute/simd_abstraction/
├── mod.rs                      # SimdEngine trait
├── simd.rs                     # Detection logic (update)
├── types.rs                    # Type bindings
├── engine128.rs                # SSE/NEON
├── engine256.rs                # AVX2
├── engine512.rs                # AVX-512
├── engine_sve256.rs            # ARM SVE 256-bit
├── engine_rvv256.rs            # NEW: RISC-V RVV 256-bit
├── engine_rvv512.rs            # NEW: RISC-V RVV 512-bit
└── tests_rvv.rs                # NEW: RVV tests

src/core/alignment/banded_swa/
├── isa_sse_neon.rs             # 128-bit kernel
├── isa_avx2.rs                 # 256-bit kernel (x86)
├── isa_avx512_int8.rs          # 512-bit kernel (x86)
├── isa_sve256.rs               # 256-bit kernel (ARM SVE)
├── isa_rvv256.rs               # NEW: 256-bit kernel (RISC-V)
├── isa_rvv512.rs               # NEW: 512-bit kernel (RISC-V)
└── dispatch.rs                 # Update for RVV routing

src/core/alignment/
├── kswv_sse_neon.rs            # 128-bit horizontal
├── kswv_avx2.rs                # 256-bit horizontal (x86)
├── kswv_sve256.rs              # 256-bit horizontal (ARM)
├── kswv_rvv256.rs              # NEW: 256-bit horizontal (RISC-V)
└── kswv_rvv512.rs              # NEW: 512-bit horizontal (RISC-V)
```

## Cross-Compilation Setup

### Target Configuration

```json
// riscv64gc-unknown-linux-gnu-rvv.json
{
    "llvm-target": "riscv64-unknown-linux-gnu",
    "data-layout": "e-m:e-p:64:64-i64:64-i128:128-n64-S128",
    "arch": "riscv64",
    "target-endian": "little",
    "target-pointer-width": "64",
    "target-c-int-width": "32",
    "os": "linux",
    "env": "gnu",
    "vendor": "unknown",
    "linker-flavor": "gcc",
    "linker": "riscv64-linux-gnu-gcc",
    "features": "+v,+zba,+zbb,+zbc,+zbs",
    "pre-link-args": {
        "gcc": ["-march=rv64gcv"]
    }
}
```

### Build Commands

```bash
# Install RISC-V toolchain
sudo apt install gcc-riscv64-linux-gnu

# Build for RVV 256-bit (SpacemiT K1)
RUSTFLAGS="-C target-feature=+v" \
cargo +nightly build --release \
    --target riscv64gc-unknown-linux-gnu \
    --features rvv256 \
    -Z build-std=core,alloc

# Run on hardware (e.g., Banana Pi BPI-F3)
scp target/riscv64gc-unknown-linux-gnu/release/ferrous-align user@bpi-f3:~/
ssh user@bpi-f3 ./ferrous-align mem reference.idx reads.fq
```

## Testing Strategy

### Emulation (QEMU)

```bash
# Install QEMU with RVV support
sudo apt install qemu-user-static

# Run tests under emulation
QEMU_CPU=rv64,v=true,vlen=256 \
cargo +nightly test --release \
    --target riscv64gc-unknown-linux-gnu \
    --features rvv256 \
    -Z build-std=core,alloc
```

### Hardware Testing

1. **Orange Pi RV2** (~$30-65) - **Primary Development Target**
   - Ky X1 (SpacemiT K1 variant), 8 cores @ 1.6 GHz, 256-bit RVV
   - 2-8 GB LPDDR4X (8GB recommended for genomics)
   - Ubuntu 24.04 official support
   - Memory-constrained: suitable for bacterial/viral genomes, unit tests
   - NVMe storage via M.2 slots for larger datasets

2. **Banana Pi BPI-F3** (~$80-100)
   - SpacemiT K1, 8 cores @ 2.0 GHz, 256-bit RVV
   - 4-16 GB RAM (16GB allows small eukaryotic genomes)
   - Run integration tests on larger genomic data

3. **Lichee Pi 3A** (~$100-150)
   - Same K1 SoC, 8-16 GB RAM
   - Modular design, more I/O options

### Memory Requirements for Testing

| Reference | Index Size | Min RAM | Suitable Boards |
|-----------|------------|---------|-----------------|
| Bacterial (~5 MB) | ~50 MB | 2 GB | All |
| Viral panel (~100 KB) | ~1 MB | 2 GB | All |
| Yeast (~12 MB) | ~120 MB | 4 GB | Orange Pi RV2 4GB+ |
| C. elegans (~100 MB) | ~1 GB | 8 GB | Orange Pi RV2 8GB, BPI-F3 |
| Human (~3 GB) | ~24 GB | 32+ GB | **Not feasible** on current boards |

### Test Matrix

| Test | QEMU | Orange Pi RV2 | BPI-F3 | SiFive |
|------|------|---------------|--------|--------|
| Unit tests (SIMD ops) | ✓ | ✓ | ✓ | ✓ |
| Integration tests (bacterial) | ✓ | ✓ | ✓ | ✓ |
| Golden reads parity (small) | - | ✓ | ✓ | ✓ |
| Benchmark throughput | - | ✓ | ✓ | ✓ |
| Human genome | - | - | - | - |

## Implementation Phases

### Phase 0: Toolchain Setup (1 week)
- [x] Acquire Orange Pi RV2 (8GB) for hardware testing (**Available**)
- [ ] Set up RISC-V cross-compilation environment
- [ ] Configure custom target JSON with RVV features
- [ ] Verify QEMU RVV emulation works
- [ ] Set up Ubuntu 24.04 on Orange Pi RV2, verify RVV detection

### Phase 1: Engine Foundation (2 weeks)
- [ ] Add `rvv`, `rvv256`, `rvv512` feature flags
- [ ] Implement `detect_rvv_vlen()` runtime detection
- [ ] Create `engine_rvv256.rs` with inline assembly operations
- [ ] Unit tests for basic operations (add, max, saturating ops)

### Phase 2: Smith-Waterman Kernel (2 weeks)
- [ ] Implement `isa_rvv256.rs` for banded Smith-Waterman
- [ ] Implement `kswv_rvv256.rs` for horizontal batching
- [ ] Update dispatch logic in `banded_swa/dispatch.rs`
- [ ] Integration tests comparing RVV vs scalar outputs

### Phase 3: Validation (1 week)
- [ ] Test on Banana Pi BPI-F3 hardware
- [ ] Benchmark throughput vs scalar baseline
- [ ] Verify GATK parity maintained
- [ ] Profile for memory bandwidth bottlenecks

### Phase 4: RVV 512-bit (Optional) (1 week)
- [ ] Create `engine_rvv512.rs` for SiFive X280
- [ ] Implement `isa_rvv512.rs` (64-way parallelism)
- [ ] Test on SiFive evaluation hardware (if available)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Rust intrinsics unstable** | High | High | Use inline assembly; C FFI bridge |
| **No hardware access** | Medium | High | QEMU emulation; acquire BPI-F3 |
| **Performance below expectations** | Medium | Medium | Profile and optimize; may not beat AVX2 |
| **Cross-compilation issues** | High | Medium | Document toolchain setup; use Docker |
| **VLA type system issues** | High | Medium | Fixed VLEN assumption; fail fast |

## Performance Expectations

### SpacemiT K1 (256-bit RVV)

Based on claimed 2× improvement over NEON:

| Operation | NEON (128-bit) | RVV (256-bit) | Speedup |
|-----------|----------------|---------------|---------|
| Lanes (i8) | 16 | 32 | 2× |
| Lanes (i16) | 8 | 16 | 2× |
| SW throughput | 1× | ~1.8-2× | Expected |

Note: Real-world speedup depends on memory bandwidth and kernel efficiency.

### Comparison with x86_64

| Platform | Vector Width | Expected Perf |
|----------|--------------|---------------|
| SSE2/NEON | 128-bit | 1× baseline |
| SpacemiT K1 (RVV) | 256-bit | ~1.5-2× |
| AVX2 | 256-bit | ~2× |
| AVX-512 | 512-bit | ~3× |

RVV on K1 should approach AVX2 performance but on a much lower power budget.

## References

- [RISC-V Vector Extension Specification](https://github.com/riscvarchive/riscv-v-spec/blob/master/v-spec.adoc)
- [RISC-V Vector Intrinsics (GCC)](https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Vector-Intrinsics.html)
- [RISC-V Vector Intrinsics (LLVM)](https://llvm.org/docs/RISCV/RISCVVectorExtension.html)
- [RVV Intrinsic Documentation](https://github.com/riscv-non-isa/rvv-intrinsic-doc)
- [Rust RISC-V Tracking Issue #114544](https://github.com/rust-lang/rust/issues/114544)
- [SpacemiT K1 Overview](https://www.spacemit.com/en/)
- [Banana Pi BPI-F3](https://wiki.banana-pi.org/Banana_Pi_BPI-F3)
- [RVV Tutorial (EuPilot)](https://eupilot.eu/wp-content/uploads/2022/11/RISC-V-VectorExtension-1-1.pdf)

## Conclusion

RISC-V RVV support is experimental but increasingly viable:

1. **Hardware available**: SpacemiT K1 boards cost $80-100
2. **256-bit vectors**: 2× throughput potential over NEON/SSE
3. **Rust challenges**: Requires nightly, inline assembly, or C FFI
4. **Modest performance**: Won't beat AVX2/AVX-512, but interesting for low-power

The main value is:
- **Ecosystem diversity**: Support emerging open-source ISA
- **Low-power alignment**: Viable for edge/embedded genomics
- **Future-proofing**: RVV will improve as hardware matures

**Priority**: P3 - Experimental, post-1.x, after ARM SVE
