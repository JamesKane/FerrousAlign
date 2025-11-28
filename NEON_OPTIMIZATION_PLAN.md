# NEON Performance Optimization Plan

**Date**: 2025-11-28

## 1. Executive Summary

A performance analysis of the FerrousAlign NEON backend on the Apple M3 Max revealed a significant performance gap compared to the SSE backend on an AMD Ryzen machine. The root cause is not an inherent limitation of NEON but rather an inefficient emulation of key SSE intrinsics in the Rust SIMD abstraction layer (`SimdEngine128`).

This document outlines the findings and proposes a design overhaul to replace the slow emulations with high-performance, NEON-native implementations.

## 2. Root Cause Analysis

The investigation into `src/core/compute/simd_abstraction/engine128.rs` identified two critical functions whose NEON implementations are major performance bottlenecks:

### 2.1. `movemask_epi8`

*   **Current Implementation (NEON):** Stores the 128-bit vector to a stack array and then performs a scalar loop to check the sign bit of each byte. This involves expensive memory access and branching inside a loop.
*   **SSE Equivalent:** A single, highly-efficient `PMOVMSKB` instruction.
*   **Impact:** High. This function is used in the main DP loop's early exit condition, and the slow implementation causes significant stalls.

### 2.2. `shuffle_epi8`

*   **Current Implementation (NEON):** Emulates the `PSHUFB` (SSSE3) instruction using a complex sequence of five NEON intrinsics (`vandq_u8`, `vqtbl1q_u8`, `vshrq_n_u8`, `vmulq_u8`, `vbicq_u8`). While it avoids a full scalar fallback, it is much slower than the single instruction on x86. The main inefficiency comes from emulating the `pshufb` behavior where a high bit in the control byte zeroes the output.
*   **SSE Equivalent:** A single, fast `PSHUFB` instruction.
*   **Impact:** Critical. This function is at the core of the DP loop for calculating match/mismatch scores, and its inefficiency directly impacts the overall throughput.

## 3. Proposed Design Overhaul

To achieve performance parity with the SSE backend, we must replace the emulated functions with efficient, NEON-native implementations.

### 3.1. High-Performance `movemask_epi8` for NEON

Instead of the store-and-loop approach, we will implement `movemask_epi8` using a bit-twiddling strategy directly on NEON registers. This avoids memory access entirely.

The proposed implementation uses a sequence of shifts and extracts to isolate the sign bits and pack them into a general-purpose register.

**Conceptual Implementation:**

```rust
#[cfg(target_arch = "aarch64")]
{
    use std::arch::aarch64::*;

    // A known good implementation from public sources (e.g., Google's highway library or SSE2NEON):
    // 1. Shift the sign bit to the LSB of each byte.
    let high_bits = vshrq_n_u8(a.as_u8(), 7);
    
    // 2. Create a bit mask for each lane position.
    const MASK_BYTES: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bit_mask = vld1q_u8(MASK_BYTES.as_ptr());
    
    // 3. AND the shifted bits with the mask to isolate the bit in the correct position.
    let masked = vandq_u8(high_bits, bit_mask);
    
    // 4. Horizontally add the two 64-bit halves of the vector.
    let res = vadd_u64(vget_low_u64(vreinterpretq_u64_u8(masked)),
                       vget_high_u64(vreinterpretq_u64_u8(masked)));
                       
    // 5. The result is the sum of the two 8-byte horizontal additions.
    (vget_lane_u8(vreinterpret_u8_u64(res), 0) as i32) |
    ((vget_lane_u8(vreinterpret_u8_u64(res), 8) as i32) << 8)
}
```
*Note: The final code would need to be carefully implemented and tested. The above is for illustration.*

This approach will be orders of magnitude faster than the current implementation.

### 3.2. Optimizing `shuffle_epi8` for NEON

The current NEON implementation for `shuffle_epi8` is close, but the emulation of the high-bit-zeroing behavior can be improved. The current code is:

```rust
let ctrl = b.as_u8();
let idx = vandq_u8(ctrl, vdupq_n_u8(0x0F));
let mut out = vqtbl1q_u8(a.as_u8(), idx);
// Build a per-lane mask of 0xFF where high bit is set in control
let high = vshrq_n_u8(ctrl, 7); // 0x00 or 0x01
let lane_mask = vmulq_u8(high, vdupq_n_u8(0xFF)); // 0x00 or 0xFF
// Clear lanes where mask is 0xFF
out = vbicq_u8(out, lane_mask);
__m128i::from_u8(out)
```

The sequence `vshrq_n_u8` -> `vmulq_u8` -> `vbicq_u8` is used to zero out lanes where the control byte has its high bit set. We can make this more direct. A sign-extending shift (`vshlq_s8`) can create a mask of `0x00` or `0xFF` more directly.

**Proposed Improvement:**

```rust
#[cfg(target_arch = "aarch64")]
unsafe {
    use std::arch::aarch64::*;
    let a_u8 = a.as_u8();
    let b_u8 = b.as_u8();

    // The lower 4 bits of the control byte are the shuffle index
    let shuffle_indices = vandq_u8(b_u8, vdupq_n_u8(0x0F));
    
    // Perform the table lookup
    let shuffled = vqtbl1q_u8(a_u8, shuffle_indices);

    // Create a mask from the high bit of the control byte.
    // If the high bit is set, the lane should be zero.
    // vcgtq_s8(zero, b_s8) creates a mask of 0xFF where b is negative (high bit set)
    let zero = vdupq_n_s8(0);
    let mask = vcgtq_s8(zero, b.as_s8());

    // Use bitwise AND with the inverted mask to clear the lanes.
    // (equivalent to blend with zero)
    let result = vbicq_u8(shuffled, mask.as_u8());

    __m128i::from_u8(result)
}
```
This approach uses `vcgtq_s8` to create the mask directly from the sign bit, which should be more efficient than the shift-and-multiply sequence.

## 4. Implementation and Validation Plan

1.  **Implement** the proposed changes in `src/core/compute/simd_abstraction/engine128.rs`.
2.  **Add Unit Tests:** Create specific unit tests for the new `movemask_epi8` and `shuffle_epi8` NEON implementations to ensure correctness against the SSE behavior.
3.  **Benchmark:** Re-run the benchmarks from `M3_MAX_NEON_BENCHMARK_FINDINGS.md` on the Apple M3 Max.
4.  **Analyze Results:** Compare the new results against the previous M3 Max results and the AMD Ryzen SSE results. The goal is to significantly close the performance gap.

By implementing these changes, we expect to see a dramatic improvement in FerrousAlign's performance on ARM64 platforms, bringing it much closer to the performance of the x86-64 SSE backend.
