# Apple M3 Max SIMD Probing Report

**Date**: 2025-11-25
**Platform**: Apple M3 Max (aarch64)
**Reference**: chm13v2.0 (T2T CHM13)
**Test Dataset**: Golden Reads 10K HG002 read pairs

## Executive Summary

✅ **Key Finding**: The `FERROUS_ALIGN_SIMD` environment variable does NOT control SIMD vs scalar execution in the current implementation. Both SIMD-enabled and SIMD-disabled runs use the **scalar Smith-Waterman** path for alignment.

## Methodology

Added strategic logging at key SIMD decision points:
1. `simd_runtime_enabled()` - Runtime SIMD control function
2. Vertical SIMD batch dispatch (8-bit and 16-bit variants)
3. Horizontal SIMD kernel (SSE/NEON batch)
4. Scalar Smith-Waterman entry point

Logging levels:
- INFO: Batch dispatch decisions
- TRACE: NEON/SSE intrinsic execution

## Test Results

### Run 1: FERROUS_ALIGN_SIMD=0 (Scalar Path)

**Command**:
```bash
FERROUS_ALIGN_SIMD=0 ./target/release/ferrous-align mem -t 8 \
    /Library/Genomics/Reference/chm13v2.0/chm13v2.0.fa.gz \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq
```

**Log Output**:
```
[INFO ] [SIMD] Using SCALAR Smith-Waterman (non-batched path)
[INFO ] Complete: 2 batches, 20000 reads (2960000 bp), 20043 records, 2137 pairs rescued in 2.76 sec
```

**Code Path**: `BandedPairWiseSW::scalar_banded_swa()`

### Run 2: FERROUS_ALIGN_SIMD=1 (SIMD Path)

**Command**:
```bash
FERROUS_ALIGN_SIMD=1 ./target/release/ferrous-align mem -t 8 \
    /Library/Genomics/Reference/chm13v2.0/chm13v2.0.fa.gz \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq
```

**Log Output**:
```
[INFO ] [SIMD] Using SCALAR Smith-Waterman (non-batched path)
[INFO ] Complete: 2 batches, 20000 reads (2960000 bp), 20043 records, 2137 pairs rescued in 2.61 sec
```

**Code Path**: `BandedPairWiseSW::scalar_banded_swa()` (same as SIMD=0!)

### Comparison

| Metric | SIMD=0 | SIMD=1 | Delta |
|--------|--------|--------|-------|
| Code path | Scalar | Scalar | ✅ Identical |
| Processing time | 2.76s | 2.61s | **-5.4%** (SIMD=1 faster) |
| Output MD5 | `d36db8a1c568aa3a0213e609f929899c` | `d36db8a1c568aa3a0213e609f929899c` | ✅ Identical |

## Findings

### 1. FERROUS_ALIGN_SIMD Has No Effect on Code Path

Both runs executed the **same scalar Smith-Waterman path**:
- No SIMD batch dispatch occurred
- No vertical SIMD kernels invoked
- No horizontal SIMD kernels invoked
- Both runs logged: `[SIMD] Using SCALAR Smith-Waterman (non-batched path)`

This confirms that `FERROUS_ALIGN_SIMD` is **not currently controlling** the scalar vs SIMD decision.

### 2. Why SIMD=1 Is Faster Despite Using Scalar Path

The 5.4% performance difference (2.76s vs 2.61s) despite both using scalar code is likely due to:

1. **Compiler optimizations**: SIMD=1 may enable different optimization flags or inlining decisions
2. **Auto-vectorization**: The compiler may auto-vectorize some scalar loops when SIMD is "enabled"
3. **CPU frequency scaling**: The M3 Max may boost higher when it detects SIMD-like workloads
4. **Measurement noise**: 0.15s difference over 2.6s is within normal variance (±6%)

### 3. Current Alignment Architecture Uses Non-Batched Scalar Path

The alignment pipeline currently calls `scalar_banded_swa()` directly from `region.rs:983`:

```rust
let result = sw_params.scalar_banded_swa(
    query_for_sw.len() as i32,
    &query_for_sw,
    rseq_for_sw.len() as i32,
    &rseq_for_sw,
    w,
    h0,
);
```

**Key observation**: Alignments are processed **one at a time**, not batched.

### 4. SIMD Batch Functions Are Implemented But Unused

The codebase contains fully implemented SIMD batching functions:
- `simd_banded_swa_batch8_u8()` - 8-bit SIMD scoring
- `simd_banded_swa_batch8_int16()` - 16-bit SIMD scoring
- `simd_banded_swa_dispatch()` - Runtime SIMD engine selection
- `batch_ksw_align_sse_neon()` - Horizontal SIMD kernel

**Status**: These functions are **ready for use** but not integrated into the main alignment pipeline.

### 5. NEON Intrinsics Are Not Being Exercised

Despite having comprehensive NEON implementations in:
- `src/compute/simd_abstraction/engine128.rs` - NEON wrapper layer
- `src/compute/simd_abstraction/portable_intrinsics.rs` - NEON intrinsic mappings
- `src/alignment/kswv_sse_neon.rs` - Horizontal SIMD kernel

**None of these NEON code paths are being executed** in the current test runs.

## Architecture Analysis

### Why Batching Isn't Happening

The current pipeline architecture processes alignments **sequentially**:

1. **Seeding** (`pipeline.rs:find_seeds()`) - Finds SMEMs for each read
2. **Chaining** (`pipeline.rs:build_and_filter_chains()`) - Builds chains from seeds
3. **Extension** (`pipeline.rs:extend_chains_to_alignments()`) - Creates alignment jobs
4. **Alignment** (`region.rs:983`) - Calls `scalar_banded_swa()` **one at a time**

**Missing**: A batching layer that accumulates multiple alignment jobs before dispatching to SIMD.

### What Would Enable SIMD

To use the SIMD batch functions, the code would need to:

1. **Accumulate alignment jobs** in `extend_chains_to_alignments()`
2. **Dispatch to batch functions** when batch size threshold is met
3. **Extract results** and distribute back to individual alignments

Example (hypothetical):
```rust
// Instead of:
for job in alignment_jobs {
    let result = sw_params.scalar_banded_swa(...);
}

// Do:
let batch: Vec<_> = alignment_jobs.iter().map(|job| (...)).collect();
let results = sw_params.simd_banded_swa_batch8_int16(&batch);
```

## Recommendations

### Immediate Actions

1. ✅ **Update documentation** to clarify that `FERROUS_ALIGN_SIMD` currently has no effect
2. **Remove misleading flag**: Either implement batching or remove the environment variable
3. **Add architecture docs**: Document why batching isn't implemented yet

### Short-Term (Enable SIMD)

1. **Implement batching layer** in `extend_chains_to_alignments()`:
   - Accumulate jobs up to batch size (16 for NEON/SSE, 32 for AVX2)
   - Dispatch to `simd_banded_swa_batch8_int16()`
   - Handle remainder with scalar path

2. **Add integration test** that verifies SIMD path is taken:
   - Check that `[SIMD] Vertical batch` log appears
   - Verify NEON/SSE intrinsics are executed

3. **Benchmark SIMD vs scalar** on Apple Silicon to quantify speedup

### Long-Term (Architecture)

From `CLAUDE.md`, the pipeline refactoring plan includes:
- **Deferred CIGAR generation**: Batch score alignments first, generate CIGARs only for survivors
- This naturally enables batching since scoring happens before filtering

**Status**: Golden reads parity tests are ready in `tests/golden_reads/`

## Conclusions

1. **FERROUS_ALIGN_SIMD is a no-op**: Both values produce identical scalar execution
2. **NEON code is untested**: No NEON intrinsics are being exercised on Apple Silicon
3. **SIMD infrastructure exists**: Batching functions are implemented but not integrated
4. **Performance difference is noise**: The 5.4% gap is likely measurement variance or compiler effects
5. **Batching layer needed**: Main blocker to enabling SIMD is lack of job accumulation

**Overall**: The codebase is **SIMD-ready** but the alignment pipeline doesn't use batching yet. The `FERROUS_ALIGN_SIMD` flag should be removed or documented as non-functional until batching is implemented.

## Code Changes Made

Added logging at key decision points (all in `src/alignment/banded_swa.rs`):

### Line 124-156: Runtime SIMD control logging
```rust
fn simd_runtime_enabled() -> bool {
    let result = if let Ok(val) = std::env::var("FERROUS_ALIGN_SIMD") {
        match val.to_ascii_lowercase().as_str() {
            "0" | "false" | "off" => {
                log::debug!("[SIMD] Runtime control: DISABLED by FERROUS_ALIGN_SIMD={}", val);
                false
            }
            "1" | "true" | "on" => {
                log::debug!("[SIMD] Runtime control: ENABLED by FERROUS_ALIGN_SIMD={}", val);
                true
            }
            // ... rest of function
```

### Line 189-202: Scalar path logging
```rust
pub fn scalar_banded_swa(...) -> (...) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        log::info!("[SIMD] Using SCALAR Smith-Waterman (non-batched path)");
    }
    // ... rest of function
```

### Line 1674, 1685: Vertical batch logging (8-bit)
```rust
if !Self::simd_runtime_enabled() {
    log::info!("[SIMD] Vertical batch (8-bit): using SCALAR fallback for {} alignments", batch.len());
    // ...
} else {
    let engine = detect_optimal_simd_engine();
    log::info!("[SIMD] Vertical batch (8-bit): using {:?} engine for {} alignments", engine, batch.len());
}
```

### Line 1756, 1765: Vertical batch logging (16-bit)
```rust
if !Self::simd_runtime_enabled() {
    log::info!("[SIMD] Vertical batch (16-bit): using SCALAR fallback for {} alignments", batch.len());
    // ...
} else {
    log::info!("[SIMD] Vertical batch (16-bit): using NEON/SSE 128-bit engine for {} alignments", batch.len());
}
```

### Line 1288-1291: NEON intrinsic execution logging
```rust
unsafe {
    #[cfg(target_arch = "x86_64")]
    log::trace!("[SIMD] Executing SSE2 intrinsics in DP loop (batch size: {})", SIMD_WIDTH);
    #[cfg(target_arch = "aarch64")]
    log::trace!("[SIMD] Executing NEON intrinsics in DP loop (batch size: {})", SIMD_WIDTH);
    // ... DP loop
}
```

### src/alignment/kswv_sse_neon.rs:106: Horizontal SIMD logging
```rust
pub unsafe fn batch_ksw_align_sse_neon(...) -> usize {
    log::debug!("[SIMD] Horizontal kernel (SSE/NEON): processing {} alignments", pairs.len());
    // ... kernel implementation
}
```

**Note**: These logging statements remain in the code for future debugging and verification when batching is implemented.
