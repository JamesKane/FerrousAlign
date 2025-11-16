# AVX2 Kernel Porting Guide

**Status**: Infrastructure complete, kernel implementation pending
**Goal**: Port C++ bwa-mem2's AVX2 banded Smith-Waterman kernel to Rust
**Expected Performance**: 1.8-2.2x speedup over current SSE implementation

---

## Current Status (2025-11-15)

### ✅ Completed Infrastructure

1. **SimdEngine Trait** (src/simd_abstraction.rs:24-129)
   - 28 operations abstracted (add, sub, max, min, load, store, etc.)
   - Zero-cost abstraction via associated types and monomorphization

2. **SimdEngine256 Implementation** (src/simd_abstraction.rs:687-877)
   - All 28 operations using `_mm256_*` AVX2 intrinsics
   - WIDTH_8 = 32, WIDTH_16 = 16 (double SSE parallelism)
   - Proper `#[target_feature(enable = "avx2")]` annotations

3. **CPU Feature Detection** (src/simd_abstraction.rs:883-935)
   - Runtime detection via `is_x86_feature_detected!("avx2")`
   - Automatic fallback to SSE on non-AVX2 CPUs

4. **Runtime Dispatch** (src/banded_swa.rs:832-908)
   - `simd_banded_swa_dispatch()`: Selects optimal implementation
   - Currently falls back to SSE (no AVX2 kernel yet)

### ⏳ Remaining Work: Implement AVX2 Kernel

Need to create `simd_banded_swa_batch32()` function that processes 32 alignments
in parallel using AVX2 instructions.

---

## C++ Reference Implementation

**File**: `/Users/jkane/Applications/bwa-mem2/src/bandedSWA.cpp`
**Function**: `BandedPairWiseSW::smithWaterman256_8` (lines 722-1150)

### Key Sections to Port

1. **Function Signature** (lines 722-733)
   ```cpp
   void BandedPairWiseSW::smithWaterman256_8(
       uint8_t seq1SoA[],        // Query sequences (Structure-of-Arrays)
       uint8_t seq2SoA[],        // Target sequences (SoA)
       uint8_t nrow,             // Number of target rows
       uint8_t ncol,             // Number of query columns
       SeqPair *p,               // Sequence pair metadata (32 pairs)
       uint8_t h0[],             // Initial scores
       uint16_t tid,             // Thread ID
       int32_t numPairs,         // Actual number of pairs (≤ 32)
       int zdrop,                // Z-drop threshold for early termination
       int32_t w,                // Band width
       uint8_t qlen[],           // Query lengths per lane
       uint8_t myband[]          // Band boundaries per lane
   )
   ```

2. **SIMD Constants Setup** (lines 735-741)
   ```cpp
   __m256i match256     = _mm256_set1_epi8(this->w_match);
   __m256i mismatch256  = _mm256_set1_epi8(this->w_mismatch);
   __m256i gapOpen256   = _mm256_set1_epi8(this->w_open);
   __m256i gapExtend256 = _mm256_set1_epi8(this->w_extend);
   __m256i gapOE256     = _mm256_set1_epi8(this->w_open + this->w_extend);
   __m256i w_ambig_256  = _mm256_set1_epi8(this->w_ambig);
   ```

3. **Query Profile Precomputation** (lines 756-806)
   - Precomputes match/mismatch scores for all target bases
   - Organized as: `profbuf[target_base][query_pos * 32 + lane]`
   - Avoids costly lookup in inner DP loop

4. **DP Matrix Initialization** (lines 808-860)
   - Initialize H, E, F matrices (32-way SoA layout)
   - First row/column initialization for gap penalties
   - Uses `_mm256_store_si256` for aligned stores

5. **Main DP Loop** (lines 862-1050)
   - Nested loop: `for i in 0..nrow { for j in 0..ncol }`
   - **MAIN_CODE8 Macro** (defined earlier in file, ~50 lines)
     - Computes H[i][j], E[i][j], F[i][j] using DP recurrence
     - Uses blend operations for conditional updates
     - Tracks max score and position per lane
   - Adaptive banding: Skip cells outside band [i-w, i+w+1]
   - Z-drop early termination: Stop lane if score drops > zdrop

6. **Result Extraction** (lines 1052-1150)
   - Extract max scores, positions from SIMD vectors
   - Store results in `SeqPair` struct array
   - Handle early-terminated lanes

### MAIN_CODE8 Macro (Critical!)

**Location**: bandedSWA.cpp:293-313 (AVX2 version)

This is the heart of the DP computation. It implements the Smith-Waterman recurrence:
```
H[i][j] = max(0,
              H[i-1][j-1] + score(query[j], target[i]),  // Match/mismatch
              E[i][j],                                    // Deletion
              F[i][j])                                    // Insertion
E[i][j] = max(H[i][j-1] - gap_open - gap_extend,
              E[i][j-1] - gap_extend)
F[i][j] = max(H[i-1][j] - gap_open - gap_extend,
              F[i-1][j] - gap_extend)
```

**AVX2 Operations Used**:
- `_mm256_add_epi8`: Add scores
- `_mm256_subs_epu8`: Saturating subtract (for gap penalties)
- `_mm256_max_epu8`: Max (for DP recurrence)
- `_mm256_cmpeq_epi8`: Comparison (for traceback)
- `_mm256_blendv_epi8`: Conditional select
- `_mm256_load_si256` / `_mm256_store_si256`: Memory access

---

## Rust Implementation Plan

### Step 1: Create Skeleton Function

Add to `src/banded_swa.rs` (after `simd_banded_swa_batch16`):

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_banded_swa_batch32_avx2(
    &self,
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
) -> Vec<OutScore> {
    use crate::simd_abstraction::SimdEngine256 as Engine;

    const SIMD_WIDTH: usize = 32; // Engine::WIDTH_8
    const MAX_SEQ_LEN: usize = 128;

    // TODO: Port C++ implementation
    // See AVX2_KERNEL_PORTING.md for detailed guide

    unimplemented!("AVX2 kernel not yet ported")
}

/// Public wrapper for AVX2 batch processing
#[cfg(target_arch = "x86_64")]
pub fn simd_banded_swa_batch32(
    &self,
    batch: &[(i32, &[u8], i32, &[u8], i32, i32)],
) -> Vec<OutScore> {
    unsafe { self.simd_banded_swa_batch32_avx2(batch) }
}
```

### Step 2: Port Data Structure Setup

1. Batch padding (32 lanes instead of 16)
2. SoA layout transformation (MAX_SEQ_LEN * 32)
3. Query profile precomputation (4 target bases × MAX_SEQ_LEN × 32)
4. DP matrix allocation (H, E, F: MAX_SEQ_LEN * 32)

### Step 3: Port DP Loop

This is the most complex part. Key changes:

1. **Replace intrinsics using SimdEngine256**:
   ```rust
   // Before (C++):
   __m256i sum = _mm256_add_epi8(a, b);

   // After (Rust):
   let sum = Engine::add_epi8(a, b);
   ```

2. **Use trait constants**:
   ```rust
   // Instead of hardcoded 32:
   for lane in 0..Engine::WIDTH_8 { ... }
   ```

3. **Handle vector types**:
   ```rust
   // C++: __m256i
   // Rust: Engine::Vec8 or Engine::Vec16
   ```

### Step 4: Port Result Extraction

Extract scores and positions from 32 lanes, handling:
- Early termination tracking
- Z-drop thresholds
- Max score positions

### Step 5: Update Dispatch

Change `banded_swa.rs:853-863`:
```rust
SimdEngineType::Engine256 => {
    // Use AVX2 kernel if available
    self.simd_banded_swa_batch32(batch)
}
```

### Step 6: Testing

1. **Unit tests**: Compare AVX2 results vs SSE results (should be identical)
2. **Performance tests**: Measure speedup with `cargo bench`
3. **Edge cases**: Empty sequences, very short sequences, Z-drop termination

---

## Common Pitfalls

### 1. Memory Alignment
- AVX2 requires 32-byte alignment for `_mm256_load_si256`
- Use `_mm256_loadu_si256` for unaligned access (slower)
- Rust `Vec<T>` is not guaranteed aligned - may need custom allocator

### 2. Buffer Sizes
- SSE buffers: `MAX_SEQ_LEN * 16`
- AVX2 buffers: `MAX_SEQ_LEN * 32` (2x memory!)
- May hit memory limits with very long sequences

### 3. Signed vs Unsigned
- C++ uses `uint8_t` for sequences, `int8_t` for scores
- Rust: Be careful with `u8` vs `i8` in intrinsics
- `_mm256_max_epu8` (unsigned) vs `_mm256_max_epi8` (signed)

### 4. Lane Masking
- Some lanes may be dummy (batch size < 32)
- Need to mask operations to avoid processing garbage data
- C++ uses blend operations for this

### 5. Early Termination
- Z-drop check requires per-lane state tracking
- Terminated lanes should not affect max score
- Need careful masking in result extraction

---

## Performance Expectations

### Theoretical Speedup
- AVX2 has 2x the lanes of SSE (32 vs 16)
- **Ideal**: 2.0x speedup

### Realistic Speedup
- Memory bandwidth bound (loading sequences)
- Increased cache pressure (2x buffer sizes)
- **Expected**: 1.8-2.2x speedup

### Bottlenecks
1. **Memory bandwidth**: Loading query/target sequences
2. **Cache misses**: Larger working set (2x buffers)
3. **Branch prediction**: Lane masking and early termination

### Measurement
```bash
# Benchmark SSE version
cargo bench --bench simd_benchmarks -- batch16

# Benchmark AVX2 version (after implementation)
cargo bench --bench simd_benchmarks -- batch32

# Compare results
# Look for ~1.8-2.2x improvement in throughput
```

---

## Estimated Effort

**Total**: ~16-24 hours for experienced Rust + SIMD developer

**Breakdown**:
- Data structure setup: 2-3 hours
- DP loop porting: 6-8 hours (most complex)
- Result extraction: 2-3 hours
- Testing and debugging: 4-6 hours
- Performance tuning: 2-4 hours

**Complexity**: High
- 400+ lines of intricate SIMD code
- Subtle correctness issues (alignment, masking, termination)
- Performance tuning (alignment, prefetching, loop unrolling)

---

## Success Criteria

1. ✅ **Correctness**: AVX2 results match SSE results exactly
2. ✅ **Performance**: 1.8-2.2x speedup over SSE on AVX2 CPUs
3. ✅ **Fallback**: Graceful fallback to SSE on non-AVX2 CPUs
4. ✅ **Tests**: All existing tests pass + new AVX2-specific tests
5. ✅ **Documentation**: Inline comments explaining AVX2-specific logic

---

## References

- **C++ Implementation**: `/Users/jkane/Applications/bwa-mem2/src/bandedSWA.cpp:722-1150`
- **MAIN_CODE8 Macro**: `/Users/jkane/Applications/bwa-mem2/src/bandedSWA.cpp:293-313`
- **SimdEngine256**: `src/simd_abstraction.rs:687-877`
- **Intel Intrinsics Guide**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- **Rust SIMD**: https://doc.rust-lang.org/core/arch/index.html
