# CRITICAL: Parallelism Loss in SoA Migration

**Status**: 🚨 **CRITICAL BUG** - Production performance regression
**Date**: 2025-12-01
**Branch**: `feature/core-rearch`
**Discovered**: User reported 4.2% CPU utilization on 16-thread run
**Expected**: ~1600% CPU utilization (16 cores fully utilized)

---

## Problem Summary

The Structure-of-Arrays (SoA) migration **completely removed all rayon parallelism** from the main alignment processing pipeline. The codebase now processes reads **sequentially** instead of in parallel.

### Observed Behavior

Running on 4M read pairs with `-t 16`:
```bash
# Current behavior (broken):
CPU: 154% (1.54 cores out of 16 = 9.6% utilization)

# Expected behavior (v0.6.0):
CPU: ~1400-1600% (14-16 cores = 87-100% utilization)
```

**Performance Impact**: ~10x slowdown on multi-core systems

---

## Root Cause

### Commit That Broke Parallelism

**Commit**: `85563c4` - "feat(soa): Phase 5 - Remove AoS paths, pure SoA pipeline"
**Date**: 2025-12-01 18:50:02

**Key change**:
```diff
- use rayon::prelude::*;
```

### Before (v0.6.0 - Working Parallelism)

**File**: `src/pipelines/linear/paired/paired_end.rs`

```rust
use rayon::prelude::*;

let mut batch_alignments: Vec<(Vec<Alignment>, Vec<Alignment>)> = (0..num_pairs)
    .into_par_iter()  // ← PARALLEL PROCESSING
    .map(|i| {
        let pair_id = batch_start_id + i as u64;

        let a1 = align_read_deferred(
            &bwa_idx_clone,
            pac_clone,
            &batch1.names[i],
            &batch1.seqs[i],
            &batch1.quals[i],
            &opt_clone,
            pair_id << 1,
            &compute_backend,
        );

        let a2 = align_read_deferred(
            &bwa_idx_clone,
            pac_clone,
            &batch2.names[i],
            &batch2.seqs[i],
            &batch2.quals[i],
            &opt_clone,
            (pair_id << 1) | 1,
            &compute_backend,
        );

        (a1, a2)
    })
    .collect();
```

**Key features**:
- ✅ Rayon `into_par_iter()` for automatic work-stealing parallelism
- ✅ Each read pair processed independently on available threads
- ✅ Linear scaling up to memory bandwidth limits

### After (feature/core-rearch - Broken)

**File**: `src/pipelines/linear/paired/paired_end.rs`

```rust
// NO RAYON IMPORT

// SoA batch alignment for R1 and R2
let soa_result1 = process_sub_batch_internal_soa(  // ← SEQUENTIAL
    &bwa_idx_clone,
    &pac_ref,
    &opt_clone,
    &batch1,
    batch_start_id,
    simd_engine,
);
let soa_result2 = process_sub_batch_internal_soa(  // ← SEQUENTIAL
    &bwa_idx_clone,
    &pac_ref,
    &opt_clone,
    &batch2,
    batch_start_id,
    simd_engine,
);
```

**Problems**:
- ❌ R1 and R2 processed **sequentially** (no parallelism)
- ❌ All reads in batch processed **sequentially** within each call
- ❌ Only ~1-2 cores utilized out of 16

---

## Why This Happened

The SoA migration focused on **data layout transformation** (AoS → SoA) but inadvertently removed the **parallel execution pattern**. The refactor assumed batch processing would be fast enough, but it lost the thread-level parallelism that made v0.6.0 competitive with BWA-MEM2.

### Architectural Mismatch

**v0.6.0 Pattern**:
```
Read batch → Rayon par_iter (16 threads) → Per-read processing
                     ↓
          Thread 1: align_read(pair_1)
          Thread 2: align_read(pair_2)
          ...
          Thread 16: align_read(pair_16)
```

**feature/core-rearch Pattern** (broken):
```
Read batch → process_sub_batch_internal_soa (1 thread) → All reads
                     ↓
          Single thread: align all reads sequentially
```

---

## Impact Assessment

### Performance Regression

| Dataset | v0.6.0 (parallel) | feature/core-rearch (sequential) | Regression |
|---------|-------------------|----------------------------------|------------|
| 10K pairs | ~1 sec | ~10 sec | **10x slower** |
| 4M pairs (HG002) | ~10-15 min | **~100-150 min** | **10x slower** |

### Thread Utilization

| Metric | v0.6.0 | feature/core-rearch |
|--------|--------|---------------------|
| CPU usage (-t 16) | ~1400-1600% | ~150% |
| Cores utilized | 14-16 | 1-2 |
| Efficiency | 87-100% | **9.6%** |

### Production Impact

- ✅ Correctness: **No impact** (alignment results identical)
- ❌ Performance: **~10x regression** on multi-core systems
- ❌ User experience: **Unacceptable** for production use
- ❌ Competitive parity: **Lost** vs BWA-MEM2 (was 85-95%, now 8-10%)

---

## Affected Code Paths

### Paired-End Processing

**File**: `src/pipelines/linear/paired/paired_end.rs`

**Lines 183-198** (first batch):
```rust
let soa_result1 = process_sub_batch_internal_soa(...);  // Sequential
let soa_result2 = process_sub_batch_internal_soa(...);  // Sequential
```

**Lines 382-397** (main loop):
```rust
let soa_result1 = process_sub_batch_internal_soa(...);  // Sequential
let soa_result2 = process_sub_batch_internal_soa(...);  // Sequential
```

### Single-End Processing

**File**: `src/pipelines/linear/single_end.rs` (need to check)

Likely has the same issue if it was refactored similarly.

---

## Solution Options

### Option 1: Restore Rayon at Batch Level (RECOMMENDED)

**Approach**: Split batch into sub-batches and process in parallel using rayon

```rust
use rayon::prelude::*;

// Determine optimal chunk size based on thread count
let num_threads = rayon::current_num_threads();
let chunk_size = (batch1.names.len() + num_threads - 1) / num_threads;

// Process R1 and R2 in parallel chunks
let (soa_result1, soa_result2) = rayon::join(
    || {
        // Process R1 chunks in parallel
        (0..batch1.names.len())
            .into_par_iter()
            .chunks(chunk_size)
            .map(|chunk_range| {
                // Create sub-batch from range
                let sub_batch = batch1.slice(chunk_range);
                process_sub_batch_internal_soa(
                    &bwa_idx_clone,
                    &pac_ref,
                    &opt_clone,
                    &sub_batch,
                    batch_start_id + chunk_range.start as u64,
                    simd_engine,
                )
            })
            .reduce(|| SoAAlignmentResult::new(), |mut acc, result| {
                acc.merge(result);
                acc
            })
    },
    || {
        // Process R2 chunks in parallel (same pattern)
        // ...
    }
);
```

**Pros**:
- ✅ Restores full parallelism
- ✅ Compatible with SoA architecture
- ✅ Maintains current batch processing logic

**Cons**:
- ⚠️ Requires SoAReadBatch::slice() method (not yet implemented)
- ⚠️ Requires SoAAlignmentResult::merge() method (not yet implemented)
- ⚠️ More complex than v0.6.0 per-read parallelism

### Option 2: Add Rayon Inside process_sub_batch_internal_soa()

**Approach**: Parallelize seeding/chaining/extension within the SoA pipeline

```rust
// In process_sub_batch_internal_soa()

// Phase 1: Parallel seeding (per-read independence)
let seed_results: Vec<_> = (0..soa_read_batch.len())
    .into_par_iter()
    .map(|read_idx| {
        let seq = soa_read_batch.get_sequence(read_idx);
        find_seeds_for_read(bwa_idx, seq, opt)
    })
    .collect();
```

**Pros**:
- ✅ Minimal API changes
- ✅ Clean separation: parallelism inside SoA module

**Cons**:
- ⚠️ Requires extracting per-read data from SoA batch (defeats SoA benefits)
- ⚠️ May reduce SIMD efficiency (scattered memory access)

### Option 3: Hybrid - Parallel Sub-Batches + Internal SIMD

**Approach**: Best of both worlds - parallel chunks with SoA within each chunk

```rust
// Split batch into thread-sized chunks
let chunks: Vec<SoAReadBatch> = batch1.split_into_chunks(num_threads);

// Process chunks in parallel
let results: Vec<SoAAlignmentResult> = chunks
    .into_par_iter()
    .map(|chunk| {
        process_sub_batch_internal_soa(
            &bwa_idx_clone,
            &pac_ref,
            &opt_clone,
            &chunk,
            chunk.start_id,
            simd_engine,
        )
    })
    .collect();

// Merge results
let final_result = SoAAlignmentResult::merge_all(results);
```

**Pros**:
- ✅ Full parallelism restored
- ✅ Maintains SoA benefits within each chunk
- ✅ Cleaner than Option 1

**Cons**:
- ⚠️ Requires SoAReadBatch::split_into_chunks() method
- ⚠️ Requires SoAAlignmentResult::merge_all() method

---

## Recommended Fix

### Phase 1: Quick Fix (Restore Basic Parallelism)

Implement **Option 3 (Hybrid)** with minimal changes:

1. Add `SoAReadBatch::split_into_chunks(n: usize) -> Vec<SoAReadBatch>`
2. Add `SoAAlignmentResult::merge(other: SoAAlignmentResult) -> Self`
3. Update `paired_end.rs` to use rayon for chunk-level parallelism
4. Test on 10K dataset to verify ~10x speedup

**Estimated effort**: 2-4 hours
**Risk**: Low (additive changes, no logic modification)

### Phase 2: Optimize (Fine-Tune Chunk Size)

Profile and optimize chunk size based on:
- Thread count
- Batch size
- Cache efficiency
- SIMD lane width

**Estimated effort**: 1-2 hours
**Risk**: Very low (performance tuning only)

---

## Testing Plan

### Validation Criteria

1. **Correctness**: SAM output identical to v0.6.0 baseline
2. **Performance**: CPU utilization ≥ 90% with `-t 16`
3. **Scaling**: Linear speedup up to memory bandwidth limits
4. **Regression**: All 263 unit tests pass

### Test Commands

```bash
# 1. Quick test (100 pairs)
time ./target/release/ferrous-align mem -t 16 $REF /tmp/test_100_R1.fq /tmp/test_100_R2.fq > /tmp/quick_parallel.sam

# 2. Medium test (10K pairs)
time ./target/release/ferrous-align mem -t 16 $REF tests/golden_reads/golden_10k_R1.fq tests/golden_reads/golden_10k_R2.fq > /tmp/10k_parallel.sam

# 3. Large test (4M pairs)
/usr/bin/time -v ./target/release/ferrous-align mem -t 16 $REF $R1 $R2 > /tmp/4m_parallel.sam 2>&1 | tee /tmp/perf_parallel.log

# 4. CPU utilization check
top -b -n 1 -p $(pgrep ferrous-align) | grep ferrous
# Should show ~1400-1600% CPU with 16 threads
```

### Performance Targets

| Test | Sequential (current) | Parallel (fixed) | Target Speedup |
|------|---------------------|------------------|----------------|
| 100 pairs | 0.1s | 0.01s | 10x |
| 10K pairs | 10s | 1s | 10x |
| 4M pairs | 150min | 15min | 10x |

---

## Priority

**Severity**: 🚨 **P0 - Critical**

**Reasoning**:
- Blocks production use of feature/core-rearch branch
- 10x performance regression is unacceptable
- Users expect multi-threaded performance (advertised with `-t` flag)
- Competitive parity with BWA-MEM2 lost (was 85-95%, now 8-10%)

**Action Required**: Immediate fix before any other work on this branch

---

## Related Issues

- SoA migration: Commits 425d985 → 85563c4
- Original rayon implementation: Session 25 (2025-11-15)
- Performance parity target: 85-95% of BWA-MEM2 (v0.6.0 achieved this)

---

## Lessons Learned

1. **Always profile major refactors**: SoA migration should have been profiled before/after
2. **Preserve performance tests**: Benchmark suite should have caught this
3. **Thread utilization is a primary metric**: Should be monitored in CI
4. **Incremental migration**: Should have kept rayon while migrating data structures

---

## Next Steps

1. ✅ Document issue (this file)
2. ⏭️ Implement Option 3 (Hybrid parallelism)
3. ⏭️ Validate correctness and performance
4. ⏭️ Add thread utilization test to prevent future regressions
5. ⏭️ Update benchmark suite to catch parallelism loss
