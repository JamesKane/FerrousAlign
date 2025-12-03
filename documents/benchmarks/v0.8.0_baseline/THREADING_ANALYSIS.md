# Threading Bottleneck Analysis

**Date**: December 3, 2025
**Branch**: `feature/pipeline-structure`
**Finding**: Only 12.9% core utilization (2 out of 16 cores)

---

## Executive Summary

**Root Cause**: Batches are processed **sequentially** in a single-threaded loop.

**Current Architecture**:
```
Batch 1: [Load] → [R1||R2 Pipeline] → [Pair] → [Rescue] → [Output] → wait...
Batch 2:                                                                  [Load] → [R1||R2 Pipeline] → ...
Batch 3:                                                                                                 [Load] → ...
```

**Only parallelism**: R1 and R2 are processed in parallel using `rayon::join` (2 threads active)

**Potential Speedup**: **4-8x** with pipeline parallelism across batches

---

## Detailed Analysis

### 1. Current Parallelism (orchestrator/paired_end/mod.rs)

#### ✅ What IS Parallelized

**Line 125-128**: R1/R2 processing uses `rayon::join`
```rust
let (result1, result2) = rayon::join(
    || self.process_single_batch(batch1, &ctx1),
    || self.process_single_batch(batch2, &ctx2),
);
```

This splits work across 2 threads:
- Thread 1: Processes R1 through seeding → chaining → extension
- Thread 2: Processes R2 through seeding → chaining → extension

#### ❌ What is NOT Parallelized

**Line 278-356**: Main batch loop is sequential
```rust
loop {
    let batch1 = reader1.read_batch(...)?;  // Sequential read
    let batch2 = reader2.read_batch(...)?;

    // Process batch (uses 2 threads via rayon::join)
    let (soa_result1, soa_result2) = self.process_pair_batches(...)?;

    // Sequential post-processing
    self.pair_alignments_aos(...);
    self.perform_mate_rescue(...);
    self.write_paired_output(...)?;

    // Next batch starts AFTER previous batch completes
}
```

**Timeline for 100K reads (2 batches)**:

| Time | Thread 1 | Thread 2 | Threads 3-16 |
|------|----------|----------|--------------|
| 0-5s | Batch 1 R1 | Batch 1 R2 | **IDLE** |
| 5-8s | Pair/Rescue/Output | (may help) | **IDLE** |
| 8-13s | Batch 2 R1 | Batch 2 R2 | **IDLE** |
| 13-16s | Pair/Rescue/Output | (may help) | **IDLE** |

**CPU Utilization**: ~12.9% (2 active threads / 16 total)

### 2. Stage-Level Parallelism

Checked if individual stages use parallel iterators:

| Stage | File | Parallelism |
|-------|------|-------------|
| Seeding | `seeding/` | ❌ None found |
| Chaining | `chaining/` | ❌ None found |
| Extension | `batch_extension/` | ✅ Some (finalize_soa.rs) |
| Pairing | `paired/pairing_aos.rs` | ❌ None found |
| Rescue | `paired/mate_rescue.rs` | ✅ Uses rayon |

**Observation**: Most compute-heavy stages (seeding, chaining) process reads **sequentially** within a batch.

### 3. Performance Profile Hotspots

From `perf report`:

| Function | % CPU | Stage | Parallelism |
|----------|-------|-------|-------------|
| `scalar_banded_swa` | 15.97% | Extension | Sequential per-alignment |
| `generate_smems_for_strand` | 15.26% | Seeding | Sequential per-read |
| `malloc`/`free` | 24.53% | All | Lock contention? |
| `get_bwt` | 8.32% | Seeding (BWT lookup) | Sequential |

**Key Insight**:
- Extension (alignment) takes 16% → Could parallelize across alignments
- Seeding takes 15% → Could parallelize across reads
- Memory allocation takes 25% → May have lock contention

---

## Quick Win Opportunities

### Option 1: Batch-Level Pipeline Parallelism ⭐ **HIGHEST IMPACT**

**Concept**: Process multiple batches concurrently using a pipeline

**Implementation**:
```rust
// Use crossbeam channels to pipeline batches
let (batch_tx, batch_rx) = crossbeam::channel::bounded(2);
let (aligned_tx, aligned_rx) = crossbeam::channel::bounded(2);

rayon::scope(|s| {
    // Reader thread
    s.spawn(|_| {
        while let Ok(batch) = read_next_batch() {
            batch_tx.send(batch).unwrap();
        }
    });

    // Alignment workers (use Rayon thread pool)
    s.spawn(|_| {
        for batch in batch_rx {
            let result = process_pair_batches(batch);
            aligned_tx.send(result).unwrap();
        }
    });

    // Output thread
    for (aligns1, aligns2) in aligned_rx {
        pair_and_output(aligns1, aligns2);
    }
});
```

**Expected Speedup**: **2-3x** (overlap I/O, compute, and output)

**Effort**: 1-2 days
**Risk**: Medium (careful channel management, error handling)

### Option 2: Parallelize Within-Batch Processing ⭐ **MEDIUM IMPACT**

**Concept**: Use `par_iter` for read-level parallelism within each stage

**Seeding Example**:
```rust
// Current (sequential)
for read in batch.iter() {
    let seeds = generate_smems(read);
    results.push(seeds);
}

// Parallel
use rayon::prelude::*;
let results: Vec<_> = batch.par_iter()
    .map(|read| generate_smems(read))
    .collect();
```

**Expected Speedup**: **2-4x** (spread work across all 16 cores)

**Effort**: 2-3 days (need to ensure thread-safety)
**Risk**: Medium (need thread-local BWT lookups, careful memory management)

### Option 3: Reduce Memory Allocation Overhead 🎯 **LOW-HANGING FRUIT**

**Observation**: 25% of CPU time in malloc/free

**Solutions**:
1. **Thread-local allocators**: Use bump allocators for per-batch temporary data
2. **Object pooling**: Reuse `Vec` allocations across batches
3. **Arena allocation**: Allocate once per batch, free in bulk

**Expected Speedup**: **5-10%** (reduce lock contention)

**Effort**: 1 day
**Risk**: Low

### Option 4: Increase Batch Size 📊 **EASIEST**

**Current**: batch_size = 100,000 reads (default from mem_opt.rs)

**Hypothesis**: Larger batches = more parallelizable work per batch

**Test**:
```bash
# Current (100K)
./target/release/ferrous-align mem -t 16 ...

# Double batch size
./target/release/ferrous-align mem -t 16 --batch-size 200000 ...
```

**Expected Speedup**: **10-20%** (amortize overhead)

**Effort**: 5 minutes (just a flag)
**Risk**: None (may increase memory)

---

## Recommended Approach

### Phase 1: Quick Wins (1-2 days)

1. **Test batch size impact** (5 min)
   - Try 200K, 500K, 1M batch sizes
   - Measure throughput vs memory

2. **Reduce allocation overhead** (1 day)
   - Add object pools for `Vec<Seed>`, `Vec<Chain>`, etc.
   - Use thread-local scratch buffers

**Expected**: 15-25% improvement with minimal risk

### Phase 2: Within-Batch Parallelism (2-3 days)

3. **Parallelize seeding stage** (1-2 days)
   - Use `par_iter` over reads in batch
   - Ensure BWT lookups are thread-safe (likely already are via shared `&Index`)

4. **Parallelize extension stage** (1 day)
   - Already partially parallel, expand to all alignments

**Expected**: 2-3x improvement (reach ~50% core utilization)

### Phase 3: Batch Pipeline (2-3 days, if needed)

5. **Implement batch-level pipelining** (2-3 days)
   - Only if Phase 1+2 don't hit 80%+ of BWA-MEM2 performance

**Expected**: Additional 1.5-2x improvement

---

## Measurements to Track

After each change, measure:

```bash
# 1. Throughput
time ./target/release/ferrous-align mem -t 16 $REF $R1 $R2 > /dev/null

# 2. Core utilization
perf stat -e context-switches,cpu-clock ./target/release/ferrous-align mem -t 16 $REF $R1 $R2

# 3. CPU percentage
/usr/bin/time -v ./target/release/ferrous-align mem -t 16 $REF $R1 $R2
# Look for "Percent of CPU this job got"
```

**Target**:
- User+Sys time ≥ 10x wall time (for 16 cores)
- Currently: 34s / 16.6s = 2.0x (only using 2 cores effectively)

---

## Risk Assessment

| Approach | Speedup | Effort | Risk | Correctness Impact |
|----------|---------|--------|------|-------------------|
| Batch size tuning | 10-20% | 5 min | None | None |
| Allocation pools | 5-10% | 1 day | Low | None |
| Within-batch parallel | 2-4x | 2-3 days | Medium | **Need testing** |
| Batch pipelining | 2-3x | 2-3 days | Medium | **Need testing** |

**Recommendation**: Start with low-risk options (batch size, allocations), then evaluate if more aggressive parallelism is needed.

---

## Next Steps

1. **Immediate** (today):
   - Test batch size impact: run with `--batch-size 200000`
   - Document baseline with different batch sizes

2. **Phase 1** (tomorrow):
   - Implement object pooling for Vec allocations
   - Benchmark improvement

3. **Phase 2** (next 2-3 days):
   - Implement `par_iter` in seeding stage
   - Ensure thread-safety and correctness

4. **Decision point**:
   - If we hit 85%+ of BWA-MEM2 → move to pairing accuracy work
   - If not → consider batch pipelining

---

**Document Version**: 1.0
**Author**: Claude Code + Performance Analysis
**Status**: Analysis Complete - Ready for Implementation
