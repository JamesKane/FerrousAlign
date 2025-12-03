# Threading Investigation Summary

**Date**: December 3, 2025
**Status**: Analysis Complete - No Quick Wins Found

---

## Key Findings

### 1. Root Cause Identified ✅

**Problem**: Only 12.9% core utilization (2 out of 16 cores)

**Why**:
- Batches are processed **sequentially** in a loop
- Only R1 and R2 within a single batch run in parallel via `rayon::join`
- Individual pipeline stages (seeding, chaining, extension) process reads sequentially

**Evidence**:
- CPU time: 34.3s (user+sys) / 16.6s (wall) = 2.07x parallelism
- Expected for 16 cores: 12-14x parallelism
- Performance profile shows 25% time in memory allocation (potential lock contention)

### 2. Batch Size Tuning: No Impact ❌

**Tested**: 100K, 200K, 500K batch sizes

**Results**:
| Batch Size | Mean Time | Relative |
|------------|-----------|----------|
| 100,000 | 16.356s | 1.00x (baseline) |
| 200,000 | 16.456s | 1.01x (+0.6%) |
| 500,000 | 16.458s | 1.01x (+0.6%) |

**Conclusion**: No measurable difference (all within noise). Makes sense for 100K read dataset (only 1-2 batches total).

---

## Available Optimization Paths

### Path 1: Within-Batch Parallelism (Recommended Next Step)

**Approach**: Use `par_iter` to parallelize read processing within each stage

**Target Stages**:
1. **Seeding** (`generate_smems_for_strand`): 15% of CPU time
2. **Extension** (`scalar_banded_swa`): 16% of CPU time

**Expected Speedup**: 2-4x (reach 50-60% core utilization)

**Effort**: 2-3 days

**Implementation**:
```rust
// In seeding stage
use rayon::prelude::*;

let seeds: Vec<_> = batch.reads.par_iter()
    .map(|read| generate_smems_for_strand(read, index))
    .collect();
```

**Risks**:
- Need to ensure BWT index access is thread-safe (likely already is via `&Index`)
- Memory allocations need to be thread-local to avoid contention
- Need extensive testing to ensure correctness

### Path 2: Batch-Level Pipeline Parallelism (More Complex)

**Approach**: Process multiple batches concurrently using channels

**Expected Speedup**: 2-3x additional (on top of Path 1)

**Effort**: 2-3 days

**Implementation**:
```rust
// Pseudocode
let (batch_tx, batch_rx) = channel();
let (aligned_tx, aligned_rx) = channel();

spawn_reader_thread(batch_tx);
spawn_alignment_workers(batch_rx, aligned_tx);
spawn_output_thread(aligned_rx);
```

**Risks**:
- Complex error handling across threads
- Careful channel sizing to avoid deadlocks
- Order preservation for reproducibility

---

## Recommendation

### ⚠️ **CORRECTION**: Threading is a Regression, Not a Design Limitation

**Updated Finding** (December 3, 2025 afternoon):

The poor threading performance (12.9% core utilization) is **NOT** a fundamental design issue requiring major refactoring. It's a **regression from the v0.8.0 code reorganization**.

**Evidence**:
- Previous versions had better parallelism
- Current implementation is missing parallel iterators in key stages
- The architecture supports parallelism; it's just not being utilized

**Root Cause**: During the pipeline restructure, stages were refactored but parallel iteration was likely not preserved in:
- Seeding stage (processing reads sequentially)
- Extension stage (processing alignments sequentially)
- Possibly chaining stage

**Fix Required**: Re-add `par_iter()` calls in stage implementations where they were removed during refactoring.

### Revised v0.8.0 Priorities

| Priority | Item | Status | Effort |
|----------|------|--------|--------|
| **#1** | Pairing Accuracy (93.80% → 97%+) | In progress | 3-5 days |
| **#2** | **Threading Regression Fix** | **Must fix for v0.8.0** | 1-2 days |
| **#3** | Performance tuning (cache, SIMD) | Not started | 2-3 days |
| **#4** | ~~Memory~~ → Already at target | Complete | N/A |

**Rationale**:
- Threading is a **regression bug**, not a feature enhancement
- Should be fixed as part of v0.8.0 to restore previous performance
- Pairing accuracy still highest priority (correctness first)

---

## If We DO Pursue Threading (v0.9.0+)

### Implementation Plan

**Phase 1**: Within-Batch Parallelism (2-3 days)

1. **Day 1**: Parallelize seeding stage
   - Add `par_iter` over reads
   - Ensure thread-safety
   - Benchmark on 100K dataset

2. **Day 2**: Parallelize extension stage
   - Add `par_iter` over alignments
   - Thread-local scratch buffers
   - Benchmark

3. **Day 3**: Testing & validation
   - Run on full datasets
   - Compare output bit-for-bit with baseline
   - Measure correctness (pairing rate, etc.)

**Phase 2**: Batch Pipelining (2-3 days, if needed)

4. **Days 4-5**: Implement pipeline architecture
5. **Day 6**: Testing & validation

**Total Effort**: 4-6 days with significant testing overhead

---

## Performance Ceiling Analysis

### Current State
- **Throughput**: 18K reads/sec (100K dataset)
- **Core utilization**: 12.9% (2 of 16 cores)

### With Within-Batch Parallelism (50% utilization)
- **Expected**: ~45K reads/sec (2.5x improvement)
- **Core utilization**: 50% (8 of 16 cores)

### With Full Parallelism (80% utilization)
- **Expected**: ~70K reads/sec (3.9x improvement)
- **Core utilization**: 80% (13 of 16 cores)

### Comparison to BWA-MEM2
- **Need BWA-MEM2 baseline** on same hardware to compare
- Current estimate: 79% of BWA-MEM2 (from README)
- With 3x speedup: Could reach 237% (2.4x faster than BWA-MEM2) 🎯

**This is attractive BUT**:
- Requires 4-6 days of work
- High risk to correctness
- Should come AFTER pairing accuracy is fixed

---

## Deliverables

Created in `documents/benchmarks/v0.8.0_baseline/`:

1. **BASELINE_SUMMARY.md** - Initial performance analysis
2. **THREADING_ANALYSIS.md** - Detailed threading investigation
3. **THREADING_SUMMARY.md** - This document (recommendations)
4. **batch_size_comparison.md** - Batch size benchmarks
5. **perf_report.txt** - Performance profile data

---

## Next Steps

### Immediate (Today)
1. ✅ Threading analysis complete
2. ✅ No quick wins found
3. ✅ Recommendation: Defer to v0.9.0

### Tomorrow
1. 🎯 **Start pairing accuracy work** (Priority #1)
2. Follow Phase 2 plan in `v0.8.0_Completion_Plan.md`
3. Generate BWA-MEM2 comparison baseline

### After Pairing is Fixed
1. Re-evaluate performance gap vs BWA-MEM2
2. Decide if threading work is worth the investment for v0.8.0
3. If yes: Implement within-batch parallelism
4. If no: Ship v0.8.0, plan threading for v0.9.0

---

**Document Version**: 1.0
**Author**: Claude Code
**Status**: Analysis Complete - Ready for Decision
