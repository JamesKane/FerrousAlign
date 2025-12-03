# v0.8.0 Baseline Performance Summary

**Date**: December 3, 2025
**Branch**: `feature/pipeline-structure`
**Commit**: (pipeline restructure complete)
**Dataset**: HG002 100K paired-end reads (200K total reads, 29.6 Mbases)
**Reference**: CHM13v2.0 (human genome, ~3.1 Gb)
**Hardware**: 16-thread system (exact CPU TBD)

---

## Performance Baseline

### Timing Results (100K Paired-End Reads)

| Run | Wall Time | Throughput (reads/sec) | Throughput (Mbases/sec) |
|-----|-----------|------------------------|-------------------------|
| Run 1 | 17.04s | 11,739 | 1.74 |
| Run 2 | 16.57s | 12,070 | 1.79 |
| Run 3 | 16.70s | 11,976 | 1.77 |
| **Average** | **16.77s** | **11,928** | **1.77** |

**Internal metrics** (from perf_stat.txt logs):
- Application reports: 18,009 reads/sec, 2.67 Mbases/sec (11.11s wall time)
- Note: Discrepancy due to index loading time included in total but not in pipeline processing

### Stage Breakdown (from logs)

| Stage | Time | % of Total |
|-------|------|-----------|
| Index loading (mmap) | 4.65s | 27.7% |
| Bootstrap insert size | ~0.5s | 3.0% |
| Main pipeline processing | 11.11s | 66.2% |
| Output writing | included | - |

**Pipeline throughput** (excluding index load): **18,009 reads/sec**

---

## Memory Baseline

### Peak Memory Usage

From `/usr/bin/time -v`:
- **Maximum resident set size**: 23,868 MB (~23.3 GB)
- **Average resident set size**: (not measured)
- **Page faults (major)**: (see time_verbose.txt)
- **Page faults (minor)**: (see time_verbose.txt)

### Memory Breakdown (Estimated)

| Component | Size (MB) | % of Total |
|-----------|-----------|------------|
| Index (mmap) | ~743 MB | 3.1% |
| Reference (.pac) | ~743 MB | 3.1% |
| Working memory | ~22,382 MB | 93.8% |

**Note**: This is **below the 24 GB target** already! The v0.8.0 goal of "reduce from ~32 GB to ~24 GB" appears to already be met.

---

## Threading Baseline

### CPU Utilization (from perf stat)

**Total time**:
- Wall time: 16.63s
- User time: 27.94s
- Sys time: 6.39s
- **Total CPU time**: 34.33s
- **CPU utilization**: 206.5% (out of 1600% max for 16 cores)

**Core utilization**: ~12.9% (only ~2 cores actively used out of 16)
⚠️ **Major optimization opportunity**: Very poor multi-threading efficiency!

### Performance Counters

| Metric | Value | Notes |
|--------|-------|-------|
| Cycles | 137.8 billion | |
| Instructions | 254.3 billion | 1.85 IPC (good) |
| Cache references | 5.15 billion | |
| Cache misses | 665.6 million | 12.92% miss rate (moderate) |
| Branches | 48.4 billion | |
| Branch misses | 405.1 million | 0.84% (excellent) |
| Context switches | 0 | (user-space only) |

**IPC**: 1.85 instructions per cycle (good - modern CPUs target 2-4)
**Cache miss rate**: 12.92% (moderate - room for improvement)
**Branch prediction**: 99.16% accuracy (excellent)

---

## Pairing Accuracy Baseline

### Current Status (from README)

| Metric | Value | BWA-MEM2 Target | Gap |
|--------|-------|-----------------|-----|
| Properly paired | 94.14% | 97.11% | -2.97pp ⚠️ |
| Mapping rate | 98.66% | 99.50% | -0.84pp |
| Mate diff chr | 1.90% | 1.51% | +0.39pp |
| Singletons | 1.05% | 0.30% | +0.75pp |
| Duplicates | 0% | 0% | 0 ✅ |

**Status**: Need to run fresh comparison on current build to verify these numbers still hold.

---

## Key Findings

### 1. ✅ Memory Already at Target

**Finding**: Peak memory is 23.3 GB, which is **already below** the 24 GB target for v0.8.0.

**Implication**: Memory optimization is **not a priority** for v0.8.0. Can defer to later version.

### 2. ⚠️ Threading is the Major Bottleneck

**Finding**: Only 206% CPU utilization out of 1600% available (16 cores) = **12.9% core utilization**

**Implication**: This is the **#1 performance opportunity**. With proper threading:
- Theoretical speedup: ~8x (if we can reach 80% utilization)
- Realistic speedup: ~4-5x (if we can reach 50-60% utilization)

**Priority**: Should be elevated to **#1 or #2** for v0.8.0, ahead of memory optimization.

### 3. Pipeline Throughput vs Wall Time

**Finding**: Pipeline reports 18K reads/sec but wall time shows 11.9K reads/sec

**Reason**: Index loading takes 4.65s (28% of total time)

**Implication**: For larger datasets, index load becomes negligible. Current throughput is likely representative of steady-state performance.

### 4. Cache Performance

**Finding**: 12.92% cache miss rate, 1.85 IPC

**Implication**: Cache locality is moderate. Some room for improvement but not critical.

### 5. Branch Prediction

**Finding**: 99.16% branch prediction accuracy

**Implication**: Control flow is very predictable. Good for CPU pipeline.

---

## Revised v0.8.0 Priorities (Based on Baselines)

### Original Plan

1. Pairing Accuracy (94.14% → 97%+)
2. Performance (79% of BWA-MEM2 → 85-90%)
3. Memory (~32 GB → 24 GB)
4. Threading (better utilization)

### Recommended Revision (Updated Dec 3 Afternoon)

1. **Pairing Accuracy** (93.80% → 97%+) - **UNCHANGED** (correctness first)
2. **Threading Regression Fix** (13% → restore previous) - **CRITICAL BUG** (broken in refactor)
3. **Performance Tuning** (cache, SIMD, batching) - **SAME LEVEL**
4. **Memory Optimization** - **DEPRIORITIZED** (already at target)

**Note**: Threading is not a design limitation but a regression from the v0.8.0 code reorganization.

### Expected Outcomes

If we achieve:
- **50% core utilization** (vs current 13%): **~4x speedup** → 47K reads/sec
- **60% core utilization**: **~5x speedup** → 59K reads/sec
- **80% core utilization** (optimistic): **~6x speedup** → 71K reads/sec

**Comparison to BWA-MEM2**: Need to establish BWA-MEM2 baseline on same hardware for apples-to-apples comparison.

---

## Next Steps

### Immediate (Today)

1. ✅ Performance baseline complete
2. ✅ Memory baseline complete
3. ✅ Threading baseline complete
4. ⏳ Run pairing accuracy comparison (need BWA-MEM2 output)

### Phase 2: Pairing Accuracy (Priority #1)

Follow plan in `v0.8.0_Completion_Plan.md` Phase 2.

### Phase 2b: Threading Investigation (NEW - Priority #1.5)

**Goal**: Understand why we're only using ~2 cores out of 16

**Investigation**:
1. Add instrumentation to orchestrators to measure stage-level parallelism
2. Profile with `perf record` to see where threads are waiting
3. Check Rayon thread pool configuration
4. Identify sequential bottlenecks

**Likely causes**:
- Sequential orchestration (stages run one after another, not pipelined)
- Small batch sizes limiting parallel work
- Lock contention in output writing
- Index access serialization

---

## Appendix: Raw Data

See files in this directory:
- `performance_baseline.md` - Timing runs
- `perf_stat.txt` - Performance counters
- `time_verbose.txt` - Detailed resource usage
- `run_simple_baseline.sh` - Reproducible script

---

**Document Version**: 1.0
**Author**: Claude Code + Profiling Tools
**Status**: Draft - Ready for Review
