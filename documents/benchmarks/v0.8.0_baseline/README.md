# v0.8.0 Baseline Performance Analysis

This directory contains baseline profiling results for the v0.8.0 pipeline restructure.

## Files

### Summary Documents
- **`BASELINE_SUMMARY.md`** - Comprehensive analysis and key findings ⭐ **START HERE**
- `performance_baseline.md` - Raw timing results (3 runs)
- `perf_stat.txt` - Performance counters and CPU statistics
- `time_verbose.txt` - Detailed resource usage from `/usr/bin/time -v`

### Scripts
- `run_simple_baseline.sh` - Run performance/threading/memory baselines (fast, ~1 min)
- `run_baseline.sh` - Full baseline with valgrind (slow, requires hyperfine)
- `compare_pairing.sh` - Compare pairing accuracy with BWA-MEM2 (requires bwa-mem2)

## Quick Start

### 1. Run Baselines

```bash
# Fast baseline (recommended)
./documents/benchmarks/v0.8.0_baseline/run_simple_baseline.sh

# View results
cat documents/benchmarks/v0.8.0_baseline/BASELINE_SUMMARY.md
```

### 2. Compare with BWA-MEM2 (Optional)

Requires `bwa-mem2` installed:

```bash
./documents/benchmarks/v0.8.0_baseline/compare_pairing.sh
```

## Key Findings

From `BASELINE_SUMMARY.md`:

### ✅ Memory Already at Target
- **Current**: 23.3 GB peak
- **Target**: 24 GB
- **Status**: ACHIEVED - deprioritize memory optimization

### ⚠️ Threading is the Major Bottleneck
- **Current**: 12.9% core utilization (2 out of 16 cores)
- **Potential**: 4-6x speedup with proper threading
- **Status**: CRITICAL - elevate to top priority

### 📊 Performance Metrics
- **Throughput**: 18,009 reads/sec (pipeline only)
- **Wall time**: 16.77s average (including index load)
- **IPC**: 1.85 (good)
- **Cache miss rate**: 12.92% (moderate)

## Recommended v0.8.0 Priorities

Based on baseline analysis:

1. **Pairing Accuracy** (94.14% → 97%+) - Correctness first
2. **Threading Optimization** (13% → 50%+ utilization) - Major speedup opportunity
3. **Performance Tuning** (cache, SIMD, batching) - Incremental improvements
4. ~~Memory Optimization~~ - Already achieved, deprioritized

## Dataset

- **Reference**: CHM13v2.0 human genome (~3.1 Gb)
- **Reads**: HG002 100K paired-end reads (200K total, 29.6 Mbases)
- **Location**: `/home/jkane/Genomics/HG002/test_100k_R{1,2}.fq`

## Hardware

- **Threads**: 16 (exact CPU TBD)
- **SIMD**: AVX2 (256-bit, 32-way parallelism)
- **Memory**: Sufficient (>24 GB available)

## Reproducibility

All scripts are self-contained and can be re-run:

```bash
# Clean previous results
rm -rf documents/benchmarks/v0.8.0_baseline/*.{txt,md,out,sam}

# Re-run
./documents/benchmarks/v0.8.0_baseline/run_simple_baseline.sh
```

## Next Steps

See `documents/v0.8.0_Completion_Plan.md` for detailed implementation plan.
