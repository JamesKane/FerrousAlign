# Performance Analysis - FerrousAlign

**Last Updated**: 2025-11-27
**Version**: v0.6.0+

## Executive Summary

FerrousAlign achieves **79% of BWA-MEM2 performance** on production workloads (4M HG002 read pairs), meeting the 70-90% target for the v0.7.0 release milestone.

| Metric | BWA-MEM2 | FerrousAlign | Ratio |
|--------|----------|--------------|-------|
| Time (4M reads) | 2:18.91 (138.9s) | 2:55.61 (175.6s) | **79%** |
| Memory | ~17 GB | ~32 GB | 1.9x higher |
| Threads | 16 | 16 | Same |

**Key Achievement**: From 5.14x slower (Nov 2024) to 1.26x slower (Nov 2025) - a **4x improvement** in relative performance.

---

## Current Benchmark Results

### 4M HG002 Read Pairs (Production Workload)

**Platform**: AMD Ryzen 9 7900X (16 threads)
**Reference**: GRCh38 (human genome, 3.1 GB)
**Data**: HG002 WGS paired-end reads (4M pairs, gzipped)

| Tool | Wall Time | CPU Time | Memory | Throughput |
|------|-----------|----------|--------|------------|
| BWA-MEM2 | 2:18.91 | N/A | ~17 GB | ~29K reads/sec |
| FerrousAlign | 2:55.61 | 2130s | 32 GB | ~23K reads/sec |

**CPU Utilization**: 2130s CPU / 175.6s wall = **12.1x** (76% of 16 threads)

### 10K Golden Reads (Small Dataset)

| Tool | Wall Time | Memory | Notes |
|------|-----------|--------|-------|
| BWA-MEM2 | 7.15s | 16.8 GB | Index loading dominates |
| FerrousAlign | 5.26s | 19.7 GB | **136% faster** |

**Note**: Small datasets show different characteristics due to cache effects and index loading overhead.

---

## Hotspot Analysis (perf profiling)

### 100K HG002 Reads Profile

| Rank | Function | % CPU | Category |
|------|----------|-------|----------|
| 1 | `simd_banded_swa_batch16_int16` | 23.2% | AVX2 SW kernel |
| 2 | `scalar_banded_swa` | 12.8% | CIGAR generation |
| 3 | `batch_ksw_align_avx2` | 12.6% | AVX2 KSW kernel |
| 4 | `generate_smems_for_strand` | 9.5% | Seeding |
| 5 | `get_bwt` | 4.7% | FM-Index |
| 6 | `generate_smems_from_position` | 4.6% | Seeding |
| 7 | `_int_malloc` | 3.4% | Memory allocation |
| 8 | `forward_only_seed_strategy` | 2.4% | Seeding |
| 9 | `__memmove_avx512_unaligned_erms` | 2.4% | Memory copy |
| 10 | `_int_free_chunk` | 2.4% | Memory deallocation |

### Category Breakdown

| Category | Total % | Components |
|----------|---------|------------|
| **SIMD Alignment** | 35.8% | batch16_int16 + batch_ksw_avx2 |
| **Scalar CIGAR** | 12.8% | generate_cigar_from_region |
| **Seeding** | ~18% | SMEMs, BWT, SA lookups |
| **Memory Ops** | ~9.5% | malloc, free, memmove |
| **Other** | ~24% | I/O, finalization, etc. |

---

## SIMD Architecture

### Supported Engines

| Engine | Width | Batch Size | Availability |
|--------|-------|------------|--------------|
| SSE2/NEON | 128-bit | 8-16 | Default (all platforms) |
| AVX2 | 256-bit | 16-32 | x86_64 (auto-detected) |
| AVX-512 | 512-bit | 32-64 | x86_64 (`--features avx512`) |

### SIMD Abstraction Layer

The codebase uses a `SimdEngine` trait abstraction for portable SIMD:

- `SimdEngine128` - SSE2/NEON (baseline)
- `SimdEngine256` - AVX2 (2x width)
- `SimdEngine512` - AVX-512 (4x width, feature-gated)

**Note**: AVX-512 kernels (`kswv_avx512.rs`, `banded_swa_avx512.rs`) intentionally use raw intrinsics for native k-mask operations, documented as acceptable deviations from the abstraction layer.

### Batch Smith-Waterman Performance

**Test**: 128 alignments, 100bp sequences

| SIMD Engine | Scalar Time | SIMD Time | Speedup | Throughput |
|-------------|-------------|-----------|---------|------------|
| ARM NEON (M3 Max) | 1.975 ms | 1.366 ms | **1.45x** | 93.7 Kelem/s |
| SSE2 | 3.10 ms | 2.10 ms | **1.48x** | 60.9 Kelem/s |
| AVX2 (Ryzen 9 7900X) | 3.11 ms | 2.08 ms | **1.50x** | 61.7 Kelem/s |
| AVX-512 | 3.11 ms | 1.64 ms | **1.90x** | 77.8 Kelem/s |

**Note**: ARM NEON shows highest absolute throughput due to Apple Silicon's efficient microarchitecture, despite having only 128-bit vector width (vs 256-bit AVX2)

---

## Optimization Opportunities

### High Impact (Potential 5-15% overall)

1. **Reduce scalar CIGAR generation overhead (12.8%)**
   - Currently one-at-a-time after batch scoring
   - Batch CIGAR generation not easily parallelizable (per-alignment traceback)
   - Pre-allocate buffers to reduce malloc/free

2. **Memory allocation reduction (9.5%)**
   - Thread-local arenas for alignment buffers
   - Reduce Vec growth/reallocation in hot paths
   - Pre-allocate workspace buffers (partially implemented)

3. **Seeding optimization (18%)**
   - Vectorize FM-Index backward search
   - Cache suffix array lookups
   - Prefetch BWT data

### Medium Impact (Potential 2-5% overall)

4. **Memory copy reduction**
   - Avoid cloning reference segments
   - Use references where possible

5. **Threading tuning**
   - Better batch sizing for thread utilization
   - NUMA-aware memory allocation

### Low Priority

6. **AVX-512 stabilization**
   - Currently requires nightly Rust
   - Wait for compiler stabilization

---

## Memory Usage

### Breakdown (4M HG002)

| Component | Size | Notes |
|-----------|------|-------|
| FM-Index | ~12 GB | BWT + checkpoints |
| Suffix Array | ~3 GB | Sampled every 8 positions |
| Reference | ~3 GB | 2-bit packed |
| Working Set | ~14 GB | Alignment buffers, batches |
| **Total** | **~32 GB** | vs BWA-MEM2 ~17 GB |

### Optimization Target

- Current: 32 GB
- Target: 24 GB (25% reduction)
- BWA-MEM2: ~17 GB

---

## Platform Comparison

### x86_64: AMD Ryzen 9 7900X (16 threads)

| Workload | BWA-MEM2 | FerrousAlign | Ratio |
|----------|----------|--------------|-------|
| 4M HG002 reads | 2:18.91 | 2:55.61 | **79%** |
| 10K Golden reads | 7.15s | 5.26s | 136% (faster) |

### Apple Silicon: M3 Max (16 threads, NEON)

**Note**: BWA-MEM2 has no NEON port, so direct comparison is not possible. Performance is inferred relative to x86 results.

| Workload | FerrousAlign | Throughput | Memory |
|----------|--------------|------------|--------|
| 10K reads (20K total) | **0.95s** alignment | 21K reads/sec | ~20 GB |
| 100K reads (200K total) | **5.6s** alignment | 36K reads/sec | ~20 GB |

**Benchmark Details (100K HG002 reads, M3 Max)**:
- Index load time: ~5-7s (mmap)
- Alignment time: 5.5-5.8s (excluding index load)
- CPU utilization: 64s CPU / 5.7s wall = **11.2x** (70% of 16 threads)
- Peak memory: ~20 GB

**Inferred BWA-MEM2 Equivalence**:
Based on the x86 ratio (79%), if BWA-MEM2 had NEON support, estimated times would be:
- 100K reads: ~4.4s (vs FerrousAlign 5.6s)
- Projected 4M reads: ~2:20 (vs FerrousAlign ~2:57)

### Cross-Platform Summary

| Platform | SIMD Engine | 100K Throughput | Notes |
|----------|-------------|-----------------|-------|
| AMD Ryzen 9 7900X | AVX2 (256-bit) | ~23K reads/sec | Primary benchmark platform |
| Apple M3 Max | NEON (128-bit) | ~36K reads/sec | Faster per-core, no BWA-MEM2 comparison |

**Key Observation**: Apple Silicon achieves higher throughput despite 128-bit SIMD (vs 256-bit AVX2) due to superior memory bandwidth and efficient core architecture

---

## Historical Progress

| Date | Version | Performance vs BWA-MEM2 | Key Changes |
|------|---------|------------------------|-------------|
| Nov 2024 | v0.4.x | 19% (5.14x slower) | Initial multi-threaded |
| Nov 2025 | v0.5.x | ~50% | SIMD routing fixes |
| Nov 2025 | v0.6.0 | **79%** | GATK parity, optimizations |

**4x improvement** in relative performance over one year.

---

## Benchmarking Methodology

### Standard Configuration

```bash
# Build with native optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run benchmark
/usr/bin/time -v ./target/release/ferrous-align mem -t 16 \
    reference.fna reads_R1.fq.gz reads_R2.fq.gz > output.sam

# Profile with perf
perf record -g ./target/release/ferrous-align mem ...
perf report --stdio --no-children --percent-limit 1
```

### Test Data

- **10K Golden Reads**: `tests/golden_reads/golden_10k_R{1,2}.fq`
- **100K Test** (Linux): `/home/jkane/Genomics/HG002/test_100k_R{1,2}.fq`
- **4M Full Dataset**: `/home/jkane/Genomics/HG002/2A1_CGATGT_L001_R{1,2}_001.fastq.gz` (Linux)
- **4M Full Dataset**: `/Users/jkane/Genomics/HG002/2A1_CGATGT_L001_R{1,2}_001.fastq.gz` (macOS)
- **Reference** (Linux): GRCh38 no-alt (`GCA_000001405.15_GRCh38_no_alt_analysis_set.fna`)
- **Reference** (macOS): `/Library/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna`

---

## Conclusion

FerrousAlign v0.6.0+ achieves **79% of BWA-MEM2 performance**, meeting the v0.7.0 release target of 70-90%. Key remaining optimization opportunities:

1. Memory allocation overhead (~9.5% CPU)
2. Scalar CIGAR generation bottleneck (~13% CPU)
3. Memory usage reduction (32 GB → 24 GB target)

The implementation is now production-viable for most use cases, with performance competitive with the C++ reference implementation.
