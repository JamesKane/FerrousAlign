# Learned Index for Suffix Array Lookup

## Overview

This document outlines the design for accelerating suffix array (SA) lookups using a learned index based on the [Sapling](https://academic.oup.com/bioinformatics/article/37/6/744/5941464) approach. Unlike BWA-MEME's heavy P-RMI (which requires 38-118 GB RAM), Sapling uses a compact piecewise linear model that adds **less than 1% memory overhead** while achieving **2x speedup**.

**Target Hardware**: Mac Studios (32-192 GB), high-end laptops (16-64 GB)

**Priority**: Post-1.x (after NPU Seed Filter)

## Executive Summary

| Aspect | BWA-MEME (P-RMI) | **Sapling (Ours)** |
|--------|------------------|-------------------|
| Memory overhead | 38-118 GB | **<1% of SA (~30 MB)** |
| Speedup | 3.32x | **2.0-2.5x** |
| Model type | 2^28 leaf models | Piecewise linear (~64K segments) |
| Training | Complex, slow | **Simple, fast (<1 min)** |
| Compatibility | Identical SAM | Identical SAM |

**Rationale**: BWA-MEME's 38 GB minimum is incompatible with our target platforms. Sapling's approach provides substantial speedup with negligible memory cost.

## Background

### The Problem: Cache Misses in SA Lookup

Binary search on the suffix array causes O(log n) random memory accesses per lookup. For the human genome (n ≈ 3 billion), this means ~32 cache misses per lookup - the dominant cost in seeding.

```
Traditional SA Lookup:
┌─────────────────────────────────────────────────────────────┐
│  Binary search: 32 iterations × cache miss ≈ 32 × 100ns    │
│  Total: ~3.2 μs per lookup                                  │
└─────────────────────────────────────────────────────────────┘

Learned Index SA Lookup:
┌─────────────────────────────────────────────────────────────┐
│  Model prediction: 1 lookup (~10ns)                         │
│  Bounded search: ~3-5 iterations × cache miss               │
│  Total: ~0.5-1.0 μs per lookup (2-3x faster)               │
└─────────────────────────────────────────────────────────────┘
```

### Why Not BWA-MEME?

BWA-MEME achieves 3.32x speedup but requires:
- **Minimum 38 GB RAM** (compact mode)
- **118 GB RAM** for full acceleration
- **64-bit suffix storage** for each SA position

This exceeds our target platforms (Mac Studios, laptops with 32-64 GB).

### Sapling's Insight

[Sapling](https://github.com/mkirsche/sapling) (Kirsche & Schatz, 2021) observed that:
1. The suffix array maps suffixes to positions in a **nearly monotonic** way
2. A simple piecewise linear function can approximate this mapping
3. Prediction errors are bounded and small (~100-1000 positions)
4. The model adds **<1% memory overhead**

## Architecture

### Piecewise Linear Model

Instead of storing millions of neural network parameters, we store a small number of linear segments:

```
Suffix (lexicographic) → SA Position (genomic)

         SA Position
              ▲
              │      ╱
              │     ╱  Segment 3
              │    ╱
              │   ╱
              │  ╱  Segment 2
              │ ╱
              │╱  Segment 1
              └──────────────────► Suffix Value
                (2-bit encoded)
```

Each segment is defined by:
- Start suffix value (u64)
- Slope (f32)
- Intercept (f32)

### Data Structures

```rust
/// Compact learned index using piecewise linear approximation.
/// Memory overhead: ~0.5-1% of suffix array size.
#[derive(Clone)]
pub struct SaplingIndex {
    /// Piecewise linear segments (typically 32K-128K segments)
    segments: Vec<LinearSegment>,

    /// Maximum prediction error across all segments
    /// Determines the bounded search range
    max_error: u32,

    /// Number of bits used for segment lookup (log2 of segment count)
    /// Typical: 15-17 bits for 32K-128K segments
    index_bits: u8,
}

/// A single linear segment of the piecewise approximation.
#[derive(Clone, Copy)]
pub struct LinearSegment {
    /// Slope: change in SA position per unit suffix value
    slope: f32,
    /// Intercept: SA position when suffix value is at segment start
    intercept: f32,
}

impl SaplingIndex {
    /// Predict SA position for a suffix.
    /// Returns (predicted_position, error_bound).
    #[inline]
    pub fn predict(&self, suffix_prefix: u64) -> (u64, u32) {
        // Use high bits of suffix to select segment
        let segment_idx = (suffix_prefix >> (64 - self.index_bits)) as usize;
        let segment = &self.segments[segment_idx.min(self.segments.len() - 1)];

        // Linear prediction: pos = slope * suffix + intercept
        let predicted = (segment.slope as f64 * suffix_prefix as f64
                        + segment.intercept as f64) as u64;

        (predicted, self.max_error)
    }

    /// Size in bytes (for memory accounting).
    pub fn size_bytes(&self) -> usize {
        self.segments.len() * std::mem::size_of::<LinearSegment>()
            + std::mem::size_of::<Self>()
    }
}
```

### Memory Analysis

For human genome (3.1 billion bases):

| Component | Size |
|-----------|------|
| Suffix array (baseline) | ~24 GB (8 bytes × 3.1B) |
| BWT + checkpoints | ~1.5 GB |
| **Sapling index (64K segments)** | **~0.5 MB** |
| **Sapling index (128K segments)** | **~1 MB** |

**Total overhead: 0.002-0.004%** of the existing index.

Even with 256K segments for higher accuracy: **~2 MB** (0.008%).

### Comparison with BWA-MEME

| Metric | BWA-MEME (min) | Sapling |
|--------|----------------|---------|
| Model size | ~2 GB | ~1 MB |
| 64-bit suffixes | ~24 GB | Not needed |
| Total overhead | 38+ GB | **<2 MB** |
| Speedup | 3.32x | 2.0-2.5x |

We trade ~1x speedup for ~20,000x less memory.

## Integration

### FM-Index Modification

```rust
// src/pipelines/linear/index/fm_index.rs

pub struct FMIndex {
    bwt: Bwt,
    suffix_array: Vec<u64>,

    /// Optional learned index (adds <1 MB for human genome)
    sapling: Option<SaplingIndex>,
}

impl FMIndex {
    /// SA lookup with optional learned index acceleration.
    #[inline]
    pub fn sa_lookup(&self, bwt_pos: u64, query_suffix: &[u8]) -> u64 {
        match &self.sapling {
            Some(sapling) => {
                // Encode query suffix to u64 (first 32 bases, 2-bit)
                let suffix_64 = encode_suffix_64(query_suffix);

                // Predict position
                let (predicted, error) = sapling.predict(suffix_64);

                // Bounded binary search
                self.bounded_sa_search(bwt_pos, predicted, error)
            }
            None => {
                // Traditional sampled SA lookup
                self.sa_lookup_traditional(bwt_pos)
            }
        }
    }

    /// Binary search within [pred-error, pred+error].
    #[inline]
    fn bounded_sa_search(&self, target_bwt: u64, predicted: u64, error: u32) -> u64 {
        let lo = predicted.saturating_sub(error as u64);
        let hi = (predicted + error as u64).min(self.suffix_array.len() as u64 - 1);

        // Typically 3-10 iterations instead of 32
        let mut lo = lo as usize;
        let mut hi = hi as usize;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.suffix_array[mid] < target_bwt {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        self.suffix_array[lo]
    }
}
```

### Training (Index Building)

Training is simple and fast - just linear regression on each segment:

```rust
/// Train Sapling index from suffix array.
/// Runs in O(n) time, typically <1 minute for human genome.
pub fn train_sapling(
    suffix_array: &[u64],
    reference: &[u8],
    num_segments: usize,  // Typically 64K-128K
) -> SaplingIndex {
    let n = suffix_array.len();
    let segment_size = n / num_segments;

    let mut segments = Vec::with_capacity(num_segments);
    let mut max_error = 0u32;

    for seg_idx in 0..num_segments {
        let start = seg_idx * segment_size;
        let end = ((seg_idx + 1) * segment_size).min(n);

        // Fit linear model to this segment
        let (slope, intercept) = fit_linear_segment(
            suffix_array,
            reference,
            start,
            end
        );

        // Compute max error in this segment
        let segment_error = compute_segment_error(
            suffix_array,
            reference,
            start,
            end,
            slope,
            intercept
        );

        max_error = max_error.max(segment_error);
        segments.push(LinearSegment { slope, intercept });
    }

    let index_bits = (num_segments as f64).log2().ceil() as u8;

    SaplingIndex {
        segments,
        max_error,
        index_bits,
    }
}

/// Simple least-squares linear regression.
fn fit_linear_segment(
    sa: &[u64],
    reference: &[u8],
    start: usize,
    end: usize,
) -> (f32, f32) {
    let n = (end - start) as f64;

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_xx = 0.0f64;

    for i in start..end {
        let x = encode_suffix_64(&reference[sa[i] as usize..]) as f64;
        let y = i as f64;

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    (slope as f32, intercept as f32)
}
```

### File Format

Extend index with `.sapling` file:

```
reference.fa.bwt.2bit.64   # BWT string (~770 MB)
reference.fa.sa            # Suffix array (~390 MB sampled)
reference.fa.pac           # Packed reference (~770 MB)
reference.fa.ann           # Annotations
reference.fa.amb           # Ambiguous bases
reference.fa.sapling       # NEW: Learned index (~1 MB)
```

`.sapling` format:

```
Header (16 bytes):
  - Magic: "FSAP" (4 bytes)
  - Version: u32
  - Num segments: u32
  - Max error: u32

Segments (num_segments * 8 bytes):
  - slope: f32
  - intercept: f32
```

### CLI Integration

```bash
# Index building (always generates .sapling - negligible cost)
ferrous-align index reference.fa

# Alignment (auto-uses .sapling if present)
ferrous-align mem reference.fa reads.fq > aligned.sam

# Disable learned index (for benchmarking)
ferrous-align mem --no-learned-index reference.fa reads.fq
```

## Performance Analysis

### Expected Results

Based on Sapling paper results:

| Genome | Binary Search | Sapling | Speedup |
|--------|---------------|---------|---------|
| Human (3.1 GB) | 1.0x | 2.1x | **2.1x** |
| E. coli (4.6 MB) | 1.0x | 2.3x | **2.3x** |
| A. thaliana (120 MB) | 1.0x | 2.0x | **2.0x** |

### Cache Behavior

| Metric | Binary Search | Sapling |
|--------|---------------|---------|
| Iterations | ~32 | ~5-8 |
| L1D misses | ~32 | ~5-8 |
| L3 misses | ~20 | ~3-5 |
| Latency | ~3.2 μs | ~1.2-1.6 μs |

### Seeding Phase Impact

SA lookup is ~30-40% of seeding time. With 2x faster lookups:

```
Before:
  SMEM generation: 60%
  SA lookup: 40%
  Total: 100%

After (Sapling):
  SMEM generation: 60%
  SA lookup: 20% (2x faster)
  Total: 80%

Seeding speedup: 1.25x
```

Combined with NPU seed filtering (~1.9x extension speedup), total pipeline improvement is substantial.

## Implementation Phases

### Phase 1: Core Implementation (1 week)
- [ ] Implement `SaplingIndex` data structure
- [ ] Implement training algorithm
- [ ] Implement `.sapling` serialization
- [ ] Unit tests for prediction accuracy

### Phase 2: FM-Index Integration (1 week)
- [ ] Modify `FMIndex::sa_lookup()`
- [ ] Add suffix encoding utility
- [ ] Integrate with index building
- [ ] Integration tests

### Phase 3: Optimization (1 week)
- [ ] SIMD-optimize bounded search (NEON/AVX2)
- [ ] Tune segment count vs error tradeoff
- [ ] Profile and optimize hot paths
- [ ] Benchmark on diverse genomes

### Phase 4: Polish (0.5 week)
- [ ] CLI flags
- [ ] Documentation
- [ ] CLAUDE.md update

## Tuning Parameters

### Segment Count vs Accuracy

| Segments | Model Size | Max Error | Search Iters |
|----------|------------|-----------|--------------|
| 32K | 256 KB | ~2000 | ~11 |
| 64K | 512 KB | ~1000 | ~10 |
| 128K | 1 MB | ~500 | ~9 |
| 256K | 2 MB | ~250 | ~8 |

**Recommendation**: 64K-128K segments (sweet spot for human genome).

### Adaptive Segment Sizing

For better accuracy, use variable-sized segments based on suffix distribution:

```rust
/// Adaptive segmentation based on suffix density.
pub fn train_sapling_adaptive(
    suffix_array: &[u64],
    reference: &[u8],
    target_error: u32,  // Target max error
) -> SaplingIndex {
    // Start with coarse segments
    // Refine segments with high error
    // Merge segments with similar slopes
    // ...
}
```

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Prediction error too high | Low | Medium | Increase segment count |
| Training too slow | Very Low | Low | Parallelize with Rayon |
| Suffix encoding edge cases | Low | Medium | Handle N bases, boundaries |
| No speedup on small genomes | Medium | Low | Disable for genomes < 100 MB |

## Future Enhancements

### Hybrid with NPU (Post-1.x)

If NPU proves beneficial for batch inference:

```rust
/// Batch SA lookups for NPU acceleration.
pub fn lookup_batch_npu(
    &self,
    suffixes: &[u64],
    npu: &dyn NpuBackend,
) -> Vec<u64> {
    // Only beneficial for very large batches (>1000)
    // due to NPU dispatch overhead
}
```

### RadixSpline (Alternative Model)

[RadixSpline](https://dl.acm.org/doi/10.1145/3401071.3401659) offers single-pass training and potentially better accuracy. Could be evaluated as a drop-in replacement for the linear model.

## References

- [Sapling: Accelerating Suffix Array Queries with Learned Data Models](https://academic.oup.com/bioinformatics/article/37/6/744/5941464) (Kirsche & Schatz, 2021)
- [Sapling GitHub Repository](https://github.com/mkirsche/sapling)
- [BWA-MEME](https://academic.oup.com/bioinformatics/article/38/9/2404/6543607) (Jung & Han, 2022) - Higher performance but 38+ GB RAM
- [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208) (Kraska et al., 2018)
- [RadixSpline: A Single-Pass Learned Index](https://dl.acm.org/doi/10.1145/3401071.3401659)

## Conclusion

The Sapling-style learned index provides:

1. **2x speedup** in SA lookup (1.25x overall seeding speedup)
2. **<1 MB memory overhead** (vs 38+ GB for BWA-MEME)
3. **Simple implementation** - piecewise linear, no neural networks
4. **Fast training** - O(n) time, <1 minute for human genome
5. **Zero accuracy loss** - identical SAM output

This approach fits our target platforms (Mac Studios, high-end laptops) while providing meaningful performance improvement. The negligible memory cost means it can be enabled by default.

**Priority**: Post-1.x, after NPU Seed Filter
