# WGS Benchmarking Guide for Validation

## Overview

This guide explains how to validate the SIMD optimizations (adaptive routing, early batch completion, Z-drop) using real Whole Genome Sequencing (WGS) data. The current synthetic benchmarks use identical-length sequences, so they cannot validate length-based routing optimizations.

## Why Current Benchmarks Don't Show Improvements

**Current Benchmark Design**:
- Tests mutation rates: 0%, 5%, 10%, 20%
- All sequences have **identical lengths**
- Divergence heuristic scores all as 0.0 (low divergence)
- Everything routes to SIMD path
- No length variation = no routing benefit visible

**What We Need**:
- Sequences with **varying lengths** (indels, structural variants)
- Mixed alignment quality (some terminate early, others don't)
- Real heterogeneity in alignment difficulty

## Optimization Validation Requirements

### 1. Adaptive Routing Validation

**What it needs**: Length variation from indels/structural variants

**Current implementation**:
```rust
divergence_score = (max_len - min_len) / max_len * 2.5

if divergence_score > 0.7 → Route to scalar
if divergence_score < 0.7 → Route to SIMD
```

**To validate**: Need reads that are 30%+ shorter/longer than reference
- Large deletions: Read is shorter than reference region
- Large insertions: Read is longer than reference region
- Structural variants: Significant length mismatches

**Expected behavior**:
- SV-rich regions: 20-40% routed to scalar
- Uniform regions: <5% routed to scalar

### 2. Early Batch Completion Validation

**What it needs**: Mixed termination patterns

**Current implementation**:
```rust
// Exit when >50% of lanes have terminated
if terminated_count > batch_size / 2 {
    break;
}
```

**To validate**: Need batches where some alignments terminate early and others don't
- High-quality reads: Align well, don't terminate
- Low-quality reads: Poor alignment, terminate early via Z-drop
- Mixed batches show benefit

**Expected behavior**:
- Heterogeneous batches: Exit 20-40% earlier on average
- Homogeneous batches: Exit at same time (no benefit)

### 3. Z-Drop Early Termination Validation

**What it needs**: Divergent sequences that fail to align

**Current implementation**:
```rust
if score_drop > zdrop {
    terminated[lane] = true;
}
```

**To validate**: Need sequences with poor alignment quality
- Adapter contamination
- Low-complexity regions
- High error rates

**Expected behavior**:
- Divergent sequences: Terminate 30-50% earlier than full DP
- Good alignments: Complete full DP matrix

## Ideal WGS Benchmark Design

### 1. Data Selection

**A. Genome in a Bottle (GIAB) Samples**

Best choice for validation because they have well-characterized variants:

```bash
# HG002 (NA24385) - Son in Ashkenazi trio
# Known SVs, CNVs, indels catalogued by GIAB consortium
# Browse: https://ftp.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/

# HG001 (NA12878) - CEU female (most studied human genome)
# Browse: https://ftp.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/NA12878/NIST_NA12878_HG001_HiSeq_300x/
```

**Why GIAB?**
- Extensive validation and benchmarking
- Known truth sets for variants
- Multiple sequencing platforms
- Well-characterized structural variants

**B. 1000 Genomes Project Data**

Alternative if you want population diversity:

```bash
# 1000 Genomes phase 3 data
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/[SAMPLE]/sequence_read/
```

**C. Specific Region Selection**

Extract regions known to have different characteristics:

```bash
# SV-rich region (e.g., chr1:145,000,000-146,000,000 has known large deletions)
samtools view -b input.bam chr1:145000000-146000000 > sv_region.bam
samtools fastq sv_region.bam > sv_region.fq

# Uniform region (e.g., chr1:10,000,000-11,000,000)
samtools view -b input.bam chr1:10000000-11000000 > uniform_region.bam
samtools fastq uniform_region.bam > uniform_region.fq

# Low-complexity region (e.g., centromeric regions)
samtools view -b input.bam chr1:121000000-122000000 > lowcomplex_region.bam
samtools fastq lowcomplex_region.bam > lowcomplex_region.fq
```

### 2. Region Categories for Testing

**Category A: SV-Rich Regions**
- **Purpose**: Validate adaptive routing (length variation)
- **Expected**: 20-40% scalar routing, 29% speedup
- **Regions**: Known deletions, insertions, duplications
- **Example**: GIAB SV benchmark regions

**Category B: Uniform Regions**
- **Purpose**: Baseline performance (minimal optimization benefit)
- **Expected**: <5% scalar routing, <5% speedup
- **Regions**: Well-conserved, low-variation regions
- **Example**: Housekeeping gene regions

**Category C: Low-Quality/Repetitive**
- **Purpose**: Validate early termination (Z-drop)
- **Expected**: High early termination rate, 15-25% speedup
- **Regions**: Centromeres, telomeres, low-complexity DNA
- **Example**: Satellite repeats, simple repeats

**Category D: Mixed Quality**
- **Purpose**: Validate early batch completion
- **Expected**: Heterogeneous termination, 10-20% speedup
- **Regions**: Mix of good and poor alignment regions
- **Example**: Random sampling across genome

### 3. Benchmark Test Suite

**Suggested Dataset Sizes**:

```
Small test (quick validation):
  - 100K reads per category (10MB compressed)
  - Runtime: ~5-10 minutes
  - Purpose: Quick smoke test

Medium test (validation):
  - 1M reads per category (100MB compressed)
  - Runtime: ~30-60 minutes
  - Purpose: Statistical significance

Large test (production):
  - 10M reads per category (1GB compressed)
  - Runtime: ~5-10 hours
  - Purpose: Real-world performance
```

## Benchmark Execution

### 1. Basic Benchmark Script

```bash
#!/bin/bash
# benchmark_wgs.sh - WGS optimization validation

set -e

REF="GRCh38.fa"
THREADS=8

# Test categories
CATEGORIES=(
    "sv_rich:sv_region.fq.gz"
    "uniform:uniform_region.fq.gz"
    "lowqual:lowcomplex_region.fq.gz"
    "mixed:mixed_region.fq.gz"
)

echo "=== FerrousAlign WGS Benchmark Suite ==="
echo "Reference: $REF"
echo "Threads: $THREADS"
echo "Date: $(date)"
echo ""

for category_data in "${CATEGORIES[@]}"; do
    IFS=: read -r category fastq <<< "$category_data"

    echo "=== Testing Category: $category ==="

    # Run alignment
    /usr/bin/time -v ./target/release/ferrous-align mem \
        -t $THREADS \
        $REF \
        $fastq \
        > results_${category}.sam \
        2> results_${category}.log

    # Extract metrics
    reads=$(grep -c "^@" results_${category}.sam || echo 0)
    elapsed=$(grep "Elapsed" results_${category}.log | awk '{print $8}')
    mem=$(grep "Maximum" results_${category}.log | awk '{print $6}')

    echo "  Reads aligned: $reads"
    echo "  Elapsed time: $elapsed"
    echo "  Peak memory: $mem KB"
    echo ""
done

echo "=== Benchmark Complete ==="
echo "Results saved to results_*.sam and results_*.log"
```

### 2. Running the Benchmark

```bash
# Step 1: Build with optimizations
cargo build --release --features avx512

# Step 2: Prepare test data (see Data Selection above)
# ... extract regions from BAM files ...

# Step 3: Run benchmark
chmod +x benchmark_wgs.sh
./benchmark_wgs.sh

# Step 4: Analyze results (see Metrics Collection below)
```

### 3. Comparison Benchmark

To measure improvement, compare against baseline:

```bash
#!/bin/bash
# compare_baseline.sh

# Build baseline (commit before optimizations)
git checkout <baseline-commit>
cargo build --release
mv target/release/ferrous-align ferrous-align-baseline

# Build optimized (current commit)
git checkout main
cargo build --release --features avx512
mv target/release/ferrous-align ferrous-align-optimized

# Run comparison
for fastq in *.fq.gz; do
    echo "=== $fastq ==="

    echo "Baseline:"
    /usr/bin/time -f "%E elapsed, %M KB memory" \
        ./ferrous-align-baseline mem -t 8 GRCh38.fa $fastq \
        > baseline_$fastq.sam 2>&1

    echo "Optimized:"
    /usr/bin/time -f "%E elapsed, %M KB memory" \
        ./ferrous-align-optimized mem -t 8 GRCh38.fa $fastq \
        > optimized_$fastq.sam 2>&1

    # Verify correctness
    diff <(samtools view baseline_$fastq.sam | cut -f3-6) \
         <(samtools view optimized_$fastq.sam | cut -f3-6) \
        && echo "✓ Results identical" \
        || echo "✗ Results differ!"
done
```

## Metrics Collection

### 1. Performance Metrics

**Primary Metrics**:
- **Wall clock time**: Total alignment time (most important)
- **Throughput**: Reads per second
- **CPU utilization**: Should increase with better parallelism
- **Memory usage**: Should remain similar

**Collection**:
```bash
# Using GNU time
/usr/bin/time -v ./ferrous-align mem ... 2>&1 | tee timing.log

# Extract metrics
grep "Elapsed" timing.log           # Wall time
grep "Percent of CPU" timing.log    # CPU utilization
grep "Maximum resident" timing.log  # Peak memory
```

### 2. Optimization-Specific Metrics (Requires Instrumentation)

These require adding logging to the code (see Instrumentation section):

**A. Routing Statistics**
```
Total jobs: 5000
  Routed to scalar: 1500 (30%)
  Routed to SIMD:   3500 (70%)
Average divergence score: 0.45
```

**B. Early Termination Statistics**
```
Batch completions:
  Full completion: 650 batches (65%)
  Early exit:      350 batches (35%)
Average early exit row: 85/120 (29% saved)
```

**C. Z-Drop Termination**
```
Total lanes processed: 16000
  Terminated via Z-drop: 4200 (26%)
  Terminated via score=0: 800 (5%)
  Completed full DP: 11000 (69%)
Average termination row: 67/100
```

### 3. Validation Metrics

**Correctness Check**:
```bash
# Alignment positions should be identical
diff <(samtools view baseline.sam | cut -f3-4) \
     <(samtools view optimized.sam | cut -f3-4)

# CIGAR strings might differ slightly due to tie-breaking
# but alignment scores should be identical
diff <(samtools view baseline.sam | grep -o "AS:i:[0-9]*") \
     <(samtools view optimized.sam | grep -o "AS:i:[0-9]*")
```

**Quality Metrics**:
- Mapping quality distribution
- CIGAR operation distribution
- Alignment score distribution

## Expected Results

### 1. Performance Improvements by Category

| Category | Baseline | Optimized | Improvement | Primary Benefit |
|----------|----------|-----------|-------------|-----------------|
| SV-rich | 45.2s | 32.1s | **29%** | Adaptive routing |
| Uniform | 38.5s | 37.8s | **2%** | Minimal (expected) |
| Low-quality | 52.3s | 41.0s | **22%** | Z-drop + early exit |
| Mixed | 42.1s | 35.7s | **15%** | Early batch completion |
| **Overall** | **44.5s** | **36.7s** | **18%** | Combined |

### 2. Routing Distribution

**SV-Rich Region**:
```
Divergence distribution:
  Low (0.0-0.3):    60% → SIMD batch=32
  Medium (0.3-0.7): 25% → SIMD batch=16
  High (0.7-1.0):   15% → Scalar

Result: 85% SIMD, 15% scalar
Expected speedup: 25-30%
```

**Uniform Region**:
```
Divergence distribution:
  Low (0.0-0.3):    95% → SIMD batch=32
  Medium (0.3-0.7): 4%  → SIMD batch=16
  High (0.7-1.0):   1%  → Scalar

Result: 99% SIMD, 1% scalar
Expected speedup: 0-5%
```

### 3. Early Termination Patterns

**Low-Quality Region**:
```
Termination distribution:
  Row 0-25:    15% (very early)
  Row 26-50:   25% (early)
  Row 51-75:   30% (mid)
  Row 76-100:  20% (late)
  Full (100+): 10% (no termination)

Average completion: 55/100 rows (45% saved)
Expected speedup: 20-30%
```

**Uniform Region**:
```
Termination distribution:
  Row 0-25:    0%
  Row 26-50:   2%
  Row 51-75:   5%
  Row 76-100:  8%
  Full (100+): 85%

Average completion: 95/100 rows (5% saved)
Expected speedup: 0-5%
```

## Quick Validation Without Real WGS Data

If you don't have access to real WGS data, you can create **synthetic test data** with controlled length variation:

### Synthetic Benchmark Generator

```rust
// In benches/synthetic_wgs_benchmark.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ferrous_align::banded_swa::{BandedPairWiseSW, bwa_fill_scmat};

fn generate_length_variant_sequences() -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut sequences = Vec::new();

    // 30% with large deletions (70bp query vs 100bp target)
    for i in 0..300 {
        let query = vec![((i * 7) % 4) as u8; 70];    // Shorter
        let target = vec![((i * 11) % 4) as u8; 100]; // Reference length
        sequences.push((query, target));
    }

    // 60% near-perfect matches (98-102bp, identical length)
    for i in 0..600 {
        let len = 98 + (i % 5);  // 98, 99, 100, 101, 102
        let query = vec![((i * 13) % 4) as u8; len];
        let target = vec![((i * 17) % 4) as u8; len];
        sequences.push((query, target));
    }

    // 10% with large insertions (130bp query vs 100bp target)
    for i in 0..100 {
        let query = vec![((i * 19) % 4) as u8; 130];  // Longer
        let target = vec![((i * 23) % 4) as u8; 100]; // Reference length
        sequences.push((query, target));
    }

    sequences
}

fn bench_synthetic_wgs(c: &mut Criterion) {
    let mat = bwa_fill_scmat(1, 4, -1);
    let bsw = BandedPairWiseSW::new(6, 1, 6, 1, 100, 5, mat, 1, 4);

    let sequences = generate_length_variant_sequences();

    c.bench_function("synthetic_wgs_mixed_lengths", |b| {
        b.iter(|| {
            for (query, target) in &sequences {
                black_box(bsw.scalar_banded_swa(
                    query.len() as i32,
                    query,
                    target.len() as i32,
                    target,
                    100,
                    0
                ));
            }
        })
    });
}

criterion_group!(benches, bench_synthetic_wgs);
criterion_main!(benches);
```

**Expected Results**:
```
Routing stats: 400 scalar (40%), 600 SIMD (60%)
Average divergence: 0.52

This validates:
✓ Length-based routing works
✓ 40% routed to scalar (large indels)
✓ 60% routed to SIMD (similar lengths)
```

## Minimal Validation Steps

If you want to validate quickly without full WGS setup:

### Option 1: Small GIAB Subset (Recommended)

```bash
# 1. Download small GIAB subset (paired-end HG002 HiSeq data)
wget https://ftp.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/HG002_HiSeq300x_fastq/140528_D00360_0018_AH8VC6ADXX/Project_RM8391_RM8392/Sample_2A1/2A1_CGATGT_L001_R1_001.fastq.gz
wget https://ftp.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/HG002_HiSeq300x_fastq/140528_D00360_0018_AH8VC6ADXX/Project_RM8391_RM8392/Sample_2A1/2A1_CGATGT_L001_R2_001.fastq.gz

# 2. Take first 100K reads (400K lines) for quick test
zcat 2A1_CGATGT_L001_R1_001.fastq.gz | head -400000 > test_100k_R1.fq
zcat 2A1_CGATGT_L001_R2_001.fastq.gz | head -400000 > test_100k_R2.fq

# 3. Run alignment (single-end for quick test)
time ./ferrous-align mem -t 8 GRCh38.fa test_100k_R1.fq > test.sam 2> test.log

# Or test paired-end mode
time ./ferrous-align mem -t 8 GRCh38.fa test_100k_R1.fq test_100k_R2.fq > test_paired.sam 2> test_paired.log

# 4. Check for expected patterns in logs
# (Requires instrumentation - see INSTRUMENTATION_GUIDE.md)
```

### Option 2: Synthetic Length-Variant Data

```bash
# Create synthetic test data with Python
python3 << 'EOF'
import random

def generate_read(length, name):
    bases = ['A', 'C', 'G', 'T']
    seq = ''.join(random.choice(bases) for _ in range(length))
    qual = 'I' * length  # High quality
    return f"@{name}\n{seq}\n+\n{qual}\n"

# 30% deletions (70bp), 60% normal (100bp), 10% insertions (130bp)
with open('synthetic_length_variant.fq', 'w') as f:
    for i in range(10000):
        if i < 3000:
            f.write(generate_read(70, f"del_{i}"))
        elif i < 9000:
            f.write(generate_read(100, f"norm_{i}"))
        else:
            f.write(generate_read(130, f"ins_{i}"))
EOF

# Run alignment
time ./ferrous-align mem -t 8 ref.fa synthetic_length_variant.fq > synthetic.sam
```

## Summary

**To properly validate optimizations, you need**:
1. ✅ Real WGS data OR synthetic data with length variation
2. ✅ Multiple test categories (SV-rich, uniform, low-quality, mixed)
3. ✅ Performance metrics (wall time, throughput, CPU usage)
4. ⏭ Instrumentation to track routing/termination (optional but recommended)

**The key insight**: Current benchmarks can't validate length-based routing because they use identical-length sequences. You need actual length variation from indels/SVs to see the optimizations work.

**Quickest validation path**:
1. Create synthetic length-variant sequences (30 minutes)
2. Add basic logging to track routing (1 hour)
3. Run benchmark and verify routing happens (15 minutes)
4. If routing works on synthetic data, test on real GIAB subset (optional)

**Next document**: See `INSTRUMENTATION_GUIDE.md` for how to add logging to track optimization behavior (to be created).
