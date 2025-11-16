# FerrousAlign Development - Active Investigation

## Current Issue

**Problem**: SMEM (Super Maximal Exact Match) generation differs from C++ bwa-mem2 reference implementation
- C++ bwa-mem2: Generates 4 long SMEMs (max 96bp) for test read `HISEQ1:18:H8VC6ADXX:1:1101:2101:2190`
- Rust implementation: Alignment behavior under investigation with real-world WGS data

**Impact**: Testing alignment correctness against human genome reference GRCh38

## What We're Solving

Ensuring FerrousAlign produces identical alignment results to C++ bwa-mem2 for production whole-genome sequencing data.

**Target Read**: `HISEQ1:18:H8VC6ADXX:1:1101:2101:2190` (148bp)
- Expected alignment: chr7:67600394
- Testing against: Full GRCh38 human reference genome (~3.1 Gbp)

## Test Data Access

### Reference Genome
```bash
/home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
```

### Test Read
```bash
/home/jkane/Genomics/HG002/test_1read.fq
```
Contains single 148bp read from HG002 whole-genome sequencing dataset.

### C++ bwa-mem2 Reference Implementation
```bash
/home/jkane/Applications/bwa-mem2/
```

## Running Tests

```bash
# Rust implementation
./target/release/ferrous-align mem -v 3 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1read.fq

# C++ reference (for comparison)
/home/jkane/Applications/bwa-mem2/bwa-mem2 mem -v 3 \
  /home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  /home/jkane/Genomics/HG002/test_1read.fq
```

**Note**: Index loading takes ~30 seconds for the 9GB human genome index files.

## Status

**Active Work**: Investigating SMEM generation algorithm to match C++ bwa-mem2 behavior
- Forward extension with k/l swapping for reverse complement BWT
- Cumulative sum calculation for BWT interval endpoints
- Maintaining BWT interval invariants during extension

**Recent Findings**:
- Simple `l = k + s` formula causes issues with k/l swap for forward extension
- C++ uses cumulative sum approach that doesn't follow standard `s = l - k` invariant
- Suggests fundamental difference in how k, l, and s fields relate in SMEM structure
