# Golden Reads Test Dataset

**Created**: 2025-11-22
**Purpose**: Parity testing for pipeline refactoring

## Contents

| File | Description |
|------|-------------|
| `golden_10k_R1.fq` | 10,000 R1 reads from HG002 WGS |
| `golden_10k_R2.fq` | 10,000 R2 reads from HG002 WGS |
| `baseline_ferrous.sam` | Ferrous-align output (current version) |
| `baseline_bwamem2.sam` | BWA-MEM2 reference output |

## Source Data

- **Sample**: HG002 (Genome in a Bottle)
- **Library**: 2A1_CGATGT_L001
- **Reference**: GRCh38 no-alt analysis set
- **Read length**: 148bp paired-end

## Baseline Statistics (2025-11-25)

### Ferrous-align (v0.6.0+)

```
20104 total alignments
20000 primary
104 supplementary
0 secondary
19991 mapped (99.44%)
19530 properly paired (97.65%)
83 singletons (0.41%)
```

### BWA-MEM2 (reference)

```
20140 total alignments
20000 primary
140 supplementary
0 secondary
20040 mapped (99.50%)
19422 properly paired (97.11%)
60 singletons (0.30%)
```

### Key Differences

| Metric | Ferrous | BWA-MEM2 | Delta |
|--------|---------|----------|-------|
| Mapped | 99.44% | 99.50% | -0.06% |
| Properly paired | 97.65% | 97.11% | **+0.54%** |
| Supplementary | 104 | 140 | -36 |
| Singletons | 83 | 60 | +23 |

**Note**: As of v0.6.0, ferrous-align now **exceeds** BWA-MEM2's proper pairing rate (97.65% vs 97.11%)! The remaining gap is primarily in supplementary alignment detection.

## Usage

### Regenerate Baselines

```bash
REF=/home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna

# Ferrous-align
./target/release/ferrous-align mem -t 16 $REF \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq \
    > tests/golden_reads/baseline_ferrous.sam

# BWA-MEM2
bwa-mem2 mem -t 16 $REF \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq \
    > tests/golden_reads/baseline_bwamem2.sam
```

### Compare Outputs

```bash
# Quick flagstat comparison
samtools flagstat tests/golden_reads/baseline_ferrous.sam
samtools flagstat tests/golden_reads/baseline_bwamem2.sam

# Field-by-field comparison (QNAME, FLAG, RNAME, POS, MAPQ, CIGAR)
paste <(grep -v "^@" baseline_ferrous.sam | cut -f1-6 | sort) \
      <(grep -v "^@" baseline_bwamem2.sam | cut -f1-6 | sort) | head -20
```

## Parity Test Criteria

For pipeline refactoring, the following must remain unchanged:

1. **Primary alignment count**: Exactly 20000
2. **Per-read fields** (for matching reads):
   - RNAME (chromosome)
   - POS (position, allow ±1 tolerance)
   - CIGAR (exact match)
   - AS tag (alignment score)
   - NM tag (edit distance)
3. **Overall metrics** (within tolerance):
   - Mapped rate: ±0.1%
   - MAPQ distribution: histogram match

## SIMD Backend Verification (2025-11-25)

All SIMD backends must produce identical alignment results. The following tests verify correctness across different instruction sets.

### Test Results

| Backend | Platform | Build Flags | Time | Output Hash | Status |
|---------|----------|-------------|------|-------------|--------|
| AVX2 (256-bit) | x86_64 | `--release` | 0.84s | `c663c0be36a839cced3d1e3b9e36c543` | ✅ Verified |
| AVX-512 (512-bit) | x86_64 | `--release --features avx512` | 2.08s | `c663c0be36a839cced3d1e3b9e36c543` | ✅ Verified |
| NEON (128-bit) | Apple M3 Max | `--release` | 7.19s | `c663c0be36a839cced3d1e3b9e36c543` | ✅ Verified |

**Key findings:**
- All three backends (AVX2, AVX-512, NEON) produce **byte-for-byte identical** alignment outputs
- AVX-512 timing is slower due to CPU frequency throttling (expected on Intel)
- NEON backend verified on Apple Silicon (M3 Max) with identical output hash
- All backends use vertical SIMD for banded Smith-Waterman
- Horizontal SIMD kernels (for batched mate rescue) are integrated but not yet invoked

### Verification Commands

```bash
# Build with AVX-512 support
RUSTFLAGS="-C target-cpu=native" cargo build --release --features avx512

# Run and verify output hash
./target/release/ferrous-align mem -t 16 $REF \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq \
    > /tmp/test.sam 2> /tmp/test.log

# Check backend used
grep "compute backend" /tmp/test.log

# Compare hash with baseline
grep -v "^@" /tmp/test.sam | md5sum
```

### NEON Validation (aarch64) ✅ VERIFIED

NEON backend has been verified on Apple Silicon (M3 Max):

```bash
# On aarch64 machine (Apple Silicon or Linux ARM64)
REF=/Library/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
cargo build --release

# Test with SIMD disabled (scalar baseline)
FERROUS_ALIGN_SIMD=0 ./target/release/ferrous-align mem $REF \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq \
    > /tmp/simd0.sam 2>&1
grep -v "^@" /tmp/simd0.sam | grep -v "^\[" | md5
# Expected: c663c0be36a839cced3d1e3b9e36c543

# Test with SIMD enabled (NEON kernel)
FERROUS_ALIGN_SIMD=1 ./target/release/ferrous-align mem $REF \
    tests/golden_reads/golden_10k_R1.fq \
    tests/golden_reads/golden_10k_R2.fq \
    > /tmp/simd1.sam 2>&1
grep -v "^@" /tmp/simd1.sam | grep -v "^\[" | md5
# Expected: c663c0be36a839cced3d1e3b9e36c543
```

**Apple M3 Max Results (2025-11-25):**
- SIMD=0 (scalar): 7.19 sec, 20104 alignments
- SIMD=1 (NEON):  17.18 sec, 20104 alignments
- Both produce identical output (MD5: `c663c0be36a839cced3d1e3b9e36c543`)

Note: SIMD=1 is currently slower because it runs both SIMD scoring AND scalar CIGAR generation for verification. Once the SIMD path is fully validated, this dual execution will be removed.

## Notes

- Golden reads are NOT committed to git (too large)
- Regenerate with `make golden-reads` or script above
- BWA-MEM2 output is for reference only; ferrous baseline is the parity target
- All SIMD backends must produce identical results (verified by MD5 hash)
