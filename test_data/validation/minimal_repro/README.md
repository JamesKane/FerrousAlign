# Minimal Reproduction Case: Zero SMEMs for Read with 'N' Base

**Bug**: Paired-end mate pairing failure due to zero SMEM generation for read with ambiguous base

**Read Pair ID**: `HISEQ1:18:H8VC6ADXX:1:1101:1150:14380`

## Files

### Input Data
- `read_1150-14380_R1.fq` - Read 1 (has 'N' at position 19)
- `read_1150-14380_R2.fq` - Read 2 (no ambiguous bases)

### Reference Output
- `read_1150-14380_bwamem2.sam` - C++ bwa-mem2 output (correct behavior)

### Analysis
- `ANALYSIS.md` - Detailed bug analysis and expected vs actual behavior

## The Bug

**R1 Sequence** (note 'N' at position 19):
```
TCACCTCTTCCAAACTTCNCATACTGCTTTCATTGGGGAAAGACCTTCACTTACTTGTAGG...
                  ^
```

**Expected** (from C++ bwa-mem2):
- R1: Maps to chrX:126402382, CIGAR=148M, MAPQ=60
- R2: Maps to chrX:126402200, CIGAR=148M, MAPQ=60
- Properly paired with insert size ~330bp

**Actual** (Rust ferrous-align before fix):
- R1: 0 SMEMs generated → unmapped
- R2: 3 SMEMs generated → mapped to chrX:126402200
- Not properly paired, missing mate information

## How to Reproduce

### Build with debug logging
```bash
cargo build --release --features debug-logging
```

### Run alignment
```bash
RUST_LOG=debug ./target/release/ferrous-align mem -v 4 \
  /path/to/reference.fa \
  test_data/validation/minimal_repro/read_1150-14380_R1.fq \
  test_data/validation/minimal_repro/read_1150-14380_R2.fq \
  > output.sam 2> output.log
```

### Check debug output
```bash
grep '\[DEBUG_READ\]' output.log
```

**Expected debug output showing the bug**:
```
[DEBUG_READ] Generating seeds for: HISEQ1:18:H8VC6ADXX:1:1101:1150:14380
[DEBUG_READ] Query length: 148
[DEBUG_READ] Generated 0 SMEM(s)  ← BUG: Should generate SMEMs around 'N'
[DEBUG_READ] Created 0 alignment(s)
```

### Compare with reference
```bash
diff output.sam read_1150-14380_bwamem2.sam
```

## Root Cause

The SMEM search algorithm fails when encountering 'N' (ambiguous) bases. C++ bwa-mem2 handles this by:
1. Splitting the read into segments around 'N' bases
2. Searching each segment independently
3. Extending seeds across 'N' bases during Smith-Waterman alignment

Our Rust implementation needs the same segment splitting logic.

## Fix Status

- ✅ Paired-end mate pairing fixed (commit 59c0747)
- ✅ SAM CIGAR format fixed (commit 59c0747)
- ❌ Zero SMEM generation for 'N' bases (investigation in progress)

## Related Files

- Bug tracking: `dev_notes/session-30-alignment-validation.md`
- Overall strategy: `dev_notes/alignment-quality-verification.md`
- Session summary: `dev_notes/session-30-summary.md`

## Reference

C++ bwa-mem2 source: `src/FMI_search.cpp::getSMEMsOnePosOneThread()`
- Lines 496-670: SMEM generation with 'N' base handling
- Lines 528-545: Segment splitting logic
