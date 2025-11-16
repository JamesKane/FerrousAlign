# Test Data for Integration Tests

## chrM.fna - Human Mitochondrial Genome (GRCh38)

- Source: GRCh38 reference (chrM)
- Length: 16,569 bp
- Use: Integration test reference (small, real biological sequence)

## queries_chrM.fq - Test Queries

Generated from chrM sequence at different positions:
1. `read_exact_100bp`: 100bp from position 1000 (expect: pos 1001, CIGAR 100M)
2. `read_exact_150bp`: 150bp from position 5000 (expect: pos 5001, CIGAR 150M)
3. `read_with_2N`: 100bp from position 10000 with 2 N substitutions
4. `read_exact_80bp`: 80bp from position 15000 (expect: pos 15001, CIGAR 80M)

## Current Status

✅ **Alignment works correctly with C++ bwa-mem2 indices**
- All test queries align to correct positions
- CIGARs match C++ bwa-mem2 output exactly

❌ **Rust index building has bug**
- Causes "index out of bounds" panic during alignment
- cp_occ array size mismatch (259 vs expected 260)
- BWT file size much smaller than C++ (27K vs 53K)
- Issue is in index building, not alignment algorithm

## Usage

```bash
# Build index with C++ bwa-mem2 (works)
/tmp/bwa-mem2-diag/bwa-mem2 index chrM.fna

# Align with Rust (works with C++ index)
./target/release/ferrous-align mem chrM.fna queries_chrM.fq > output.sam

# Build index with Rust (currently broken)
./target/release/ferrous-align index chrM.fna  # ⚠️ Creates incomplete index
```
