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

## Baseline Statistics (2025-11-22)

### Ferrous-align (current)

```
20116 total alignments
20000 primary
116 supplementary
0 secondary
19929 mapped (99.07%)
18056 properly paired (90.28%)
137 singletons (0.69%)
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
| Mapped | 99.07% | 99.50% | -0.43% |
| Properly paired | 90.28% | 97.11% | -6.83% |
| Supplementary | 116 | 140 | -24 |
| Singletons | 137 | 60 | +77 |

**Note**: The 6.83% gap in proper pairing rate suggests differences in insert size estimation or pair scoring. This is a known area for improvement.

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

## Notes

- Golden reads are NOT committed to git (too large)
- Regenerate with `make golden-reads` or script above
- BWA-MEM2 output is for reference only; ferrous baseline is the parity target
