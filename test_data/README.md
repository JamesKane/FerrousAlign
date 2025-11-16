# Test Data Directory

This directory contains stable, version-controlled test data for integration tests. Unlike temporary test directories that are created and destroyed during test runs, these files persist and provide reproducible test scenarios across different machines and builds.

## Directory Structure

```
test_data/
├── paired_end/           # Paired-end sequencing test data
│   ├── ref.fa           # Reference genome (503bp)
│   ├── read1.fq         # Read 1 FASTQ (50bp)
│   └── read2.fq         # Read 2 FASTQ (50bp)
└── README.md            # This file
```

## Paired-End Test Data (`paired_end/`)

### Reference Sequence (`ref.fa`)

**Length**: 503 bp
**Format**: FASTA
**Chromosome**: chr1

The reference sequence is designed with **unique, non-repeating patterns** to ensure reads map to specific, predictable positions:

```
>chr1
ACGTTAGCGATCGATAGCTGCATGCTAGCGATCGATCGATAGCTGATCGA...
```

**Key Properties**:
- Non-repetitive sequence to avoid ambiguous alignments
- Sufficient length (503bp) for realistic insert size calculations
- Patterns designed to distinguish forward and reverse strands

### Read Files

#### Read 1 (`read1.fq`)

**Format**: FASTQ
**Length**: 50 bp
**Read name**: `read1/1`
**Expected mapping**: Position 1-50 (forward strand)

```
@read1/1
ACGTTAGCGATCGATAGCTGCATGCTAGCGATCGATCGATAGCTGATCGA
+
##################################################
```

**Sequence**: Extracted from reference positions 1-50 (0-based: 0-49)

#### Read 2 (`read2.fq`)

**Format**: FASTQ
**Length**: 50 bp
**Read name**: `read1/2`
**Expected mapping**: Position 251-300 (reverse strand)

```
@read1/2
TCGATCAGCTAGCATCGATCAGCTAGCATCGATCGCTAGCATCGATCGAT
+
##################################################
```

**Sequence**: Reverse complement of reference positions 251-300 (0-based: 250-299)

### Expected Paired-End Behavior

**Orientation**: FR (Forward-Reverse) - Most common for Illumina sequencing

**Insert Size**: ~300 bp
- Read 1 starts at position 1
- Read 2 ends at position 300
- Distance between read starts: 250 bp
- Template length (TLEN): 300 bp (including both reads)

**Expected SAM Flags**:

| Read | FLAG | Decimal | Description |
|------|------|---------|-------------|
| Read 1 | 0x63 | 99 | paired + properly paired + read1 + mate reverse |
| Read 2 | 0x93 | 147 | paired + properly paired + reverse + read2 |

**FLAG Breakdown**:
- `0x1` (1): Paired-end read
- `0x2` (2): Properly paired (both reads mapped correctly)
- `0x10` (16): Read on reverse strand
- `0x20` (32): Mate on reverse strand
- `0x40` (64): First read in pair
- `0x80` (128): Second read in pair

**Expected SAM Output**:

```
read1/1  99   chr1  1    60  50M  =    251  300   ACGTTA...  ######...
read1/2  147  chr1  251  60  50M  =    1    -300  TCGATC...  ######...
```

**Fields**:
- RNEXT: `=` (same chromosome)
- PNEXT: Mate's position
- TLEN: +300 for read1 (leftmost), -300 for read2 (rightmost)

### Insert Size Statistics (Phase 2)

When processed through `mem_pestat` equivalent:
- Orientation detected: FR
- Insert size: 300 bp
- This single pair provides the baseline for insert size distribution
- With more pairs, statistics would include:
  - Mean insert size
  - Standard deviation
  - Percentiles (25th, 50th, 75th)
  - Proper pair boundaries

## Usage in Tests

### Building the Index

The index is built automatically by tests if it doesn't exist:

```rust
let test_data_dir = PathBuf::from("test_data/paired_end");
let ref_prefix = test_data_dir.join("ref");
let ref_fasta_path = test_data_dir.join("ref.fa");

if !ref_prefix.with_extension("bwt.2bit.64").exists() {
    bwa_mem2_rust::bwa_index::bwa_index(&ref_fasta_path, &ref_prefix)?;
}
```

### Running Alignment

```rust
let read1_path = test_data_dir.join("read1.fq");
let read2_path = test_data_dir.join("read2.fq");

let output = Command::new("target/release/bwa-mem2-rust")
    .arg(ref_prefix.to_str().unwrap())
    .arg(read1_path.to_str().unwrap())
    .arg(read2_path.to_str().unwrap())
    .output()?;
```

## Maintenance

**DO NOT DELETE** these files or the generated index files (`.bwt.2bit.64`, `.ann`, `.amb`, `.pac`, `.sa`).

**Index files are gitignored** but will be regenerated automatically by tests if missing.

**To regenerate test data**:
1. Only modify if test requirements change
2. Ensure sequences remain unique and non-repetitive
3. Update this README if changes are made
4. Verify all tests still pass after changes

## Test Coverage

These test files are used by:
- `tests/paired_end_integration_test.rs::test_paired_end_fr_orientation`
- Future paired-end tests (as they are enabled)

## Design Principles

1. **Reproducible**: Same data across all environments
2. **Minimal**: Small enough for fast tests (503bp reference vs GB-sized genomes)
3. **Unique**: No ambiguous alignments
4. **Realistic**: Matches real Illumina paired-end sequencing patterns
5. **Self-contained**: No external dependencies
6. **Version-controlled**: Changes tracked in git
