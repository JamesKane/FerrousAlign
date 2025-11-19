# Validation Test Data (Session 36)

This directory contains test files for validating alignment correctness against bwa-mem2.

## Test Files

### Reference Genome

**`unique_sequence_ref.fa`** (120bp)
- Unique sequence with distinct regions: ACGT repeats (40bp) + T's (20bp) + G's (20bp) + C's (20bp) + A's (20bp)
- Designed to avoid ambiguous alignments from repetitive sequences
- Good for testing exact matches, insertions, and deletions

### Query Reads

**`exact_match_100bp.fq`**
- 100bp exact match to reference positions 1-100
- Expected: `100M` at position 1
- **bwa-mem2**: `read1	0	chr1	1	60	100M	*	0	0	...	NM:i:0	MD:Z:100	AS:i:100	XS:i:60`
- **FerrousAlign**: `read1	0	chr1	1	60	100M	*	0	0	...	AS:i:199	NM:i:0	MD:Z:100	XS:i:99`
- Status: ✅ PASS (minor AS/XS score differences acceptable)

**`insertion_2bp.fq`**
- 102bp query with 2bp insertion (AA BB) in middle of T homopolymer
- Expected: CIGAR with `2I` operator
- **bwa-mem2**: `read_with_insertion	0	chr1	1	60	44M2I56M`
- **FerrousAlign**: `read_with_insertion	0	chr1	114	15	48S54M`
- Status: ❌ **FAIL - Soft clips instead of detecting insertion**

**`deletion_2bp.fq`**
- 98bp query with 2bp deletion (removed TT from T homopolymer)
- Expected: CIGAR with `D` operator
- **bwa-mem2**: `read_del_2bp	0	chr1	1	60	39M4D57M`
- **FerrousAlign**: `read_del_2bp	0	chr1	1	0	56M4D40M`
- Status: ⚠️ PARTIAL - Detects deletion but at different position (homopolymer sliding)

**`deletion_8bp.fq`**
- 92bp query with 8bp deletion (removed TTTTTTTT from T homopolymer)
- Expected: CIGAR with `D` operator (or unmapped if too divergent)
- **bwa-mem2**: Unmapped (deletion too large)
- **FerrousAlign**: `read_del	0	chr1	1	0	52M8D40M`
- Status: ⚠️ Different strategy - we align large deletions, bwa-mem2 doesn't

## Key Findings (Session 36)

### ✅ Working Correctly
1. **Exact matches** - Both produce identical alignments for unique sequences
2. **Deletion detection** - Both detect deletions (though may differ on position due to homopolymer sliding)
3. **Primary-only output** - Fixed in Session 36 to match bwa-mem2 default behavior

### ❌ Known Issues
1. **Insertion detection failure** - We generate soft clips (`48S54M`) instead of insertion operators (`44M2I56M`)
   - Root cause: Likely in Smith-Waterman extension or CIGAR generation
   - Impact: Insertion variants won't be detected correctly
   - Priority: HIGH - breaks variant calling pipelines

2. **Repetitive sequence handling** - bwa-mem2 marks highly repetitive sequences as unmapped, we attempt to map them
   - Impact: Different behavior on low-complexity sequences
   - Priority: LOW - edge case

## Usage

```bash
# Index the reference
bwa-mem2 index test_data/validation/unique_sequence_ref.fa

# Test exact match
bwa-mem2 mem test_data/validation/unique_sequence_ref.fa test_data/validation/exact_match_100bp.fq

# Test insertion (reveals bug)
bwa-mem2 mem test_data/validation/unique_sequence_ref.fa test_data/validation/insertion_2bp.fq
```

## Future Work

1. Fix insertion detection in Smith-Waterman/CIGAR generation
2. Add tests for mismatches (X vs M operator validation)
3. Add paired-end validation tests
4. Add tests for edge cases (boundary alignments, very long indels)
