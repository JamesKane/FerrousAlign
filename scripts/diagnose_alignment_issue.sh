#!/bin/bash
# Diagnostic script for alignment quality regression
# Investigates why read HISEQ1:18:H8VC6ADXX:1:1101:10009:11965 aligns to wrong location

set -euo pipefail

REF=/home/jkane/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
BWA=/home/jkane/Applications/bwa-mem2/bwa-mem2
FERROUS=./target/release/ferrous-align

# Insert size stats from BWA-MEM2: mean=731, std=158, low=1413, high=62
INSERT_STATS="731,158,1413,62"

echo "============================================"
echo "ALIGNMENT DIAGNOSTIC FOR PROBLEM READ PAIR"
echo "============================================"
echo ""
echo "Read: HISEQ1:18:H8VC6ADXX:1:1101:10009:11965"
echo ""
echo "Expected (BWA-MEM2):"
echo "  R1: chr5:50141532, CIGAR: 148M"
echo "  R2: chr5:50141965, CIGAR: 148M"
echo ""
echo "Actual (Ferrous):"
echo "  R1: chr5:49956952, CIGAR: 1S129M18S"
echo "  R2: chr5:50141983, CIGAR: 18S130M"
echo ""
echo "Hypothesis: Seeding, chaining, or extension differs from BWA-MEM2"
echo ""

# Build if needed
if [ ! -f "$FERROUS" ]; then
    echo "Building Ferrous..."
    cargo build --release
fi

echo "============================================"
echo "PHASE 1: BWA-MEM2 BASELINE"
echo "============================================"
echo ""

$BWA mem -t 1 "$REF" /tmp/debug_pair_R1.fq /tmp/debug_pair_R2.fq > /tmp/bwa_debug_pair.sam 2>&1

echo "BWA-MEM2 Output:"
grep -v "^@" /tmp/bwa_debug_pair.sam | awk '{print "  "$1, "flag="$2, "chr="$3, "pos="$4, "CIGAR="$6, "AS:i:"$12}'
echo ""

echo "============================================"
echo "PHASE 2: FERROUS DEBUG RUN (SINGLE-THREADED)"
echo "============================================"
echo ""

# Run with maximum debugging enabled using -v 4 flag
timeout 30 $FERROUS mem -t 1 -v 4 -I "$INSERT_STATS" \
    "$REF" /tmp/debug_pair_R1.fq /tmp/debug_pair_R2.fq \
    > /tmp/ferrous_debug_pair.sam 2> /tmp/ferrous_debug_pair.log

echo "Ferrous Output:"
grep -v "^@" /tmp/ferrous_debug_pair.sam | awk '{print "  "$1, "flag="$2, "chr="$3, "pos="$4, "CIGAR="$6, "score="$5}'
echo ""

echo "============================================"
echo "PHASE 3: SEEDING ANALYSIS"
echo "============================================"
echo ""

echo "Extracting SMEM generation logs..."
echo ""

# Extract R1 seed information
echo "R1 Seeds (forward strand):"
grep "HISEQ1:18:H8VC6ADXX:1:1101:10009:11965.*SMEM OUTPUT.*Phase" /tmp/ferrous_debug_pair.log | head -10 || echo "  (No SMEM output found)"
echo ""

# Extract seed conversion summary
echo "Seed Conversion Summary:"
grep "SEED_CONVERSION.*10009:11965" /tmp/ferrous_debug_pair.log || echo "  (No conversion logs found)"
echo ""

# Count total seeds generated
echo "Total seeds per read:"
grep "Created.*seeds from.*SMEMs.*10009:11965" /tmp/ferrous_debug_pair.log || echo "  (No seed count found)"
echo ""

echo "============================================"
echo "PHASE 4: CHAINING ANALYSIS"
echo "============================================"
echo ""

echo "Chain generation logs:"
grep -E "chain_seeds.*10009:11965|CHAIN_VALIDATION.*10009:11965" /tmp/ferrous_debug_pair.log | head -20 || echo "  (No chain logs found)"
echo ""

echo "============================================"
echo "PHASE 5: EXTENSION ANALYSIS"
echo "============================================"
echo ""

echo "Extension/alignment logs:"
grep -E "batch_ksw.*10009:11965|Extension.*10009:11965|CIGAR.*10009:11965" /tmp/ferrous_debug_pair.log | head -20 || echo "  (No extension logs found)"
echo ""

echo "============================================"
echo "PHASE 6: DETAILED LOG SECTIONS"
echo "============================================"
echo ""

echo "Full debug log saved to: /tmp/ferrous_debug_pair.log"
echo "SAM output saved to: /tmp/ferrous_debug_pair.sam"
echo ""
echo "To examine specific pipeline stages:"
echo "  Seeding:   grep 'SMEM\|SEED_CONVERSION' /tmp/ferrous_debug_pair.log | less"
echo "  Chaining:  grep 'chain_seeds\|CHAIN' /tmp/ferrous_debug_pair.log | less"
echo "  Extension: grep 'batch_ksw\|Extension\|CIGAR' /tmp/ferrous_debug_pair.log | less"
echo "  Pairing:   grep 'mem_pair\|proper_pair' /tmp/ferrous_debug_pair.log | less"
echo ""

echo "============================================"
echo "SUMMARY"
echo "============================================"
echo ""

# Extract key statistics
BWA_R1_POS=$(grep -v "^@" /tmp/bwa_debug_pair.sam | head -1 | awk '{print $4}')
BWA_R1_CIGAR=$(grep -v "^@" /tmp/bwa_debug_pair.sam | head -1 | awk '{print $6}')
FERROUS_R1_POS=$(grep -v "^@" /tmp/ferrous_debug_pair.sam | head -1 | awk '{print $4}')
FERROUS_R1_CIGAR=$(grep -v "^@" /tmp/ferrous_debug_pair.sam | head -1 | awk '{print $6}')

echo "R1 Alignment Comparison:"
echo "  BWA-MEM2: pos=$BWA_R1_POS, CIGAR=$BWA_R1_CIGAR"
echo "  Ferrous:  pos=$FERROUS_R1_POS, CIGAR=$FERROUS_R1_CIGAR"
echo ""

if [ "$BWA_R1_POS" = "$FERROUS_R1_POS" ] && [ "$BWA_R1_CIGAR" = "$FERROUS_R1_CIGAR" ]; then
    echo "✅ PASS: Alignments match!"
else
    echo "❌ FAIL: Alignments differ"
    echo ""
    echo "Position difference: $((FERROUS_R1_POS - BWA_R1_POS)) bases"
    echo ""
    echo "Next steps:"
    echo "  1. Review seeding logs to see if correct seeds were generated"
    echo "  2. Check if correct chains were created and scored"
    echo "  3. Verify extension parameters match BWA-MEM2"
fi
echo ""
