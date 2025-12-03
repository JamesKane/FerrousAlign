#!/bin/bash
# Compare pairing accuracy between FerrousAlign and BWA-MEM2
# Run this after establishing baselines to identify pairing discrepancies

set -euo pipefail

# Configuration
REF="/home/jkane/Genomics/Reference/chm13v2.0/chm13v2.0.fa.gz"
R1_100K="/home/jkane/Genomics/HG002/test_100k_R1.fq"
R2_100K="/home/jkane/Genomics/HG002/test_100k_R2.fq"
THREADS=16
FERROUS="./target/release/ferrous-align"
OUTDIR="./documents/benchmarks/v0.8.0_baseline"

echo "========================================="
echo "Pairing Accuracy Comparison"
echo "========================================="

# Check if bwa-mem2 is available
if ! command -v bwa-mem2 &> /dev/null; then
    echo "ERROR: bwa-mem2 not found in PATH"
    echo "Please install bwa-mem2 or add it to PATH"
    exit 1
fi

# Generate FerrousAlign output
echo "[1/4] Running FerrousAlign..."
$FERROUS mem -t $THREADS $REF $R1_100K $R2_100K > "$OUTDIR/ferrous.sam" 2>&1

# Generate BWA-MEM2 output
echo "[2/4] Running BWA-MEM2..."
bwa-mem2 mem -t $THREADS $REF $R1_100K $R2_100K > "$OUTDIR/bwa_mem2.sam" 2>&1

# Extract properly paired read names
echo "[3/4] Extracting properly paired reads..."
samtools view -f 2 "$OUTDIR/bwa_mem2.sam" | cut -f1 | sort -u > "$OUTDIR/bwa_paired.txt"
samtools view -f 2 "$OUTDIR/ferrous.sam" | cut -f1 | sort -u > "$OUTDIR/ferrous_paired.txt"

# Compare
echo "[4/4] Comparing pairing decisions..."

BWA_PAIRED=$(wc -l < "$OUTDIR/bwa_paired.txt")
FERROUS_PAIRED=$(wc -l < "$OUTDIR/ferrous_paired.txt")
ONLY_BWA=$(comm -23 "$OUTDIR/bwa_paired.txt" "$OUTDIR/ferrous_paired.txt" | wc -l)
ONLY_FERROUS=$(comm -13 "$OUTDIR/bwa_paired.txt" "$OUTDIR/ferrous_paired.txt" | wc -l)

{
    echo "# Pairing Accuracy Comparison"
    echo ""
    echo "**Date**: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "**Dataset**: 100K paired-end reads (200K total)"
    echo ""
    echo "## Summary"
    echo ""
    echo "| Metric | BWA-MEM2 | FerrousAlign | Difference |"
    echo "|--------|----------|--------------|------------|"
    echo "| Properly paired reads | $BWA_PAIRED | $FERROUS_PAIRED | $((FERROUS_PAIRED - BWA_PAIRED)) |"
    echo "| Paired in BWA-MEM2 only | - | - | $ONLY_BWA |"
    echo "| Paired in FerrousAlign only | - | - | $ONLY_FERROUS |"
    echo ""
    echo "## Pairing Rate"
    echo ""
    echo "- **BWA-MEM2**: $(echo "scale=2; $BWA_PAIRED / 100000 * 100" | bc)%"
    echo "- **FerrousAlign**: $(echo "scale=2; $FERROUS_PAIRED / 100000 * 100" | bc)%"
    echo ""
    echo "## Discrepancies"
    echo ""
    echo "Reads paired differently:"
    echo "- Paired in BWA-MEM2 but not FerrousAlign: $ONLY_BWA reads"
    echo "- Paired in FerrousAlign but not BWA-MEM2: $ONLY_FERROUS reads"
    echo ""
    echo "See \`mismatched_pairs_*.txt\` for read IDs."
} > "$OUTDIR/pairing_comparison.md"

# Save mismatched read IDs
comm -23 "$OUTDIR/bwa_paired.txt" "$OUTDIR/ferrous_paired.txt" > "$OUTDIR/mismatched_pairs_bwa_only.txt"
comm -13 "$OUTDIR/bwa_paired.txt" "$OUTDIR/ferrous_paired.txt" > "$OUTDIR/mismatched_pairs_ferrous_only.txt"

# Generate flagstat for both
samtools flagstat "$OUTDIR/bwa_mem2.sam" > "$OUTDIR/bwa_flagstat.txt"
samtools flagstat "$OUTDIR/ferrous.sam" > "$OUTDIR/ferrous_flagstat.txt"

echo ""
echo "========================================="
echo "Comparison complete!"
echo "========================================="
echo "Results:"
echo "  - pairing_comparison.md (summary)"
echo "  - mismatched_pairs_bwa_only.txt ($ONLY_BWA reads)"
echo "  - mismatched_pairs_ferrous_only.txt ($ONLY_FERROUS reads)"
echo "  - bwa_flagstat.txt / ferrous_flagstat.txt"
echo ""
cat "$OUTDIR/pairing_comparison.md"
