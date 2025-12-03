#!/bin/bash
# v0.8.0 Simple Baseline Profiling Script (no hyperfine dependency)
# Run this to establish performance, memory, and threading baselines

set -euo pipefail

# Configuration
REF="/home/jkane/Genomics/Reference/chm13v2.0/chm13v2.0.fa.gz"
R1_100K="/home/jkane/Genomics/HG002/test_100k_R1.fq"
R2_100K="/home/jkane/Genomics/HG002/test_100k_R2.fq"
THREADS=16
FERROUS="./target/release/ferrous-align"
OUTDIR="./documents/benchmarks/v0.8.0_baseline"

echo "========================================="
echo "v0.8.0 Baseline Profiling"
echo "========================================="
echo "Reference: $REF"
echo "Test data: 100K paired-end reads"
echo "Threads: $THREADS"
echo "Output: $OUTDIR"
echo ""

# Ensure binary is built
if [ ! -f "$FERROUS" ]; then
    echo "ERROR: Binary not found. Run 'cargo build --release' first."
    exit 1
fi

# 1. Performance Baseline (manual timing, 3 runs)
echo "[1/4] Performance profiling (3 runs)..."
{
    echo "# v0.8.0 Performance Baseline"
    echo ""
    echo "**Date**: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "**Command**: \`$FERROUS mem -t $THREADS $REF $R1_100K $R2_100K\`"
    echo "**Dataset**: 100K paired-end reads"
    echo ""
    echo "## Timing Results"
    echo ""
} > "$OUTDIR/performance_baseline.md"

for i in 1 2 3; do
    echo "  Run $i/3..."
    START=$(date +%s.%N)
    $FERROUS mem -t $THREADS $REF $R1_100K $R2_100K > /dev/null 2>&1
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    echo "Run $i: ${ELAPSED}s" >> "$OUTDIR/performance_baseline.md"
    echo "    ${ELAPSED}s"
done

# Calculate average (simple bc calculation)
{
    echo ""
    echo "## Summary"
    echo ""
    echo "See individual run times above."
} >> "$OUTDIR/performance_baseline.md"

# 2. Detailed timing with /usr/bin/time
echo ""
echo "[2/4] Detailed resource usage..."
/usr/bin/time -v $FERROUS mem -t $THREADS $REF $R1_100K $R2_100K > /dev/null \
    2> "$OUTDIR/time_verbose.txt"

# Extract key metrics
echo "  Peak memory: $(grep 'Maximum resident' $OUTDIR/time_verbose.txt | awk '{print $6/1024 " MB"}')"
echo "  Wall time: $(grep 'Elapsed' $OUTDIR/time_verbose.txt | awk '{print $8}')"

# 3. Threading profiling with perf
echo ""
echo "[3/4] Threading profiling with perf..."
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches \
    $FERROUS mem -t $THREADS $REF $R1_100K $R2_100K > /dev/null \
    2> "$OUTDIR/perf_stat.txt" || echo "  WARNING: perf failed, continuing..."

# 4. Quick memory check (skip full valgrind for now - too slow)
echo ""
echo "[4/4] Quick memory check skipped (use run_memory_profile.sh for full analysis)"

echo ""
echo "========================================="
echo "Baseline profiling complete!"
echo "========================================="
echo "Results in: $OUTDIR/"
echo ""
echo "Files created:"
echo "  - performance_baseline.md (timing results)"
echo "  - time_verbose.txt (detailed resource usage)"
echo "  - perf_stat.txt (threading statistics)"
echo ""
echo "Next steps:"
echo "  1. Review performance_baseline.md"
echo "  2. Run run_memory_profile.sh for memory baseline (slow)"
echo "  3. Run pairing accuracy comparison"
