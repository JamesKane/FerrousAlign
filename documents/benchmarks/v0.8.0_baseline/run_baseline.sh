#!/bin/bash
# v0.8.0 Baseline Profiling Script
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

# 1. Performance Baseline
echo "[1/4] Performance profiling..."
hyperfine --warmup 2 --runs 5 \
    --export-markdown "$OUTDIR/performance_baseline.md" \
    "$FERROUS mem -t $THREADS $REF $R1_100K $R2_100K > /dev/null" \
    2>&1 | tee "$OUTDIR/hyperfine_output.txt"

# 2. Detailed timing with /usr/bin/time
echo ""
echo "[2/4] Detailed resource usage..."
/usr/bin/time -v $FERROUS mem -t $THREADS $REF $R1_100K $R2_100K > /dev/null \
    2> "$OUTDIR/time_verbose.txt"

# 3. Memory profiling (single-threaded for clearer profile)
echo ""
echo "[3/4] Memory profiling (single-threaded, may take 5-10 min)..."
echo "  Using valgrind --tool=massif..."
valgrind --tool=massif \
    --massif-out-file="$OUTDIR/massif.out" \
    --time-unit=B \
    $FERROUS mem -t 1 $REF $R1_100K $R2_100K > /dev/null \
    2> "$OUTDIR/massif_stderr.txt"

echo "  Generating massif report..."
ms_print "$OUTDIR/massif.out" > "$OUTDIR/massif_report.txt"

# 4. Threading profiling
echo ""
echo "[4/4] Threading profiling with perf..."
if command -v perf &> /dev/null; then
    perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches \
        $FERROUS mem -t $THREADS $REF $R1_100K $R2_100K > /dev/null \
        2> "$OUTDIR/perf_stat.txt"
else
    echo "  WARNING: perf not available, skipping threading profile"
fi

echo ""
echo "========================================="
echo "Baseline profiling complete!"
echo "========================================="
echo "Results in: $OUTDIR/"
echo ""
echo "Next steps:"
echo "  1. Review performance_baseline.md for timing"
echo "  2. Review massif_report.txt for memory usage"
echo "  3. Review perf_stat.txt for threading stats"
echo "  4. Run pairing accuracy comparison separately"
