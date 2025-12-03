#!/bin/bash
# Analyze mismatched reads

cd /home/jkane/RustroverProjects/FerrousAlign/documents/benchmarks/v0.8.0_baseline

echo "Analyzing first 5 mismatched reads..."
echo ""

head -5 mismatched_bwa_only_full.txt | while read READ; do
    echo "=== $READ ==="
    echo "BWA-MEM2 (FLAG MAPQ POS):"
    grep "^$READ	" bwa_mem2.sam | cut -f1,2,3,4,5,7,8,9 | head -2
    echo ""
    echo "FerrousAlign (FLAG MAPQ POS):"
    grep "^$READ	" ferrous.sam | cut -f1,2,3,4,5,7,8,9 | head -2
    echo ""
    echo "---"
    echo ""
done
