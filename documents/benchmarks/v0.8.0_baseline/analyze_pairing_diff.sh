#!/bin/bash

# Compare pairing between BWA-MEM2 and FerrousAlign
# Find reads where they disagree on PROPER_PAIR flag

BWA_SAM="/tmp/bwa_mem2_output.sam"
FERROUS_SAM="/tmp/ferrous_is_alt_fix_only.sam"

echo "Extracting properly paired reads from BWA-MEM2..."
samtools view -f 0x02 "$BWA_SAM" | cut -f1 | sort | uniq > /tmp/bwa_proper.txt

echo "Extracting properly paired reads from FerrousAlign..."
samtools view -f 0x02 "$FERROUS_SAM" | cut -f1 | sort | uniq > /tmp/ferrous_proper.txt

echo "Finding reads where BWA-MEM2 has proper pair but FerrousAlign doesn't..."
comm -23 /tmp/bwa_proper.txt /tmp/ferrous_proper.txt > /tmp/bwa_only_proper.txt

echo "Finding reads where FerrousAlign has proper pair but BWA-MEM2 doesn't..."
comm -13 /tmp/bwa_proper.txt /tmp/ferrous_proper.txt > /tmp/ferrous_only_proper.txt

echo ""
echo "=== Summary ==="
echo "BWA-MEM2 properly paired:     $(wc -l < /tmp/bwa_proper.txt)"
echo "FerrousAlign properly paired: $(wc -l < /tmp/ferrous_proper.txt)"
echo "BWA-only proper:              $(wc -l < /tmp/bwa_only_proper.txt)"
echo "Ferrous-only proper:          $(wc -l < /tmp/ferrous_only_proper.txt)"

echo ""
echo "First 10 reads where BWA-MEM2 has proper pair but FerrousAlign doesn't:"
head -10 /tmp/bwa_only_proper.txt
