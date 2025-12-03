#!/bin/bash
# Compare alignment scores between BWA-MEM2 and FerrousAlign for mismatched reads

cd /home/jkane/RustroverProjects/FerrousAlign/documents/benchmarks/v0.8.0_baseline

echo "Read,BWA_AS,Ferrous_AS,BWA_MAPQ,Ferrous_MAPQ,BWA_Pos,Ferrous_Pos" > alignment_score_comparison.csv

head -100 mismatched_bwa_only_full.txt | while read READ; do
    BWA_LINE=$(grep "^$READ	" bwa_mem2.sam | head -1)
    FERR_LINE=$(grep "^$READ	" ferrous.sam | head -1)

    if [ -n "$BWA_LINE" ] && [ -n "$FERR_LINE" ]; then
        BWA_AS=$(echo "$BWA_LINE" | grep -o "AS:i:[0-9]*" | head -1 | cut -d: -f3)
        FERR_AS=$(echo "$FERR_LINE" | grep -o "AS:i:[0-9]*" | head -1 | cut -d: -f3)
        BWA_MAPQ=$(echo "$BWA_LINE" | cut -f5)
        FERR_MAPQ=$(echo "$FERR_LINE" | cut -f5)
        BWA_POS=$(echo "$BWA_LINE" | cut -f3,4 | tr '\t' ':')
        FERR_POS=$(echo "$FERR_LINE" | cut -f3,4 | tr '\t' ':')

        echo "$READ,$BWA_AS,$FERR_AS,$BWA_MAPQ,$FERR_MAPQ,$BWA_POS,$FERR_POS"
    fi
done >> alignment_score_comparison.csv

echo "Analysis complete. Saved to alignment_score_comparison.csv"
echo ""
echo "Summary statistics:"
echo "Total analyzed: $(tail -n +2 alignment_score_comparison.csv | wc -l)"
echo ""
echo "Sample (first 10):"
head -11 alignment_score_comparison.csv | column -t -s,
