#!/bin/bash
# Validation test script - compares FerrousAlign output against bwa-mem2
# Usage: ./run_validation.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="$SCRIPT_DIR/unique_sequence_ref.fa"
BWA_MEM2="/home/jkane/Applications/bwa-mem2/bwa-mem2"
FERROUS_ALIGN="$SCRIPT_DIR/../../target/release/ferrous-align"

echo "=== FerrousAlign Validation Tests ==="
echo "Reference: $REF"
echo ""

# Build indices if not present
if [ ! -f "$REF.bwt.2bit.64" ]; then
    echo "Building bwa-mem2 index..."
    $BWA_MEM2 index "$REF" 2>&1 | tail -3
    echo ""
fi

# Function to compare alignments
compare_alignment() {
    local test_name=$1
    local query_file=$2
    local expected_cigar=$3

    echo "--- Test: $test_name ---"

    # Run bwa-mem2
    local bwa_output=$($BWA_MEM2 mem "$REF" "$query_file" 2>/dev/null | grep -v "^@")
    local bwa_cigar=$(echo "$bwa_output" | cut -f6 | head -1)

    # Run FerrousAlign
    local ferrous_output=$($FERROUS_ALIGN mem "$REF" "$query_file" 2>/dev/null | grep -v "^@")
    local ferrous_cigar=$(echo "$ferrous_output" | cut -f6 | head -1)

    echo "bwa-mem2:      $bwa_cigar"
    echo "FerrousAlign:  $ferrous_cigar"

    if [ "$bwa_cigar" = "$ferrous_cigar" ]; then
        echo -e "${GREEN}✓ PASS${NC} - CIGARs match"
    elif [ -z "$bwa_cigar" ]; then
        echo -e "${YELLOW}⚠ DIFF${NC} - bwa-mem2 unmapped, FerrousAlign: $ferrous_cigar"
    else
        echo -e "${RED}✗ FAIL${NC} - CIGAR mismatch"
        echo "Expected pattern: $expected_cigar"
    fi
    echo ""
}

# Run tests
compare_alignment "Exact Match (100bp)" \
    "$SCRIPT_DIR/exact_match_100bp.fq" \
    "100M"

compare_alignment "Insertion (2bp)" \
    "$SCRIPT_DIR/insertion_2bp.fq" \
    "*M*I*M (insertion operator)"

compare_alignment "Deletion (2bp)" \
    "$SCRIPT_DIR/deletion_2bp.fq" \
    "*M*D*M (deletion operator)"

compare_alignment "Deletion (8bp)" \
    "$SCRIPT_DIR/deletion_8bp.fq" \
    "*M*D*M or unmapped"

echo "=== Summary ==="
echo "See README.md for detailed expected outputs and known issues"
