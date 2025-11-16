#!/bin/bash
# Simple synthetic test to validate SMEM generation and SA position mapping

set -e

echo "=== Creating synthetic reference ==="
# Create a simple reference: "AAAA" repeated, then "CCCC" repeated, then "GGGG" repeated, then "TTTT" repeated
# Total: 400 bases (100 of each)
# This makes it easy to know where each pattern should be
cat > test_synthetic_ref.fasta << 'EOF'
>test_chr
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
EOF

echo "Reference created: 400 bases"
echo "  Positions 0-99:   AAAA (100x)"
echo "  Positions 100-199: CCCC (100x)"
echo "  Positions 200-299: GGGG (100x)"
echo "  Positions 300-399: TTTT (100x)"

echo ""
echo "=== Building index ==="
./target/release/ferrous-align index test_synthetic_ref.fasta 2>&1 | grep -E "INFO|ERROR|Building"

# Check if all index files were created
echo ""
echo "=== Checking index files ==="
for ext in amb ann bwt.2bit.64 pac sa; do
    if [ -f "test_synthetic_ref.fasta.$ext" ]; then
        size=$(stat -f%z "test_synthetic_ref.fasta.$ext" 2>/dev/null || stat -c%s "test_synthetic_ref.fasta.$ext")
        echo "✓ test_synthetic_ref.fasta.$ext ($size bytes)"
    else
        echo "✗ test_synthetic_ref.fasta.$ext MISSING!"
    fi
done

echo ""
echo "=== Creating test queries ==="
# Query 1: "AAAAAAAAAAAAAAAAAAAAAA" (22 A's) - should match at position 0-77
cat > test_query1.fasta << 'EOF'
>query1_22A
AAAAAAAAAAAAAAAAAAAAAA
EOF
echo "Query 1: 22 A's - should match at positions 0-77"

# Query 2: "CCCCCCCCCCCCCCCCCCCCCC" (22 C's) - should match at position 100-177
cat > test_query2.fasta << 'EOF'
>query2_22C
CCCCCCCCCCCCCCCCCCCCCC
EOF
echo "Query 2: 22 C's - should match at positions 100-177"

# Query 3: Unique pattern that appears exactly once
cat > test_query3.fasta << 'EOF'
>query3_unique
AAAAAAAAACCCCCCCCCCGGGG
EOF
echo "Query 3: AAAAAAAAACCCCCCCCCCGGGG - should match at position 92 (overlaps A/C/G boundary)"

echo ""
echo "=== Testing Query 1 (22 A's) ==="
./target/release/ferrous-align mem -T 0 -v 4 test_synthetic_ref.fasta test_query1.fasta \
    2>&1 | grep -E "SMEM|PERFECT MATCH|NO PERFECT MATCH|Generated|Using" | head -20

echo ""
echo "=== Testing Query 2 (22 C's) ==="
./target/release/ferrous-align mem -T 0 -v 4 test_synthetic_ref.fasta test_query2.fasta \
    2>&1 | grep -E "SMEM|PERFECT MATCH|NO PERFECT MATCH|Generated|Using" | head -20

echo ""
echo "=== Testing Query 3 (unique pattern) ==="
./target/release/ferrous-align mem -T 0 -v 4 test_synthetic_ref.fasta test_query3.fasta \
    2>&1 | grep -E "SMEM|PERFECT MATCH|NO PERFECT MATCH|Generated|Using|Reference at SA" | head -30

echo ""
echo "=== Cleanup ==="
rm -f test_synthetic_ref.fasta test_synthetic_ref.fasta.* test_query*.fasta
echo "Done!"
