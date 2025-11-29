#!/usr/bin/env python3
"""
Compare two SAM files for alignment concordance.

Usage: python3 compare_sam_outputs.py baseline.sam test.sam [label]

Concordance criteria:
- Same reference name (RNAME)
- Position within ±5bp tolerance
- Same strand (FLAG bit 0x10)
- Same mapping status (mapped vs unmapped)
"""

import sys
from collections import defaultdict

POS_TOLERANCE = 5
MAPQ_TOLERANCE = 5

def parse_sam(filename):
    """Parse SAM file and return dict of read_name -> list of alignments."""
    alignments = defaultdict(list)

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('@'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 11:
                continue

            qname = fields[0]
            flag = int(fields[1])
            rname = fields[2]
            pos = int(fields[3])
            mapq = int(fields[4])

            # Determine if this is read1 or read2
            is_read1 = (flag & 0x40) != 0
            is_read2 = (flag & 0x80) != 0
            is_reverse = (flag & 0x10) != 0
            is_unmapped = (flag & 0x4) != 0
            is_secondary = (flag & 0x100) != 0
            is_supplementary = (flag & 0x800) != 0

            # Skip secondary and supplementary for primary comparison
            if is_secondary or is_supplementary:
                continue

            # Create unique key for read1 vs read2
            read_key = f"{qname}/{'1' if is_read1 else '2'}"

            alignments[read_key].append({
                'rname': rname,
                'pos': pos,
                'mapq': mapq,
                'is_reverse': is_reverse,
                'is_unmapped': is_unmapped,
                'flag': flag,
            })

    return alignments

def compare_alignments(base_aln, test_aln):
    """Compare two alignments and return concordance status."""
    # Both unmapped
    if base_aln['is_unmapped'] and test_aln['is_unmapped']:
        return 'concordant', 'both_unmapped'

    # One mapped, one unmapped
    if base_aln['is_unmapped'] != test_aln['is_unmapped']:
        return 'discordant', 'mapping_status_differs'

    # Different chromosome
    if base_aln['rname'] != test_aln['rname']:
        return 'discordant', f"different_chr:{base_aln['rname']}vs{test_aln['rname']}"

    # Different strand
    if base_aln['is_reverse'] != test_aln['is_reverse']:
        return 'discordant', 'different_strand'

    # Position difference
    pos_diff = abs(base_aln['pos'] - test_aln['pos'])
    if pos_diff > POS_TOLERANCE:
        return 'discordant', f"pos_diff:{pos_diff}"

    # All checks passed
    return 'concordant', f"pos_diff:{pos_diff}"

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} baseline.sam test.sam [label]")
        sys.exit(1)

    baseline_file = sys.argv[1]
    test_file = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else "Comparison"

    print(f"\n{'='*60}")
    print(f"SAM Comparison: {label}")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_file}")
    print(f"Test:     {test_file}")
    print()

    # Parse both files
    baseline = parse_sam(baseline_file)
    test = parse_sam(test_file)

    # Compare
    concordant = 0
    discordant = 0
    missing_in_test = 0
    missing_in_baseline = 0
    discordant_details = []

    all_reads = set(baseline.keys()) | set(test.keys())

    for read_key in sorted(all_reads):
        base_alns = baseline.get(read_key, [])
        test_alns = test.get(read_key, [])

        if not base_alns:
            missing_in_baseline += 1
            continue

        if not test_alns:
            missing_in_test += 1
            continue

        # Compare primary alignments (first in list)
        base_aln = base_alns[0]
        test_aln = test_alns[0]

        status, reason = compare_alignments(base_aln, test_aln)

        if status == 'concordant':
            concordant += 1
        else:
            discordant += 1
            if len(discordant_details) < 20:  # Limit output
                discordant_details.append({
                    'read': read_key,
                    'reason': reason,
                    'baseline': f"{base_aln['rname']}:{base_aln['pos']}:{'-' if base_aln['is_reverse'] else '+'}",
                    'test': f"{test_aln['rname']}:{test_aln['pos']}:{'-' if test_aln['is_reverse'] else '+'}",
                })

    total = concordant + discordant
    concordance_pct = (concordant / total * 100) if total > 0 else 0

    # Print results
    print(f"Results:")
    print(f"  Total reads compared: {total}")
    print(f"  Concordant:           {concordant} ({concordant/total*100:.2f}%)" if total > 0 else "  Concordant: 0")
    print(f"  Discordant:           {discordant} ({discordant/total*100:.2f}%)" if total > 0 else "  Discordant: 0")
    print(f"  Missing in test:      {missing_in_test}")
    print(f"  Missing in baseline:  {missing_in_baseline}")
    print()
    print(f"  CONCORDANCE: {concordance_pct:.2f}%")

    if concordance_pct >= 99.0:
        print(f"  STATUS: PASS (≥99%)")
    else:
        print(f"  STATUS: FAIL (<99%)")

    # Print discordant examples
    if discordant_details:
        print(f"\nDiscordant Examples (first {len(discordant_details)}):")
        for d in discordant_details:
            print(f"  {d['read']}: {d['reason']}")
            print(f"    baseline: {d['baseline']}")
            print(f"    test:     {d['test']}")

    print()

    # Return exit code based on pass/fail
    sys.exit(0 if concordance_pct >= 99.0 else 1)

if __name__ == '__main__':
    main()
