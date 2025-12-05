#!/usr/bin/env python3
"""
Inspect a specific read's alignment in two SAM files.

Usage: python3 inspect_read.py <read_name> <baseline.sam> <test.sam>

Example:
  python3 scripts/inspect_read.py HISEQ1:18:H8VC6ADXX:1:1101:10007:56060 \
      documents/benchmarks/v0.8.0_baseline/bwa_mem2.sam /tmp/ferrous_output.sam
"""

import sys
import os

def parse_cigar(cigar):
    """Parse CIGAR string into operations."""
    ops = []
    num = ""
    for c in cigar:
        if c.isdigit():
            num += c
        else:
            ops.append((int(num), c))
            num = ""
    return ops

def get_optional_tags(fields):
    """Extract optional tags from SAM fields."""
    tags = {}
    for field in fields[11:]:
        if ':' in field:
            parts = field.split(':', 2)
            if len(parts) >= 3:
                tags[parts[0]] = parts[2]
    return tags

def format_alignment(fields):
    """Format a SAM alignment for display."""
    if len(fields) < 11:
        return "  Invalid SAM record"

    qname = fields[0]
    flag = int(fields[1])
    rname = fields[2]
    pos = int(fields[3])
    mapq = int(fields[4])
    cigar = fields[5]
    rnext = fields[6]
    pnext = int(fields[7])
    tlen = int(fields[8])
    seq = fields[9]
    qual = fields[10]

    # Decode flags
    flags = []
    if flag & 0x1: flags.append("paired")
    if flag & 0x2: flags.append("proper_pair")
    if flag & 0x4: flags.append("unmapped")
    if flag & 0x8: flags.append("mate_unmapped")
    if flag & 0x10: flags.append("reverse")
    if flag & 0x20: flags.append("mate_reverse")
    if flag & 0x40: flags.append("read1")
    if flag & 0x80: flags.append("read2")
    if flag & 0x100: flags.append("secondary")
    if flag & 0x800: flags.append("supplementary")

    tags = get_optional_tags(fields)

    # Calculate alignment span
    cigar_ops = parse_cigar(cigar) if cigar != "*" else []
    ref_consumed = sum(n for n, op in cigar_ops if op in "MDN=X")

    is_primary = not (flag & 0x100) and not (flag & 0x800)
    type_str = "PRIMARY" if is_primary else ("secondary" if flag & 0x100 else "supplementary")

    lines = []
    lines.append(f"  [{type_str}] {rname}:{pos}-{pos + ref_consumed - 1} ({'+' if not (flag & 0x10) else '-'})")
    lines.append(f"    FLAG: {flag} ({', '.join(flags)})")
    lines.append(f"    MAPQ: {mapq}")
    lines.append(f"    CIGAR: {cigar}")
    lines.append(f"    Mate: {rnext}:{pnext} TLEN:{tlen}")

    # Show important tags
    important_tags = ['AS', 'XS', 'NM', 'MD', 'XA']
    tag_strs = [f"{t}={tags[t]}" for t in important_tags if t in tags]
    if tag_strs:
        lines.append(f"    Tags: {', '.join(tag_strs)}")

    # Show sequence (truncated)
    if len(seq) > 60:
        lines.append(f"    SEQ: {seq[:30]}...{seq[-30:]}")
    else:
        lines.append(f"    SEQ: {seq}")

    return "\n".join(lines)

def find_read(sam_file, read_name):
    """Find all alignments for a read in a SAM file."""
    alignments = []

    with open(sam_file, 'r') as f:
        for line in f:
            if line.startswith('@'):
                continue
            fields = line.strip().split('\t')
            if len(fields) >= 11 and fields[0] == read_name:
                alignments.append(fields)

    return alignments

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    read_name = sys.argv[1]
    baseline_sam = sys.argv[2]
    test_sam = sys.argv[3]

    print(f"\n{'='*70}")
    print(f"Read: {read_name}")
    print(f"{'='*70}")

    # Find in baseline
    print(f"\n--- Baseline ({os.path.basename(baseline_sam)}) ---")
    baseline_alns = find_read(baseline_sam, read_name)
    if not baseline_alns:
        print("  (not found)")
    else:
        for fields in baseline_alns:
            print(format_alignment(fields))

    # Find in test
    print(f"\n--- Test ({os.path.basename(test_sam)}) ---")
    test_alns = find_read(test_sam, read_name)
    if not test_alns:
        print("  (not found)")
    else:
        for fields in test_alns:
            print(format_alignment(fields))

    # Summary comparison
    print(f"\n--- Comparison ---")
    if baseline_alns and test_alns:
        # Compare primary alignments
        base_primary = [a for a in baseline_alns if not (int(a[1]) & 0x900)]
        test_primary = [a for a in test_alns if not (int(a[1]) & 0x900)]

        if base_primary and test_primary:
            for i, (b, t) in enumerate(zip(base_primary, test_primary)):
                b_flag = int(b[1])
                t_flag = int(t[1])
                read_num = "/1" if b_flag & 0x40 else "/2" if b_flag & 0x80 else ""

                b_chr, b_pos = b[2], int(b[3])
                t_chr, t_pos = t[2], int(t[3])

                if b_chr == t_chr and abs(b_pos - t_pos) <= 5:
                    print(f"  Read{read_num}: CONCORDANT ({b_chr}:{b_pos} vs {t_chr}:{t_pos}, diff={abs(b_pos - t_pos)})")
                elif b_chr == t_chr:
                    print(f"  Read{read_num}: DISCORDANT position ({b_chr}:{b_pos} vs {t_chr}:{t_pos}, diff={abs(b_pos - t_pos)})")
                else:
                    print(f"  Read{read_num}: DISCORDANT chromosome ({b_chr}:{b_pos} vs {t_chr}:{t_pos})")

    print()

if __name__ == '__main__':
    main()
