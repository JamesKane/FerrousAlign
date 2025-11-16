# Instrumentation Guide

## Overview

FerrousAlign includes comprehensive logging to track SIMD optimization behavior. This guide explains how to use the instrumentation to validate performance improvements and debug optimization issues.

## Logging Framework

Uses Rust's `log` crate with `env_logger` for structured logging.

**Log Levels**:
- **ERROR** (`-v 1`): Critical failures only
- **WARN** (`-v 2`): Warnings + errors
- **INFO** (`-v 3`): Progress + statistics (default)
- **DEBUG** (`-v 4+`): Detailed optimization metrics

## Quick Start

### Basic Usage

```bash
# Default (INFO level) - shows routing statistics
./ferrous-align mem -v 3 ref.idx reads.fq > output.sam 2> output.log

# Detailed (DEBUG level) - shows all optimization details
./ferrous-align mem -v 4 ref.idx reads.fq > output.sam 2> output.log

# Analyze logs
grep "Adaptive routing" output.log
grep "batch completion" output.log
```

### Example Output

**INFO Level** (routing statistics):
```
[INFO ] Adaptive routing: 1000 total jobs, 300 scalar (30.0%), 700 SIMD (70.0%), avg_divergence=0.450
[INFO ] Adaptive routing: 1000 total jobs, 50 scalar (5.0%), 950 SIMD (95.0%), avg_divergence=0.120
```

**DEBUG Level** (detailed metrics):
```
[DEBUG] Processing 300 high-divergence jobs with scalar
[DEBUG] Processing 700 low-divergence jobs with SIMD (batch_size=16)
[DEBUG] AVX2 batch completion: 18/32 lanes terminated, exit_row=85/120 (29.2% saved), early_exit=true
[DEBUG] AVX2 batch completion: 2/32 lanes terminated, exit_row=119/120 (0.8% saved), early_exit=false
[DEBUG] AVX-512 batch completion: 35/64 lanes terminated, exit_row=67/100 (33.0% saved), early_exit=true
```

## Metrics Explained

### 1. Adaptive Routing Statistics

**Logged by**: `src/align.rs::execute_adaptive_alignments()`

**Log Level**: INFO (shows in default verbosity)

**Format**:
```
Adaptive routing: {total} total jobs, {scalar_count} scalar ({scalar_%}%), {simd_count} SIMD ({simd_%}%), avg_divergence={score}
```

**Fields**:
- `total jobs`: Number of alignment jobs in this batch
- `scalar`: Jobs routed to scalar execution (high divergence)
- `SIMD`: Jobs routed to SIMD execution (low divergence)
- `avg_divergence`: Average length mismatch ratio (0.0-1.0)

**Interpretation**:

| avg_divergence | Routing Behavior | Meaning |
|----------------|------------------|---------|
| 0.00-0.10 | 95%+ SIMD | Uniform sequence lengths (minimal indels) |
| 0.10-0.30 | 80-95% SIMD | Low divergence (small indels) |
| 0.30-0.50 | 50-80% SIMD | Medium divergence (moderate indels) |
| 0.50-0.70 | 20-50% SIMD | High divergence (large indels) |
| 0.70-1.00 | <20% SIMD | Very high divergence (structural variants) |

**Example Analysis**:

```bash
# Extract routing stats
grep "Adaptive routing" output.log | \
  awk '{print $9, $11, $13}' | \
  head -10

# Output:
300 (30.0%), avg_divergence=0.450  # SV-rich region: 30% scalar routing
50 (5.0%), avg_divergence=0.120    # Uniform region: 5% scalar routing
```

**What to Look For**:
- ✅ **Expected**: SV-rich regions show 20-40% scalar routing
- ✅ **Expected**: Uniform regions show <5% scalar routing
- ⚠️ **Unexpected**: All regions show 0% scalar → No length variation in data
- ⚠️ **Unexpected**: All regions show >80% scalar → Excessive divergence threshold

### 2. Batch Completion Statistics

**Logged by**: `src/banded_swa_avx2.rs::simd_banded_swa_batch32()` and `src/banded_swa_avx512.rs::simd_banded_swa_batch64()`

**Log Level**: DEBUG (requires `-v 4`)

**Format**:
```
{AVX2|AVX-512} batch completion: {terminated}/{batch_size} lanes terminated, exit_row={row}/{max} ({percent}% saved), early_exit={bool}
```

**Fields**:
- `terminated/batch_size`: How many lanes finished early (e.g., 18/32)
- `exit_row/max`: Which DP row we exited at (e.g., 85/120)
- `percent saved`: Percentage of computation avoided (e.g., 29.2%)
- `early_exit`: Whether we exited before completing full DP matrix

**Interpretation**:

| Terminated % | Percent Saved | Meaning |
|--------------|---------------|---------|
| 0-25% | 0-5% | Homogeneous batch (all align similarly) |
| 25-50% | 5-15% | Slightly heterogeneous |
| 50-75% | 15-35% | Mixed quality (early exit triggered) |
| 75-100% | 35-60% | Very heterogeneous (significant savings) |

**Example Analysis**:

```bash
# Extract early exit statistics
grep "batch completion" output.log | \
  awk '{print $4, $6, $7}' | \
  head -10

# Output:
18/32 (29.2% saved)   # Moderate heterogeneity
2/32 (0.8% saved)     # Homogeneous batch
35/64 (33.0% saved)   # Good heterogeneity (AVX-512)
```

**What to Look For**:
- ✅ **Expected**: Mixed-quality data shows 15-35% savings
- ✅ **Expected**: Uniform data shows <5% savings
- ⚠️ **Unexpected**: Always 0% saved → No early termination happening
- ⚠️ **Unexpected**: Always >50% saved → Threshold too low (tune to 75%)

### 3. SIMD Engine Selection

**Logged by**: `benches/simd_engine_comparison.rs` (runtime CPU detection)

**Log Level**: STDERR (always shown during benchmarks)

**Format**:
```
=== SIMD Feature Detection ===
Platform: x86_64
SSE2:     true
AVX2:     true
AVX-512F: true
AVX-512BW:true
Expected SIMD engine: AVX-512 (512-bit, 64-way parallelism)
```

**What to Look For**:
- Verify correct SIMD features detected for your CPU
- Confirm expected engine matches your hardware capabilities
- Check if `--features avx512` was used when building

## Usage Examples

### Example 1: Validate Adaptive Routing

**Scenario**: Test if routing works with length-variant data

```bash
# Create synthetic test data with length variation
python3 << 'EOF'
import random
bases = ['A', 'C', 'G', 'T']

with open('length_variant.fq', 'w') as f:
    # 30% deletions (70bp)
    for i in range(3000):
        seq = ''.join(random.choice(bases) for _ in range(70))
        f.write(f"@del_{i}\n{seq}\n+\n{'I'*70}\n")

    # 60% normal (100bp)
    for i in range(6000):
        seq = ''.join(random.choice(bases) for _ in range(100))
        f.write(f"@norm_{i}\n{seq}\n+\n{'I'*100}\n")

    # 10% insertions (130bp)
    for i in range(1000):
        seq = ''.join(random.choice(bases) for _ in range(130))
        f.write(f"@ins_{i}\n{seq}\n+\n{'I'*130}\n")
EOF

# Run alignment with INFO logging
./ferrous-align mem -v 3 ref.idx length_variant.fq > test.sam 2> test.log

# Check routing statistics
grep "Adaptive routing" test.log

# Expected output:
# Adaptive routing: 10000 total jobs, 4000 scalar (40.0%), 6000 SIMD (60.0%), avg_divergence=0.520
```

**Interpretation**:
- ✅ 40% scalar routing confirms length-based routing works
- ✅ avg_divergence=0.520 indicates mixed length distribution

### Example 2: Measure Early Termination

**Scenario**: Track early batch completion on mixed-quality data

```bash
# Run with DEBUG logging
./ferrous-align mem -v 4 ref.idx mixed_quality.fq > test.sam 2> test.log

# Analyze early termination patterns
grep "batch completion" test.log | \
  awk -F'[(%]' '{sum+=$NF; count++} END {print "Avg savings:", sum/count "%"}'

# Example output:
# Avg savings: 18.5%
```

**Interpretation**:
- 18.5% average savings indicates moderate heterogeneity
- Expected for real WGS data with varying alignment quality

### Example 3: Compare Regions

**Scenario**: Compare SV-rich vs uniform regions

```bash
# Process SV-rich region
./ferrous-align mem -v 3 ref.idx sv_region.fq > sv.sam 2> sv.log

# Process uniform region
./ferrous-align mem -v 3 ref.idx uniform_region.fq > uniform.sam 2> uniform.log

# Compare routing
echo "=== SV-Rich Region ==="
grep "Adaptive routing" sv.log | head -5

echo "=== Uniform Region ==="
grep "Adaptive routing" uniform.log | head -5
```

**Expected Output**:

```
=== SV-Rich Region ===
Adaptive routing: 1000 total jobs, 350 scalar (35.0%), 650 SIMD (65.0%), avg_divergence=0.480
Adaptive routing: 1000 total jobs, 280 scalar (28.0%), 720 SIMD (72.0%), avg_divergence=0.420

=== Uniform Region ===
Adaptive routing: 1000 total jobs, 25 scalar (2.5%), 975 SIMD (97.5%), avg_divergence=0.080
Adaptive routing: 1000 total jobs, 30 scalar (3.0%), 970 SIMD (97.0%), avg_divergence=0.090
```

**Interpretation**:
- SV-rich: 28-35% scalar routing (expected for structural variants)
- Uniform: 2.5-3.0% scalar routing (expected for conserved regions)
- Clear differentiation validates adaptive routing

## Analysis Scripts

### Script 1: Routing Summary

```bash
#!/bin/bash
# analyze_routing.sh - Summarize routing statistics

LOG_FILE=$1

echo "=== Routing Summary ==="
echo ""

# Calculate average routing percentages
echo "Average Routing Distribution:"
grep "Adaptive routing" $LOG_FILE | \
  awk '{
    scalar += $9;
    simd += $11;
    count++;
  }
  END {
    print "  Scalar:", scalar/count "%";
    print "  SIMD:  ", simd/count "%";
  }'

echo ""
echo "Divergence Distribution:"
grep "Adaptive routing" $LOG_FILE | \
  awk -F'=' '{print $2}' | \
  sort -n | \
  awk '{
    sum += $1;
    if ($1 < 0.3) low++;
    else if ($1 < 0.7) med++;
    else high++;
    count++;
  }
  END {
    print "  Low (<0.3):   ", low, "/", count, "(" low/count*100 "%)";
    print "  Medium (0.3-0.7):", med, "/", count, "(" med/count*100 "%)";
    print "  High (>0.7):  ", high, "/", count, "(" high/count*100 "%)";
    print "  Average:      ", sum/count;
  }'
```

**Usage**:
```bash
chmod +x analyze_routing.sh
./analyze_routing.sh output.log
```

### Script 2: Early Termination Summary

```bash
#!/bin/bash
# analyze_termination.sh - Summarize early termination statistics

LOG_FILE=$1

echo "=== Early Termination Summary ==="
echo ""

grep "batch completion" $LOG_FILE | \
  awk -F'[/ %()]' '{
    terminated = $2;
    batch_size = $3;
    percent_saved = $8;

    total_terminated += terminated;
    total_batch_size += batch_size;
    total_percent += percent_saved;
    count++;

    if (percent_saved > 20) high_savings++;
    if (percent_saved > 0) early_exit++;
  }
  END {
    print "Total batches processed:", count;
    print "Early exits:", early_exit, "/", count, "(" early_exit/count*100 "%)";
    print "High savings (>20%):", high_savings, "/", count, "(" high_savings/count*100 "%)";
    print "Avg lanes terminated:", total_terminated/count, "/", total_batch_size/count;
    print "Avg computation saved:", total_percent/count "%";
  }'
```

**Usage**:
```bash
chmod +x analyze_termination.sh
./analyze_termination.sh output.log
```

## Troubleshooting

### Issue: No routing logs appear

**Symptom**: `grep "Adaptive routing" output.log` returns nothing

**Causes**:
1. Verbosity too low (need `-v 3` or higher)
2. No alignment jobs processed (empty input)
3. Using old code without instrumentation

**Solution**:
```bash
# Ensure INFO level logging
./ferrous-align mem -v 3 ref.idx reads.fq > output.sam 2> output.log

# Verify logs are going to stderr
ls -lh output.log

# Check if any logs present
head output.log
```

### Issue: All sequences route to SIMD (0% scalar)

**Symptom**: `Adaptive routing: ... 0 scalar (0.0%), ...`

**Cause**: All sequences have identical or very similar lengths

**Explanation**: Adaptive routing uses length-based divergence heuristic. If query and target always have the same length, divergence score = 0.0, so everything routes to SIMD.

**Solution**: Test with data that has length variation (indels, SVs):
```bash
# Use real WGS data with structural variants
# Or create synthetic length-variant data (see WGS_BENCHMARKING_GUIDE.md)
```

### Issue: No early termination (always 0% saved)

**Symptom**: `batch completion: 0/32 lanes terminated, ... (0.0% saved)`

**Causes**:
1. Z-drop threshold too high (zdrop=100 is very permissive)
2. All sequences align perfectly (no divergence)
3. Early termination not triggered

**Solution**:
```bash
# Check zdrop value (default is 100)
# Try lowering for testing:
# (Would require code modification to expose zdrop parameter)

# Or test with known divergent sequences
```

### Issue: Excessive early termination (>60% saved)

**Symptom**: `batch completion: 30/32 lanes terminated, ... (75.0% saved)`

**Cause**: Z-drop threshold too low OR majority threshold too low

**Analysis**: This might actually be good! Means we're successfully avoiding wasted computation.

**Validation**: Check alignment quality:
```bash
# Ensure alignments are still correct
samtools flagstat output.sam

# Check mapping quality distribution
samtools view output.sam | cut -f5 | sort | uniq -c
```

## Advanced Usage

### Enable Trace-Level Logging

For maximum detail, use `RUST_LOG` environment variable:

```bash
# Enable TRACE level for specific modules
RUST_LOG=ferrous_align::align=trace ./ferrous-align mem ref.idx reads.fq > output.sam 2> output.log

# Enable TRACE for all modules (very verbose!)
RUST_LOG=trace ./ferrous-align mem ref.idx reads.fq > output.sam 2> output.log
```

### Log to File with Timestamps

```bash
# Add timestamps with ts utility
./ferrous-align mem -v 4 ref.idx reads.fq 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' | tee output.log
```

### Structured Log Analysis with jq

If you add JSON logging (future enhancement):

```bash
# Parse JSON logs
cat output.log | jq '.[] | select(.routing) | .routing'
```

## Performance Impact

**Overhead of Instrumentation**:
- INFO level: <0.1% (negligible)
- DEBUG level: 0.5-1.0% (still minimal)
- TRACE level: 2-5% (noticeable but acceptable for debugging)

**Recommendation**: Use INFO level for production, DEBUG for validation

## Summary

**To validate optimizations**:

1. ✅ Run with `-v 3` (INFO) to see routing statistics
2. ✅ Run with `-v 4` (DEBUG) to see detailed metrics
3. ✅ Use analysis scripts to summarize log data
4. ✅ Compare SV-rich vs uniform regions
5. ✅ Verify routing happens when expected (length variation)

**Key metrics to track**:
- **Routing**: % scalar vs SIMD
- **Divergence**: Average length mismatch
- **Early termination**: % computation saved
- **Batch completion**: Lanes terminated

**Next steps**:
- See `WGS_BENCHMARKING_GUIDE.md` for validation strategy
- See `ADAPTIVE_ROUTING_ANALYSIS.md` for implementation details
- See `ZDROP_ANALYSIS.md` for optimization recommendations
