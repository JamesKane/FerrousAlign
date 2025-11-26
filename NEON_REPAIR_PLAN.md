# NEON Intrinsics Repair Plan

**Date**: 2025-11-26
**Branch**: feature/session49-frontloading-optimization
**Platform**: Apple M3 (aarch64)

## Current Status (Session 49)

### Fixed: `_mm_blendv_epi8` NEON Implementation

**Root Cause**: The NEON translation of `_mm_blendv_epi8` was fundamentally incorrect.

**SSE Semantics**: `_mm_blendv_epi8(a, b, mask)` - for each byte, if mask's **MSB (bit 7)** is 1, select from `b`; otherwise select from `a`.

**Original (Broken) NEON**:
```rust
// WRONG: vbslq_u8 uses ALL bits of mask, not just MSB
__m128i(simd_arch::vbslq_u8(mask.0, b.0, a.0))
```

**Fixed NEON**:
```rust
// Expand MSB to all 8 bits via arithmetic right shift
let mask_expanded = simd_arch::vshrq_n_s8::<7>(mask.as_s8());
let mask_u8 = simd_arch::vreinterpretq_u8_s8(mask_expanded);
__m128i(simd_arch::vbslq_u8(mask_u8, b.0, a.0))
```

**Impact**: This fix reduced horizontal SIMD mismatches from **100%** to **~40%**.

### Current Results

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Mismatch rate (100 pairs) | 791/791 (100%) | 449/791 (57%) |
| Mismatch rate (1K pairs) | N/A | 1204/3044 (40%) |
| Properly paired (1K pairs) | ~95% | **96.20%** |

### Remaining Issue: Score Differences

After the blendv fix, alignment **positions** (te, qe) often match between scalar and SIMD, but **scores** still differ:

```
MISMATCH[0]: scalar(score=11, te=1029, qe=136) vs simd(score=4, te=1029, qe=136)
```

The SIMD scores are consistently lower than scalar. This suggests an issue in score accumulation, possibly related to:
1. Shift arithmetic in the 8-bit scoring path
2. Ambiguous base handling differences (SIMD uses `w_ambig=-1`, matrix uses 0)
3. Another undiscovered intrinsic issue

### Tests Added

Added comprehensive NEON intrinsics tests in `tests/intrinsics.rs`:
- `test_cmpgt_epu8` - unsigned 8-bit greater-than
- `test_cmpge_epu8` - unsigned 8-bit greater-or-equal
- `test_adds_epu8` - saturating unsigned add
- `test_subs_epu8` - saturating unsigned subtract
- `test_max_epu8` - unsigned 8-bit max
- `test_unpacklo_epi8` - byte interleave (low)
- `test_unpackhi_epi8` - byte interleave (high)

All 18 intrinsics tests pass on aarch64.

## Next Steps

1. **Add scalar fallback flag** for horizontal SIMD to establish baseline
2. **Debug score accumulation** - trace through DP loop to find where scores diverge
3. **Validate vertical SIMD** (banded_swa.rs) on NEON

---

## Original Problem Summary

The ARM NEON intrinsic translations have not been validated for correctness, unlike the x86-64 paths (SSE/AVX2/AVX-512) where initial SIMD work was validated and all three backends produce 100% identical output.

The algorithms are correct - the issue is in the portable intrinsics layer that translates SSE intrinsics to NEON equivalents.

Two code paths use these NEON translations and need validation:

| Component | File | Status |
|-----------|------|--------|
| Vertical SIMD (banded_swa) | `src/alignment/banded_swa.rs` | ❓ Unvalidated |
| Horizontal SIMD (kswv_batch) | `src/alignment/kswv_sse_neon.rs` | ⚠️ ~40% mismatch (was 100%) |

## Architecture Context

### Portable Intrinsics Layer

The codebase uses a portable SIMD abstraction in `src/compute/simd_abstraction/`:

```
src/compute/simd_abstraction/
├── mod.rs                    # Platform detection, re-exports
├── engine128.rs              # SimdEngine128 trait + implementations
├── portable_intrinsics.rs    # SSE-to-NEON translations
└── (avx2, avx512 modules)    # x86-only, not relevant here
```

On ARM, `__m128i` is a wrapper struct around `uint8x16_t` (NEON 128-bit vector), and SSE intrinsics are translated to NEON equivalents.

### Critical Intrinsics Status

| SSE Intrinsic | NEON Translation | Status |
|---------------|------------------|--------|
| `_mm_blendv_epi8` | MSB expansion + vbsl | ✅ Fixed |
| `_mm_max_epi16` | `vmaxq_s16` | ✅ Tested |
| `_mm_add_epi16` | `vaddq_s16` | ✅ Tested |
| `_mm_sub_epi16` | `vsubq_s16` | ✅ Tested |
| `_mm_cmpgt_epi16` | `vcgtq_s16` | ✅ Tested |
| `_mm_cmpgt_epu8` | `vcgtq_u8` | ✅ Tested |
| `_mm_cmpge_epu8` | `vcgeq_u8` | ✅ Tested |
| `_mm_adds_epu8` | `vqaddq_u8` | ✅ Tested |
| `_mm_subs_epu8` | `vqsubq_u8` | ✅ Tested |
| `_mm_max_epu8` | `vmaxq_u8` | ✅ Tested |
| `_mm_shuffle_epi8` | `vqtbl1q_u8` + mask | ✅ Tested |
| `_mm_movemask_epi8` | Manual extraction | ✅ Tested |
| `_mm_unpacklo_epi8` | `vzip1q_u8` | ✅ Tested |
| `_mm_unpackhi_epi8` | `vzip2q_u8` | ✅ Tested |
| `_mm_slli_si128` | Byte copy fallback | ✅ Tested |
| `_mm_srli_si128` | Byte copy fallback | ✅ Tested |
| `_mm_alignr_epi8` | `vextq_u8` | ✅ Tested |

## Test Data

Located in `/Users/jkane/RustroverProjects/FerrousAlign/test_data/neon_validation/`:

| File | Description |
|------|-------------|
| `100_R1.fq`, `100_R2.fq` | 100 HG002 read pairs |
| `1k_R1.fq`, `1k_R2.fq` | 1,000 HG002 read pairs |
| `10k_R1.fq`, `10k_R2.fq` | 10,000 HG002 read pairs |

Reference: `/Library/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna`

## Validation Commands

```bash
# Build with native NEON
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run intrinsics unit tests
cargo test --test intrinsics -- --nocapture

# Run with debug logging to see mismatches
./target/release/ferrous-align mem -t 1 -v 4 \
  /Library/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  test_data/neon_validation/100_R1.fq \
  test_data/neon_validation/100_R2.fq \
  2>&1 | grep "MISMATCH"

# Check properly paired rate
./target/release/ferrous-align mem -t 1 \
  /Library/Genomics/Reference/b38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  test_data/neon_validation/1k_R1.fq \
  test_data/neon_validation/1k_R2.fq \
  2>/dev/null | samtools flagstat -
```

## Success Criteria

| Criterion | Target | Current |
|-----------|--------|---------|
| kswv SIMD vs scalar match rate | 100% | ~60% |
| Properly paired rate (1K pairs) | ≥97% | 96.20% |
| All intrinsics tests pass | Yes | ✅ Yes |

## Files Modified

1. `src/compute/simd_abstraction/portable_intrinsics.rs` - Fixed `_mm_blendv_epi8`
2. `tests/intrinsics.rs` - Added comprehensive NEON intrinsics tests
3. `src/alignment/paired/mate_rescue.rs` - Fixed missing `debug` parameter

## Notes

- The x86-64 paths (SSE/AVX2/AVX-512) are validated and correct - do not modify
- NEON code paths are gated by `#[cfg(target_arch = "aarch64")]` - changes will not affect x86
- The blendv fix is critical - it was causing complete failure of conditional selection
- Score differences remain but alignment positions are largely correct
- Need scalar fallback flag to properly baseline the horizontal SIMD path
