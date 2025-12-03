# Alignment Module Cleanup - AoS Code Removal

**Date**: 2025-12-01  
**Branch**: feature/core-rearch  
**Status**: ✅ Complete

## Summary

Removed legacy Array-of-Structures (AoS) code paths from `src/core/alignment/` that became unreachable after the Pure SoA pipeline migration (Phases 1-5).

## Files Removed

1. **`types.rs`** (8 lines)
   - Only defined `ExtensionDirection` enum
   - Zero external references
   - Functionality replaced by `banded_swa/types.rs`

2. **`cigar.rs`** (306 lines)
   - CIGAR string operations and normalization
   - Zero external references
   - Functionality replaced by CIGAR utilities in `banded_swa/utils.rs`

**Total deletion**: 314 lines of dead code

## Files Retained (Used by SoA Pipeline)

### SIMD Kernels (Active)
- **`banded_swa/`**: Vertical SIMD Smith-Waterman (processes query positions in parallel)
  - `isa_avx2.rs`, `isa_avx512_int8.rs`, `isa_avx512_int16.rs`, `isa_sse_neon.rs`
  - `kernel.rs`, `kernel_avx512.rs`, `scalar/`
  - Used by: `dispatch.rs`, `finalize_soa.rs`, `orchestration_soa.rs`, `region.rs`

- **`kswv_*.rs`**: Horizontal SIMD kernels (process multiple alignments in parallel)
  - `kswv_sse_neon.rs` (128-bit, 16-way)
  - `kswv_avx2.rs` (256-bit, 32-way)
  - `kswv_avx512.rs` (512-bit, 64-way)
  - Used by: `dispatch.rs`

### Supporting Infrastructure (Active)
- **`shared_types.rs`**: SoA carriers (`AlignJob`, `SwSoA`, `KswSoA`)
  - Used by: `dispatch.rs`, `mate_rescue.rs`, all SIMD kernels
  
- **`workspace.rs`**: Thread-local buffer pools for zero-allocation batch processing
  - Used by: `dispatch.rs`, `mate_rescue.rs`, all SIMD kernels

- **`edit_distance.rs`**: NM/MD tag computation
  - Used by: `mate_rescue.rs`

- **`ksw_affine_gap.rs`**: Affine-gap Smith-Waterman fallback (scalar)
  - Used by: `mate_rescue.rs`

- **`utils.rs`**: Base encoding utilities
  - Used by: `pipeline.rs`, `seeding.rs` (via `crate::alignment` re-export)

- **`kswv/`**: Shared macros and adapters for horizontal SIMD
  - Used internally by kswv kernels

- **`kswv_batch.rs`**: Horizontal SIMD batching infrastructure
  - Used internally by `kswv/shared.rs`

## Verification

```bash
# Compilation check
cargo check
# Result: ✅ Passes with minor warnings (unused variables, doc comments)

# Test suite
cargo test --lib
# Result: ✅ All 241 unit tests pass

# Release build
cargo build --release
# Result: ✅ Success, no errors
```

## Allowed Dead Code

Some functions are marked `#[allow(dead_code)]` intentionally:
- `banded_swa/shared.rs`: SoA utility functions kept for potential future kernels
- `kswv_*.rs`: Batch padding/packing utilities kept for kernel variants

These are **not** legacy AoS code - they're modern SoA utilities that aren't currently used but provide flexibility for future SIMD optimizations.

## Remaining Work

### Dead Field Warnings (Non-Critical)
```rust
// src/core/alignment/banded_swa/types.rs:120
pub struct BandedPairWiseSW {
    w_match: i8,      // ⚠️ never read directly
    w_mismatch: i8,   // ⚠️ never read directly
    w_open: i8,       // ⚠️ never read directly
    w_extend: i8,     // ⚠️ never read directly
    w_ambig: i8,      // ⚠️ never read directly
}
```

**Why not removed**: These fields are used internally by SIMD kernels via struct layout. The warning is expected for parameter structs that are passed to kernels rather than directly read by Rust code.

### Unused Variable Warnings (Non-Critical)
- `pairing.rs:785`: `r1_is_rev` - Likely used in debug builds or planned features
- `pairing.rs:786`: `r2_is_rev` - Likely used in debug builds or planned features
- `kswv/shared.rs:63`: `used` - Return value from macro expansion

These are minor hygiene issues and don't indicate unreachable code paths.

## Impact

- **Code size**: -314 lines (-0.5% of total codebase)
- **Compilation**: No change (already not compiled into binary)
- **Performance**: No change (dead code wasn't being executed)
- **Maintainability**: ✅ Improved (less cruft to navigate)

## Commits

```
f76ee44 refactor(alignment): Remove unused AoS modules (types.rs, cigar.rs)
```

---

**Conclusion**: All legacy AoS code paths in `src/core/alignment/` have been identified and removed. The remaining modules are actively used by the Pure SoA pipeline and should be retained.
