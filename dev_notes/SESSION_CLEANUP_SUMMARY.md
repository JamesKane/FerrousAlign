# Session Cleanup Summary - AoS Code Removal

**Date**: 2025-12-01  
**Branch**: feature/core-rearch  
**Session Goal**: Remove all unreachable AoS code paths after Pure SoA pipeline migration

## Work Completed

### 1. Alignment Module Cleanup ✅

**Files Removed**:
- `src/core/alignment/types.rs` (8 lines) - Unused ExtensionDirection enum
- `src/core/alignment/cigar.rs` (306 lines) - Unused CIGAR utilities

**Total deletion**: 314 lines of dead code

**Commits**:
```
f76ee44 refactor(alignment): Remove unused AoS modules (types.rs, cigar.rs)
d369930 docs: Document alignment module cleanup and AoS removal
```

### 2. Batch Extension Cleanup ✅

**Function Removed**:
- `soa_views_for_job()` - Diagnostic helper that always returned empty slices
- Unused import: `ExtensionJobBatch` in soa.rs

**Total deletion**: 21 lines

**Commits**:
```
31c785b refactor(batch_extension): Remove unused soa_views_for_job function
```

### 3. Minor Hygiene Fixes ✅

**Removed**:
- Unused import: `crate::core::alignment::shared_types::KswSoA` in dispatch.rs
- Unused variables: `r1_is_rev`, `r2_is_rev` in pairing.rs (lines 785-786)

**Total deletion**: 6 lines

**Compiler warnings**: 9 → 6 (-33%)

**Commits**:
```
59edf74 refactor: Remove unused imports and variables
```

## Total Impact

- **Lines deleted**: 341 lines of dead code
- **Files removed**: 2 files (types.rs, cigar.rs)
- **Functions removed**: 1 function (soa_views_for_job)
- **Compiler warnings**: Reduced from 9 to 6
- **Build status**: ✅ cargo check passes
- **Test status**: ✅ All 241 unit tests pass

## Files Audited

### Alignment Module (`src/core/alignment/`)
✅ All files audited for AoS code paths:
- **Kept (active)**: banded_swa/, kswv_*, workspace.rs, shared_types.rs, edit_distance.rs, ksw_affine_gap.rs, utils.rs
- **Removed (dead)**: types.rs, cigar.rs

### Batch Extension Module (`src/pipelines/linear/batch_extension/`)
✅ All files audited for unused functions:
- **Kept (active)**: collect_soa.rs, dispatch.rs, finalize_soa.rs, orchestration_soa.rs, soa.rs (make_batch_soa only), types.rs
- **Removed (dead)**: soa_views_for_job function

### Paired Module (`src/pipelines/linear/paired/`)
✅ Audited for deprecated code:
- No commented-out functions found
- No DEPRECATED/FIXME markers found
- Minor unused variable cleanup (r1_is_rev, r2_is_rev)

## Remaining Warnings (Non-Critical)

### Dead Code Warnings (6 total)
1. **Unused doc comment** (banded_swa/isa_avx2.rs:109)
   - Macro-generated code, rustdoc limitation
   
2. **Unused variables** (kswv/shared.rs:63)
   - `used` variable from macro expansion
   - Affects SSE/AVX2/AVX-512 kernels (3 warnings)
   
3. **Dead fields** (banded_swa/types.rs:120)
   - `BandedPairWiseSW` struct fields (w_match, w_mismatch, w_open, w_extend, w_ambig)
   - Used internally by SIMD kernels via struct layout
   - Cannot be removed - expected for parameter structs

### Why Not Fixed
- **Macro-generated warnings**: Rustdoc limitation with macro expansions
- **Parameter struct fields**: Used by SIMD kernels via memory layout, not Rust field access
- **Low priority**: These warnings don't indicate bugs or performance issues

## Verification

```bash
# Build check
cargo check
# Result: ✅ Passes (6 warnings, all non-critical)

# Unit tests
cargo test --lib
# Result: ✅ All 241 tests pass

# Release build
cargo build --release
# Result: ✅ Success

# No unused functions detected
cargo build --release 2>&1 | grep "function.*never used"
# Result: (empty)
```

## Conclusion

All reachable AoS code paths have been successfully removed from the codebase. The remaining modules are:

1. **Active SoA code**: All SIMD kernels, workspace management, batch processing
2. **Allowed dead code**: Intentionally kept utilities marked with `#[allow(dead_code)]` for future use
3. **Parameter structs**: Fields used via memory layout by SIMD kernels

The Pure SoA pipeline is now free of legacy AoS cruft and ready for concordance debugging.

## Next Steps

As documented in `SoA_PIPELINE_CONCORDANCE_FINDINGS.md`:

1. 🔍 Investigate seeding stage differences (81% concordance, need >98%)
2. 🔍 Compare SoA vs AoS batch processing logic in seeding/chaining/extension
3. 🐛 Fix identified bugs causing alignment position discordance
4. ✅ Re-test and validate against BWA-MEM2 baseline

**DO NOT MERGE** to main until concordance >98%.
