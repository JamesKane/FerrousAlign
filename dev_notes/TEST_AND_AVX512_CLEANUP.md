# Test Suite and AVX-512 Feature Cleanup

**Date**: 2025-12-01  
**Branch**: feature/core-rearch  
**Goal**: Fix AVX-512 feature and remove deprecated AoS test code

## AVX-512 Feature Fixes ✅

### Issues Found
1. **Import Error** (`isa_avx512_int8.rs:7`):
   - Imported `sw_kernel_avx512_with_ws` from `kernel` module  
   - Should import from `kernel_avx512` module
   
2. **Missing Parameter** (`kernel_avx512.rs:15,38`):
   - Function signature missing `num_jobs: usize` parameter
   - Called function expected 3 args, got 2
   
3. **Invalid Field Access** (`kernel_avx512.rs:46`):
   - Tried to access `params.batch.len()`
   - `KernelParams` has no `batch` field
   - Should use `num_jobs` parameter instead

### Fixes Applied
```rust
// isa_avx512_int8.rs - Fixed import
use crate::core::alignment::banded_swa::kernel_avx512::sw_kernel_avx512_with_ws;

// kernel_avx512.rs - Added num_jobs parameter
pub unsafe fn sw_kernel_avx512_with_ws<const W: usize, E: SwSimd>(
    params: &KernelParams<'_>,
    num_jobs: usize,  // ← Added
    ws: &mut dyn WorkspaceArena,
) -> Vec<OutScore>

// kernel_avx512.rs - Fixed lane calculation
let lanes = num_jobs.min(params.qlen.len()).min(W);  // ← Was params.batch.len()
```

### Verification
```bash
cargo +nightly build --features avx512
# Result: ✅ Compiles successfully

cargo +nightly test --lib --features avx512  
# Result: ✅ 240 tests passed (vs 237 without avx512)
```

**Commit**: `6cabb63` - fix(avx512): Fix AVX-512 feature compilation errors

## Deprecated Code Removal ✅

### Removed from dispatch.rs (80 lines)
**Panic-only stubs** (no functionality, just deprecation messages):
- `simd_banded_swa_dispatch()` - Legacy AoS dispatcher
- `simd_banded_swa_dispatch_int16()` - Legacy i16 AoS dispatcher
- `simd_banded_swa_dispatch_with_cigar()` - Legacy scalar+CIGAR dispatcher

**Unused imports**:
- `AlignmentResult` - Only used by removed functions
- `ExtensionDirection` - Only used by removed functions

### Removed from isa_avx512_int8.rs (87 lines)
**Deprecated function**:
- `simd_banded_swa_batch64()` - AoS entry point (79 lines)
  - Manually converted AoS tuples to AlignJob structs
  - Transposed to SoA layout via workspace
  - Called kernel (duplicated logic from macro)

**Unused imports** (after removal):
- `AlignJob`, `KernelConfig`, `Banding`, `GapPenalties`, `ScoringMatrix`
- `OwnedSwSoA`, `with_workspace`, `KernelParams`, `OutScore`

**Result**: File reduced from 100 lines → 13 lines

### Removed from isa_avx512_int16.rs (51 lines)
**Deprecated function**:
- `simd_banded_swa_batch32_int16()` - AoS i16 entry point

**Deprecated test**:
- `test_simd_banded_swa_batch32_int16_basic()` - Test of deprecated function
  - Only test in file
  - Entire test module removed

**Unused imports** (after removal):
- `crate::generate_swa_entry_i16` (AoS macro)
- `with_workspace`, `OutScore`, `std::arch::x86_64::*`

**Result**: File reduced from 67 lines → 16 lines

## Remaining Work

### Other ISA Modules (Not Yet Cleaned)
- `isa_avx2.rs`: Has deprecated `simd_banded_swa_batch32()` + test
- `isa_sse_neon.rs`: Has deprecated `simd_banded_swa_batch16()` + `simd_banded_swa_batch8_int16()`

These follow the same pattern and can be cleaned up similarly.

### Pattern for Cleanup
1. Remove deprecated `#[deprecated]` function (typically 70-90 lines)
2. Remove associated test (if exists)
3. Remove now-unused imports
4. Keep only `generate_swa_entry_*_soa!` macro calls
5. Verify compilation with `cargo +nightly check --features avx512`

## Impact Summary

**Code Removed**: 218 lines of deprecated AoS code
- dispatch.rs: -80 lines
- isa_avx512_int8.rs: -87 lines  
- isa_avx512_int16.rs: -51 lines

**AVX-512 Status**: ✅ Fully functional
- Compilation: Fixed
- Tests: 240 passing (3 more than baseline)
- Feature gate: Working correctly

**Test Suite Status**: ✅ Clean
- Unit tests: 237 passing (without avx512)
- AVX-512 tests: 240 passing (with avx512)
- No deprecated test failures

## Commits

```
013d10a refactor(banded_swa): Remove deprecated AoS entry points and tests
6cabb63 fix(avx512): Fix AVX-512 feature compilation errors
```

## Next Steps (if continuing cleanup)

1. Remove deprecated AoS functions from `isa_avx2.rs`
2. Remove deprecated AoS functions from `isa_sse_neon.rs`
3. Search for any other `#[deprecated]` markers in codebase
4. Verify all tests still pass: `cargo test --lib`
5. Verify AVX-512 still works: `cargo +nightly test --lib --features avx512`

---

**Conclusion**: AVX-512 feature is now fully functional and 218 lines of deprecated AoS code have been removed. The cleanup demonstrates the SoA migration is complete for AVX-512, with all entry points now using the modern macro-based SoA API.
