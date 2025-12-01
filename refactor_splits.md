### File skeletons and step‑by‑step edits to satisfy the size guard
Below are exact module skeletons and snippets you can paste, with minimal path churn. Apply in the proposed order to steadily bring each file under 500 LOC while keeping tests green.

---

### A) Core alignment: banded_swa folder structure
Target directory: `src/core/alignment/banded_swa/`

Create these files (or split existing ones into them):

1) `src/core/alignment/banded_swa/mod.rs`
```rust
//! Banded Smith–Waterman: shared kernel + thin ISA wrappers + dispatch

pub mod shared;      // pad_batch, soa_transform (no‑op when pre‑SoA), pack_outscores
pub mod kernel;      // SwSimd trait + sw_kernel + KernelParams
mod engines;         // SwEngine128/256/512 adapters (no public API)
pub mod scalar;      // scalar fallback (core only)

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod isa_sse_neon;
#[cfg(target_arch = "x86_64")]
pub mod isa_avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod isa_avx512_int8;    // int8 AVX‑512 wrapper
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod isa_avx512_int16;   // (optional) int16 AVX‑512 path (can defer)

pub mod dispatch;    // BandedPairWiseSW API and runtime selection

// Re‑exports (keep external API stable)
pub use dispatch::*;
pub use kernel::{KernelParams};
```

2) `src/core/alignment/banded_swa/engines.rs` (move adapters out of kernel.rs)
```rust
use crate::core::compute::simd_abstraction as simd;
use super::kernel::SwSimd;

#[derive(Copy, Clone)]
pub struct SwEngine128;
#[derive(Copy, Clone)]
pub struct SwEngine256;
#[derive(Copy, Clone)]
pub struct SwEngine512;

// Implementations mirror what you already had in kernel.rs
// Keep only int8 lanes ops needed by sw_kernel
impl SwSimd for SwEngine128 { /* … SimdEngine128 mapping … */ }
impl SwSimd for SwEngine256 { /* … SimdEngine256 mapping … */ }
impl SwSimd for SwEngine512 { /* … SimdEngine512 mapping … */ }
```

3) `src/core/alignment/banded_swa/kernel.rs` (keep only core items)
```rust
//! Shared banded SW kernel (int8 lanes)

use crate::alignment::banded_swa::OutScore;

pub trait SwSimd: Copy {
    type V8: Copy;
    const LANES: usize;
    unsafe fn setzero_epi8() -> Self::V8; /* … rest of minimal ops … */
}

pub struct KernelParams<'a> { /* existing fields */ }

#[inline]
pub unsafe fn sw_kernel<const W: usize, E: SwSimd>(p: &KernelParams<'_>) -> Vec<OutScore> {
    // Move only the finalized inner loop here. All adapters/tests live elsewhere.
}
```

4) `src/core/alignment/banded_swa/dispatch.rs`
```rust
//! Public API entry points and runtime selection

use super::{kernel::{sw_kernel, KernelParams}, shared, engines::{SwEngine128, SwEngine256, SwEngine512}};
use crate::alignment::banded_swa::OutScore;

pub struct BandedPairWiseSW { /* existing fields (if any) */ }

impl BandedPairWiseSW {
    pub fn simd_banded_swa_dispatch_soa<const W: usize>(&self, batch: &crate::pipelines::linear::batch_extension::ExtensionJobBatch) -> Vec<OutScore> {
        // Build KernelParams from pre‑SoA batch and call sw_kernel::<W, Engine>() based on W
        unimplemented!("wire to SoA batch");
    }
    // Keep legacy AoS dispatch if still needed (will call shared::pad_batch + soa_transform)
}
```

5) ISA wrappers (thin) — keep each ≤ 200 LOC
- `src/core/alignment/banded_swa/isa_sse_neon.rs`
```rust
//! SSE/NEON thin wrapper (16 lanes)
use crate::alignment::banded_swa::OutScore;
use crate::generate_swa_entry;
use super::engines::SwEngine128;

generate_swa_entry!(
    name = simd_banded_swa_batch16,
    width = 16,
    engine = SwEngine128,
    cfg = cfg(any(target_arch = "x86_64", target_arch = "aarch64")),
    target_feature = "", // not required; abstraction selects SSE/NEON
);
```

- `src/core/alignment/banded_swa/isa_avx2.rs`
```rust
//! AVX2 thin wrapper (32 lanes)
use crate::alignment::banded_swa::OutScore;
use crate::generate_swa_entry;
use super::engines::SwEngine256;

generate_swa_entry!(
    name = simd_banded_swa_batch32,
    width = 32,
    engine = SwEngine256,
    cfg = cfg(target_arch = "x86_64"),
    target_feature = "avx2",
);
```

- `src/core/alignment/banded_swa/isa_avx512_int8.rs`
```rust
//! AVX‑512 int8 thin wrapper (64 lanes)
use crate::alignment::banded_swa::OutScore;
use crate::generate_swa_entry;
use super::engines::SwEngine512;

generate_swa_entry!(
    name = simd_banded_swa_batch64,
    width = 64,
    engine = SwEngine512,
    cfg = cfg(all(target_arch = "x86_64", feature = "avx512")),
    target_feature = "avx512bw,avx512f",
);
```

- `src/core/alignment/banded_swa/isa_avx512_int16.rs` (optional placeholder)
```rust
//! AVX‑512 int16 path (placeholder/manual for now)
// Keep minimal until an i16 shared kernel lands
```

6) `src/core/alignment/banded_swa/scalar.rs`
- Keep only the scalar DP kernel and tiny helpers; move large tests to `tests/scalar_vs_simd.rs`.

7) Tests — move to `tests/`
Create integration tests (call public APIs):
- `tests/kernel_parity.rs` — small cases comparing scalar vs SIMD and cross‑ISA
- `tests/avx2_parity.rs`, `tests/sse_neon_parity.rs`, `tests/avx512_parity.rs` — deterministic + randomized small batches
- `tests/pipeline_batch_extension.rs` — covers SoA construction + dispatch

Each should import via `use ferrous_align::alignment::banded_swa::*;` (or specific fns).

---

### B) Pipelines: split batch_extension into a folder
Target: `src/pipelines/linear/batch_extension/`

1) Replace monolithic file with a module folder:
- Create `src/pipelines/linear/batch_extension/mod.rs`
```rust
pub mod types;
pub mod collect;
pub mod convert;
pub mod soa;
pub mod dispatch;
pub mod distribute;

pub use types::*;   // keep external API stable
pub use dispatch::*;
```

2) Move code into:
- `types.rs` — `BatchedExtensionJob`, `BatchExtensionResult`, `ExtensionJobBatch` (+ SoA fields)
- `collect.rs` — `collect_extension_jobs_for_read`, batch collection helpers
- `convert.rs` — AoS→SoA ingest once; helpers returning lane arrays (`qlen/tlen/w/h0`)
- `soa.rs` — `make_batch_soa::<W>()`, `soa_views_for_job()`, validation
- `dispatch.rs` — `execute_batch_simd_scoring`, `dispatch_simd_scoring_soa` (thin; calls core dispatch)
- `distribute.rs` — `distribute_extension_results`

3) Update imports throughout:
- Replace `crate::pipelines::linear::batch_extension::{…}` with `crate::pipelines::linear::batch_extension::{types::*, soa::*, dispatch::*}` as needed.
- Keep a tiny `src/pipelines/linear/batch_extension.rs` shim if you want to avoid any import churn for now:
```rust
pub mod batch_extension;
pub use batch_extension::*;
```
But recommended: update imports once and delete the old file.

4) Move tests out of the monolith into `tests/pipeline_batch_extension.rs`.

---

### C) Practical order (fastest to reduce LOC)
1) Move inline tests from the four oversized core files into `tests/`.
2) Create `engines.rs` and move SwEngine impls there.
3) Split AVX‑512 into `isa_avx512_int8.rs` (wrapper) and (optional) `isa_avx512_int16.rs`.
4) Split pipeline `batch_extension` into folder: start with `types.rs`, `soa.rs`, `dispatch.rs`, then `collect.rs`/`convert.rs`/`distribute.rs`.
5) Rerun size guard and tests after each step.

Commands handy:
- List tests: `cargo test -p ferrous-align -- --list | head -200`
- Run only pipeline tests: `cargo test -p ferrous-align pipeline_batch_extension`
- Size guard: `sh scripts/size_guard.sh`

---

### D) Tips and guardrails
- Keep extensive comments as `//!` module docs; move deep background to `docs/`.
- Target ≤ 400 LOC per file to leave headroom.
- Maintain public API via `pub use` re‑exports in `mod.rs` files.
- Avoid duplicating helper fns; keep them in `shared.rs` or `soa.rs` and import where needed.

If you’d like, I can generate exact `mod.rs` and stub files for the pipeline folder or the banded_swa engines based on your current code layout to accelerate the split even further.