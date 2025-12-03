### Objective
Update the linear pipeline to use the new SoA-first banded_swa and kswv kernels by default, retire legacy AoS paths, consolidate dispatch and memory provisioning, and remove redundant code. Maintain API stability at the pipeline boundary while simplifying internals.

### High-level strategy
- Make SoA the single internal representation for both banded_swa and kswv in the pipeline.
- Centralize runtime dispatch (width selection + i8/i16 routing) in one place in the pipeline, mirroring bwa-mem2 limits (≤128 → i8; >128 → i16).
- Source all buffers from the thread‑local workspace to achieve zero per‑call allocations after warmup.
- Keep legacy AoS ISA wrappers as thin shims for now, but route pipeline calls exclusively to SoA entries.
- Remove duplicated helpers, constants, and one‑off transposes once the pipeline is on SoA.

### Detailed plan (phased)
1. Unify dispatch policy in the pipeline (single point of truth)
- Actions
  - In `src/pipelines/linear/batch_extension/dispatch.rs`, create a single function (e.g., `dispatch_banded_swa_soa`) that:
    - Computes `max_len` across batch jobs.
    - Selects i8/i16 path based on `max_len ≤ 128` (bwa‑mem2 parity).
    - Selects SIMD width by `SimdEngineType` and calls the SoA entries only:
      - i8: `simd_banded_swa_batch16_soa/32_soa/64_soa`
      - i16: `simd_banded_swa_batch8_int16_soa/16_int16_soa/32_int16_soa`
  - Similarly, centralize kswv dispatch into `dispatch_kswv_soa` using `KswSoA` and SoA wrappers (`kswv_batch16_soa/32_soa/64_soa`).
- Acceptance
  - One dispatch site per algorithm (banded_swa, kswv); wrappers no longer contain policy logic.

2. Make SoA the default data path
- Actions
  - Replace any ad‑hoc `pad_batch`/`soa_transform` usage inside pipeline code with:
    - banded_swa: workspace‑backed `BandedSoAProvider` now (short term), then migrate to `AlignmentWorkspace` (see step 3) for full zero‑alloc.
    - kswv: `AlignmentWorkspace::ensure_and_transpose_ksw` (already present) to produce `KswSoA`.
  - Ensure `execute_batch_simd_scoring` produces SoA once (per batch) and passes it down; avoid per‑call Vecs.
- Acceptance
  - Pipeline no longer calls legacy AoS entries.
  - All SoA slices are stride‑sized and padded (0xFF) and come from the workspace/providor.

3. Eliminate remaining per‑call allocations (banded_swa SoA provider)
- Actions
  - Move `BandedSoAProvider` into the thread‑local `AlignmentWorkspace` alongside its aligned vectors, exposing `ws.ensure_and_transpose_banded::<W>(&[AlignJob], stride) -> SwSoA`.
  - Update ISA AoS shims (if still used externally) to invoke SoA via `with_workspace(|ws| ...)` and to pass rows via `sw_kernel_with_ws`.
- Acceptance
  - Repeated calls with same shapes do not trigger new allocations for either rows or SoA buffers (verified by pointer‑equality tests similar to `tests/workspace_alloc.rs`).

4. Standardize configuration passing
- Actions
  - Prefer `KernelConfig` bundles in pipeline and wrappers; continue to fill legacy scalars for backward compatibility until all callsites are migrated.
  - Keep a single source of scoring matrix and penalties in the pipeline and pass them through SoA entries.
- Acceptance
  - No code path derives penalties/matrix from multiple places; wrappers simply forward values.

5. AVX‑512 specific routing
- Actions
  - Where `feature = "avx512"` and CPU supports `avx512bw`, route banded_swa W64 calls to the `sw_kernel_avx512_with_ws` path (already wired) via the SoA entry.
  - For kswv, keep W64 benches/tests guarded; dispatch remains SoA‑based with optional env guard for unstable W64 while it’s being stabilized.
- Acceptance
  - Behavior matches existing code paths; no regressions in non‑AVX‑512 environments.

6. Clean up redundancy and dead code
- Remove/retire
  - Legacy AoS‑only pipeline paths that hand‑roll transposes for banded_swa or kswv.
  - Duplicate constants (`MAX_SEQ_LEN8`, etc.) scattered across modules; keep a single definition per algorithm in a shared place (e.g., `shared_types` or `workspace`).
  - Redundant helper functions: in pipeline code, eliminate lingering calls to `pad_batch/soa_transform` (keep these helpers only for benches/tests where directly useful).
  - Per‑ISA wrapper logic that computes dispatch policy; wrappers should be thin and policy‑free (pipeline decides and calls the appropriate SoA entry).
  - Any old `AoS` entry points that are not used by public APIs anymore; if externally visible, mark `#[deprecated]` for one release cycle before removal.
- Consolidate
  - `KernelParams`/`KernelParams16` usage: prefer SoA carriers + config in the pipeline; wrappers should adapt only when calling kernels.
  - Workspace buffer getters (`ksw_buffers_*`) via a single selector (`ksw_buffers_for_width`) where possible (already added) and remove per‑ISA duplicate getters from call sites.

7. Tests
- Update/add tests
  - Pipeline integration tests (existing `dispatch_policy.rs` already checks i8→i16 routing); add a pipeline end‑to‑end test that verifies SoA path is used and results map back correctly.
  - Parity tests for kswv SoA adapter (already added) remain; add one pipeline‑level test using `ensure_and_transpose_ksw`.
  - Keep AVX‑512 feature‑gated tests for banded_swa parity (already added).
- Acceptance
  - `cargo test` green on x86_64 and aarch64 (where applicable) with CI matrices.

8. Bench and docs
- Ensure benches now call SoA entries consistently (already done for `align_perf.rs`).
- Update `PERFORMANCE.md` notes to reflect “pipeline defaults to SoA” and that AoS shims are legacy.

### Concrete cleanup checklist
- Pipeline
  - [ ] Replace any remaining AoS calls in `linear/batch_extension` with SoA.
  - [ ] Remove per‑call `BandedSoAProvider::new()` use from wrappers; use `with_workspace` instead.
  - [ ] Collapse duplicated dispatch logic into a single function per algorithm.
- Banded SWA
  - [ ] Move SoA provider into `AlignmentWorkspace`; delete standalone provider struct or keep only as a thin wrapper over workspace for external APIs.
  - [ ] Remove `pad_batch/soa_transform` from runtime paths; keep in benches/tests only.
- KSWV
  - [ ] Ensure all call sites use `ensure_and_transpose_ksw` and SoA wrappers.
  - [ ] Remove ad‑hoc ksw SoA allocation paths.
- Constants/Config
  - [ ] Single definition for i8 threshold (128) and KSW qlen limit, referenced by both pipeline and wrappers (no duplication).
  - [ ] Prefer `KernelConfig` everywhere internally; wrappers fill scalars for legacy.
- Wrappers
  - [ ] Remove or deprecate AoS exporting functions if not used by public API; ensure SoA entries are exported and documented.

### Risks and mitigations
- Risk: Hidden external users still calling AoS wrappers directly.
  - Mitigation: Keep AoS entries for one deprecation cycle and log a warning in debug builds; document migration.
- Risk: Performance regressions from extra copies.
  - Mitigation: Ensure all SoA comes from workspace; add reuse tests (already added for kswv; mirror for banded_swa SoA after migration).
- Risk: AVX‑512 kswv W64 instability under load.
  - Mitigation: Keep runtime/env guard for benches; do not change pipeline defaults until stabilized.

### Milestones and PR slicing
- PR1: Centralize pipeline dispatch and route to SoA for both algorithms; no functional changes to kernels.
- PR2: Move banded_swa SoA provisioning into `AlignmentWorkspace` and eliminate per‑call allocations; add reuse tests.
- PR3: Cleanup sweep: remove redundant helpers/constants and deprecate unused AoS entries; docs update.
- PR4: Optional AVX‑512 polish (if needed) and final performance validation update in `PERFORMANCE.md`.

### Acceptance criteria (global)
- Pipeline calls SoA entries exclusively for banded_swa and kswv.
- Single dispatch policy implementation with bwa‑mem2 limits; wrappers are policy‑free.
- Zero per‑call allocations on repeated shapes after warmup for both algorithms (workspace‑backed).
- All tests green; benches run on supported hardware; docs updated.
- Final code deliverables must pass the `sh scripts/size_guard.sh` gate.
