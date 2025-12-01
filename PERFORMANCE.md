## Baselines — 2025‑12‑01

Host and toolchain
- CPU: AVX‑512BW available (is_x86_feature_detected!("avx512bw") == true)
- OS: Linux (exact distro/version omitted)
- Rust: edition 2024 (see Cargo.toml), stable toolchain
- Features: benches run both without and with `--features avx512`

Bench suites
- Criterion bench: `benches/align_perf.rs`
  - banded_swa SoA entries:
    - i8 W16 (SSE/NEON), i8 W32 (AVX2), i16 W16 (AVX2), i8 W64 (AVX‑512; feature‑guarded)
  - kswv SoA entries:
    - i8 W16 (SSE/NEON), i8 W32 (AVX2), i8 W64 (AVX‑512; feature‑guarded)
  - Lengths: {64, 128, 151, 256, 400} (i8 paths are routed only for ≤128 by policy)
  - Band widths used: {10, 20, 50}

Dispatch policy (bwa‑mem2 parity)
- Use i8 kernels only when max(qlen, tlen) ≤ 128; otherwise route to i16 kernels.
- This applies to wrappers and to pipeline dispatch.

AVX‑512 mask fast‑path
- Implemented behind `core::alignment::banded_swa::kernel_avx512::sw_kernel_avx512_with_ws` façade.
- SoA entries for W64 call this façade when `feature = "avx512"` is enabled and CPU supports AVX‑512BW.

Results summary (high level)
- banded_swa i8:
  - W32 (AVX2) shows expected speedup over W16 (SSE/NEON) across ≤128bp cases.
  - W64 (AVX‑512) parity‑tested vs generic kernel (feature‑gated test) and bench‑guarded; numbers collected where supported.
- banded_swa i16:
  - W16 (AVX2, 16‑bit) covers >128bp lengths up to 400bp; stable throughput across bands 20/50.
- kswv i8:
  - W16/W32 benches are stable and produce results.
  - W64 (AVX‑512) bench is currently unstable on this host and can segfault under load; temporarily skipped in benches via `SKIP_KSWV_W64=1` env guard while we investigate.

How to reproduce
1) Non‑AVX‑512 benches
```
cargo bench --bench align_perf
```
2) AVX‑512 benches (enable feature; auto‑guarded at runtime)
```
cargo bench --bench align_perf --features avx512
```
3) Temporarily skip unstable kswv W64 benches
```
SKIP_KSWV_W64=1 cargo bench --bench align_perf --features avx512
```

Extracting a concise Criterion summary (Markdown)
- After running benches, you can extract per-benchmark mean times into Markdown tables:
```
ruby scripts/crit_extract.rb                      # all groups
ruby scripts/crit_extract.rb --group banded_swa_soa
ruby scripts/crit_extract.rb --write PERFORMANCE.md  # append summary to this file
```

Artifacts
- Criterion JSON/HTML reports are under `target/criterion/` (not committed).
- Use the HTML reports in `target/criterion/report/index.html` to browse detailed charts.

Notes and next steps
- Increase measurement time or reduce sample size for 400bp cases to avoid Criterion warnings; adjust in `configure()` or per‑group.
- Add an AVX‑512 kswv W64 parity smoke test (feature‑gated) mirroring the banded_swa parity test to aid stabilization.
- Optional: move banded_swa SoA provisioning into the thread‑local `AlignmentWorkspace` for full zero‑alloc on repeated shapes.
