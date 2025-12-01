//! Shared types for SIMD alignment pipelines (banded_swa and kswv)
//!
//! Non-breaking scaffolding that standardizes:
//! - AoS job descriptors (`AlignJob`)
//! - SoA carriers for u8/i16 scoring (`SwSoA`, `SwSoA16`)
//! - Kernel configuration bundles (`KernelConfig` and sub-structs)
//! - Reuse/arena traits for SoA providers and DP workspace (`SoAProvider`, `WorkspaceArena`)
//!
//! Notes:
//! - SoA buffers should be sized as `max_len * lanes` and 64-byte aligned for
//!   optimal SIMD access. Implementations of the provider/workspace traits are
//!   responsible for alignment and capacity management.

/// Per-alignment metadata used before transposition (Array-of-Structures).
#[derive(Clone, Copy, Debug)]
pub struct AlignJob<'a> {
    pub query: &'a [u8],
    pub target: &'a [u8],
    pub qlen: usize,
    pub tlen: usize,
    /// Band width (per-lane override). Negative or zero may indicate unbanded.
    pub band: i32,
    /// Initial score h0 (per-lane), in the kernel's score width domain.
    pub h0: i32,
}

/// Gap penalty bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GapPenalties {
    pub o_del: i32,
    pub e_del: i32,
    pub o_ins: i32,
    pub e_ins: i32,
}

/// Banding / z-drop parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Banding {
    /// Default/common band; per-lane overrides are carried in SoA carriers.
    pub band: i32,
    pub zdrop: i32,
}

/// Scoring matrix (typically 5x5: A,C,G,T,N) and its dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScoringMatrix<'a> {
    pub mat5x5: &'a [i8; 25],
    pub m: i32,
}

/// Kernel configuration (policy) shared by all SIMD paths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelConfig<'a> {
    pub gaps: GapPenalties,
    pub banding: Banding,
    pub scoring: ScoringMatrix<'a>,
}

/// Structure-of-Arrays carrier for u8 scoring kernels.
#[derive(Clone, Copy, Debug)]
pub struct SwSoA<'a> {
    pub query_soa: &'a [u8],  // len == max_qlen * lanes
    pub target_soa: &'a [u8], // len == max_tlen * lanes
    pub qlen: &'a [i8],       // per-lane (clamped where needed)
    pub tlen: &'a [i8],
    pub band: &'a [i8],
    pub h0:   &'a [i8],
    pub lanes: usize,
    pub max_qlen: i32,
    pub max_tlen: i32,
}

/// Structure-of-Arrays carrier for i16 scoring kernels.
#[derive(Clone, Copy, Debug)]
pub struct SwSoA16<'a> {
    pub query_soa: &'a [i16],
    pub target_soa: &'a [i16],
    pub qlen: &'a [i16],
    pub tlen: &'a [i16],
    pub band: &'a [i16],
    pub h0:   &'a [i16],
    pub lanes: usize,
    pub max_qlen: i32,
    pub max_tlen: i32,
}

/// Structure-of-Arrays carrier for kswv (horizontal SIMD) kernels using u8 lanes.
///
/// This mirrors `SwSoA` but names the sequences as `ref_soa`/`query_soa` to
/// match ksw terminology. Sequences are laid out as `pos * lanes + lane` and
/// padded with 0xFF sentinel up to `max_*len` for safe loads.
#[derive(Clone, Copy, Debug)]
pub struct KswSoA<'a> {
    /// Reference (target) sequences in SoA layout: `ref_soa[pos * lanes + lane]`.
    pub ref_soa: &'a [u8],
    /// Query sequences in SoA layout: `query_soa[pos * lanes + lane]`.
    pub query_soa: &'a [u8],
    /// Per-lane query lengths (clamped to i8 domain where applicable).
    pub qlen: &'a [i8],
    /// Per-lane reference lengths (clamped to i8 domain where applicable).
    pub tlen: &'a [i8],
    /// Optional per-lane band width (some ksw variants use this as window size).
    pub band: &'a [i8],
    /// Optional per-lane initial score (not always used in ksw paths).
    pub h0: &'a [i8],
    /// SIMD stride (lanes) for the prepared SoA.
    pub lanes: usize,
    /// Maximum clamped query length across lanes.
    pub max_qlen: i32,
    /// Maximum clamped reference length across lanes.
    pub max_tlen: i32,
}

/// Kernel configuration for kswv paths. Reuses the generic `KernelConfig`.
pub type KswKernelConfig<'a> = KernelConfig<'a>;

/// Provider of reusable SoA buffers. Implementations should guarantee 64-byte
/// alignment and capacity sufficient for `(lanes, max_qlen, max_tlen)`.
pub trait SoAProvider {
    /// Ensure capacity and transpose the provided jobs into SoA layout.
    /// Returns read-only SoA views valid until the next mutation of the provider.
    fn ensure_and_transpose<'a>(&'a mut self, jobs: &[AlignJob<'a>], lanes: usize) -> SwSoA<'a>;
}

/// Reusable DP workspace (rows and temporaries). Implementations must ensure
/// 64-byte alignment for all returned slices.
pub trait WorkspaceArena {
    /// Ensure internal buffers for the given shape and element size.
    fn ensure_rows(&mut self, lanes: usize, qmax: usize, tmax: usize, elem_size: usize);

    /// Optional typed getters (u8 scoring). Return (H,E,F) rows sized `qmax*lanes`.
    fn rows_u8(&mut self) -> Option<(&mut [i8], &mut [i8], &mut [i8])> { None }

    /// Optional typed getters (i16 scoring). Return (H,E,F) rows sized `qmax*lanes`.
    fn rows_u16(&mut self) -> Option<(&mut [i16], &mut [i16], &mut [i16])> { None }
}

/// Facade for a per-thread memory pool that can hand out both SoA buffers and
/// DP workspace for a given engine/width. Concrete implementations live in the
/// pipeline code; this trait allows kernels to be agnostic to ownership.
pub trait AlignmentMemoryPool: SoAProvider + WorkspaceArena {}
