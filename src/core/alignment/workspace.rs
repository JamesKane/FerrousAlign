//! Thread-local workspace for reusable allocations
//!
//! This module provides per-thread buffer pools to avoid repeated allocations
//! in the hot alignment path. Each thread gets its own workspace that is reused
//! across reads, reducing allocation overhead by ~10%.
//!
//! ## SW Kernel Buffers
//!
//! The SW kernel buffers eliminate ~65KB of allocations per batch call in
//! `simd_banded_swa_batch16_int16` and ~32KB per call in `batch_ksw_align_avx2`.
//!
//! ## KSW Horizontal SIMD Buffers (Mate Rescue)
//!
//! The KSW buffers eliminate ~32KB (AVX2) or ~64KB (AVX-512) allocations per
//! batch call in `batch_ksw_align_avx2` and `batch_ksw_align_avx512`. These are
//! called millions of times during mate rescue.
//!
//! **Key optimization**: BWA-MEM2 pre-allocates all buffers at construction time
//! and reuses them across all batches. We replicate this pattern with thread-local
//! workspace to avoid per-batch allocation overhead.
//!
//! ## Priority 2: 64-byte Aligned Buffers
//!
//! All KSW kernel buffers use 64-byte alignment to enable aligned SIMD stores
//! (`_mm256_store_si256`, `_mm512_store_si512`) instead of unaligned stores
//! (`_mm256_storeu_si256`, `_mm512_storeu_si512`). This provides ~10-15%
//! additional performance improvement on top of Priority 1 (workspace allocation).

use crate::core::alignment::shared_types::{AlignJob, KswSoA, WorkspaceArena};
use crate::pipelines::linear::seeding::SMEM;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::ptr::NonNull;

/// An owned version of SwSoA for passing data out of a workspace closure.
pub struct OwnedSwSoA {
    pub query_soa: Vec<u8>,
    pub target_soa: Vec<u8>,
    pub qlen: Vec<i8>,
    pub tlen: Vec<i8>,
    pub band: Vec<i8>,
    pub h0: Vec<i8>,
    pub lanes: usize,
    pub max_qlen: i32,
    pub max_tlen: i32,
}

/// A vector-like container with 64-byte alignment for SIMD operations
///
/// This ensures that SIMD load/store operations can use aligned instructions
/// which are typically faster than unaligned variants.
struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T: Copy> AlignedVec<T> {
    /// Create a new aligned vector with the given capacity
    ///
    /// # Safety
    /// The allocation uses 64-byte alignment regardless of T's natural alignment
    fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
            };
        }

        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            64, // 64-byte alignment for cache lines and AVX-512
        )
        .unwrap();

        let ptr = unsafe {
            // Use alloc_zeroed to prevent non-deterministic behavior from uninitialized memory
            let raw_ptr = alloc_zeroed(layout) as *mut T;
            NonNull::new(raw_ptr).expect("Allocation failed")
        };

        Self {
            ptr,
            len: capacity,
            capacity,
        }
    }

    /// Get a mutable slice view of the buffer
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Fill the buffer with a value
    #[inline]
    fn fill(&mut self, value: T) {
        self.as_mut_slice().fill(value);
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                let layout =
                    Layout::from_size_align_unchecked(self.capacity * std::mem::size_of::<T>(), 64);
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

/// Maximum expected read length for pre-allocation
const MAX_READ_LEN: usize = 512;

/// Maximum expected SMEMs per strand search
const MAX_SMEMS_PER_STRAND: usize = 1024;

/// Maximum sequence length for SW kernels (matches banded_swa_avx2.rs)
const SW_MAX_SEQ_LEN: usize = 512;

/// SIMD width for 16-bit batch16 kernel (AVX2: 256-bit / 16-bit = 16 lanes)
const SW_SIMD_WIDTH_16: usize = 16;

/// SIMD width for 16-bit batch32 kernel (AVX-512: 512-bit / 16-bit = 32 lanes)
const SW_SIMD_WIDTH_32: usize = 32;

/// SIMD width for 8-bit SSE/NEON KSW kernel (128-bit / 8-bit = 16 lanes)
const KSW_SIMD_WIDTH_SSE_NEON: usize = 16;

/// SIMD width for 8-bit AVX2 KSW kernel (256-bit / 8-bit = 32 lanes)
const KSW_SIMD_WIDTH_AVX2: usize = 32;

/// SIMD width for 8-bit AVX-512 KSW kernel (512-bit / 8-bit = 64 lanes)
const KSW_SIMD_WIDTH_AVX512: usize = 64;

/// Maximum sequence length for KSW kernel workspace pre-allocation
///
/// Set to 512 to handle:
/// - Modern 150bp PE reads (common today)
/// - Mate rescue sequences (can be 200-400bp)
/// - Long reads from newer sequencers
///
/// Note: Sequences exceeding this will fall back to aligned allocation
/// (not scalar fallback - that only happens for extremely long reads)
const KSW_MAX_SEQ_LEN: usize = 512;

// Thread-local workspace for alignment buffers
thread_local! {
    static WORKSPACE: RefCell<AlignmentWorkspace> = RefCell::new(AlignmentWorkspace::new());
}

/// Reusable buffers for the alignment pipeline
pub struct AlignmentWorkspace {
    /// Encoded query sequence (2-bit packed)
    pub encoded_query: Vec<u8>,
    /// Encoded reverse complement
    pub encoded_query_rc: Vec<u8>,
    /// Previous SMEM array buffer (for generate_smems_for_strand)
    pub smem_prev_buf: Vec<SMEM>,
    /// Current SMEM array buffer (for generate_smems_for_strand)
    pub smem_curr_buf: Vec<SMEM>,
    /// All SMEMs collected during seeding
    pub all_smems: Vec<SMEM>,
    /// Re-seeding candidates (middle_pos, min_intv)
    pub reseed_candidates: Vec<(usize, u64)>,

    // ========================================================================
    // SW Kernel Buffers (simd_banded_swa_batch16_int16)
    // ========================================================================
    // These eliminate ~65KB of allocations per batch call
    /// Query sequences in SoA layout (16-bit, 16 lanes)
    pub sw_query_soa_16: Vec<i16>,
    /// Target sequences in SoA layout (16-bit, 16 lanes)
    pub sw_target_soa_16: Vec<i16>,
    /// H matrix for DP (16-bit, 16 lanes)
    pub sw_h_matrix_16: Vec<i16>,
    /// E matrix for DP (16-bit, 16 lanes)
    pub sw_e_matrix_16: Vec<i16>,

    // ========================================================================
    // SW Kernel Buffers for AVX-512 (simd_banded_swa_batch32_int16)
    // ========================================================================
    // These eliminate ~130KB of allocations per batch call (32 lanes vs 16)
    /// Query sequences in SoA layout (16-bit, 32 lanes for AVX-512)
    pub sw_query_soa_32: Vec<i16>,
    /// Target sequences in SoA layout (16-bit, 32 lanes for AVX-512)
    pub sw_target_soa_32: Vec<i16>,
    /// H matrix for DP (16-bit, 32 lanes for AVX-512)
    pub sw_h_matrix_32: Vec<i16>,
    /// E matrix for DP (16-bit, 32 lanes for AVX-512)
    pub sw_e_matrix_32: Vec<i16>,

    // ========================================================================
    // KSW Kernel Buffers for SSE/NEON (batch_ksw_align_sse_neon)
    // ========================================================================
    // These eliminate ~33KB of allocations per batch call
    // Size: (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON = 513 * 16 = 8,208 bytes each
    // 64-byte aligned for aligned SIMD stores
    /// H0 buffer for horizontal SIMD (8-bit, 16 lanes)
    ksw_h0_buf_sse_neon: AlignedVec<u8>,
    /// H1 buffer for horizontal SIMD (8-bit, 16 lanes)
    ksw_h1_buf_sse_neon: AlignedVec<u8>,
    /// F buffer for horizontal SIMD (8-bit, 16 lanes)
    ksw_f_buf_sse_neon: AlignedVec<u8>,
    /// Row max buffer for horizontal SIMD (8-bit, 16 lanes)
    ksw_row_max_buf_sse_neon: AlignedVec<u8>,

    // ========================================================================
    // KSW Kernel Buffers for AVX2 (batch_ksw_align_avx2)
    // ========================================================================
    // These eliminate ~66KB of allocations per batch call
    // Size: (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2 = 513 * 32 = 16,416 bytes each
    // 64-byte aligned for aligned SIMD stores (_mm256_store_si256)
    /// H0 buffer for horizontal SIMD (8-bit, 32 lanes)
    ksw_h0_buf_avx2: AlignedVec<u8>,
    /// H1 buffer for horizontal SIMD (8-bit, 32 lanes)
    ksw_h1_buf_avx2: AlignedVec<u8>,
    /// F buffer for horizontal SIMD (8-bit, 32 lanes)
    ksw_f_buf_avx2: AlignedVec<u8>,
    /// Row max buffer for horizontal SIMD (8-bit, 32 lanes)
    ksw_row_max_buf_avx2: AlignedVec<u8>,

    // ========================================================================
    // KSW Kernel Buffers for AVX-512 (batch_ksw_align_avx512)
    // ========================================================================
    // These eliminate ~132KB of allocations per batch call
    // Size: (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512 = 513 * 64 = 32,832 bytes each
    // 64-byte aligned for aligned SIMD stores (_mm512_store_si512)
    /// H0 buffer for horizontal SIMD (8-bit, 64 lanes)
    ksw_h0_buf_avx512: AlignedVec<u8>,
    /// H1 buffer for horizontal SIMD (8-bit, 64 lanes)
    ksw_h1_buf_avx512: AlignedVec<u8>,
    /// F buffer for horizontal SIMD (8-bit, 64 lanes)
    ksw_f_buf_avx512: AlignedVec<u8>,
    /// Row max buffer for horizontal SIMD (8-bit, 64 lanes)
    ksw_row_max_buf_avx512: AlignedVec<u8>,

    // SoA sequence buffers for KSWV (reused across calls; 64B aligned)
    ksw_query_soa_16: AlignedVec<u8>,
    ksw_ref_soa_16: AlignedVec<u8>,
    ksw_query_soa_32: AlignedVec<u8>,
    ksw_ref_soa_32: AlignedVec<u8>,
    ksw_query_soa_64: AlignedVec<u8>,
    ksw_ref_soa_64: AlignedVec<u8>,

    // Per-lane scalar arrays for KSWV SoA carrier
    ksw_qlen: Vec<i8>,
    ksw_tlen: Vec<i8>,
    ksw_band: Vec<i8>,
    ksw_h0: Vec<i8>,
    ksw_lanes: usize,
    ksw_max_qlen: i32,
    ksw_max_tlen: i32,

    // ========================================================================
    // Banded SW (generic) reusable DP rows (64-byte aligned)
    // ========================================================================
    /// H/E/F rows for int8 scoring kernels (size managed via ensure_rows)
    banded_h_u8: AlignedVec<i8>,
    banded_e_u8: AlignedVec<i8>,
    banded_f_u8: AlignedVec<i8>,
    /// H/E/F rows for int16 scoring kernels (size managed via ensure_rows)
    banded_h_i16: AlignedVec<i16>,
    banded_e_i16: AlignedVec<i16>,
    banded_f_i16: AlignedVec<i16>,
    /// Tracked shape for current allocations (for debugging/validation)
    banded_shape: (usize, usize, usize), // (lanes, qmax, tmax)

    // Banded SWA SoA provider fields
    banded_query_soa: AlignedVec<u8>,
    banded_target_soa: AlignedVec<u8>,
    banded_qlen: Vec<i8>,
    banded_tlen: Vec<i8>,
    banded_band: Vec<i8>,
    banded_h0: Vec<i8>,
    banded_lanes: usize,
    banded_max_qlen: i32,
    banded_max_tlen: i32,
}

impl AlignmentWorkspace {
    /// Create a new workspace with pre-allocated buffers
    pub fn new() -> Self {
        Self {
            encoded_query: Vec::with_capacity(MAX_READ_LEN),
            encoded_query_rc: Vec::with_capacity(MAX_READ_LEN),
            smem_prev_buf: Vec::with_capacity(MAX_SMEMS_PER_STRAND),
            smem_curr_buf: Vec::with_capacity(MAX_SMEMS_PER_STRAND),
            all_smems: Vec::with_capacity(MAX_SMEMS_PER_STRAND * 2),
            reseed_candidates: Vec::with_capacity(64),

            // SW kernel buffers (16-bit, 16 lanes) - ~65KB total
            sw_query_soa_16: vec![0i16; SW_MAX_SEQ_LEN * SW_SIMD_WIDTH_16],
            sw_target_soa_16: vec![0i16; SW_MAX_SEQ_LEN * SW_SIMD_WIDTH_16],
            sw_h_matrix_16: vec![0i16; SW_MAX_SEQ_LEN * SW_SIMD_WIDTH_16],
            sw_e_matrix_16: vec![0i16; SW_MAX_SEQ_LEN * SW_SIMD_WIDTH_16],

            // SW kernel buffers for AVX-512 (16-bit, 32 lanes) - ~130KB total
            sw_query_soa_32: vec![0i16; SW_MAX_SEQ_LEN * SW_SIMD_WIDTH_32],
            sw_target_soa_32: vec![0i16; SW_MAX_SEQ_LEN * SW_SIMD_WIDTH_32],
            sw_h_matrix_32: vec![0i16; SW_MAX_SEQ_LEN * SW_SIMD_WIDTH_32],
            sw_e_matrix_32: vec![0i16; SW_MAX_SEQ_LEN * SW_SIMD_WIDTH_32],

            // KSW kernel buffers for SSE/NEON (8-bit, 16 lanes) - ~33KB total
            // 64-byte aligned for aligned SIMD stores
            ksw_h0_buf_sse_neon: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON,
            ),
            ksw_h1_buf_sse_neon: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON,
            ),
            ksw_f_buf_sse_neon: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON,
            ),
            ksw_row_max_buf_sse_neon: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON,
            ),

            // KSW kernel buffers for AVX2 (8-bit, 32 lanes) - ~66KB total
            // 64-byte aligned for aligned SIMD stores (_mm256_store_si256)
            ksw_h0_buf_avx2: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2),
            ksw_h1_buf_avx2: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2),
            ksw_f_buf_avx2: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2),
            ksw_row_max_buf_avx2: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2,
            ),

            // KSW kernel buffers for AVX-512 (8-bit, 64 lanes) - ~132KB total
            // 64-byte aligned for aligned SIMD stores (_mm512_store_si512)
            ksw_h0_buf_avx512: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512,
            ),
            ksw_h1_buf_avx512: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512,
            ),
            ksw_f_buf_avx512: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512,
            ),
            ksw_row_max_buf_avx512: AlignedVec::with_capacity(
                (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512,
            ),

            // KSW SoA sequence buffers (initially empty; sized on demand)
            ksw_query_soa_16: AlignedVec::with_capacity(0),
            ksw_ref_soa_16: AlignedVec::with_capacity(0),
            ksw_query_soa_32: AlignedVec::with_capacity(0),
            ksw_ref_soa_32: AlignedVec::with_capacity(0),
            ksw_query_soa_64: AlignedVec::with_capacity(0),
            ksw_ref_soa_64: AlignedVec::with_capacity(0),

            ksw_qlen: Vec::new(),
            ksw_tlen: Vec::new(),
            ksw_band: Vec::new(),
            ksw_h0: Vec::new(),
            ksw_lanes: 0,
            ksw_max_qlen: 0,
            ksw_max_tlen: 0,

            // Banded SW generic rows (empty; sized on demand)
            banded_h_u8: AlignedVec::with_capacity(0),
            banded_e_u8: AlignedVec::with_capacity(0),
            banded_f_u8: AlignedVec::with_capacity(0),
            banded_h_i16: AlignedVec::with_capacity(0),
            banded_e_i16: AlignedVec::with_capacity(0),
            banded_f_i16: AlignedVec::with_capacity(0),
            banded_shape: (0, 0, 0),

            // Banded SWA SoA provider fields
            banded_query_soa: AlignedVec::with_capacity(0),
            banded_target_soa: AlignedVec::with_capacity(0),
            banded_qlen: Vec::new(),
            banded_tlen: Vec::new(),
            banded_band: Vec::new(),
            banded_h0: Vec::new(),
            banded_lanes: 0,
            banded_max_qlen: 0,
            banded_max_tlen: 0,
        }
    }

    /// Clear all buffers for reuse (keeps capacity)
    pub fn clear(&mut self) {
        self.encoded_query.clear();
        self.encoded_query_rc.clear();
        self.smem_prev_buf.clear();
        self.smem_curr_buf.clear();
        self.all_smems.clear();
        self.reseed_candidates.clear();
        // Note: SW kernel buffers are not cleared here as they are
        // fully overwritten by the kernel before use
    }

    /// Reset SW kernel buffers to zero for AVX2 (call before batch processing)
    #[inline]
    pub fn reset_sw_buffers(&mut self) {
        self.sw_query_soa_16.fill(0);
        self.sw_target_soa_16.fill(0);
        self.sw_h_matrix_16.fill(0);
        self.sw_e_matrix_16.fill(0);
    }

    /// Reset SW kernel buffers to zero for AVX-512 (call before batch processing)
    #[inline]
    pub fn reset_sw_buffers_avx512(&mut self) {
        self.sw_query_soa_32.fill(0);
        self.sw_target_soa_32.fill(0);
        self.sw_h_matrix_32.fill(0);
        self.sw_e_matrix_32.fill(0);
    }

    /// Reset KSW kernel buffers to zero for SSE/NEON (call before batch processing)
    #[inline]
    pub fn reset_ksw_buffers_sse_neon(&mut self) {
        self.ksw_h0_buf_sse_neon.fill(0);
        self.ksw_h1_buf_sse_neon.fill(0);
        self.ksw_f_buf_sse_neon.fill(0);
        self.ksw_row_max_buf_sse_neon.fill(0);
    }

    /// Return ISA-appropriate KSW workspace buffers (H0, H1, F, RowMax) for a given SIMD width.
    /// The returned slices are 64-byte aligned and sized for `(KSW_MAX_SEQ_LEN+1) * lanes`.
    #[inline]
    pub fn ksw_buffers_for_width(
        &mut self,
        lanes: usize,
    ) -> (&mut [u8], &mut [u8], &mut [u8], &mut [u8]) {
        match lanes {
            KSW_SIMD_WIDTH_SSE_NEON => (
                self.ksw_h0_buf_sse_neon.as_mut_slice(),
                self.ksw_h1_buf_sse_neon.as_mut_slice(),
                self.ksw_f_buf_sse_neon.as_mut_slice(),
                self.ksw_row_max_buf_sse_neon.as_mut_slice(),
            ),
            KSW_SIMD_WIDTH_AVX2 => (
                self.ksw_h0_buf_avx2.as_mut_slice(),
                self.ksw_h1_buf_avx2.as_mut_slice(),
                self.ksw_f_buf_avx2.as_mut_slice(),
                self.ksw_row_max_buf_avx2.as_mut_slice(),
            ),
            KSW_SIMD_WIDTH_AVX512 => (
                self.ksw_h0_buf_avx512.as_mut_slice(),
                self.ksw_h1_buf_avx512.as_mut_slice(),
                self.ksw_f_buf_avx512.as_mut_slice(),
                self.ksw_row_max_buf_avx512.as_mut_slice(),
            ),
            _ => (
                self.ksw_h0_buf_sse_neon.as_mut_slice(),
                self.ksw_h1_buf_sse_neon.as_mut_slice(),
                self.ksw_f_buf_sse_neon.as_mut_slice(),
                self.ksw_row_max_buf_sse_neon.as_mut_slice(),
            ),
        }
    }

    /// Reset KSW kernel buffers to zero for AVX2 (call before batch processing)
    #[inline]
    pub fn reset_ksw_buffers_avx2(&mut self) {
        self.ksw_h0_buf_avx2.fill(0);
        self.ksw_h1_buf_avx2.fill(0);
        self.ksw_f_buf_avx2.fill(0);
        self.ksw_row_max_buf_avx2.fill(0);
    }

    /// Reset KSW kernel buffers to zero for AVX-512 (call before batch processing)
    #[inline]
    pub fn reset_ksw_buffers_avx512(&mut self) {
        self.ksw_h0_buf_avx512.fill(0);
        self.ksw_h1_buf_avx512.fill(0);
        self.ksw_f_buf_avx512.fill(0);
        self.ksw_row_max_buf_avx512.fill(0);
    }

    /// Get mutable references to SSE/NEON KSW buffers
    ///
    /// Returns (h0, h1, f, row_max) tuples suitable for the kswv_sse_neon kernel.
    /// The caller must ensure the workspace is not borrowed elsewhere.
    /// All buffers are 64-byte aligned.
    #[inline]
    pub fn ksw_buffers_sse_neon(&mut self) -> (&mut [u8], &mut [u8], &mut [u8], &mut [u8]) {
        (
            self.ksw_h0_buf_sse_neon.as_mut_slice(),
            self.ksw_h1_buf_sse_neon.as_mut_slice(),
            self.ksw_f_buf_sse_neon.as_mut_slice(),
            self.ksw_row_max_buf_sse_neon.as_mut_slice(),
        )
    }

    /// Get mutable references to AVX2 KSW buffers
    ///
    /// Returns (h0, h1, f, row_max) tuples suitable for the kswv_avx2 kernel.
    /// The caller must ensure the workspace is not borrowed elsewhere.
    /// All buffers are 64-byte aligned for aligned SIMD stores (_mm256_store_si256).
    #[inline]
    pub fn ksw_buffers_avx2(&mut self) -> (&mut [u8], &mut [u8], &mut [u8], &mut [u8]) {
        (
            self.ksw_h0_buf_avx2.as_mut_slice(),
            self.ksw_h1_buf_avx2.as_mut_slice(),
            self.ksw_f_buf_avx2.as_mut_slice(),
            self.ksw_row_max_buf_avx2.as_mut_slice(),
        )
    }

    /// Get mutable references to AVX-512 KSW buffers
    ///
    /// Returns (h0, h1, f, row_max) tuples suitable for the kswv_avx512 kernel.
    /// The caller must ensure the workspace is not borrowed elsewhere.
    /// All buffers are 64-byte aligned for aligned SIMD stores (_mm512_store_si512).
    #[inline]
    pub fn ksw_buffers_avx512(&mut self) -> (&mut [u8], &mut [u8], &mut [u8], &mut [u8]) {
        (
            self.ksw_h0_buf_avx512.as_mut_slice(),
            self.ksw_h1_buf_avx512.as_mut_slice(),
            self.ksw_f_buf_avx512.as_mut_slice(),
            self.ksw_row_max_buf_avx512.as_mut_slice(),
        )
    }

    /// Maximum supported query/reference length for KSW kernels
    #[inline]
    pub const fn ksw_max_seq_len() -> usize {
        KSW_MAX_SEQ_LEN
    }

    #[inline]
    fn ensure_aligned_capacity_u8(buf: &mut AlignedVec<i8>, needed: usize) {
        if buf.capacity < needed {
            *buf = AlignedVec::with_capacity(needed);
        } else {
            // update logical length if we ever use it as a slice view
            buf.len = needed;
        }
    }

    #[inline]
    fn ensure_aligned_capacity_i16(buf: &mut AlignedVec<i16>, needed: usize) {
        if buf.capacity < needed {
            *buf = AlignedVec::with_capacity(needed);
        } else {
            buf.len = needed;
        }
    }

    #[inline]
    fn ensure_aligned_capacity_u8_bytes(buf: &mut AlignedVec<u8>, needed: usize) {
        if buf.capacity < needed {
            *buf = AlignedVec::with_capacity(needed);
        } else {
            buf.len = needed;
        }
    }

    #[inline]
    fn ensure_ksw_scalar_capacity(&mut self, lanes: usize) {
        if self.ksw_qlen.len() != lanes {
            self.ksw_qlen.resize(lanes, 0);
        }
        if self.ksw_tlen.len() != lanes {
            self.ksw_tlen.resize(lanes, 0);
        }
        if self.ksw_band.len() != lanes {
            self.ksw_band.resize(lanes, 0);
        }
        if self.ksw_h0.len() != lanes {
            self.ksw_h0.resize(lanes, 0);
        }
        self.ksw_lanes = lanes;
    }

    /// Ensure KSW SoA sequence buffers for the given stride and lengths, and return mutable slices
    #[inline]
    fn ensure_ksw_soa_buffers(
        &mut self,
        lanes: usize,
        max_q: usize,
        max_t: usize,
    ) -> (&mut [u8], &mut [u8]) {
        let q_needed = lanes.saturating_mul(max_q.max(1));
        let t_needed = lanes.saturating_mul(max_t.max(1));
        match lanes {
            KSW_SIMD_WIDTH_SSE_NEON => {
                Self::ensure_aligned_capacity_u8_bytes(&mut self.ksw_query_soa_16, q_needed);
                Self::ensure_aligned_capacity_u8_bytes(&mut self.ksw_ref_soa_16, t_needed);
                (
                    self.ksw_query_soa_16.as_mut_slice(),
                    self.ksw_ref_soa_16.as_mut_slice(),
                )
            }
            KSW_SIMD_WIDTH_AVX2 => {
                Self::ensure_aligned_capacity_u8_bytes(&mut self.ksw_query_soa_32, q_needed);
                Self::ensure_aligned_capacity_u8_bytes(&mut self.ksw_ref_soa_32, t_needed);
                (
                    self.ksw_query_soa_32.as_mut_slice(),
                    self.ksw_ref_soa_32.as_mut_slice(),
                )
            }
            KSW_SIMD_WIDTH_AVX512 => {
                Self::ensure_aligned_capacity_u8_bytes(&mut self.ksw_query_soa_64, q_needed);
                Self::ensure_aligned_capacity_u8_bytes(&mut self.ksw_ref_soa_64, t_needed);
                (
                    self.ksw_query_soa_64.as_mut_slice(),
                    self.ksw_ref_soa_64.as_mut_slice(),
                )
            }
            _ => {
                // Fallback to the smallest supported width
                Self::ensure_aligned_capacity_u8_bytes(&mut self.ksw_query_soa_16, q_needed);
                Self::ensure_aligned_capacity_u8_bytes(&mut self.ksw_ref_soa_16, t_needed);
                (
                    self.ksw_query_soa_16.as_mut_slice(),
                    self.ksw_ref_soa_16.as_mut_slice(),
                )
            }
        }
    }
}

impl WorkspaceArena for AlignmentWorkspace {
    /// Ensure internal DP rows for the requested shape and element size.
    /// All rows are 64-byte aligned by construction of AlignedVec.
    fn ensure_rows(&mut self, lanes: usize, qmax: usize, tmax: usize, elem_size: usize) {
        let _ = tmax; // reserved for future use (row-per-target optimizations)
        let needed = lanes.saturating_mul(qmax.max(1));
        if elem_size == std::mem::size_of::<i8>() {
            Self::ensure_aligned_capacity_u8(&mut self.banded_h_u8, needed);
            Self::ensure_aligned_capacity_u8(&mut self.banded_e_u8, needed);
            Self::ensure_aligned_capacity_u8(&mut self.banded_f_u8, needed);
        } else if elem_size == std::mem::size_of::<i16>() {
            Self::ensure_aligned_capacity_i16(&mut self.banded_h_i16, needed);
            Self::ensure_aligned_capacity_i16(&mut self.banded_e_i16, needed);
            Self::ensure_aligned_capacity_i16(&mut self.banded_f_i16, needed);
        }
        self.banded_shape = (lanes, qmax, tmax);
    }

    fn rows_u8(&mut self) -> Option<(&mut [i8], &mut [i8], &mut [i8])> {
        let h = self.banded_h_u8.as_mut_slice();
        let e = self.banded_e_u8.as_mut_slice();
        let f = self.banded_f_u8.as_mut_slice();
        debug_assert_eq!((h.as_ptr() as usize) & 63, 0, "H row not 64B aligned");
        debug_assert_eq!((e.as_ptr() as usize) & 63, 0, "E row not 64B aligned");
        debug_assert_eq!((f.as_ptr() as usize) & 63, 0, "F row not 64B aligned");
        Some((h, e, f))
    }

    fn rows_u16(&mut self) -> Option<(&mut [i16], &mut [i16], &mut [i16])> {
        let h = self.banded_h_i16.as_mut_slice();
        let e = self.banded_e_i16.as_mut_slice();
        let f = self.banded_f_i16.as_mut_slice();
        debug_assert_eq!((h.as_ptr() as usize) & 63, 0, "H16 row not 64B aligned");
        debug_assert_eq!((e.as_ptr() as usize) & 63, 0, "E16 row not 64B aligned");
        debug_assert_eq!((f.as_ptr() as usize) & 63, 0, "F16 row not 64B aligned");
        Some((h, e, f))
    }
}

// =========================================================================
// KSWV SoA adapter (workspace-backed, 64B-aligned, zero per-call allocs)
// =========================================================================
impl AlignmentWorkspace {
    /// Ensure capacity and transpose AlignJob descriptors into an owned Banded SWA SoA view.
    pub fn ensure_and_transpose_banded_owned(
        &mut self,
        jobs: &[AlignJob],
        lanes: usize,
    ) -> OwnedSwSoA {
        let stride_lanes = lanes; // desired SIMD stride (e.g., 16/32)
        let job_count = jobs.len(); // actual number of active lanes

        // Determine maximum lengths across provided jobs
        let mut max_q = 0usize;
        let mut max_t = 0usize;
        for i in 0..job_count {
            max_q = max_q.max(jobs[i].qlen);
            max_t = max_t.max(jobs[i].tlen);
        }
        // Ensure capacities using the SIMD stride (pad inactive lanes)
        self.ensure_banded_soa_capacity(stride_lanes, max_q, max_t);

        let qbuf = self.banded_query_soa.as_mut_slice();
        let tbuf = self.banded_target_soa.as_mut_slice();

        // Initialize padding to 0xFF (common sentinel used by kernels)
        qbuf.fill(0xFF);
        tbuf.fill(0xFF);

        // Per-lane scalars and transpose for active jobs; pad the rest with zeros/defaults
        for lane in 0..stride_lanes {
            if lane < job_count {
                let j = &jobs[lane];
                let qn = j.qlen.min(max_q);
                let tn = j.tlen.min(max_t);
                // Write per-lane scalars (clamped to i8 range for u8 scoring path)
                self.banded_qlen[lane] = (j.qlen.min(127)).try_into().unwrap_or(127);
                self.banded_tlen[lane] = (j.tlen.min(127)).try_into().unwrap_or(127);
                self.banded_band[lane] = j.band.clamp(i32::MIN, 127) as i8;
                self.banded_h0[lane] = j.h0.clamp(i32::MIN, 127) as i8;

                // Transpose sequences into SoA: pos*stride + lane
                for pos in 0..qn {
                    qbuf[pos * stride_lanes + lane] = j.query[pos];
                }
                for pos in 0..tn {
                    tbuf[pos * stride_lanes + lane] = j.target[pos];
                }
            } else {
                // Inactive lanes: zero scalars; buffers already set to 0xFF
                self.banded_qlen[lane] = 0;
                self.banded_tlen[lane] = 0;
                self.banded_band[lane] = 0;
                self.banded_h0[lane] = 0;
            }
        }

        OwnedSwSoA {
            query_soa: self.banded_query_soa.as_mut_slice().to_vec(),
            target_soa: self.banded_target_soa.as_mut_slice().to_vec(),
            qlen: self.banded_qlen.clone(),
            tlen: self.banded_tlen.clone(),
            band: self.banded_band.clone(),
            h0: self.banded_h0.clone(),
            lanes: stride_lanes,
            max_qlen: self.banded_max_qlen,
            max_tlen: self.banded_max_tlen,
        }
    }

    #[inline]
    fn ensure_banded_soa_capacity(&mut self, lanes: usize, max_q: usize, max_t: usize) {
        let q_needed = lanes.saturating_mul(max_q);
        let t_needed = lanes.saturating_mul(max_t);
        if self.banded_query_soa.capacity < q_needed {
            self.banded_query_soa = AlignedVec::with_capacity(q_needed);
        } else {
            self.banded_query_soa.len = q_needed;
        }
        if self.banded_target_soa.capacity < t_needed {
            self.banded_target_soa = AlignedVec::with_capacity(t_needed);
        } else {
            self.banded_target_soa.len = t_needed;
        }
        // per-lane arrays
        if self.banded_qlen.len() != lanes {
            self.banded_qlen.resize(lanes, 0);
        }
        if self.banded_tlen.len() != lanes {
            self.banded_tlen.resize(lanes, 0);
        }
        if self.banded_band.len() != lanes {
            self.banded_band.resize(lanes, 0);
        }
        if self.banded_h0.len() != lanes {
            self.banded_h0.resize(lanes, 0);
        }
        self.banded_lanes = lanes;
        self.banded_max_qlen = max_q as i32;
        self.banded_max_tlen = max_t as i32;
    }

    /// Ensure capacity and transpose AlignJob descriptors into a KSWV SoA view.
    /// Buffers are sized as `max_len * lanes` and padded with 0xFF sentinel.
    pub fn ensure_and_transpose_ksw<'a>(
        &'a mut self,
        jobs: &[AlignJob<'a>],
        lanes: usize,
    ) -> KswSoA<'a> {
        let stride_lanes = lanes;
        let job_count = jobs.len().min(lanes);

        // Determine maximum lengths across provided jobs
        let mut max_q = 0usize;
        let mut max_t = 0usize;
        for i in 0..job_count {
            max_q = max_q.max(jobs[i].qlen);
            max_t = max_t.max(jobs[i].tlen);
        }

        // Ensure per-lane scalar arrays first (no borrows held across buffer writes)
        self.ensure_ksw_scalar_capacity(stride_lanes);
        for lane in 0..stride_lanes {
            if lane < job_count {
                let j = &jobs[lane];
                self.ksw_qlen[lane] = (j.qlen.min(127)).try_into().unwrap_or(127);
                self.ksw_tlen[lane] = (j.tlen.min(127)).try_into().unwrap_or(127);
                self.ksw_band[lane] = j.band.clamp(i32::MIN, 127) as i8;
                self.ksw_h0[lane] = j.h0.clamp(i32::MIN, 127) as i8;
            } else {
                self.ksw_qlen[lane] = 0;
                self.ksw_tlen[lane] = 0;
                self.ksw_band[lane] = 0;
                self.ksw_h0[lane] = 0;
            }
        }

        self.ksw_max_qlen = max_q as i32;
        self.ksw_max_tlen = max_t as i32;

        // Ensure sequence buffers and copy sequences (scope mutable borrows locally)
        let (query_slice_len, ref_slice_len) =
            (stride_lanes * max_q.max(1), stride_lanes * max_t.max(1));
        {
            let (qbuf, tbuf) = self.ensure_ksw_soa_buffers(stride_lanes, max_q, max_t);
            debug_assert_eq!(qbuf.len(), query_slice_len);
            debug_assert_eq!(tbuf.len(), ref_slice_len);
            qbuf.fill(0xFF);
            tbuf.fill(0xFF);
            for lane in 0..job_count {
                let j = &jobs[lane];
                let qn = j.qlen.min(max_q);
                let tn = j.tlen.min(max_t);
                for pos in 0..qn {
                    qbuf[pos * stride_lanes + lane] = j.query[pos];
                }
                for pos in 0..tn {
                    tbuf[pos * stride_lanes + lane] = j.target[pos];
                }
            }
        }

        // After the local scope, no mutable borrows remain; obtain immutable views
        let (ref_soa_slice, query_soa_slice): (&[u8], &[u8]) = match stride_lanes {
            KSW_SIMD_WIDTH_SSE_NEON => (
                &self.ksw_ref_soa_16.as_mut_slice()[..ref_slice_len],
                &self.ksw_query_soa_16.as_mut_slice()[..query_slice_len],
            ),
            KSW_SIMD_WIDTH_AVX2 => (
                &self.ksw_ref_soa_32.as_mut_slice()[..ref_slice_len],
                &self.ksw_query_soa_32.as_mut_slice()[..query_slice_len],
            ),
            KSW_SIMD_WIDTH_AVX512 => (
                &self.ksw_ref_soa_64.as_mut_slice()[..ref_slice_len],
                &self.ksw_query_soa_64.as_mut_slice()[..query_slice_len],
            ),
            _ => (
                &self.ksw_ref_soa_16.as_mut_slice()[..ref_slice_len],
                &self.ksw_query_soa_16.as_mut_slice()[..query_slice_len],
            ),
        };

        KswSoA {
            ref_soa: ref_soa_slice,
            query_soa: query_soa_slice,
            qlen: &self.ksw_qlen[..stride_lanes],
            tlen: &self.ksw_tlen[..stride_lanes],
            band: &self.ksw_band[..stride_lanes],
            h0: &self.ksw_h0[..stride_lanes],
            lanes: stride_lanes,
            max_qlen: self.ksw_max_qlen,
            max_tlen: self.ksw_max_tlen,
        }
    }
}

impl Default for AlignmentWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a closure with the thread-local workspace
///
/// # Example
/// ```ignore
/// use crate::alignment::workspace::with_workspace;
///
/// with_workspace(|ws| {
///     ws.clear();
///     // Use ws.encoded_query, ws.all_smems, etc.
/// });
/// ```
pub fn with_workspace<F, R>(f: F) -> R
where
    F: FnOnce(&mut AlignmentWorkspace) -> R,
{
    WORKSPACE.with(|ws| f(&mut ws.borrow_mut()))
}
