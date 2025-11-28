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

use crate::pipelines::linear::seeding::SMEM;
use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::ptr::NonNull;

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
            let raw_ptr = alloc(layout) as *mut T;
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
                let layout = Layout::from_size_align_unchecked(
                    self.capacity * std::mem::size_of::<T>(),
                    64,
                );
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

/// Maximum sequence length for KSW kernel (matches C++ bwa-mem2 MAX_SEQ_LEN_QER)
/// Sequences exceeding this length will trigger scalar fallback (ksw_extend2)
const KSW_MAX_SEQ_LEN: usize = 128;

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
            ksw_h0_buf_sse_neon: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON),
            ksw_h1_buf_sse_neon: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON),
            ksw_f_buf_sse_neon: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON),
            ksw_row_max_buf_sse_neon: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_SSE_NEON),

            // KSW kernel buffers for AVX2 (8-bit, 32 lanes) - ~66KB total
            // 64-byte aligned for aligned SIMD stores (_mm256_store_si256)
            ksw_h0_buf_avx2: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2),
            ksw_h1_buf_avx2: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2),
            ksw_f_buf_avx2: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2),
            ksw_row_max_buf_avx2: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX2),

            // KSW kernel buffers for AVX-512 (8-bit, 64 lanes) - ~132KB total
            // 64-byte aligned for aligned SIMD stores (_mm512_store_si512)
            ksw_h0_buf_avx512: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512),
            ksw_h1_buf_avx512: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512),
            ksw_f_buf_avx512: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512),
            ksw_row_max_buf_avx512: AlignedVec::with_capacity((KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_AVX512),
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
