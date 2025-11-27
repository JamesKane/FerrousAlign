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

use crate::pipelines::linear::seeding::SMEM;
use std::cell::RefCell;

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

/// SIMD width for 8-bit batch32 kernel (KSW)
const KSW_SIMD_WIDTH_8: usize = 32;

/// Maximum sequence length for KSW kernel
const KSW_MAX_SEQ_LEN: usize = 256;

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
    // KSW Kernel Buffers (batch_ksw_align_avx2)
    // ========================================================================
    // These eliminate ~32KB of allocations per batch call

    /// H0 buffer for horizontal SIMD (8-bit, 32 lanes)
    pub ksw_h0_buf: Vec<u8>,
    /// H1 buffer for horizontal SIMD (8-bit, 32 lanes)
    pub ksw_h1_buf: Vec<u8>,
    /// F buffer for horizontal SIMD (8-bit, 32 lanes)
    pub ksw_f_buf: Vec<u8>,
    /// Row max buffer for horizontal SIMD (8-bit, 32 lanes)
    pub ksw_row_max_buf: Vec<u8>,
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

            // KSW kernel buffers (8-bit, 32 lanes) - ~32KB total
            ksw_h0_buf: vec![0u8; (KSW_MAX_SEQ_LEN + 1) * KSW_SIMD_WIDTH_8],
            ksw_h1_buf: vec![0u8; KSW_MAX_SEQ_LEN * KSW_SIMD_WIDTH_8],
            ksw_f_buf: vec![0u8; KSW_MAX_SEQ_LEN * KSW_SIMD_WIDTH_8],
            ksw_row_max_buf: vec![0u8; KSW_MAX_SEQ_LEN * KSW_SIMD_WIDTH_8],
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

    /// Reset KSW kernel buffers to zero (call before batch processing)
    #[inline]
    pub fn reset_ksw_buffers(&mut self) {
        self.ksw_h0_buf.fill(0);
        self.ksw_h1_buf.fill(0);
        self.ksw_f_buf.fill(0);
        self.ksw_row_max_buf.fill(0);
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
