// ============================================================================
// BATCHED KSW (Smith-Waterman) for Mate Rescue
// ============================================================================
//
// This module implements horizontal SIMD batching for mate rescue alignments,
// matching BWA-MEM2's kswv.cpp architecture.
//
// Key concepts:
// 1. Structure of Arrays (SoA) - sequences transposed for SIMD efficiency
// 2. Batch processing - 16-64 alignments per SIMD operation (runtime detected)
// 3. Pre-allocated memory pools - avoid per-alignment allocation
//
// BWA-MEM2 reference: src/kswv.cpp, src/kswv.h
// ============================================================================

use std::alloc::{Layout, alloc, dealloc};
use crate::compute::simd_abstraction::simd::{SimdEngineType, get_simd_batch_sizes};

/// Maximum sequence lengths for buffer allocation
pub const MAX_SEQ_LEN_REF: usize = 2048;
pub const MAX_SEQ_LEN_QER: usize = 512;

/// Sequence pair metadata for batch processing
/// Matches BWA-MEM2's SeqPair struct (kswv.h:83-93)
#[derive(Debug, Clone, Default)]
pub struct SeqPair {
    /// Index into reference sequence buffer
    pub ref_idx: usize,
    /// Index into query sequence buffer
    pub query_idx: usize,
    /// Unique pair ID
    pub id: usize,
    /// Reference sequence length
    pub ref_len: i32,
    /// Query sequence length
    pub query_len: i32,
    /// Initial score (h0)
    pub h0: i32,
    /// Best alignment score (output)
    pub score: i32,
    /// Target end position (output)
    pub te: i32,
    /// Query end position (output)
    pub qe: i32,
    /// Second best score (output)
    pub score2: i32,
    /// Second best target end (output)
    pub te2: i32,
    /// Target start position (output)
    pub tb: i32,
    /// Query start position (output)
    pub qb: i32,
}

/// Alignment result matching BWA-MEM2's kswr_t
#[derive(Debug, Clone, Copy, Default)]
pub struct KswResult {
    pub score: i32,
    pub te: i32,
    pub qe: i32,
    pub score2: i32,
    pub te2: i32,
    pub tb: i32,
    pub qb: i32,
}

/// Structure of Arrays (SoA) buffer for batched SIMD processing
///
/// Layout: For position `k` across `batch_size` sequences:
///   data[k * batch_size + seq_idx] = sequence[seq_idx][k]
///
/// This allows a single SIMD load to fetch the same position from
/// all sequences in the batch simultaneously.
#[derive(Debug)]
pub struct SoABuffer {
    /// Reference sequences in SoA layout
    ref_data: *mut u8,
    /// Query sequences in SoA layout
    query_data: *mut u8,
    /// Layout for deallocation
    ref_layout: Layout,
    query_layout: Layout,
    /// Maximum dimension
    max_ref_len: usize,
    max_query_len: usize,
    batch_size: usize,
}

impl SoABuffer {
    /// Create a new SoA buffer with 64-byte alignment for SIMD
    ///
    /// # Arguments
    /// * `max_ref_len` - Maximum reference sequence length
    /// * `max_query_len` - Maximum query sequence length
    /// * `engine` - SIMD engine type (determines batch size)
    ///
    /// The batch size is automatically determined from the SIMD engine:
    /// - Engine128 (SSE/NEON): 16 sequences
    /// - Engine256 (AVX2): 32 sequences
    /// - Engine512 (AVX-512): 64 sequences
    pub fn new(max_ref_len: usize, max_query_len: usize, engine: SimdEngineType) -> Self {
        // Get batch size from SIMD engine capabilities
        let (max_batch, _) = get_simd_batch_sizes(engine);
        let batch_size = max_batch;

        let ref_size = max_ref_len * batch_size;
        let query_size = max_query_len * batch_size;

        // 64-byte alignment for cache line and AVX-512
        let ref_layout = Layout::from_size_align(ref_size, 64)
            .expect("Invalid layout for ref buffer");
        let query_layout = Layout::from_size_align(query_size, 64)
            .expect("Invalid layout for query buffer");

        let ref_data = unsafe { alloc(ref_layout) };
        let query_data = unsafe { alloc(query_layout) };

        assert!(!ref_data.is_null(), "Failed to allocate ref buffer");
        assert!(!query_data.is_null(), "Failed to allocate query buffer");

        // Initialize to 0xFF (padding value matching BWA-MEM2)
        unsafe {
            std::ptr::write_bytes(ref_data, 0xFF, ref_size);
            std::ptr::write_bytes(query_data, 0xFF, query_size);
        }

        Self {
            ref_data,
            query_data,
            ref_layout,
            query_layout,
            max_ref_len,
            max_query_len,
            batch_size,
        }
    }

    /// Transpose sequences from AoS to SoA layout
    ///
    /// # Arguments
    /// * `pairs` - Sequence pair metadata
    /// * `ref_seqs` - Reference sequences (AoS)
    /// * `query_seqs` - Query sequences (AoS)
    ///
    /// After this call, position `k` of sequence `j` is at:
    ///   ref_data[k * batch_size + j]
    ///   query_data[k * batch_size + j]
    pub fn transpose(
        &mut self,
        pairs: &[SeqPair],
        ref_seqs: &[&[u8]],
        query_seqs: &[&[u8]],
    ) {
        let num_pairs = pairs.len().min(self.batch_size);

        // Clear buffers with padding value
        unsafe {
            std::ptr::write_bytes(self.ref_data, 0xFF, self.max_ref_len * self.batch_size);
            std::ptr::write_bytes(self.query_data, 0xFF, self.max_query_len * self.batch_size);
        }

        // Transpose reference sequences
        for (j, (pair, ref_seq)) in pairs.iter().zip(ref_seqs.iter()).enumerate().take(num_pairs) {
            let len = (pair.ref_len as usize).min(ref_seq.len()).min(self.max_ref_len);
            for k in 0..len {
                let base = ref_seq[k];
                // Handle ambiguous bases (N = 4)
                let val = if base >= 4 { 4 } else { base };
                unsafe {
                    *self.ref_data.add(k * self.batch_size + j) = val;
                }
            }
        }

        // Transpose query sequences
        for (j, (pair, query_seq)) in pairs.iter().zip(query_seqs.iter()).enumerate().take(num_pairs) {
            let len = (pair.query_len as usize).min(query_seq.len()).min(self.max_query_len);
            for k in 0..len {
                let base = query_seq[k];
                let val = if base >= 4 { 4 } else { base };
                unsafe {
                    *self.query_data.add(k * self.batch_size + j) = val;
                }
            }
        }
    }

    /// Get pointer to reference SoA data
    #[inline]
    pub fn ref_ptr(&self) -> *const u8 {
        self.ref_data
    }

    /// Get pointer to query SoA data
    #[inline]
    pub fn query_ptr(&self) -> *const u8 {
        self.query_data
    }

    /// Get batch size
    #[inline]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl Drop for SoABuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.ref_data.is_null() {
                dealloc(self.ref_data, self.ref_layout);
            }
            if !self.query_data.is_null() {
                dealloc(self.query_data, self.query_layout);
            }
        }
    }
}

// Safety: SoABuffer manages raw pointers but owns the memory exclusively
unsafe impl Send for SoABuffer {}
unsafe impl Sync for SoABuffer {}

/// Pre-allocated memory pool for Smith-Waterman DP matrices
/// Matches BWA-MEM2's kswv class memory allocation pattern
#[derive(Debug)]
pub struct KswvMemoryPool {
    /// F scores (gap extension)
    f_buf: *mut i16,
    /// H scores row 0
    h0_buf: *mut i16,
    /// H scores row 1
    h1_buf: *mut i16,
    /// H max scores
    hmax_buf: *mut i16,
    /// Row maximums
    row_max_buf: *mut i16,
    /// Layout for deallocation
    layout: Layout,
    /// Dimensions
    max_query_len: usize,
    batch_size: usize,
    num_threads: usize,
}

impl KswvMemoryPool {
    /// Create a new memory pool for the given SIMD engine
    ///
    /// # Arguments
    /// * `max_query_len` - Maximum query sequence length
    /// * `engine` - SIMD engine type (determines batch size)
    /// * `num_threads` - Number of threads to allocate buffers for
    ///
    /// The batch size is automatically determined from the SIMD engine:
    /// - Engine128 (SSE/NEON): 16 sequences
    /// - Engine256 (AVX2): 32 sequences
    /// - Engine512 (AVX-512): 64 sequences
    pub fn new(max_query_len: usize, engine: SimdEngineType, num_threads: usize) -> Self {
        // Get batch size from SIMD engine capabilities
        let (max_batch, _) = get_simd_batch_sizes(engine);
        let batch_size = max_batch;

        // Each thread needs: batch_size * max_query_len * sizeof(i16) per buffer
        let per_thread_size = batch_size * max_query_len * std::mem::size_of::<i16>();
        let total_size = per_thread_size * num_threads;

        let layout = Layout::from_size_align(total_size, 64)
            .expect("Invalid layout for memory pool");

        let f_buf = unsafe { alloc(layout) as *mut i16 };
        let h0_buf = unsafe { alloc(layout) as *mut i16 };
        let h1_buf = unsafe { alloc(layout) as *mut i16 };
        let hmax_buf = unsafe { alloc(layout) as *mut i16 };
        let row_max_buf = unsafe { alloc(layout) as *mut i16 };

        assert!(!f_buf.is_null(), "Failed to allocate F buffer");
        assert!(!h0_buf.is_null(), "Failed to allocate H0 buffer");
        assert!(!h1_buf.is_null(), "Failed to allocate H1 buffer");
        assert!(!hmax_buf.is_null(), "Failed to allocate Hmax buffer");
        assert!(!row_max_buf.is_null(), "Failed to allocate rowMax buffer");

        Self {
            f_buf,
            h0_buf,
            h1_buf,
            hmax_buf,
            row_max_buf,
            layout,
            max_query_len,
            batch_size,
            num_threads,
        }
    }

    /// Get thread-local buffer slices
    ///
    /// # Safety
    /// Caller must ensure thread_id < num_threads
    #[inline]
    pub unsafe fn get_thread_buffers(&self, thread_id: usize) -> ThreadBuffers {
        let offset = thread_id * self.batch_size * self.max_query_len;
        ThreadBuffers {
            f: self.f_buf.add(offset),
            h0: self.h0_buf.add(offset),
            h1: self.h1_buf.add(offset),
            hmax: self.hmax_buf.add(offset),
            row_max: self.row_max_buf.add(offset),
            len: self.batch_size * self.max_query_len,
        }
    }
}

impl Drop for KswvMemoryPool {
    fn drop(&mut self) {
        unsafe {
            if !self.f_buf.is_null() { dealloc(self.f_buf as *mut u8, self.layout); }
            if !self.h0_buf.is_null() { dealloc(self.h0_buf as *mut u8, self.layout); }
            if !self.h1_buf.is_null() { dealloc(self.h1_buf as *mut u8, self.layout); }
            if !self.hmax_buf.is_null() { dealloc(self.hmax_buf as *mut u8, self.layout); }
            if !self.row_max_buf.is_null() { dealloc(self.row_max_buf as *mut u8, self.layout); }
        }
    }
}

unsafe impl Send for KswvMemoryPool {}
unsafe impl Sync for KswvMemoryPool {}

/// Thread-local buffer pointers
#[derive(Debug, Clone, Copy)]
pub struct ThreadBuffers {
    pub f: *mut i16,
    pub h0: *mut i16,
    pub h1: *mut i16,
    pub hmax: *mut i16,
    pub row_max: *mut i16,
    pub len: usize,
}

// ============================================================================
// BATCH KERNEL (placeholder - to be implemented with SIMD intrinsics)
// ============================================================================

/// Batched Smith-Waterman alignment using horizontal SIMD
///
/// Processes `batch_size` alignments in parallel using SIMD registers.
/// Each SIMD lane handles one alignment.
///
/// # Arguments
/// * `soa` - SoA buffer with transposed sequences
/// * `pairs` - Sequence pair metadata (lengths, h0)
/// * `results` - Output results (score, te, qe, tb, qb)
/// * `engine` - SIMD engine type (determines which kernel to use)
/// * `scoring` - Scoring parameters (match, mismatch, gap open/extend)
///
/// # Returns
/// Number of alignments processed
///
/// # SIMD Dispatch
///
/// This function dispatches to the appropriate SIMD kernel based on the engine:
/// - `Engine128`: SSE/NEON kernel (16 sequences in parallel)
/// - `Engine256`: AVX2 kernel (32 sequences in parallel)
/// - `Engine512`: AVX-512 kernel (64 sequences in parallel)
pub fn batch_ksw_align(
    soa: &SoABuffer,
    pairs: &mut [SeqPair],
    results: &mut [KswResult],
    engine: SimdEngineType,
    match_score: i8,
    mismatch_penalty: i8,
    gap_open: i32,
    gap_extend: i32,
) -> usize {
    // Dispatch to appropriate SIMD kernel based on engine type
    match engine {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        SimdEngineType::Engine512 => {
            use crate::alignment::kswv_avx512;

            // Call AVX-512 kernel (64-way horizontal SIMD)
            unsafe {
                kswv_avx512::batch_ksw_align_avx512(
                    soa.ref_ptr(),
                    soa.query_ptr(),
                    pairs[0].ref_len as i16,  // nrow
                    pairs[0].query_len as i16, // ncol
                    pairs,
                    results,
                    match_score,
                    mismatch_penalty,
                    gap_open,
                    gap_extend,
                    gap_open,   // o_ins (same as o_del for now)
                    gap_extend, // e_ins (same as e_del for now)
                    -1,         // w_ambig
                    0,          // phase
                )
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdEngineType::Engine256 => {
            // TODO: Implement AVX2 kernel (32-way horizontal SIMD)
            batch_ksw_align_fallback(soa, pairs, results, match_score, mismatch_penalty, gap_open, gap_extend)
        }
        SimdEngineType::Engine128 => {
            // TODO: Implement SSE/NEON kernel (16-way horizontal SIMD)
            batch_ksw_align_fallback(soa, pairs, results, match_score, mismatch_penalty, gap_open, gap_extend)
        }
    }
}

/// Fallback scalar implementation for batch_ksw_align
///
/// This is a temporary placeholder used while the actual SIMD kernels are being
/// implemented. It processes alignments sequentially to validate the infrastructure.
///
/// # Note
/// This function should NOT be used in production - it's only for testing the
/// SoA buffer and API infrastructure.
fn batch_ksw_align_fallback(
    soa: &SoABuffer,
    pairs: &mut [SeqPair],
    results: &mut [KswResult],
    _match_score: i8,
    _mismatch_penalty: i8,
    _gap_open: i32,
    _gap_extend: i32,
) -> usize {
    let num_pairs = pairs.len().min(soa.batch_size()).min(results.len());

    // Placeholder: Just return dummy results
    // Real SIMD implementations will process all pairs in parallel
    for i in 0..num_pairs {
        results[i] = KswResult {
            score: 0,
            te: -1,
            qe: -1,
            score2: -1,
            te2: -1,
            tb: -1,
            qb: -1,
        };

        // Copy results back to pair
        pairs[i].score = results[i].score;
        pairs[i].te = results[i].te;
        pairs[i].qe = results[i].qe;
        pairs[i].score2 = results[i].score2;
        pairs[i].te2 = results[i].te2;
        pairs[i].tb = results[i].tb;
        pairs[i].qb = results[i].qb;
    }

    num_pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_buffer_allocation() {
        // Test with Engine128 (16 sequences)
        let engine = SimdEngineType::Engine128;
        let soa = SoABuffer::new(256, 128, engine);
        assert!(!soa.ref_ptr().is_null());
        assert!(!soa.query_ptr().is_null());
        assert_eq!(soa.batch_size(), 16);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_soa_buffer_avx2() {
        // Test with Engine256 (32 sequences)
        let engine = SimdEngineType::Engine256;
        let soa = SoABuffer::new(256, 128, engine);
        assert!(!soa.ref_ptr().is_null());
        assert!(!soa.query_ptr().is_null());
        assert_eq!(soa.batch_size(), 32);
    }

    #[test]
    fn test_soa_transpose() {
        let engine = SimdEngineType::Engine128;
        let mut soa = SoABuffer::new(16, 16, engine);
        let batch_size = soa.batch_size();

        let pairs = vec![
            SeqPair { ref_len: 4, query_len: 4, ..Default::default() },
            SeqPair { ref_len: 3, query_len: 3, ..Default::default() },
        ];

        let ref_seqs: Vec<&[u8]> = vec![
            &[0, 1, 2, 3],  // ACGT
            &[3, 2, 1],     // TGC
        ];

        let query_seqs: Vec<&[u8]> = vec![
            &[0, 0, 1, 1],  // AACC
            &[2, 2, 3],     // GGT
        ];

        soa.transpose(&pairs, &ref_seqs, &query_seqs);

        // Check SoA layout: position 0 of all sequences
        unsafe {
            // ref[0][0] = 0 (A), ref[1][0] = 3 (T)
            assert_eq!(*soa.ref_ptr().add(0 * batch_size + 0), 0);
            assert_eq!(*soa.ref_ptr().add(0 * batch_size + 1), 3);

            // ref[0][1] = 1 (C), ref[1][1] = 2 (G)
            assert_eq!(*soa.ref_ptr().add(1 * batch_size + 0), 1);
            assert_eq!(*soa.ref_ptr().add(1 * batch_size + 1), 2);
        }
    }

    #[test]
    fn test_memory_pool() {
        let engine = SimdEngineType::Engine128;
        let pool = KswvMemoryPool::new(128, engine, 4);

        unsafe {
            let buffers = pool.get_thread_buffers(0);
            assert!(!buffers.f.is_null());
            assert!(!buffers.h0.is_null());
        }
    }

    #[test]
    fn test_batch_size_matches_engine() {
        // Engine128: should be 16
        let engine128 = SimdEngineType::Engine128;
        let soa128 = SoABuffer::new(256, 128, engine128);
        assert_eq!(soa128.batch_size(), 16);

        #[cfg(target_arch = "x86_64")]
        {
            // Engine256: should be 32
            let engine256 = SimdEngineType::Engine256;
            let soa256 = SoABuffer::new(256, 128, engine256);
            assert_eq!(soa256.batch_size(), 32);
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        {
            // Engine512: should be 64
            let engine512 = SimdEngineType::Engine512;
            let soa512 = SoABuffer::new(256, 128, engine512);
            assert_eq!(soa512.batch_size(), 64);
        }
    }
}
