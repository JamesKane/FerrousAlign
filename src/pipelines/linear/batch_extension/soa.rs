use super::types::BatchedExtensionJob;

/// Build SoA (Structure-of-Arrays) interleaved buffers for the entire batch.
///
/// This transforms the AoS (Array-of-Structures) sequence data into a
/// SIMD-friendly SoA layout, where data for each lane is contiguous within
/// chunks of size W.
///
/// # Arguments
/// * `W` - SIMD width (e.g., 16 for AVX2, 32 for AVX-512)
pub fn make_batch_soa<const W: usize>(
    jobs: &[BatchedExtensionJob],
    query_seqs: &[u8],
    ref_seqs: &[u8],
) -> (Vec<u8>, Vec<u8>, Vec<usize>) {
    if jobs.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut query_soa = Vec::new();
    let mut target_soa = Vec::new();
    let mut pos_offsets = Vec::new();

    // Process jobs in chunks of W
    for jobs_chunk in jobs.chunks(W) {
        let chunk_base_offset_q = query_soa.len();
        let chunk_base_offset_t = target_soa.len();

        // Determine max lengths for this chunk (no cap - i16 path handles longer seqs)
        let max_qlen = jobs_chunk
            .iter()
            .map(|j| j.query_len)
            .max()
            .unwrap_or(0) as usize;
        let max_tlen = jobs_chunk
            .iter()
            .map(|j| j.ref_len)
            .max()
            .unwrap_or(0) as usize;

        // Store metadata for this chunk
        pos_offsets.push(chunk_base_offset_q);
        pos_offsets.push(chunk_base_offset_t);
        pos_offsets.push(max_qlen);
        pos_offsets.push(max_tlen);

        // Resize SoA buffers for the new chunk
        query_soa.resize(chunk_base_offset_q + max_qlen * W, 0xFF);
        target_soa.resize(chunk_base_offset_t + max_tlen * W, 0xFF);

        for (lane, job) in jobs_chunk.iter().enumerate() {
            let q_seq = &query_seqs[job.query_offset..job.query_offset + job.query_len as usize];
            let t_seq = &ref_seqs[job.ref_offset..job.ref_offset + job.ref_len as usize];

            // Interleave query sequence
            let q_len = (job.query_len as usize).min(max_qlen);
            for i in 0..q_len {
                query_soa[chunk_base_offset_q + i * W + lane] = q_seq[i];
            }

            // Interleave target sequence
            let t_len = (job.ref_len as usize).min(max_tlen);
            for i in 0..t_len {
                target_soa[chunk_base_offset_t + i * W + lane] = t_seq[i];
            }
        }
    }
    (query_soa, target_soa, pos_offsets)
}
