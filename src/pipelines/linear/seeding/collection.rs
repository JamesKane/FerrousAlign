//! Seed collection and batch processing.
//!
//! Contains the main `find_seeds_batch` function that orchestrates
//! SMEM generation and conversion to seeds.

use crate::alignment::utils::{base_to_code, reverse_complement_code};
use crate::alignment::workspace::with_workspace;
use crate::core::io::soa_readers::SoAReadBatch;
use crate::pipelines::linear::index::index::BwaIndex;
use crate::pipelines::linear::mem_opt::MemOpt;

use super::bwt::get_sa_entries;
use super::smem::{forward_only_seed_strategy, generate_smems_for_strand, generate_smems_from_position};
use super::types::{Seed, SoASeedBatch, SoAEncodedQueryBatch, SMEM};

/// Match C++ SEEDS_PER_READ limit (see bwa-mem2/src/macro.h)
const SEEDS_PER_READ: usize = 500;

/// Find seeds for a batch of reads.
///
/// This is the main entry point for seeding. It:
/// 1. Encodes query sequences
/// 2. Generates SMEMs using bidirectional FM-index search
/// 3. Performs re-seeding for long unique matches
/// 4. Optionally runs forward-only seeding (3rd round)
/// 5. Converts SMEMs to seeds with reference positions
pub fn find_seeds_batch(
    bwa_idx: &BwaIndex,
    read_batch: &SoAReadBatch,
    opt: &MemOpt,
) -> (SoASeedBatch, SoAEncodedQueryBatch, SoAEncodedQueryBatch) {
    let num_reads = read_batch.len();
    let total_query_len: usize = read_batch.read_boundaries.iter().map(|(_, len)| *len).sum();

    let mut soa_seed_batch = SoASeedBatch::with_capacity(num_reads * 50, num_reads);
    let mut soa_encoded_query_batch =
        SoAEncodedQueryBatch::with_capacity(total_query_len, num_reads);
    let mut soa_encoded_query_rc_batch =
        SoAEncodedQueryBatch::with_capacity(total_query_len, num_reads);

    for read_idx in 0..num_reads {
        let (seq_start, query_len) = read_batch.read_boundaries[read_idx];
        let query_name = &read_batch.names[read_idx];
        let query_seq = &read_batch.seqs[seq_start..(seq_start + query_len)];

        // Create encoded versions of the query sequence
        let mut encoded_query = Vec::with_capacity(query_len);
        let mut encoded_query_rc = Vec::with_capacity(query_len);
        for &base in query_seq {
            let code = base_to_code(base);
            encoded_query.push(code);
            encoded_query_rc.push(reverse_complement_code(code));
        }
        encoded_query_rc.reverse();

        // Store encoded queries in SoA batches
        let current_encoded_query_start = soa_encoded_query_batch.encoded_seqs.len();
        soa_encoded_query_batch
            .encoded_seqs
            .extend_from_slice(&encoded_query);
        soa_encoded_query_batch
            .query_boundaries
            .push((current_encoded_query_start, query_len));

        let current_encoded_query_rc_start = soa_encoded_query_rc_batch.encoded_seqs.len();
        soa_encoded_query_rc_batch
            .encoded_seqs
            .extend_from_slice(&encoded_query_rc);
        soa_encoded_query_rc_batch
            .query_boundaries
            .push((current_encoded_query_rc_start, query_len));

        if query_len == 0 {
            soa_seed_batch
                .read_seed_boundaries
                .push((soa_seed_batch.query_pos.len(), 0));
            continue;
        }

        let min_seed_len = opt.min_seed_len;
        let min_intv = 1u64;

        log::debug!(
            "{query_name}: Starting SMEM generation: min_seed_len={min_seed_len}, min_intv={min_intv}, query_len={query_len}"
        );

        log::debug!(
            "SMEM_VALIDATION {}: Parameters: min_seed_len={}, max_occ={}, split_factor={:.2}, split_width={}, max_mem_intv={}",
            query_name,
            opt.min_seed_len,
            opt.max_occ,
            opt.split_factor,
            opt.split_width,
            opt.max_mem_intv
        );

        let mut max_smem_count = 0usize;
        let current_read_seed_start_idx = soa_seed_batch.query_pos.len();

        // Use thread-local workspace buffers
        let mut all_smems = with_workspace(|ws| {
            ws.all_smems.clear();

            generate_smems_for_strand(
                bwa_idx,
                query_name,
                query_len,
                &encoded_query,
                false,
                min_seed_len,
                min_intv,
                &mut ws.all_smems,
                &mut max_smem_count,
                &mut ws.smem_prev_buf,
                &mut ws.smem_curr_buf,
            );

            std::mem::take(&mut ws.all_smems)
        });

        let pass1_count = all_smems.len();
        log::debug!("SMEM_VALIDATION {query_name}: Pass 1 (initial) generated {pass1_count} SMEMs");
        log_smems_debug(query_name, "Pass1", &all_smems, 0);

        // Re-seeding pass
        let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;
        let split_width = opt.split_width as u64;

        let mut reseed_candidates: Vec<(usize, u64)> = Vec::with_capacity(32);

        for smem in all_smems.iter() {
            let smem_len = smem.query_end - smem.query_start + 1;
            if smem_len >= split_len && smem.interval_size <= split_width {
                let middle_pos = ((smem.query_start + smem.query_end + 1) >> 1) as usize;
                let new_min_intv = smem.interval_size + 1;

                log::debug!(
                    "{}: Re-seed candidate: smem m={}, n={}, len={}, s={}, middle_pos={}, new_min_intv={}",
                    query_name,
                    smem.query_start,
                    smem.query_end,
                    smem_len,
                    smem.interval_size,
                    middle_pos,
                    new_min_intv
                );

                reseed_candidates.push((middle_pos, new_min_intv));
            }
        }

        let initial_smem_count = all_smems.len();
        with_workspace(|ws| {
            for (middle_pos, new_min_intv) in &reseed_candidates {
                generate_smems_from_position(
                    bwa_idx,
                    query_name,
                    query_len,
                    &encoded_query,
                    false,
                    min_seed_len,
                    *new_min_intv,
                    *middle_pos,
                    &mut all_smems,
                    &mut ws.smem_prev_buf,
                    &mut ws.smem_curr_buf,
                );
            }
        });

        let pass2_added = all_smems.len() - initial_smem_count;
        if pass2_added > 0 {
            log::debug!(
                "SMEM_VALIDATION {}: Pass 2 (re-seeding) added {} new SMEMs (total: {})",
                query_name,
                pass2_added,
                all_smems.len()
            );
            log_smems_debug(query_name, "Pass2", &all_smems, initial_smem_count);
        } else {
            log::debug!("SMEM_VALIDATION {query_name}: Pass 2 (re-seeding) added 0 new SMEMs");
        }

        // 3rd round seeding
        let smems_before_3rd_round = all_smems.len();
        let mut used_3rd_round_seeding = false;

        if opt.max_mem_intv > 0 {
            used_3rd_round_seeding = true;
            log::debug!(
                "{}: Running 3rd round seeding (max_mem_intv={}) with {} existing SMEMs",
                query_name,
                opt.max_mem_intv,
                all_smems.len()
            );

            forward_only_seed_strategy(
                bwa_idx,
                query_name,
                query_len,
                &encoded_query,
                false,
                min_seed_len,
                opt.max_mem_intv,
                &mut all_smems,
            );

            let pass3_added = all_smems.len() - smems_before_3rd_round;
            if pass3_added > 0 {
                log::debug!(
                    "SMEM_VALIDATION {}: Pass 3 (forward-only) added {} new SMEMs (total: {})",
                    query_name,
                    pass3_added,
                    all_smems.len()
                );
                log_smems_debug(query_name, "Pass3", &all_smems, smems_before_3rd_round);
            } else {
                log::debug!(
                    "SMEM_VALIDATION {query_name}: Pass 3 (forward-only) added 0 new SMEMs"
                );
            }
        } else {
            log::debug!(
                "SMEM_VALIDATION {query_name}: Pass 3 (forward-only) skipped (max_mem_intv=0)"
            );
        }

        // Filter SMEMs
        let mut unique_filtered_smems: Vec<SMEM> = Vec::with_capacity(all_smems.len());
        all_smems.sort_by_key(|smem| {
            (
                smem.query_start,
                smem.query_end,
                smem.bwt_interval_start,
                smem.is_reverse_complement,
            )
        });

        let effective_max_occ = if used_3rd_round_seeding {
            let min_occ = all_smems
                .iter()
                .map(|s| s.interval_size)
                .min()
                .unwrap_or(opt.max_occ as u64);
            let relaxed_threshold = (min_occ + 1).max(opt.max_occ as u64);
            log::debug!(
                "{}: 3rd round seeding used, relaxing max_occ filter from {} to {} (min_occ={})",
                query_name,
                opt.max_occ,
                relaxed_threshold,
                min_occ
            );
            relaxed_threshold
        } else {
            opt.max_occ as u64
        };

        for smem in all_smems.iter() {
            let seed_len = smem.query_end - smem.query_start + 1;
            let occurrences = smem.interval_size;
            if seed_len >= opt.min_seed_len && occurrences <= effective_max_occ {
                unique_filtered_smems.push(*smem);
            }
        }

        log::debug!(
            "SMEM_VALIDATION {}: Filtering summary: {} total SMEMs -> {} unique (min_seed_len={}, max_occ={})",
            query_name,
            all_smems.len(),
            unique_filtered_smems.len(),
            opt.min_seed_len,
            effective_max_occ
        );

        log::debug!(
            "{}: Generated {} SMEMs, filtered to {} unique",
            query_name,
            all_smems.len(),
            unique_filtered_smems.len()
        );

        log_smems_debug(query_name, "SMEM_OVERLAP", &unique_filtered_smems, 0);

        let mut sorted_smems = unique_filtered_smems;
        sorted_smems.sort_by_key(|smem| -(smem.query_end - smem.query_start + 1));

        let is_highly_repetitive = sorted_smems.len() <= 4
            && sorted_smems
                .iter()
                .all(|s| s.query_end - s.query_start > (query_len as i32 * 3 / 4));

        let seeds_per_smem = if sorted_smems.is_empty() {
            SEEDS_PER_READ
        } else if is_highly_repetitive {
            SEEDS_PER_READ
        } else {
            (SEEDS_PER_READ / sorted_smems.len()).max(1)
        };

        let mut current_read_seeds: Vec<Seed> = Vec::with_capacity(SEEDS_PER_READ);
        let mut seeds_per_smem_count = Vec::with_capacity(sorted_smems.len());

        for (smem_idx, smem) in sorted_smems.iter().enumerate() {
            let seeds_before = current_read_seeds.len();

            let max_positions_this_smem = (seeds_per_smem as u32).min(opt.max_occ as u32);

            let ref_positions = get_sa_entries(
                bwa_idx,
                smem.bwt_interval_start,
                smem.interval_size,
                max_positions_this_smem,
            );

            let seed_len = smem.query_end - smem.query_start;
            let l_pac = bwa_idx.bns.packed_sequence_length;

            if log::log_enabled!(log::Level::Debug) {
                log::debug!(
                    "SEED_CONVERSION {}: SMEM[{}] query[{}..{}] -> {} ref positions (requested: {})",
                    query_name,
                    smem_idx,
                    smem.query_start,
                    smem.query_end,
                    ref_positions.len(),
                    max_positions_this_smem
                );
            }

            let mut skipped_boundary = 0;
            for (pos_idx, ref_pos) in ref_positions.iter().enumerate() {
                let rid = bwa_idx.bns.pos_to_rid(*ref_pos, *ref_pos + seed_len as u64);
                if rid < 0 {
                    skipped_boundary += 1;
                    continue;
                }

                let is_rev = *ref_pos >= l_pac;

                if log::log_enabled!(log::Level::Debug) {
                    log::debug!(
                        "SEED_CONVERSION {}:   Seed[{}]: ref_pos={} is_rev={} rid={} chr={}",
                        query_name,
                        pos_idx,
                        ref_pos,
                        is_rev,
                        rid,
                        if rid >= 0 {
                            bwa_idx.bns.annotations[rid as usize].name.as_str()
                        } else {
                            "N/A"
                        }
                    );
                }

                let seed = Seed {
                    query_pos: smem.query_start,
                    ref_pos: *ref_pos,
                    len: seed_len,
                    is_rev,
                    interval_size: smem.interval_size,
                    rid,
                };
                current_read_seeds.push(seed);

                if current_read_seeds.len() >= SEEDS_PER_READ {
                    log::debug!(
                        "{query_name}: Hit SEEDS_PER_READ limit ({SEEDS_PER_READ}), truncating"
                    );
                    break;
                }
            }

            let seeds_added = current_read_seeds.len() - seeds_before;
            seeds_per_smem_count.push((smem_idx, seeds_added, skipped_boundary));

            if current_read_seeds.len() >= SEEDS_PER_READ {
                break;
            }
        }

        log::debug!(
            "SEED_CONVERSION {}: Total {} SMEMs -> {} seeds ({} seeds/SMEM limit)",
            query_name,
            sorted_smems.len(),
            current_read_seeds.len(),
            seeds_per_smem
        );

        if log::log_enabled!(log::Level::Debug) {
            for &(idx, count, skipped) in seeds_per_smem_count.iter().take(10) {
                if count > 0 || skipped > 0 {
                    log::debug!(
                        "SEED_CONVERSION {query_name}:   SMEM[{idx}] -> {count} seeds ({skipped} skipped at boundary)"
                    );
                }
            }
            if seeds_per_smem_count.len() > 10 {
                log::debug!(
                    "SEED_CONVERSION {}:   ... ({} more SMEMs)",
                    query_name,
                    seeds_per_smem_count.len() - 10
                );
            }
        }

        if max_smem_count > query_len {
            log::debug!(
                "{query_name}: SMEM buffer grew beyond initial capacity! max_smem_count={max_smem_count} > query_len={query_len}"
            );
        }

        log::debug!(
            "{}: Created {} seeds from {} SMEMs",
            query_name,
            current_read_seeds.len(),
            sorted_smems.len()
        );

        // Populate SoASeedBatch
        for seed in current_read_seeds {
            soa_seed_batch.query_pos.push(seed.query_pos);
            soa_seed_batch.ref_pos.push(seed.ref_pos);
            soa_seed_batch.len.push(seed.len);
            soa_seed_batch.is_rev.push(seed.is_rev);
            soa_seed_batch.interval_size.push(seed.interval_size);
            soa_seed_batch.rid.push(seed.rid);
        }
        let num_seeds_for_read = soa_seed_batch.query_pos.len() - current_read_seed_start_idx;
        soa_seed_batch
            .read_seed_boundaries
            .push((current_read_seed_start_idx, num_seeds_for_read));
    }

    (
        soa_seed_batch,
        soa_encoded_query_batch,
        soa_encoded_query_rc_batch,
    )
}

/// Helper to log SMEMs in debug mode.
fn log_smems_debug(query_name: &str, pass_name: &str, smems: &[SMEM], skip: usize) {
    if !log::log_enabled!(log::Level::Debug) {
        return;
    }

    log::debug!(
        "{} {}: {} SMEMs after filtering:",
        pass_name,
        query_name,
        smems.len() - skip
    );

    for (idx, smem) in smems.iter().skip(skip).enumerate().take(10) {
        log::debug!(
            "{} {}:   SMEM[{}]: query[{}..{}] len={} bwt=[{}, {}) size={}",
            pass_name,
            query_name,
            idx,
            smem.query_start,
            smem.query_end,
            smem.query_end - smem.query_start + 1,
            smem.bwt_interval_start,
            smem.bwt_interval_end,
            smem.interval_size
        );
    }

    if smems.len() - skip > 10 {
        log::debug!(
            "{} {}:   ... ({} more SMEMs)",
            pass_name,
            query_name,
            smems.len() - skip - 10
        );
    }
}
