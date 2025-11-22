use crate::alignment::banded_swa::BandedPairWiseSW;
use crate::alignment::banded_swa::merge_cigar_operations;
use crate::alignment::chaining::Chain;
use crate::alignment::chaining::cal_max_gap;
use crate::alignment::chaining::chain_seeds;
use crate::alignment::chaining::filter_chains;
use crate::alignment::extension::AlignmentJob;
use crate::alignment::extension::execute_adaptive_alignments;
use crate::alignment::extension::execute_scalar_alignments;
use crate::alignment::finalization::Alignment;
use crate::alignment::finalization::generate_sa_tags;
use crate::alignment::finalization::generate_xa_tags;
use crate::alignment::finalization::mark_secondary_alignments;
use crate::alignment::finalization::sam_flags;
use crate::alignment::seeding::SMEM;
use crate::alignment::seeding::Seed;
use crate::alignment::seeding::generate_smems_for_strand;
use crate::alignment::seeding::generate_smems_from_position;
use crate::alignment::seeding::get_sa_entries;
use crate::alignment::utils::base_to_code;
use crate::alignment::utils::reverse_complement_code;
use crate::index::BwaIndex;
use crate::mem_opt::MemOpt;
use crate::utils::hash_64;

// ============================================================================
// SEED GENERATION (SMEM EXTRACTION)
// ============================================================================
//
// This section contains the main seed generation pipeline:
// - SMEM (Supermaximal Exact Match) extraction using FM-Index
// - Bidirectional search (forward and reverse complement)
// - Seed extension and filtering
// ============================================================================

pub fn generate_seeds(
    bwa_idx: &BwaIndex,
    pac_data: &[u8], // Pre-loaded PAC data for MD tag generation
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
) -> Vec<Alignment> {
    // Call the new refactored pipeline
    align_read(
        bwa_idx, pac_data, query_name, query_seq, query_qual, opt, true, // use_batched_simd
    )
}

// ============================================================================
// REFACTORED PIPELINE
// ============================================================================

// Data structures for refactoring
#[derive(Debug, Clone)]
struct SeedJobMapping {
    seed_idx: usize,              // Index into chain.seeds
    left_job_idx: Option<usize>,  // LEFT extension job index
    right_job_idx: Option<usize>, // RIGHT extension job index
}

#[derive(Debug, Clone)]
struct ChainJobMapping {
    seed_jobs: Vec<SeedJobMapping>, // Multiple seeds per chain
}

type RawAlignment = (i32, Vec<(u8, i32)>, Vec<u8>, Vec<u8>);

struct ExtensionResult {
    extended_cigars: Vec<RawAlignment>,
    chain_to_job_map: Vec<ChainJobMapping>,
    filtered_chains: Vec<Chain>,
    sorted_seeds: Vec<Seed>,
    encoded_query: Vec<u8>,
    encoded_query_rc: Vec<u8>,
}

/// A new, top-level function that clearly shows the alignment pipeline.
/// This is the refactored `generate_seeds`.
pub fn align_read(
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
    use_batched_simd: bool,
) -> Vec<Alignment> {
    // 1. SEEDING: Find maximal exact matches (SMEMs)
    let (seeds, encoded_query, encoded_query_rc) = find_seeds(bwa_idx, query_name, query_seq, opt);

    // 2. CHAINING: Group seeds into potential alignment chains
    let (chains, sorted_seeds) = build_and_filter_chains(seeds, opt, query_seq.len());

    // 3. EXTENSION: Perform banded Smith-Waterman on the chains
    let extension_result = extend_chains_to_alignments(
        bwa_idx,
        query_name,
        opt,
        chains,
        sorted_seeds,
        encoded_query,
        encoded_query_rc,
        use_batched_simd,
    );

    // 4. FINALIZATION: Score, clean up, and format the final alignments
    let final_alignments = finalize_alignments(
        extension_result,
        bwa_idx,
        pac_data,
        query_name,
        query_seq,
        query_qual,
        opt,
    );

    final_alignments
}

/// Stage 1: Seed Finding
fn find_seeds(
    bwa_idx: &BwaIndex,
    query_name: &str,
    query_seq: &[u8],
    opt: &MemOpt,
) -> (Vec<Seed>, Vec<u8>, Vec<u8>) {
    let query_len = query_seq.len();
    if query_len == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    #[cfg(feature = "debug-logging")]
    let is_debug_read = query_name.contains("1150:14380");

    #[cfg(feature = "debug-logging")]
    if is_debug_read {
        log::debug!("[DEBUG_READ] Generating seeds for: {}", query_name);
        log::debug!("[DEBUG_READ] Query length: {}", query_len);
    }

    // Create encoded versions of the query sequence
    let mut encoded_query = Vec::with_capacity(query_len);
    let mut encoded_query_rc = Vec::with_capacity(query_len); // Reverse complement
    for &base in query_seq {
        let code = base_to_code(base);
        encoded_query.push(code);
        encoded_query_rc.push(reverse_complement_code(code));
    }
    encoded_query_rc.reverse();

    let mut all_smems: Vec<SMEM> = Vec::new();
    let min_seed_len = opt.min_seed_len;
    let min_intv = 1u64;

    log::debug!(
        "{}: Starting SMEM generation: min_seed_len={}, min_intv={}, query_len={}",
        query_name,
        min_seed_len,
        min_intv,
        query_len
    );

    let mut max_smem_count = 0usize;

    // Process forward and reverse complement strands
    generate_smems_for_strand(
        bwa_idx,
        query_name,
        query_len,
        &encoded_query,
        false,
        min_seed_len,
        min_intv,
        &mut all_smems,
        &mut max_smem_count,
    );
    generate_smems_for_strand(
        bwa_idx,
        query_name,
        query_len,
        &encoded_query_rc,
        true,
        min_seed_len,
        min_intv,
        &mut all_smems,
        &mut max_smem_count,
    );

    // Re-seeding pass: For long unique SMEMs, re-seed from middle to find split alignments
    // This matches C++ bwamem.cpp:695-714
    let split_len = (opt.min_seed_len as f32 * opt.split_factor) as i32;
    let split_width = opt.split_width as u64;

    // Collect re-seeding candidates from initial SMEMs
    let mut reseed_candidates: Vec<(usize, u64, bool)> = Vec::new(); // (middle_pos, min_intv, is_rc)

    for smem in all_smems.iter() {
        let smem_len = smem.query_end - smem.query_start + 1;
        // Re-seed if: length >= split_len AND interval_size <= split_width
        if smem_len >= split_len && smem.interval_size <= split_width {
            // Calculate middle position: (start + end + 1) >> 1 to match C++
            let middle_pos = ((smem.query_start + smem.query_end + 1) >> 1) as usize;
            let new_min_intv = smem.interval_size + 1;

            log::debug!(
                "{}: Re-seed candidate: smem m={}, n={}, len={}, s={}, middle_pos={}, new_min_intv={}, is_rc={}",
                query_name,
                smem.query_start,
                smem.query_end,
                smem_len,
                smem.interval_size,
                middle_pos,
                new_min_intv,
                smem.is_reverse_complement
            );

            reseed_candidates.push((middle_pos, new_min_intv, smem.is_reverse_complement));
        }
    }

    // Execute re-seeding for each candidate
    let initial_smem_count = all_smems.len();
    for (middle_pos, new_min_intv, is_rc) in reseed_candidates {
        let encoded = if is_rc {
            &encoded_query_rc
        } else {
            &encoded_query
        };

        generate_smems_from_position(
            bwa_idx,
            query_name,
            query_len,
            encoded,
            is_rc,
            min_seed_len,
            new_min_intv,
            middle_pos,
            &mut all_smems,
        );
    }

    if all_smems.len() > initial_smem_count {
        log::debug!(
            "{}: Re-seeding added {} new SMEMs (total: {})",
            query_name,
            all_smems.len() - initial_smem_count,
            all_smems.len()
        );
    }

    // Filter SMEMs
    let mut unique_filtered_smems: Vec<SMEM> = Vec::new();
    all_smems.sort_by_key(|smem| {
        (
            smem.query_start,
            smem.query_end,
            smem.bwt_interval_start,
            smem.is_reverse_complement,
        )
    });

    // NOTE: split_factor and split_width control RE-SEEDING for chimeric detection,
    // NOT seed filtering. The basic filter (min_seed_len + max_occ) is sufficient.
    // The previous "chimeric filter" was incorrectly discarding valid seeds.
    // See C++ bwamem.cpp:639-695 - split logic is for creating additional sub-seeds,
    // not for removing seeds that pass the basic quality checks.

    if let Some(mut prev_smem) = all_smems.first().cloned() {
        let mut process_smem = |smem: SMEM, _is_first: bool| {
            let seed_len = smem.query_end - smem.query_start + 1;
            let occurrences = smem.interval_size;
            // Keep seeds that pass basic quality filter (min_seed_len AND max_occ)
            let keep = seed_len >= opt.min_seed_len && occurrences <= opt.max_occ as u64;
            if keep {
                unique_filtered_smems.push(smem);
            }
        };
        process_smem(prev_smem, true);
        for i in 1..all_smems.len() {
            let current_smem = all_smems[i];
            if current_smem != prev_smem {
                process_smem(current_smem, false);
            }
            prev_smem = current_smem;
        }
    }

    log::debug!(
        "{}: Generated {} SMEMs, filtered to {} unique",
        query_name,
        all_smems.len(),
        unique_filtered_smems.len()
    );

    let mut sorted_smems = unique_filtered_smems;
    sorted_smems.sort_by_key(|smem| -(smem.query_end - smem.query_start + 1));

    // Match C++ SEEDS_PER_READ limit (see bwa-mem2/src/macro.h)
    const SEEDS_PER_READ: usize = 500;

    let mut seeds = Vec::new();
    for smem in sorted_smems.iter() {
        // Use the new get_sa_entries function to get multiple reference positions
        let ref_positions = crate::alignment::seeding::get_sa_entries(
            bwa_idx,
            smem.bwt_interval_start,
            smem.interval_size,
            opt.max_occ as u32,
        );

        let seed_len = smem.query_end - smem.query_start;
        for ref_pos in ref_positions {
            // Compute rid (chromosome ID) - skip seeds that span chromosome boundaries
            // Matches C++ bwamem.cpp:911-914
            let rid = bwa_idx.bns.pos_to_rid(ref_pos, ref_pos + seed_len as u64);
            if rid < 0 {
                // Seed spans multiple chromosomes or forward-reverse boundary - skip
                continue;
            }

            let seed = Seed {
                query_pos: smem.query_start,
                ref_pos,
                len: seed_len,
                is_rev: smem.is_reverse_complement,
                interval_size: smem.interval_size,
                rid,
            };
            seeds.push(seed);

            // Hard limit on seeds per read to prevent memory explosion
            if seeds.len() >= SEEDS_PER_READ {
                log::debug!(
                    "{}: Hit SEEDS_PER_READ limit ({}), truncating",
                    query_name,
                    SEEDS_PER_READ
                );
                break;
            }
        }

        if seeds.len() >= SEEDS_PER_READ {
            break;
        }
    }

    if max_smem_count > query_len {
        log::debug!(
            "{}: SMEM buffer grew beyond initial capacity! max_smem_count={} > query_len={}",
            query_name,
            max_smem_count,
            query_len
        );
    }

    (seeds, encoded_query, encoded_query_rc)
}

/// Stage 2: Chaining
fn build_and_filter_chains(
    seeds: Vec<Seed>,
    opt: &MemOpt,
    query_len: usize,
) -> (Vec<Chain>, Vec<Seed>) {
    // Chain seeds together and then filter them
    let (mut chained_results, sorted_seeds) = chain_seeds(seeds, opt);
    log::debug!("Chaining produced {} chains", chained_results.len());

    let filtered_chains = filter_chains(&mut chained_results, &sorted_seeds, opt, query_len as i32);
    log::debug!(
        "Chain filtering kept {} chains (from {} total)",
        filtered_chains.len(),
        chained_results.len()
    );

    (filtered_chains, sorted_seeds)
}

/// Stage 3: Seed Extension / Alignment
fn extend_chains_to_alignments(
    bwa_idx: &BwaIndex,
    _query_name: &str,
    opt: &MemOpt,
    chains: Vec<Chain>,
    seeds: Vec<Seed>,
    encoded_query: Vec<u8>,
    encoded_query_rc: Vec<u8>,
    use_batched_simd: bool,
) -> ExtensionResult {
    let sw_params = BandedPairWiseSW::new(
        opt.o_del,
        opt.e_del,
        opt.o_ins,
        opt.e_ins,
        opt.zdrop,
        5,
        opt.pen_clip5,
        opt.pen_clip3,
        opt.mat,
        opt.a as i8,
        -(opt.b as i8),
    );

    let mut alignment_jobs = Vec::new();
    let mut chain_to_job_map: Vec<ChainJobMapping> = Vec::new();

    for (chain_idx, chain) in chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            chain_to_job_map.push(ChainJobMapping {
                seed_jobs: Vec::new(),
            });
            continue;
        }

        let full_query = if chain.is_rev {
            &encoded_query_rc
        } else {
            &encoded_query
        };
        let query_len = full_query.len() as i32;

        let l_pac = bwa_idx.bns.packed_sequence_length;
        let (mut rmax_0, mut rmax_1) = (l_pac << 1, 0u64);

        for &seed_idx in &chain.seeds {
            let seed = &seeds[seed_idx];
            let left_margin = seed.query_pos + cal_max_gap(opt, seed.query_pos);
            let b = seed.ref_pos.saturating_sub(left_margin as u64);
            let remaining_query = query_len - seed.query_pos - seed.len;
            let right_margin = remaining_query + cal_max_gap(opt, remaining_query);
            let e = seed.ref_pos + seed.len as u64 + right_margin as u64;
            rmax_0 = rmax_0.min(b);
            rmax_1 = rmax_1.max(e);
        }
        rmax_1 = rmax_1.min(l_pac << 1);
        if rmax_0 < l_pac && l_pac < rmax_1 {
            if seeds[chain.seeds[0]].ref_pos < l_pac {
                rmax_1 = l_pac;
            } else {
                rmax_0 = l_pac;
            }
        }

        let rseq = match bwa_idx.bns.get_reference_segment(rmax_0, rmax_1 - rmax_0) {
            Ok(seq) => seq,
            Err(_) => {
                chain_to_job_map.push(ChainJobMapping {
                    seed_jobs: Vec::new(),
                });
                continue;
            }
        };

        let mut seed_job_mappings = Vec::new();
        for &seed_chain_idx in chain.seeds.iter().rev() {
            let seed = &seeds[seed_chain_idx];
            let mut left_job_idx = None;
            let mut right_job_idx = None;

            if seed.query_pos > 0 {
                let tmp = (seed.ref_pos - rmax_0) as usize;
                if tmp > 0 && tmp <= rseq.len() {
                    left_job_idx = Some(alignment_jobs.len());
                    alignment_jobs.push(AlignmentJob {
                        seed_idx: chain_idx,
                        query: full_query[0..seed.query_pos as usize].to_vec(),
                        target: rseq[0..tmp].to_vec(),
                        band_width: opt.w,
                        query_offset: 0,
                        direction: Some(crate::alignment::banded_swa::ExtensionDirection::Left),
                        seed_len: seed.len,
                    });
                }
            }

            let seed_query_end = seed.query_pos + seed.len;
            if seed_query_end < query_len {
                let re = ((seed.ref_pos + seed.len as u64) - rmax_0) as usize;
                if re < rseq.len() {
                    right_job_idx = Some(alignment_jobs.len());
                    alignment_jobs.push(AlignmentJob {
                        seed_idx: chain_idx,
                        query: full_query[seed_query_end as usize..].to_vec(),
                        target: rseq[re..].to_vec(),
                        band_width: opt.w,
                        query_offset: seed_query_end,
                        direction: Some(crate::alignment::banded_swa::ExtensionDirection::Right),
                        seed_len: seed.len,
                    });
                }
            }
            seed_job_mappings.push(SeedJobMapping {
                seed_idx: seed_chain_idx,
                left_job_idx,
                right_job_idx,
            });
        }
        chain_to_job_map.push(ChainJobMapping {
            seed_jobs: seed_job_mappings,
        });
    }

    let extended_cigars: Vec<RawAlignment> = if use_batched_simd && alignment_jobs.len() >= 8 {
        execute_adaptive_alignments(&sw_params, &alignment_jobs)
    } else {
        execute_scalar_alignments(&sw_params, &alignment_jobs)
    }
    .into_iter()
    .map(|(s, c, ra, qa)| (s, c, ra, qa))
    .collect();

    ExtensionResult {
        extended_cigars,
        chain_to_job_map,
        filtered_chains: chains,
        sorted_seeds: seeds,
        encoded_query,
        encoded_query_rc,
    }
}

/// Stage 4: Finalization
fn finalize_alignments(
    extension_result: ExtensionResult,
    bwa_idx: &BwaIndex,
    pac_data: &[u8],
    query_name: &str,
    query_seq: &[u8],
    query_qual: &str,
    opt: &MemOpt,
) -> Vec<Alignment> {
    let ExtensionResult {
        extended_cigars,
        chain_to_job_map,
        filtered_chains,
        sorted_seeds: seeds,
        encoded_query,
        encoded_query_rc,
    } = extension_result;

    let alignment_scores: Vec<i32> = extended_cigars
        .iter()
        .map(|(score, _, _, _)| *score)
        .collect();
    let alignment_cigars: Vec<Vec<(u8, i32)>> = extended_cigars
        .iter()
        .map(|(_, cigar, _, _)| cigar.clone())
        .collect();

    let mut alignments = Vec::new();

    for (chain_idx, chain) in filtered_chains.iter().enumerate() {
        if chain.seeds.is_empty() {
            continue;
        }

        let mapping = &chain_to_job_map[chain_idx];
        let full_query = if chain.is_rev {
            &encoded_query_rc
        } else {
            &encoded_query
        };
        let query_len = full_query.len() as i32;

        let mut best_score = 0;
        let mut best_alignment_data = None;

        for seed_job in &mapping.seed_jobs {
            let seed = &seeds[seed_job.seed_idx];
            let mut combined_cigar = Vec::new();
            let mut combined_score = 0;
            let mut alignment_start_pos = seed.ref_pos;
            let (mut query_start_aligned, mut query_end_aligned) = (0, query_len);

            // Track which extensions exist for proper score calculation
            let has_left = seed_job.left_job_idx.is_some();
            let has_right = seed_job.right_job_idx.is_some();

            if let Some(left_idx) = seed_job.left_job_idx {
                let left_cigar = &alignment_cigars[left_idx];
                combined_cigar.extend(left_cigar.iter().cloned());
                combined_score += alignment_scores[left_idx];
                let left_ref_len: i32 = left_cigar
                    .iter()
                    .filter_map(|&(op, len)| {
                        if op == b'M' || op == b'D' {
                            Some(len)
                        } else {
                            None
                        }
                    })
                    .sum();
                alignment_start_pos = alignment_start_pos.saturating_sub(left_ref_len as u64);
            } else if seed.query_pos > 0 {
                combined_cigar.push((b'S', seed.query_pos));
                query_start_aligned = seed.query_pos;
            }

            combined_cigar.push((b'M', seed.len));

            // Score calculation: Each extension score includes h0 = seed.len
            // - Both extensions: left_score + right_score - seed.len (avoid double-counting h0)
            // - One extension only: score already includes seed via h0
            // - No extensions: just seed.len
            if !has_left && !has_right {
                combined_score += seed.len;
            } else if has_left && has_right {
                // Both extensions include h0=seed.len, so subtract one to avoid double-counting
                combined_score -= seed.len;
            }
            // If only one extension exists, h0 is already counted once (correct)

            let seed_end = seed.query_pos + seed.len;
            if let Some(right_idx) = seed_job.right_job_idx {
                combined_cigar.extend(alignment_cigars[right_idx].iter().cloned());
                combined_score += alignment_scores[right_idx];
            } else if seed_end < query_len {
                combined_cigar.push((b'S', query_len - seed_end));
                query_end_aligned = seed_end;
            }

            let cigar_for_candidate = merge_cigar_operations(combined_cigar);
            if combined_score > best_score {
                best_score = combined_score;
                best_alignment_data = Some((
                    cigar_for_candidate,
                    combined_score,
                    alignment_start_pos,
                    query_start_aligned,
                    query_end_aligned,
                ));
            }
        }

        if let Some((mut cigar, score, start_pos, q_start, q_end)) = best_alignment_data {
            let cigar_len: i32 = cigar
                .iter()
                .filter_map(|&(op, len)| {
                    if matches!(op as char, 'M' | 'I' | 'S' | '=' | 'X') {
                        Some(len)
                    } else {
                        None
                    }
                })
                .sum();
            if cigar_len != query_len {
                if let Some(op) = cigar
                    .iter_mut()
                    .rev()
                    .find(|op| matches!(op.0 as char, 'S' | 'M'))
                {
                    op.1 += query_len - cigar_len;
                }
            }

            // Calculate actual query bounds from CIGAR
            // query_start = sum of leading S/H operations
            // query_end = query_len - sum of trailing S/H operations
            let mut actual_query_start = 0i32;
            let mut actual_query_end = query_len;

            // Sum leading clips
            for &(op, len) in cigar.iter() {
                if op == b'S' || op == b'H' {
                    actual_query_start += len;
                } else {
                    break;
                }
            }

            // Sum trailing clips
            for &(op, len) in cigar.iter().rev() {
                if op == b'S' || op == b'H' {
                    actual_query_end -= len;
                } else {
                    break;
                }
            }

            // Update query bounds
            let q_start = actual_query_start;
            let q_end = actual_query_end;

            let (pos_f, is_rev) = bwa_idx.bns.bns_depos(start_pos as i64);
            let rid = bwa_idx.bns.bns_pos2rid(pos_f);

            // DEBUG: Trace coordinate conversion
            log::debug!(
                "{}: COORD_TRACE: start_pos={}, l_pac={}, pos_f={}, is_rev={}, rid={}",
                query_name, start_pos, bwa_idx.bns.packed_sequence_length, pos_f, is_rev, rid
            );

            let (ref_name, ref_id, chr_pos) =
                if rid >= 0 && (rid as usize) < bwa_idx.bns.annotations.len() {
                    let ann = &bwa_idx.bns.annotations[rid as usize];
                    log::debug!(
                        "{}: COORD_TRACE: rid={}, ann.name={}, ann.offset={}, chr_pos={}",
                        query_name, rid, ann.name, ann.offset, pos_f - ann.offset as i64
                    );
                    (
                        ann.name.clone(),
                        rid as usize,
                        (pos_f - ann.offset as i64) as u64,
                    )
                } else {
                    ("unknown_ref".to_string(), 0, 0)
                };

            let md_tag = if !pac_data.is_empty() {
                let ref_len: i32 = cigar
                    .iter()
                    .filter_map(|&(op, len)| {
                        if matches!(op as char, 'M' | 'D') {
                            Some(len)
                        } else {
                            None
                        }
                    })
                    .sum();
                let ref_aligned = bwa_idx.bns.bns_get_seq(
                    pac_data,
                    start_pos as i64,
                    start_pos as i64 + ref_len as i64,
                );
                Alignment::generate_md_tag(
                    &ref_aligned,
                    &full_query[q_start as usize..q_end as usize],
                    &cigar,
                )
            } else {
                cigar
                    .iter()
                    .filter_map(|&(op, len)| if op == b'M' { Some(len) } else { None })
                    .sum::<i32>()
                    .to_string()
            };

            let nm = Alignment::calculate_exact_nm(&md_tag, &cigar);

            alignments.push(Alignment {
                query_name: query_name.to_string(),
                flag: if chain.is_rev { sam_flags::REVERSE } else { 0 },
                ref_name,
                ref_id,
                pos: chr_pos,
                mapq: 60,
                score,
                cigar,
                rnext: "*".to_string(),
                pnext: 0,
                tlen: 0,
                seq: String::from_utf8_lossy(query_seq).to_string(),
                qual: query_qual.to_string(),
                tags: vec![
                    ("AS".to_string(), format!("i:{}", score)),
                    ("NM".to_string(), format!("i:{}", nm)),
                    ("MD".to_string(), format!("Z:{}", md_tag)),
                ],
                query_start: q_start,
                query_end: q_end,
                seed_coverage: chain.weight,
                hash: hash_64((chr_pos << 1) | (if chain.is_rev { 1 } else { 0 })),
                frac_rep: chain.frac_rep,
            });
        }
    }

    if !alignments.is_empty() {
        // Filter alignments by minimum score threshold (matches C++ bwamem.cpp:1538)
        let before_filter = alignments.len();
        alignments.retain(|a| a.score >= opt.t);
        if before_filter != alignments.len() {
            log::debug!(
                "{}: Filtered {} alignments below score threshold {} (before: {}, after: {})",
                query_name,
                before_filter - alignments.len(),
                opt.t,
                before_filter,
                alignments.len()
            );
        }

        alignments.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.hash.cmp(&b.hash)));
        mark_secondary_alignments(&mut alignments, opt);

        let xa_tags = generate_xa_tags(&alignments, opt);
        let sa_tags = generate_sa_tags(&alignments);

        for aln in alignments.iter_mut() {
            if aln.flag & sam_flags::SECONDARY == 0 {
                if let Some(sa_tag) = sa_tags.get(&aln.query_name) {
                    aln.tags.push(("SA".to_string(), sa_tag.clone()));
                } else if let Some(xa_tag) = xa_tags.get(&aln.query_name) {
                    aln.tags.push(("XA".to_string(), xa_tag.clone()));
                }
            }
        }
    } else {
        alignments.push(Alignment {
            query_name: query_name.to_string(),
            flag: sam_flags::UNMAPPED,
            ref_name: "*".to_string(),
            ref_id: 0,
            pos: 0,
            mapq: 0,
            score: 0,
            cigar: Vec::new(),
            rnext: "*".to_string(),
            pnext: 0,
            tlen: 0,
            seq: String::from_utf8_lossy(query_seq).to_string(),
            qual: query_qual.to_string(),
            tags: vec![
                ("AS".to_string(), "i:0".to_string()),
                ("NM".to_string(), "i:0".to_string()),
            ],
            query_start: 0,
            query_end: 0,
            seed_coverage: 0,
            hash: 0,
            frac_rep: 0.0,
        });
    }

    alignments
}

#[cfg(test)]
mod tests {
    use crate::alignment::pipeline::sam_flags;
    use crate::index::BwaIndex;

    #[test]
    fn test_generate_seeds_basic() {
        use crate::mem_opt::MemOpt;
        use std::path::Path;

        let prefix = Path::new("test_data/test_ref.fa");
        if !prefix.exists() {
            eprintln!("Skipping test_generate_seeds_basic - test data not found");
            return;
        }

        let bwa_idx = match BwaIndex::bwa_idx_load(&prefix) {
            Ok(idx) => idx,
            Err(_) => {
                eprintln!("Skipping test_generate_seeds_basic - could not load index");
                return;
            }
        };

        let mut opt = MemOpt::default();
        opt.min_seed_len = 10; // Ensure our seed is long enough

        let query_name = "test_query";
        let query_seq = b"ACGTACGTACGT"; // 12bp
        let query_qual = "IIIIIIIIIIII";

        // Test doesn't need real MD tags, use empty pac_data
        let pac_data: &[u8] = &[];
        let alignments =
            super::generate_seeds(&bwa_idx, pac_data, query_name, query_seq, query_qual, &opt);

        assert!(
            !alignments.is_empty(),
            "Expected at least one alignment for a matching query"
        );

        let primary_alignment = alignments
            .iter()
            .find(|a| a.flag & sam_flags::SECONDARY == 0);
        assert!(primary_alignment.is_some(), "Expected a primary alignment");

        let pa = primary_alignment.unwrap();
        assert_eq!(pa.ref_name, "test_sequence");
        assert!(
            pa.score > 0,
            "Expected a positive score for a good match, got {}",
            pa.score
        );
        assert!(pa.pos < 60, "Position should be within reference length");
        assert_eq!(pa.cigar_string(), "12M", "Expected a perfect match CIGAR");
    }
}
