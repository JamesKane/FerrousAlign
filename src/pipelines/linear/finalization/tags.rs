//! XA and SA tag generation.
//!
//! Implements C++ mem_gen_alt (bwamem_extra.cpp:130-183).

use super::alignment::Alignment;
use super::sam_flags;
use crate::pipelines::linear::mem_opt::MemOpt;
use std::collections::HashMap;

/// Attach XA and SA tags to alignments
pub fn attach_xa_sa_tags(alignments: &mut [Alignment], opt: &MemOpt) {
    let xa_tags = generate_xa_tags(alignments, opt);
    let sa_tags = generate_sa_tags(alignments);

    for aln in alignments.iter_mut() {
        if aln.flag & sam_flags::SECONDARY == 0 {
            if let Some(sa_tag) = sa_tags.get(&aln.query_name) {
                aln.tags.push(("SA".to_string(), sa_tag.clone()));
            } else if let Some(xa_tag) = xa_tags.get(&aln.query_name) {
                aln.tags.push(("XA".to_string(), xa_tag.clone()));
            }
        }
    }
}

/// Generate XA tags for alternative alignments
pub fn generate_xa_tags(alignments: &[Alignment], opt: &MemOpt) -> HashMap<String, String> {
    let mut xa_tags: HashMap<String, String> = HashMap::new();

    if alignments.is_empty() {
        return xa_tags;
    }

    let mut by_read: HashMap<String, Vec<&Alignment>> = HashMap::new();
    for aln in alignments {
        by_read.entry(aln.query_name.clone()).or_default().push(aln);
    }

    for (read_name, read_alns) in by_read.iter() {
        let primary = read_alns.iter().find(|a| a.flag & sam_flags::SECONDARY == 0);

        if primary.is_none() {
            continue;
        }

        let primary_score = primary.unwrap().score;
        let xa_threshold = (primary_score as f32 * opt.xa_drop_ratio) as i32;

        let mut secondaries: Vec<&Alignment> = read_alns
            .iter()
            .filter(|a| (a.flag & sam_flags::SECONDARY != 0) && (a.score >= xa_threshold))
            .cloned()
            .collect();

        if secondaries.is_empty() {
            continue;
        }

        secondaries.sort_by(|a, b| b.score.cmp(&a.score));

        let max_hits = opt.max_xa_hits as usize;
        if secondaries.len() > max_hits {
            secondaries.truncate(max_hits);
        }

        let xa_entries: Vec<String> = secondaries.iter().map(|aln| aln.to_xa_entry()).collect();

        if !xa_entries.is_empty() {
            let xa_tag = format!("Z:{};", xa_entries.join(";"));
            xa_tags.insert(read_name.clone(), xa_tag);
        }
    }

    xa_tags
}

/// Generate SA tags for chimeric/split-read alignments
pub fn generate_sa_tags(alignments: &[Alignment]) -> HashMap<String, String> {
    let mut all_sa_tags: HashMap<String, String> = HashMap::new();

    if alignments.is_empty() {
        return all_sa_tags;
    }

    let mut alignments_by_read: HashMap<String, Vec<&Alignment>> = HashMap::new();
    for aln in alignments {
        alignments_by_read.entry(aln.query_name.clone()).or_default().push(aln);
    }

    for (read_name, read_alns) in alignments_by_read.iter() {
        let non_secondary_alns: Vec<&Alignment> = read_alns
            .iter()
            .filter(|a| (a.flag & sam_flags::SECONDARY) == 0)
            .cloned()
            .collect();

        if non_secondary_alns.len() < 2 {
            continue;
        }

        let mut sorted_alns = non_secondary_alns.clone();
        sorted_alns.sort_by(|a, b| {
            a.ref_name.cmp(&b.ref_name)
                .then_with(|| a.pos.cmp(&b.pos))
                .then_with(|| (a.flag & sam_flags::REVERSE).cmp(&(b.flag & sam_flags::REVERSE)))
        });

        let mut sa_parts: Vec<String> = Vec::new();
        for aln in sorted_alns.iter() {
            let nm_value = aln
                .tags
                .iter()
                .find(|(tag_name, _)| tag_name == "NM")
                .map(|(_, val)| val.clone())
                .unwrap_or_else(|| "i:0".to_string())
                .strip_prefix("i:")
                .unwrap_or("0")
                .parse::<i32>()
                .unwrap_or(0);

            let strand = if (aln.flag & sam_flags::REVERSE) != 0 { '-' } else { '+' };
            sa_parts.push(format!(
                "{},{},{},{},{},{}",
                aln.ref_name,
                aln.pos + 1,
                strand,
                aln.cigar_string(),
                aln.mapq,
                nm_value
            ));
        }

        if !sa_parts.is_empty() {
            let sa_tag_value = format!("Z:{};", sa_parts.join(";"));
            all_sa_tags.insert(read_name.clone(), sa_tag_value);
        }
    }

    all_sa_tags
}
