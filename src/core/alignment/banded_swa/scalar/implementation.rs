use crate::alignment::banded_swa::BandedPairWiseSW;
use crate::alignment::banded_swa::types::{TB_DEL, TB_INS, TB_MATCH};
use crate::alignment::banded_swa::{EhT, OutScore};

pub fn scalar_banded_swa(
    sw_params: &BandedPairWiseSW,
    qlen: i32,
    query: &[u8],
    tlen: i32,
    target: &[u8],
    w: i32,
    h0: i32,
) -> (OutScore, Vec<(u8, i32)>, Vec<u8>, Vec<u8>) {
    // Note: This is the direct scalar implementation.
    // For runtime SIMD dispatch, use banded_swa() instead.

    // Handle degenerate cases: empty sequences
    if tlen == 0 || qlen == 0 {
        // For empty sequences, return zero score and empty CIGAR
        // This is a biologically invalid case, but we handle it gracefully
        let out_score = OutScore {
            score: 0,
            target_end_pos: 0,
            query_end_pos: 0,
            gtarget_end_pos: 0,
            global_score: 0,
            max_offset: 0,
        };
        return (out_score, Vec::new(), Vec::new(), Vec::new());
    }

    // CRITICAL: Validate that qlen matches query.len() and tlen matches target.len()
    // This prevents index out of bounds errors.
    // Clamp to actual lengths to prevent panic.
    let qlen = (qlen as usize).min(query.len()) as i32;
    let tlen = (tlen as usize).min(target.len()) as i32;

    if qlen == 0 || tlen == 0 {
        let out_score = OutScore {
            score: 0,
            target_end_pos: 0,
            query_end_pos: 0,
            gtarget_end_pos: 0,
            global_score: 0,
            max_offset: 0,
        };
        return (out_score, Vec::new(), Vec::new(), Vec::new());
    }

    let oe_del = sw_params.o_del() + sw_params.e_del();
    let oe_ins = sw_params.o_ins() + sw_params.e_ins();

    // Allocate memory for query profile and eh_t array
    let mut qp = vec![0i8; (qlen * sw_params.alphabet_size()) as usize];
    let mut eh = vec![EhT::default(); (qlen + 1) as usize];
    // Traceback matrix: (tlen+1) x (qlen+1)
    let mut tb = vec![vec![0u8; (qlen + 1) as usize]; (tlen + 1) as usize]; // Initialize with 0 (MATCH)

    // Generate the query profile
    for k in 0..sw_params.alphabet_size() {
        let p_row_start = (k * qlen) as usize; // Corrected: k * qlen
        for j in 0..qlen as usize {
            // CRITICAL: Clamp query[j] to valid range [0, 4] to prevent out-of-bounds access to sw_params.mat
            // Query values should be 0=A, 1=C, 2=G, 3=T, 4=N, but clamp to be safe
            let base_code = (query[j] as i32).min(4);
            qp[p_row_start + j] =
                sw_params.scoring_matrix()[(k * sw_params.alphabet_size() + base_code) as usize];
        }
    }

    // Fill the first row (initialization for DP)
    eh[0].h = h0;
    eh[1].h = if h0 > oe_ins { h0 - oe_ins } else { 0 };
    for j in 2..=(qlen as usize) {
        if eh[j - 1].h > sw_params.e_ins() {
            eh[j].h = eh[j - 1].h - sw_params.e_ins();
        } else {
            eh[j].h = 0;
        }
    }

    // Initialize traceback matrix edges
    // Top row (i=0): insertions (gaps in target)
    for j in 1..=(qlen as usize) {
        tb[0][j] = TB_INS;
    }
    // Left column (j=0): deletions (gaps in query)
    for i in 1..=(tlen as usize) {
        tb[i][0] = TB_DEL;
    }

    // DP loop
    let mut max_score = h0;
    let mut max_i = -1;
    let mut max_j = -1;
    let mut _max_ie = -1;
    let mut current_gscore = -1;
    let mut current_max_off = 0;

    let mut beg = 0;
    let mut end = qlen;

    for i in 0..tlen {
        let mut f = 0;
        let mut _h1 = 0;
        let mut m_val = 0;
        let mut mj = -1;

        // CRITICAL: Clamp target base code to valid range [0, 4]
        let target_base = (target[i as usize] as i32).min(4);
        let q_row_start = (target_base * qlen) as usize;
        let q_slice = &qp[q_row_start..(q_row_start + qlen as usize)];

        // Apply the band and the constraint
        let mut current_beg = beg;
        let mut current_end = end;

        if current_beg < i - w {
            current_beg = i - w;
        }
        if current_end > i + w + 1 {
            current_end = i + w + 1;
        }
        if current_end > qlen {
            current_end = qlen;
        }

        // Compute the first column
        if current_beg == 0 {
            _h1 = h0 - (sw_params.o_del() + sw_params.e_del() * (i + 1));
            if _h1 < 0 {
                _h1 = 0;
            }
        } else {
            _h1 = 0;
        }

        for j in current_beg..current_end {
            let p_idx = j as usize;
            let mut m_score = eh[p_idx].h;
            let mut e = eh[p_idx].e;

            eh[p_idx].h = _h1; // Store H from previous row, current column

            let mut _current_tb = TB_MATCH; // Default to match

            // Calculate M (match/mismatch) score
            m_score += q_slice[j as usize] as i32;
            if m_score < 0 {
                m_score = 0;
            } // Local alignment: clamp to 0

            // Determine max of M, E, F using simple comparisons
            // (Replaces expensive sort_by_key call that was O(n log n) per cell)
            let (h, tb_code) = if m_score >= e && m_score >= f {
                (m_score, TB_MATCH)
            } else if e >= f {
                (e, TB_DEL)
            } else {
                (f, TB_INS)
            };
            _h1 = h;
            _current_tb = tb_code;

            // Update traceback matrix
            tb[i as usize + 1][j as usize + 1] = _current_tb;

            if m_val < h {
                m_val = h;
                mj = j;
            }

            // Update E (deletion)
            let mut t_del = m_score - oe_del;
            t_del = if t_del > 0 { t_del } else { 0 };
            e -= sw_params.e_del();
            e = if e > t_del { e } else { t_del };
            eh[p_idx].e = e;

            // Update F (insertion)
            let mut t_ins = m_score - oe_ins;
            t_ins = if t_ins > 0 { t_ins } else { 0 };
            f -= sw_params.e_ins();
            f = if f > t_ins { f } else { t_ins };
        }
        eh[current_end as usize].h = _h1;
        eh[current_end as usize].e = 0;

        if current_end == qlen && current_gscore < _h1 {
            current_gscore = _h1;
            _max_ie = i;
        }

        if m_val == 0 {
            break;
        }

        if m_val > max_score {
            max_score = m_val;
            max_i = i;
            max_j = mj;
            current_max_off = current_max_off.max((mj - i).abs());
        } else if sw_params.zdrop() > 0 {
            let diff_i = i - max_i;
            let diff_j = mj - max_j;
            if diff_i > diff_j {
                if max_score - m_val - (diff_i - diff_j) * sw_params.e_del() > sw_params.zdrop() {
                    break;
                }
            } else if max_score - m_val - (diff_j - diff_i) * sw_params.e_ins() > sw_params.zdrop()
            {
                break;
            }
        }

        // Update beg and end for the next round
        let mut new_beg = current_beg;
        while new_beg < current_end && eh[new_beg as usize].h == 0 && eh[new_beg as usize].e == 0 {
            new_beg += 1;
        }
        beg = new_beg;

        let mut new_end = current_end;
        while new_end >= beg && eh[new_end as usize].h == 0 && eh[new_end as usize].e == 0 {
            new_end -= 1;
        }
        end = (new_end + 2).min(qlen);
    }

    // Backtrack to generate CIGAR
    // Use local alignment mode: start from max score position (max_i, max_j)
    // Soft clipping will be added as post-processing for unaligned regions
    let mut cigar = Vec::new();
    let use_global_end = false; // Local alignment: start from max score position
    let mut curr_i = if use_global_end {
        _max_ie + 1
    } else {
        max_i + 1
    };
    let mut curr_j = if use_global_end { qlen } else { max_j + 1 };

    const MAX_TRACEBACK_ITERATIONS: i32 = 10000; // Safety limit to prevent infinite loops
    let mut iteration_count = 0;

    // Vectors to capture aligned bases for MD tag generation
    // These will be reversed at the end (like CIGAR)
    let mut ref_aligned = Vec::new();
    let mut query_aligned = Vec::new();

    while curr_i > 0 || curr_j > 0 {
        // Safety check: prevent infinite loops
        if iteration_count >= MAX_TRACEBACK_ITERATIONS {
            log::error!(
                "Traceback exceeded MAX_TRACEBACK_ITERATIONS ({MAX_TRACEBACK_ITERATIONS}) - possible infinite loop!"
            );
            log::error!("  curr_i={curr_i}, curr_j={curr_j}, qlen={qlen}, tlen={tlen}");
            break;
        }
        iteration_count += 1;

        let prev_i = curr_i;
        let prev_j = curr_j;

        let tb_code = tb[curr_i as usize][curr_j as usize];
        match tb_code {
            TB_MATCH => {
                // Count consecutive alignment positions (both matches and mismatches use 'M')
                // This matches bwa-mem2 behavior which uses MIDSH operations (no X or =)
                // Also capture the aligned bases for MD tag generation
                let mut alignment_count = 0;

                while curr_i > 0 && curr_j > 0 && tb[curr_i as usize][curr_j as usize] == TB_MATCH {
                    alignment_count += 1;
                    // Capture aligned bases (1-indexed in arrays, so subtract 1)
                    // Note: bases are added in reverse order, will be reversed later
                    ref_aligned.push(target[(curr_i - 1) as usize]);
                    query_aligned.push(query[(curr_j - 1) as usize]);
                    curr_i -= 1;
                    curr_j -= 1;
                }

                // Emit alignment positions as 'M' (both matches and mismatches)
                if alignment_count > 0 {
                    cigar.push((b'M', alignment_count));
                }
            }
            TB_DEL => {
                // Deletion: gap in query, target base consumed
                let mut count = 0;
                while curr_i > 0 && tb[curr_i as usize][curr_j as usize] == TB_DEL {
                    count += 1;
                    // Capture deleted reference base (target consumes, query doesn't)
                    ref_aligned.push(target[(curr_i - 1) as usize]);
                    // No query base consumed for deletion
                    curr_i -= 1;
                }
                if count > 0 {
                    cigar.push((b'D', count));
                }
            }
            TB_INS => {
                // Insertion: gap in target, query base consumed
                let mut count = 0;
                while curr_j > 0 && tb[curr_i as usize][curr_j as usize] == TB_INS {
                    count += 1;
                    // Capture inserted query base (query consumes, target doesn't)
                    query_aligned.push(query[(curr_j - 1) as usize]);
                    // No reference base consumed for insertion
                    curr_j -= 1;
                }
                if count > 0 {
                    cigar.push((b'I', count));
                }
            }
            _ => break, // Should not happen
        }

        // Safety check: ensure we made progress
        if curr_i == prev_i && curr_j == prev_j {
            log::warn!(
                "Traceback made no progress at curr_i={curr_i}, curr_j={curr_j}, tb_code={tb_code}"
            );
            break; // Exit to prevent infinite loop
        }
    }

    cigar.reverse(); // CIGAR is usually represented from start to end

    // Reverse aligned sequences to match CIGAR direction (start to end)
    ref_aligned.reverse();
    query_aligned.reverse();

    // --- Apply Clipping Penalty Logic (C++ bwa-mem2 bwamem.cpp:2498, 2570, 2638, 2715, 2784, 2853) ---
    // Decision: Should we soft-clip or extend to query boundaries?
    // C++ logic: if (gscore <= 0 || gscore <= score - pen_clip), soft-clip; else extend
    //
    // Note: current_gscore is the score when reaching qlen (query end)
    // This logic applies primarily to the 3' end extension decision
    // For 5' end, we would need separate left extension (not implemented yet)
    //
    // For now, apply clipping penalty decision to the entire alignment:
    // - If global extension is better (gscore > score - pen_clip3), don't soft-clip 3' end
    // - If global extension is poor (gscore <= score - pen_clip3), soft-clip as before

    let query_start = curr_j; // Where alignment started in query (local)
    let query_end = max_j + 1; // Where alignment ended in query (local, exclusive)

    let mut final_cigar = Vec::new();

    // === 5' END CLIPPING DECISION ===
    // Check if we should soft-clip the 5' end or assume global alignment to position 0
    // C++ bwa-mem2 logic (bwamem.cpp:2498): if (gscore <= 0 || gscore <= score - pen_clip5)
    // Currently we only have global score for 3' end (reaching qlen), not for 5' end
    // So for 5' end, we fall back to always soft-clipping (matching current behavior)
    // TODO: Implement separate left extension to get 5' global score
    if query_start > 0 {
        final_cigar.push((b'S', query_start));
    }

    // Add the core alignment operations
    final_cigar.extend_from_slice(&cigar);

    // === 3' END CLIPPING DECISION ===
    // Check if we should soft-clip the 3' end or extend to qlen
    // C++ logic (bwamem.cpp:2715, 2784, 2853): if (gscore <= 0 || gscore <= score - pen_clip3), soft-clip
    let should_clip_3prime = if query_end < qlen {
        // Only apply clipping penalty logic if alignment doesn't already reach query end
        current_gscore <= 0 || current_gscore <= (max_score - sw_params.pen_clip3())
    } else {
        false // Already at query end, no clipping needed
    };

    if should_clip_3prime && query_end < qlen {
        // Use local alignment: soft-clip the 3' end
        final_cigar.push((b'S', qlen - query_end));
        log::debug!(
            "3' clipping: gscore={} <= score={} - pen_clip3={} = {}, soft-clipping {} bases",
            current_gscore,
            max_score,
            sw_params.pen_clip3(),
            max_score - sw_params.pen_clip3(),
            qlen - query_end
        );
    } else if query_end < qlen {
        // ==================================================================
        // GLOBAL EXTENSION: Use M operations instead of soft-clipping
        // ==================================================================
        // When global extension is preferred (gscore > score - pen_clip3),
        // BWA-MEM2 extends to the query boundary (qb=0 or qe=qlen).
        //
        // BWA-MEM2 bwamem.cpp:2498-2504:
        //   if (gscore <= 0 || gscore <= score - pen_clip5) {
        //       a->qb -= qle; a->rb -= tle;  // LOCAL: soft-clip
        //   } else {
        //       a->qb = 0; a->rb -= gtle;    // GLOBAL: extend to boundary
        //   }
        //
        // We achieve this by using M operations for the remaining bases.
        // The M operation covers both matches and mismatches in SAM format.
        // ==================================================================
        let remaining_bases = qlen - query_end;
        log::debug!(
            "3' global extension: gscore={} > score={} - pen_clip3={} = {}, extending {} bases as M",
            current_gscore,
            max_score,
            sw_params.pen_clip3(),
            max_score - sw_params.pen_clip3(),
            remaining_bases
        );
        // Use M operations for global extension (covers matches/mismatches)
        final_cigar.push((b'M', remaining_bases));
    }

    // Debug logging for problematic CIGARs (all insertions)
    if cigar.len() == 1 && cigar[0].0 == b'I' {
        log::warn!(
            "PATHOLOGICAL CIGAR: Single insertion of {} bases! qlen={}, tlen={}, max_i={}, max_j={}, score={}",
            cigar[0].1,
            qlen,
            tlen,
            max_i,
            max_j,
            max_score
        );
        log::warn!("  Query preview: {:?}", &query[..10.min(query.len())]);
        log::warn!("  Target preview: {:?}", &target[..10.min(target.len())]);
        log::warn!("  query_start={query_start}, query_end={query_end}, qlen={qlen}");
    }

    let out_score = OutScore {
        score: max_score,
        target_end_pos: max_i + 1,
        query_end_pos: max_j + 1,
        gtarget_end_pos: _max_ie + 1, // Global target end position (where query reached qlen)
        global_score: current_gscore,
        max_offset: current_max_off,
    };

    (out_score, final_cigar, ref_aligned, query_aligned)
}
