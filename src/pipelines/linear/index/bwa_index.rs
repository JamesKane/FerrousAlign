use std::fs::File;
use std::io;
use std::io::Read;
use std::path::Path;

pub const BWA_IDX_ALL: i32 = 0x7;

pub fn bwa_index(fasta_file: &Path, prefix: &Path) -> io::Result<()> {
    log::info!(
        "Building index for {} with prefix {}",
        fasta_file.display(),
        prefix.display()
    );

    // Step 1: Parse FASTA and generate BntSeq and PAC array
    let bns = super::bntseq::BntSeq::bns_fasta2bntseq(
        fasta_file, prefix, false, // for_only is false
    )?;
    bns.bns_dump(prefix)?;

    // Step 2: Construct BWT from PAC array
    // Read the PAC array
    let pac_file_path = prefix.with_extension("pac");
    let mut pac_file = File::open(&pac_file_path)?;
    let mut pac_data = Vec::new();
    pac_file.read_to_end(&mut pac_data)?;

    // MIMIC C++ IMPLEMENTATION EXACTLY:
    // C++ calls: saisxx(reference_seq.c_str(), suffix_array + 1, pac_len)
    // This builds SA for ONLY the bases (without sentinel), then manually sets SA[0] = pac_len

    // CRITICAL FIX: Build bidirectional BWT (forward + reverse complement)
    // C++ bwa-mem2 builds BWT on BIDIRECTIONAL sequence to search both strands
    // Convert PAC to bases (0=A, 1=C, 2=G, 3=T), but use shifted values for SA construction
    // bio crate needs a lexicographically smallest sentinel at the end
    // So shift bases to 1,2,3,4 and use 0 as sentinel
    let mut text_for_sais: Vec<u8> =
        Vec::with_capacity((2 * bns.packed_sequence_length + 1) as usize);

    // Add forward strand
    for i in 0..bns.packed_sequence_length {
        let byte_idx = (i / 4) as usize;
        // CRITICAL: Match C++ bwa-mem2 bit order (same as extraction in bntseq.rs)
        let bit_offset = ((!(i % 4)) & 3) * 2;
        let base = (pac_data[byte_idx] >> bit_offset) & 0x03;
        text_for_sais.push(base + 1); // Shift: A=1, C=2, G=3, T=4
    }

    // Add reverse complement strand
    for i in (0..bns.packed_sequence_length).rev() {
        let byte_idx = (i / 4) as usize;
        let bit_offset = ((!(i % 4)) & 3) * 2;
        let base = (pac_data[byte_idx] >> bit_offset) & 0x03;
        let complement = (3 - base) + 1; // Complement: A<->T (0<->3), C<->G (1<->2), then shift
        text_for_sais.push(complement);
    }

    text_for_sais.push(0); // Sentinel: lexicographically smallest

    eprintln!("\n=== RUST INDEX BUILD TRACE ===");
    eprintln!("[RUST] l_pac = {}", bns.packed_sequence_length);
    eprintln!(
        "[RUST] Building SA for {} bases (with sentinel)",
        text_for_sais.len()
    );
    eprintln!(
        "[RUST] text_for_sais length: forward={}, RC={}, total_with_sentinel={}",
        bns.packed_sequence_length,
        bns.packed_sequence_length,
        text_for_sais.len()
    );
    eprintln!(
        "[RUST] text_for_sais[0..20] = {:?}",
        &text_for_sais[..text_for_sais.len().min(20)]
    );
    eprintln!(
        "[RUST] text_for_sais[{}..{}] (RC start) = {:?}",
        bns.packed_sequence_length as usize,
        (bns.packed_sequence_length as usize + 10).min(text_for_sais.len()),
        &text_for_sais[bns.packed_sequence_length as usize
            ..(bns.packed_sequence_length as usize + 10).min(text_for_sais.len())]
    );
    eprintln!(
        "[RUST] text_for_sais[{}] (sentinel) = {:?}",
        text_for_sais.len() - 1,
        text_for_sais.last()
    );

    // Use bio crate (well-tested bioinformatics library) for suffix array construction
    use bio::data_structures::suffix_array::suffix_array;

    // Build SA using bio crate (includes sentinel)
    let sa_with_sentinel = suffix_array(&text_for_sais);
    let sa_full: Vec<i32> = sa_with_sentinel.iter().map(|&x| x as i32).collect();

    eprintln!(
        "[RUST] SA construction complete, length = {}",
        sa_full.len()
    );
    eprintln!("[RUST] SA[0..10] = {:?}", &sa_full[..sa_full.len().min(10)]);
    eprintln!("[RUST] Sampled SA positions (every 8th):");
    for i in (0..80).step_by(8) {
        if i < sa_full.len() {
            let sa_val = sa_full[i];
            let region = if sa_val == text_for_sais.len() as i32 - 1 {
                "SENTINEL"
            } else if sa_val >= bns.packed_sequence_length as i32 {
                "RC"
            } else {
                "FWD"
            };
            eprintln!("  [RUST] SA[{i}] = {sa_val} ({region})");
        }
    }

    // Build BWT from SA (mimics build_fm_index in C++)
    let seq_len_with_sentinel = text_for_sais.len();
    let mut bwt_output: Vec<i32> = vec![0; seq_len_with_sentinel];

    for i in 0..seq_len_with_sentinel {
        let sa_val = sa_full[i];
        if sa_val == 0 {
            // Suffix starts at position 0, BWT char wraps to last position (sentinel)
            // C++ sets bwt[i] = 4 for sentinel
            bwt_output[i] = 4;
        } else {
            // BWT[i] = text[SA[i] - 1]
            // text_for_sais has values 1,2,3,4 (shifted) or 0 (sentinel)
            let bwt_char = text_for_sais[(sa_val - 1) as usize];
            // Convert back to 0,1,2,3 or 4 for sentinel
            bwt_output[i] = if bwt_char == 0 {
                4
            } else {
                (bwt_char - 1) as i32
            };
        }
    }

    eprintln!(
        "[RUST] BWT construction complete, length = {}",
        bwt_output.len()
    );
    eprintln!(
        "[RUST] BWT[0..20] = {:?}",
        &bwt_output[..bwt_output.len().min(20)]
    );
    eprintln!("[RUST] BWT character counts:");
    let mut bwt_char_counts = [0; 5];
    for &ch in bwt_output.iter() {
        if (0..5).contains(&ch) {
            bwt_char_counts[ch as usize] += 1;
        }
    }
    eprintln!(
        "  [RUST] A(0)={}, C(1)={}, G(2)={}, T(3)={}, Sentinel(4)={}",
        bwt_char_counts[0],
        bwt_char_counts[1],
        bwt_char_counts[2],
        bwt_char_counts[3],
        bwt_char_counts[4]
    );

    // Pack BWT into 2-bit format
    let mut packed_bwt_data = Vec::with_capacity(bwt_output.len().div_ceil(4));
    let mut current_byte = 0u8;
    for (idx, &base) in bwt_output.iter().enumerate() {
        // base is 0,1,2,3 for A,C,G,T or 4 for sentinel
        // For packing, sentinel (4) is stored as 0 (will be handled specially later)
        let stored_base = if base == 4 { 0 } else { base as u8 };
        let bit_offset = (idx % 4) * 2;
        current_byte |= (stored_base & 0x03) << bit_offset;
        if (idx + 1) % 4 == 0 || idx == bwt_output.len() - 1 {
            packed_bwt_data.push(current_byte);
            current_byte = 0;
        }
    }

    // Calculate l2 array (count of each base)
    let mut l2_counts = [0u64; 4]; // For A, C, G, T
    for &base_code in bwt_output.iter() {
        if (0..4).contains(&base_code) {
            l2_counts[base_code as usize] += 1;
        }
        // base_code == 4 is sentinel, not counted
    }

    let mut l2: [u64; 5] = [0; 5];
    l2[0] = 0; // Count of bases smaller than 'A' is 0
    for i in 0..4 {
        l2[i + 1] = l2[i] + l2_counts[i];
    }

    eprintln!("[RUST] L2 array (cumulative base counts):");
    eprintln!("  [RUST] l2[0]={} (before A)", l2[0]);
    eprintln!("  [RUST] l2[1]={} (before C, #A={})", l2[1], l2_counts[0]);
    eprintln!("  [RUST] l2[2]={} (before G, #C={})", l2[2], l2_counts[1]);
    eprintln!("  [RUST] l2[3]={} (before T, #G={})", l2[3], l2_counts[2]);
    eprintln!("  [RUST] l2[4]={} (total, #T={})", l2[4], l2_counts[3]);

    // Find sentinel position in BWT (where SA[i] == 0, meaning BWT[i] wraps to sentinel)
    let mut sentinel_index = 0i64;
    for i in 0..seq_len_with_sentinel {
        if sa_full[i] == 0 {
            sentinel_index = i as i64;
            eprintln!("[RUST] Found sentinel at BWT position {sentinel_index} (SA[{i}]=0)");
            break;
        }
    }
    eprintln!("=== END RUST INDEX BUILD TRACE ===\n");

    // Create Bwt struct
    let mut bwt = super::bwt::Bwt::new_from_bwt_data(
        packed_bwt_data,
        l2,
        seq_len_with_sentinel as u64,
        sentinel_index as u64,
    );

    // Step 3: Construct Suffix Array (sampled) from sa_full
    pub const SA_INTV: u32 = 8; // Default SA interval
    bwt.bwt_cal_sa(SA_INTV as i32, &sa_full);

    // Calculate cp_occ (pass sentinel_index to skip it in bitmask construction)
    let cp_occ = bwt.calculate_cp_occ(sentinel_index as u64);

    // Create BwaIndex and dump it
    let bwa_index_to_dump = super::index::BwaIndex {
        bwt,
        bns,
        cp_occ,
        sentinel_index,
        min_seed_len: 19,
        mmap_bwt: None, // Not needed for building
        mmap_pac: None,
    };
    bwa_index_to_dump.dump(prefix)?;

    log::info!("Index building complete.");
    Ok(())
}
