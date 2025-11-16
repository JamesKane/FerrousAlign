use std::io;
use std::path::Path;
use std::fs::File;
use std::io::Read;

pub const BWA_IDX_ALL: i32 = 0x7;

pub fn bwa_index(fasta_file: &Path, prefix: &Path) -> io::Result<()> {
    log::info!("Building index for {} with prefix {}", fasta_file.display(), prefix.display());

    // Step 1: Parse FASTA and generate BntSeq and PAC array
    let bns = crate::bntseq::BntSeq::bns_fasta2bntseq(
        File::open(fasta_file)?,
        prefix,
        false, // for_only is false
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

    // Convert PAC to bases (0=A, 1=C, 2=G, 3=T), but use shifted values for SA construction
    // bio crate needs a lexicographically smallest sentinel at the end
    // So shift bases to 1,2,3,4 and use 0 as sentinel
    let mut text_for_sais: Vec<u8> = Vec::with_capacity((bns.l_pac + 1) as usize);
    for i in 0..bns.l_pac {
        let byte_idx = (i / 4) as usize;
        let bit_offset = (i % 4) * 2;
        let base = (pac_data[byte_idx] >> bit_offset) & 0x03;
        text_for_sais.push((base + 1) as u8);  // Shift: A=1, C=2, G=3, T=4
    }
    text_for_sais.push(0);  // Sentinel: lexicographically smallest

    // eprintln!("DEBUG: Building SA for {} bases (with sentinel)", text_for_sais.len());
    // eprintln!("DEBUG: text_for_sais[:17] = {:?}", &text_for_sais[..text_for_sais.len().min(17)]);

    // Use bio crate (well-tested bioinformatics library) for suffix array construction
    use bio::data_structures::suffix_array::suffix_array;

    // Build SA using bio crate (includes sentinel)
    let sa_with_sentinel = suffix_array(&text_for_sais);
    let sa_full: Vec<i32> = sa_with_sentinel.iter().map(|&x| x as i32).collect();

    // eprintln!("DEBUG: Full SA (first 20): {:?}", &sa_full[..sa_full.len().min(20)]);

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
            bwt_output[i] = if bwt_char == 0 { 4 } else { (bwt_char - 1) as i32 };
        }
    }

    // eprintln!("DEBUG: BWT (first 20): {:?}", &bwt_output[..bwt_output.len().min(20)]);

    // Pack BWT into 2-bit format
    let mut packed_bwt_data = Vec::with_capacity((bwt_output.len() + 3) / 4);
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
        if base_code >= 0 && base_code < 4 {
            l2_counts[base_code as usize] += 1;
        }
        // base_code == 4 is sentinel, not counted
    }

    let mut l2: [u64; 5] = [0; 5];
    l2[0] = 0; // Count of bases smaller than 'A' is 0
    for i in 0..4 {
        l2[i + 1] = l2[i] + l2_counts[i];
    }

    // eprintln!("DEBUG L2: Base counts in BWT: A={}, C={}, G={}, T={}",
    //           l2_counts[0], l2_counts[1], l2_counts[2], l2_counts[3]);
    // eprintln!("DEBUG L2: l2 array (cumulative): {:?}", l2);
    // eprintln!("DEBUG L2: l2[0]={} (0), l2[1]={} (#A), l2[2]={} (#A+#C), l2[3]={} (#A+#C+#G), l2[4]={} (total)",
    //           l2[0], l2[1], l2[2], l2[3], l2[4]);

    // Find sentinel position in BWT (where SA[i] == 0, meaning BWT[i] wraps to sentinel)
    let mut sentinel_index = 0i64;
    for i in 0..seq_len_with_sentinel {
        if sa_full[i] == 0 {
            sentinel_index = i as i64;
            // eprintln!("DEBUG: Found sentinel at BWT position {} (SA[{}]=0)", sentinel_index, i);
            break;
        }
    }

    // Create Bwt struct
    let mut bwt = crate::bwt::Bwt::new_from_bwt_data(
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
    let bwa_index_to_dump = crate::mem::BwaIndex {
        bwt,
        bns,
        cp_occ,
        sentinel_index,
        min_seed_len: 19,
    };
    bwa_index_to_dump.dump(prefix)?;

    log::info!("Index building complete.");
    Ok(())
}