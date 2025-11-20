// Index management module
//
// This module handles BWA index loading, dumping, and the main BwaIndex structure
// that combines BWT, reference sequences, and FM-Index checkpoints.

use crate::bntseq::BntSeq;
use crate::bwt::Bwt;
use crate::fm_index::{CP_SHIFT, CpOcc};
use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// Main BWA index structure combining BWT, reference sequences, and FM-Index data
pub struct BwaIndex {
    pub bwt: Bwt,
    pub bns: BntSeq,
    pub cp_occ: Vec<CpOcc>,
    pub sentinel_index: i64,
    pub min_seed_len: i32, // Kept for backwards compatibility, but will use MemOpt
}

impl BwaIndex {
    /// Load BWA index from disk
    ///
    /// Loads the .bwt.2bit.64 file containing:
    /// - BWT sequence length and cumulative counts (l2 array)
    /// - FM-Index checkpoints (cp_occ array)
    /// - Sampled suffix array (sa_ms_byte and sa_ls_word)
    /// - Sentinel index position
    ///
    /// Also loads reference sequences from .ann, .amb, and .pac files via BntSeq
    pub fn bwa_idx_load(prefix: &Path) -> io::Result<Self> {
        let mut bwt = Bwt::new();
        let bns = BntSeq::bns_restore(prefix)?;

        let cp_file_name = PathBuf::from(prefix.to_string_lossy().to_string() + ".bwt.2bit.64");
        // Use 16MB buffer for index loading (default 8KB causes 1.2M syscalls for 9.4GB file!)
        const INDEX_BUFFER_SIZE: usize = 16 * 1024 * 1024; // 16 MB
        let mut cp_file = BufReader::with_capacity(INDEX_BUFFER_SIZE, File::open(&cp_file_name)?);

        let mut buf_i64 = [0u8; 8];
        let mut buf_u64 = [0u8; 8];
        let mut buf_u8 = [0u8; 1];
        let mut buf_u32 = [0u8; 4];

        // 1. Read seq_len
        cp_file.read_exact(&mut buf_i64)?;
        bwt.seq_len = i64::from_le_bytes(buf_i64) as u64;

        // 2. Read count array (l2)
        for i in 0..5 {
            cp_file.read_exact(&mut buf_i64)?;
            bwt.cumulative_count[i] = i64::from_le_bytes(buf_i64) as u64;
        }

        // CRITICAL: Match C++ bwa-mem2 behavior - add 1 to all count values
        // See FMI_search.cpp:435 - this is required for correct SMEM generation
        for i in 0..5 {
            bwt.cumulative_count[i] += 1;
        }

        // 3. Read cp_occ array
        let cp_occ_size = (bwt.seq_len >> CP_SHIFT) + 1;
        let mut cp_occ: Vec<CpOcc> = Vec::with_capacity(cp_occ_size as usize);
        for _ in 0..cp_occ_size {
            let mut checkpoint_counts = [0i64; 4];
            for i in 0..4 {
                cp_file.read_exact(&mut buf_i64)?;
                checkpoint_counts[i] = i64::from_le_bytes(buf_i64);
            }
            let mut bwt_encoding_bits = [0u64; 4];
            for i in 0..4 {
                cp_file.read_exact(&mut buf_u64)?;
                bwt_encoding_bits[i] = u64::from_le_bytes(buf_u64);
            }
            cp_occ.push(CpOcc {
                checkpoint_counts,
                bwt_encoding_bits,
            });
        }

        // In C++, SA_COMPX is 3 (defined in macro.h), so sa_intv is 8
        let sa_compx = 3;
        let _sa_intv = 1 << sa_compx; // sa_intv = 8
        // C++ uses: ((ref_seq_len >> SA_COMPX) + 1)
        // which equals: (ref_seq_len / 8) + 1
        let sa_len = (bwt.seq_len >> sa_compx) + 1;

        // 4. Read sa_ms_byte array
        bwt.sa_high_bytes.reserve_exact(sa_len as usize);
        for _ in 0..sa_len {
            cp_file.read_exact(&mut buf_u8)?;
            let val = u8::from_le_bytes(buf_u8) as i8;
            bwt.sa_high_bytes.push(val);
        }

        // 5. Read sa_ls_word array
        bwt.sa_low_words.reserve_exact(sa_len as usize);
        for _ in 0..sa_len {
            cp_file.read_exact(&mut buf_u32)?;
            let val = u32::from_le_bytes(buf_u32);
            bwt.sa_low_words.push(val);
        }

        // 6. Read sentinel_index
        cp_file.read_exact(&mut buf_i64)?;
        let sentinel_index = i64::from_le_bytes(buf_i64);
        bwt.primary = sentinel_index as u64;

        // Set other bwt fields that were not in the file
        bwt.sa_sample_interval = 1 << sa_compx;
        bwt.sa_sample_count = sa_len;

        // Debug: verify SA values look reasonable
        if bwt.sa_high_bytes.len() > 10 {
            log::debug!(
                "Loaded SA samples: n_sa={}, sa_intv={}",
                bwt.sa_sample_count,
                bwt.sa_sample_interval
            );
            log::debug!("First 5 SA values:");
            for i in 0..5.min(bwt.sa_high_bytes.len()) {
                let sa_val = ((bwt.sa_high_bytes[i] as i64) << 32) | (bwt.sa_low_words[i] as i64);
                log::debug!("  SA[{}] = {}", i * bwt.sa_sample_interval as usize, sa_val);
            }
        }

        Ok(BwaIndex {
            bwt,
            bns,
            cp_occ,
            sentinel_index,
            min_seed_len: 1, // Initialize with default value
        })
    }

    /// Dump BWA index to disk
    ///
    /// Writes the .bwt.2bit.64 file in the format compatible with C++ bwa-mem2
    pub fn dump(&self, prefix: &Path) -> io::Result<()> {
        let bwt_file_path = prefix.with_extension("bwt.2bit.64");
        let mut file = File::create(&bwt_file_path)?;

        // Match C++ FMI_search::build_fm_index format
        // 1. ref_seq_len (i64)
        file.write_all(&(self.bwt.seq_len as i64).to_le_bytes())?;

        // 2. count array (l2) (5 * i64)
        for i in 0..5 {
            file.write_all(&(self.bwt.cumulative_count[i] as i64).to_le_bytes())?;
        }

        // 3. cp_occ array
        for cp_occ_entry in self.cp_occ.iter() {
            for i in 0..4 {
                file.write_all(&cp_occ_entry.checkpoint_counts[i].to_le_bytes())?;
            }
            for i in 0..4 {
                file.write_all(&cp_occ_entry.bwt_encoding_bits[i].to_le_bytes())?;
            }
        }

        // 4. sa_ms_byte array
        for val in self.bwt.sa_high_bytes.iter() {
            file.write_all(&val.to_le_bytes())?;
        }

        // 5. sa_ls_word array
        for val in self.bwt.sa_low_words.iter() {
            file.write_all(&val.to_le_bytes())?;
        }

        // 6. sentinel_index (i64)
        file.write_all(&self.sentinel_index.to_le_bytes())?;

        Ok(())
    }
}
