// Index management module
//
// This module handles BWA index loading, dumping, and the main BwaIndex structure
// that combines BWT, reference sequences, and FM-Index checkpoints.
//
// Memory-mapped I/O is used for index loading to:
// 1. Eliminate explicit loading time (kernel handles paging)
// 2. Improve cache behavior through kernel prefetching
// 3. Reduce memory pressure (pages loaded on demand)

use super::bntseq::BntSeq;
use super::bwt::Bwt;
use super::fm_index::{CP_SHIFT, CpOcc};
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Main BWA index structure combining BWT, reference sequences, and FM-Index data
pub struct BwaIndex {
    pub bwt: Bwt,
    pub bns: BntSeq,
    pub cp_occ: Vec<CpOcc>,
    pub sentinel_index: i64,
    pub min_seed_len: i32, // Kept for backwards compatibility, but will use MemOpt
    /// Memory-mapped index file (kept alive to maintain kernel page caching)
    #[allow(dead_code)]
    pub(crate) mmap_bwt: Option<Mmap>,
    /// Memory-mapped PAC file
    #[allow(dead_code)]
    pub(crate) mmap_pac: Option<Mmap>,
}

impl BwaIndex {
    /// Load BWA index from disk using memory-mapped I/O
    ///
    /// Uses mmap for the .bwt.2bit.64 file to:
    /// - Eliminate explicit loading time (kernel handles paging on demand)
    /// - Improve cache behavior through kernel prefetching
    /// - Reduce memory pressure (pages loaded only when accessed)
    ///
    /// The mmap handles are kept alive in the BwaIndex struct to maintain
    /// kernel page caching throughout the alignment process.
    pub fn bwa_idx_load(prefix: &Path) -> io::Result<Self> {
        use std::time::Instant;
        let start = Instant::now();

        let mut bwt = Bwt::new();
        let bns = BntSeq::bns_restore(prefix)?;

        // Memory-map the .bwt.2bit.64 file
        let cp_file_name = PathBuf::from(prefix.to_string_lossy().to_string() + ".bwt.2bit.64");
        let file = File::open(&cp_file_name)?;
        let mmap = unsafe { Mmap::map(&file)? };

        log::debug!(
            "Memory-mapped .bwt.2bit.64: {} bytes ({:.1} MB)",
            mmap.len(),
            mmap.len() as f64 / 1024.0 / 1024.0
        );

        let mut offset = 0usize;

        // Helper to read scalar values from mmap
        fn read_i64(mmap: &[u8], offset: &mut usize) -> i64 {
            let val = i64::from_le_bytes(mmap[*offset..*offset + 8].try_into().unwrap());
            *offset += 8;
            val
        }

        // 1. Read seq_len
        bwt.seq_len = read_i64(&mmap, &mut offset) as u64;

        // 2. Read count array (l2)
        for i in 0..5 {
            bwt.cumulative_count[i] = read_i64(&mmap, &mut offset) as u64;
        }

        // CRITICAL: Match C++ bwa-mem2 behavior - add 1 to all count values
        // See FMI_search.cpp:435 - this is required for correct SMEM generation
        for i in 0..5 {
            bwt.cumulative_count[i] += 1;
        }

        // 3. Read cp_occ array using bulk copy
        // CpOcc struct is 64 bytes: 4×i64 + 4×u64
        let cp_occ_size = (bwt.seq_len >> CP_SHIFT) + 1;
        let cp_occ_bytes = cp_occ_size as usize * std::mem::size_of::<CpOcc>();
        let mut cp_occ: Vec<CpOcc> = Vec::with_capacity(cp_occ_size as usize);
        unsafe {
            let src = mmap[offset..offset + cp_occ_bytes].as_ptr() as *const CpOcc;
            cp_occ.set_len(cp_occ_size as usize);
            std::ptr::copy_nonoverlapping(src, cp_occ.as_mut_ptr(), cp_occ_size as usize);
        }
        offset += cp_occ_bytes;

        // In C++, SA_COMPX is 3 (defined in macro.h), so sa_intv is 8
        let sa_compx = 3;
        let sa_len = (bwt.seq_len >> sa_compx) + 1;

        // 4. Read sa_ms_byte array using bulk copy
        let sa_high_bytes_len = sa_len as usize;
        bwt.sa_high_bytes = Vec::with_capacity(sa_high_bytes_len);
        unsafe {
            let src = mmap[offset..offset + sa_high_bytes_len].as_ptr() as *const i8;
            bwt.sa_high_bytes.set_len(sa_high_bytes_len);
            std::ptr::copy_nonoverlapping(src, bwt.sa_high_bytes.as_mut_ptr(), sa_high_bytes_len);
        }
        offset += sa_high_bytes_len;

        // 5. Read sa_ls_word array using bulk copy
        let sa_low_words_len = sa_len as usize;
        let sa_low_words_bytes = sa_low_words_len * std::mem::size_of::<u32>();
        bwt.sa_low_words = Vec::with_capacity(sa_low_words_len);
        unsafe {
            let src = mmap[offset..offset + sa_low_words_bytes].as_ptr() as *const u32;
            bwt.sa_low_words.set_len(sa_low_words_len);
            std::ptr::copy_nonoverlapping(src, bwt.sa_low_words.as_mut_ptr(), sa_low_words_len);
        }
        offset += sa_low_words_bytes;

        // 6. Read sentinel_index
        let sentinel_index = read_i64(&mmap, &mut offset);
        bwt.primary = sentinel_index as u64;

        // Set other bwt fields
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

        let elapsed = start.elapsed();
        log::info!(
            "Index loaded via mmap in {:.3}s (seq_len={}, cp_occ={}, sa={})",
            elapsed.as_secs_f64(),
            bwt.seq_len,
            cp_occ.len(),
            sa_len
        );

        Ok(BwaIndex {
            bwt,
            bns,
            cp_occ,
            sentinel_index,
            min_seed_len: 1,
            mmap_bwt: Some(mmap),
            mmap_pac: None, // PAC is loaded separately in BntSeq
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
