use super::fm_index::{CP_SHIFT, CpOcc};
use crate::utils::BinaryWrite;
use std::io::{self, Write};
use std::path::Path;

pub type BwtInt = u64;

/// BWT (Burrows-Wheeler Transform) index structure
/// Corresponds to C++ bwt_t (bwt.h:49-61)
#[derive(Debug)]
pub struct Bwt {
    /// Primary index of BWT: S^{-1}(0)
    pub primary: BwtInt,
    /// Cumulative count array C() - occurrence counts for each base
    pub cumulative_count: [BwtInt; 5],
    /// Total sequence length
    pub seq_len: BwtInt,
    /// Size of BWT data (approximately seq_len/4 due to 2-bit packing)
    pub bwt_size: BwtInt,
    /// Occurrence count lookup table (256 entries for fast computation)
    pub cnt_table: [u32; 256],
    /// Suffix array sampling interval (typically 8 or 16)
    pub sa_sample_interval: i32,
    /// Number of suffix array samples stored
    pub sa_sample_count: BwtInt,
    /// Packed BWT data (2-bit encoding, 4 bases per byte)
    pub bwt_data: Vec<u8>,
    /// Suffix array high bytes (compressed format)
    pub sa_high_bytes: Vec<i8>,
    /// Suffix array low words (compressed format)
    pub sa_low_words: Vec<u32>,
}

#[path = "bwt_test.rs"]
mod bwt_test;

impl Default for Bwt {
    fn default() -> Self {
        Bwt {
            primary: 0,
            cumulative_count: [0; 5],
            seq_len: 0,
            bwt_size: 0,
            cnt_table: [0; 256],
            sa_sample_interval: 0,
            sa_sample_count: 0,
            bwt_data: Vec::new(),
            sa_high_bytes: Vec::new(),
            sa_low_words: Vec::new(),
        }
    }
}

impl Bwt {
    pub fn new() -> Self {
        Self::default() // Use the default implementation
    }

    pub fn new_from_bwt_data(bwt_data: Vec<u8>, l2: [u64; 5], seq_len: u64, primary: u64) -> Self {
        let bwt_size = bwt_data.len() as u64;

        // Initialize cnt_table
        let mut cnt_table = [0; 256];
        for i in 0..16 {
            cnt_table[i * 16] = i as u32;
        }

        Bwt {
            cumulative_count: l2, // Use the passed l2
            seq_len,
            primary,
            sa_sample_interval: 0, // Default value
            sa_sample_count: 0,    // Default value
            bwt_size,
            cnt_table,
            sa_high_bytes: Vec::new(),
            sa_low_words: Vec::new(),
            bwt_data,
        }
    }

    pub fn bwt_dump_bwt(&self, prefix: &Path) -> io::Result<()> {
        let bwt_file_path = prefix.with_extension("bwt.2bit.64");
        let mut file = std::fs::File::create(&bwt_file_path)?;

        // Write seq_len, primary, l2, sa_intv, n_sa
        file.write_u64_le(self.seq_len)?;
        file.write_u64_le(self.primary)?;
        file.write_u64_array_le(&self.cumulative_count)?;
        file.write_u32_le(self.sa_sample_interval as u32)?; // Convert i32 to u32 for writing
        file.write_u64_le(self.sa_sample_count)?;

        // Write bwt_data
        file.write_all(&self.bwt_data)?;

        // Write sa_ms_byte and sa_ls_word
        for val in &self.sa_high_bytes {
            file.write_i8_le(*val)?;
        }
        for val in &self.sa_low_words {
            file.write_u32_le(*val)?;
        }

        Ok(())
    }

    pub fn bwt_cal_sa(&mut self, sa_intv: i32, sa_temp: &[i32]) {
        self.sa_sample_interval = sa_intv;
        self.sa_sample_count = self.seq_len.div_ceil(sa_intv as u64);

        self.sa_high_bytes.reserve(self.sa_sample_count as usize);
        self.sa_low_words.reserve(self.sa_sample_count as usize);

        // eprintln!("bwt_cal_sa: seq_len={}, sa_intv={}, n_sa={}, sa_temp.len()={}",
        //           self.seq_len, sa_intv, self.sa_sample_count, sa_temp.len());

        for i in 0..self.sa_sample_count as usize {
            let sa_index = i * sa_intv as usize;
            if sa_index >= sa_temp.len() {
                log::warn!("sa_index {} >= sa_temp.len() {}", sa_index, sa_temp.len());
                break;
            }
            let sa_val = sa_temp[sa_index] as i64;
            // eprintln!("  i={}, sa_index={}, sa_temp[{}]={}, sa_val={}", i, sa_index, sa_index, sa_temp[sa_index], sa_val);
            self.sa_high_bytes.push((sa_val >> 32) as i8);
            self.sa_low_words.push(sa_val as u32);
        }
    }

    pub fn get_bwt_base(&self, pos: u64) -> u8 {
        let byte_idx = (pos / 4) as usize;
        let bit_offset = (pos % 4) * 2;
        (self.bwt_data[byte_idx] >> bit_offset) & 0x03
    }

    pub fn calculate_cp_occ(&self, sentinel_index: u64) -> Vec<CpOcc> {
        let cp_occ_size = (self.seq_len >> CP_SHIFT) + 1;
        let mut cp_occ: Vec<CpOcc> = Vec::with_capacity(cp_occ_size as usize);

        let mut cumulative_counts = [0i64; 4]; // Cumulative counts up to current position
        let mut block_one_hot_bwt_str = [0u64; 4]; // Bitmask for the current block
        let mut current_pos = 0u64;

        // Initialize the first CpOcc entry (for position 0)
        cp_occ.push(CpOcc {
            checkpoint_counts: [0; 4], // Counts before position 0 are all 0
            bwt_encoding_bits: [0; 4], // Bitmask for the first block
        });

        let mut checkpoint_index = 0; // Track which checkpoint we're in

        for &byte in self.bwt_data.iter() {
            for bit_offset in (0..8).step_by(2) {
                // Iterate through 2-bit bases in the byte
                if current_pos >= self.seq_len {
                    break; // Reached end of sequence
                }

                let base_code = (byte >> bit_offset) & 0x03; // Extract 2-bit base

                // Skip the sentinel position (stored as 0 but should not be counted as base 0)
                if current_pos != sentinel_index {
                    // Update bitmask for the current block
                    let offset_in_block = current_pos % (1 << CP_SHIFT);
                    block_one_hot_bwt_str[base_code as usize] |= 1u64 << (63 - offset_in_block);

                    // Update cumulative counts
                    cumulative_counts[base_code as usize] += 1;
                }

                current_pos += 1;

                if current_pos % (1 << CP_SHIFT) == 0 {
                    // Checkpoint reached: update the current checkpoint's bitmask, then create next checkpoint
                    cp_occ[checkpoint_index].bwt_encoding_bits = block_one_hot_bwt_str;

                    // Create next checkpoint with current cumulative counts
                    cp_occ.push(CpOcc {
                        checkpoint_counts: cumulative_counts, // Cumulative counts up to the end of this block
                        bwt_encoding_bits: [0u64; 4], // Will be filled as we process next block
                    });

                    checkpoint_index += 1;
                    block_one_hot_bwt_str = [0u64; 4]; // Reset bitmask for the next block
                }
            }
        }

        // For sequences that don't fill a complete checkpoint block, update the last checkpoint's bitmask
        if checkpoint_index < cp_occ.len() {
            cp_occ[checkpoint_index].bwt_encoding_bits = block_one_hot_bwt_str;
        }

        cp_occ
    }
}
