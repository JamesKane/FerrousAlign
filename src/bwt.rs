use crate::align::{CP_SHIFT, CpOcc};
use crate::utils::BinaryWrite;
use std::io::{self, Write};
use std::path::Path;

pub type BwtInt = u64;

#[derive(Debug)] // Keep Debug derive for easy printing
pub struct Bwt {
    pub primary: BwtInt,
    pub l2: [BwtInt; 5], // C(), cumulative count
    pub seq_len: BwtInt,
    pub bwt_size: BwtInt,
    pub cnt_table: [u32; 256], // This will be skipped in serialization/deserialization
    pub sa_intv: i32,
    pub n_sa: BwtInt,
    pub bwt_data: Vec<u8>,    // Store the packed BWT data
    pub sa_ms_byte: Vec<i8>,  // Suffix array most significant byte
    pub sa_ls_word: Vec<u32>, // Suffix array least significant word
}

#[path = "bwt_test.rs"]
mod bwt_test;

impl Default for Bwt {
    fn default() -> Self {
        Bwt {
            primary: 0,
            l2: [0; 5],
            seq_len: 0,
            bwt_size: 0,
            cnt_table: [0; 256], // Initialize with zeros
            sa_intv: 0,
            n_sa: 0,
            bwt_data: Vec::new(),
            sa_ms_byte: Vec::new(),
            sa_ls_word: Vec::new(),
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
            l2, // Use the passed l2
            seq_len,
            primary,
            sa_intv: 0, // Default value
            n_sa: 0,    // Default value
            bwt_size,
            cnt_table,
            sa_ms_byte: Vec::new(),
            sa_ls_word: Vec::new(),
            bwt_data,
        }
    }

    pub fn bwt_dump_bwt(&self, prefix: &Path) -> io::Result<()> {
        let bwt_file_path = prefix.with_extension("bwt.2bit.64");
        let mut file = std::fs::File::create(&bwt_file_path)?;

        // Write seq_len, primary, l2, sa_intv, n_sa
        file.write_u64_le(self.seq_len)?;
        file.write_u64_le(self.primary)?;
        file.write_u64_array_le(&self.l2)?;
        file.write_u32_le(self.sa_intv as u32)?; // Convert i32 to u32 for writing
        file.write_u64_le(self.n_sa)?;

        // Write bwt_data
        file.write_all(&self.bwt_data)?;

        // Write sa_ms_byte and sa_ls_word
        for val in &self.sa_ms_byte {
            file.write_i8_le(*val)?;
        }
        for val in &self.sa_ls_word {
            file.write_u32_le(*val)?;
        }

        Ok(())
    }

    pub fn bwt_cal_sa(&mut self, sa_intv: i32, sa_temp: &[i32]) {
        self.sa_intv = sa_intv;
        self.n_sa = (self.seq_len + sa_intv as u64 - 1) / sa_intv as u64;

        self.sa_ms_byte.reserve(self.n_sa as usize);
        self.sa_ls_word.reserve(self.n_sa as usize);

        // eprintln!("bwt_cal_sa: seq_len={}, sa_intv={}, n_sa={}, sa_temp.len()={}",
        //           self.seq_len, sa_intv, self.n_sa, sa_temp.len());

        for i in 0..self.n_sa as usize {
            let sa_index = i * sa_intv as usize;
            if sa_index >= sa_temp.len() {
                log::warn!("sa_index {} >= sa_temp.len() {}", sa_index, sa_temp.len());
                break;
            }
            let sa_val = sa_temp[sa_index] as i64;
            // eprintln!("  i={}, sa_index={}, sa_temp[{}]={}, sa_val={}", i, sa_index, sa_index, sa_temp[sa_index], sa_val);
            self.sa_ms_byte.push((sa_val >> 32) as i8);
            self.sa_ls_word.push(sa_val as u32);
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
            cp_count: [0; 4],        // Counts before position 0 are all 0
            one_hot_bwt_str: [0; 4], // Bitmask for the first block
        });

        let mut checkpoint_index = 0; // Track which checkpoint we're in

        for (_byte_idx, &byte) in self.bwt_data.iter().enumerate() {
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
                    cp_occ[checkpoint_index].one_hot_bwt_str = block_one_hot_bwt_str;

                    // Create next checkpoint with current cumulative counts
                    cp_occ.push(CpOcc {
                        cp_count: cumulative_counts, // Cumulative counts up to the end of this block
                        one_hot_bwt_str: [0u64; 4],  // Will be filled as we process next block
                    });

                    checkpoint_index += 1;
                    block_one_hot_bwt_str = [0u64; 4]; // Reset bitmask for the next block
                }
            }
        }

        // For sequences that don't fill a complete checkpoint block, update the last checkpoint's bitmask
        if checkpoint_index < cp_occ.len() {
            cp_occ[checkpoint_index].one_hot_bwt_str = block_one_hot_bwt_str;
        }

        cp_occ
    }
}
