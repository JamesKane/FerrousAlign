//! SAM flag bit masks (SAM specification v1.6).
//!
//! Used for setting and querying alignment flags in SAM/BAM format.

pub const PAIRED: u16 = 0x1;         // Template having multiple segments in sequencing
pub const PROPER_PAIR: u16 = 0x2;    // Each segment properly aligned according to the aligner
pub const UNMAPPED: u16 = 0x4;       // Segment unmapped
pub const MATE_UNMAPPED: u16 = 0x8;  // Next segment in the template unmapped
pub const REVERSE: u16 = 0x10;       // SEQ being reverse complemented
pub const MATE_REVERSE: u16 = 0x20;  // SEQ of the next segment reverse complemented
pub const FIRST_IN_PAIR: u16 = 0x40; // The first segment in the template
pub const SECOND_IN_PAIR: u16 = 0x80; // The last segment in the template
pub const SECONDARY: u16 = 0x100;    // Secondary alignment
pub const QCFAIL: u16 = 0x200;       // Not passing filters
pub const DUPLICATE: u16 = 0x400;    // PCR or optical duplicate
pub const SUPPLEMENTARY: u16 = 0x800; // Supplementary alignment
