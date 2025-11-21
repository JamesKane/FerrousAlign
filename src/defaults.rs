// src/defaults.rs

// Algorithmic Constants
pub const MIN_SEED_LEN: i32 = 19;
pub const BAND_WIDTH: i32 = 100;
pub const OFF_DIAGONAL_DROPOFF: i32 = 100;
pub const RESEED_FACTOR: f32 = 1.5;
pub const SEED_OCCURRENCE_3RD: u64 = 20;
pub const MAX_OCCURRENCES: i32 = 500;
pub const DROP_CHAIN_FRACTION: f32 = 0.50;
pub const MIN_CHAIN_WEIGHT: i32 = 0;
pub const MAX_MATE_RESCUES: i32 = 50;

// Scoring Constants
pub const MATCH_SCORE: i32 = 1;
pub const MISMATCH_PENALTY: i32 = 4;
pub const GAP_OPEN_PENALTIES: &str = "6,6";
pub const GAP_EXTEND_PENALTIES: &str = "1,1";
pub const CLIPPING_PENALTIES: &str = "5,5";
pub const UNPAIRED_PENALTY: i32 = 17;

// Other Constants
pub const VERBOSITY: i32 = 3;
pub const MIN_SCORE: i32 = 30;
pub const MAX_XA_HITS: &str = "5,200";
