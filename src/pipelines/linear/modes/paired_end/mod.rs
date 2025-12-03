//! Paired-end mode processing
//!
//! Hybrid AoS/SoA pipeline for correct mate pairing.
//!
//! # Architecture
//!
//! ```text
//! [SoA] Alignment → [AoS] Pairing → [SoA] Mate Rescue → [AoS] Output
//! ```
//!
//! The pairing stage MUST use AoS to maintain per-read alignment boundaries.
//! Pure SoA pairing causes 96% duplicate reads due to index corruption.

// Implementation will be added in later phases
