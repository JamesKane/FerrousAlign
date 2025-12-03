//! Seed chaining module.
//!
//! This module implements B-tree based seed chaining with O(n log n) complexity.
//! Matches C++ bwa-mem2's mem_chain_seeds (bwamem.cpp:806-974).
//!
//! # Module Organization
//!
//! - `types` - Core data structures (`Chain`, `SoAChainBatch`)
//! - `btree` - B-tree based chaining algorithm
//! - `filter` - Chain filtering by weight and overlap
//! - `weight` - Chain weight calculation
//!
//! # Algorithm Overview
//!
//! 1. Sort seeds by (query_pos, query_end)
//! 2. For each seed:
//!    a. Find the chain with closest reference position using B-tree
//!    b. Try to merge seed into that chain (test_and_merge)
//!    c. If merge fails, create a new chain
//! 3. Filter chains by weight and overlap

mod btree;
mod filter;
mod types;
mod weight;

// Re-export types
pub use types::{Chain, SoAChainBatch, MAX_CHAINS_PER_READ, MAX_SEEDS_PER_READ};

// Re-export B-tree chaining functions
pub use btree::{chain_seeds, chain_seeds_batch, chain_seeds_with_l_pac};

// Re-export filtering functions
pub use filter::{filter_chains, filter_chains_batch};

// Re-export weight calculation
pub use weight::{cal_max_gap, calculate_chain_weight, calculate_chain_weight_soa};
