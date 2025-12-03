//! Pipeline stage abstraction layer
//!
//! This module defines the `PipelineStage` trait that all pipeline stages implement.
//! Stages transform data from one type to another, with explicit support for
//! both SoA (Structure-of-Arrays) and AoS (Array-of-Structures) layouts.
//!
//! # Stage Pipeline
//!
//! ```text
//! Loading → Seeding → Chaining → Extension → Finalization
//! ```
//!
//! Each stage is independently testable and can be swapped for alternative
//! implementations (e.g., GPU-accelerated extension).

// Submodules will be added in Phase 1-4
// pub mod loading;
// pub mod seeding;
// pub mod chaining;
// pub mod extension;
// pub mod finalization;
