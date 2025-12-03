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
//!
//! # Hybrid AoS/SoA Architecture
//!
//! The paired-end pipeline requires different data layouts for different stages:
//! - **SoA** for compute-heavy stages (alignment, mate rescue) - SIMD benefits
//! - **AoS** for logic-heavy stages (pairing, output) - correctness requirement
//!
//! Stage implementations declare their input/output types, and the orchestrator
//! handles any necessary conversions at stage boundaries.

use crate::core::compute::ComputeContext;
use crate::pipelines::linear::index::index::BwaIndex;
use crate::pipelines::linear::mem_opt::MemOpt;
use std::fmt;

// ============================================================================
// STAGE TRAIT
// ============================================================================

/// A pipeline stage that transforms input data to output data.
///
/// Each stage in the alignment pipeline implements this trait, allowing:
/// - Independent testing of each stage
/// - Swappable implementations (e.g., CPU vs GPU extension)
/// - Clear data flow through explicit input/output types
///
/// # Type Parameters
///
/// Stages are defined with associated types for flexibility:
/// - `Input`: The data type consumed by this stage
/// - `Output`: The data type produced by this stage
///
/// # Example
///
/// ```ignore
/// struct SeedingStage;
///
/// impl PipelineStage for SeedingStage {
///     type Input = SoAReadBatch;
///     type Output = SoASeedBatch;
///
///     fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError> {
///         // ... seeding implementation
///     }
///
///     fn name(&self) -> &'static str {
///         "Seeding"
///     }
/// }
/// ```
pub trait PipelineStage {
    /// The input type consumed by this stage
    type Input;

    /// The output type produced by this stage
    type Output;

    /// Process a batch of data through this stage.
    ///
    /// # Arguments
    /// * `input` - The input data to process
    /// * `ctx` - Shared context containing index, options, and compute backend
    ///
    /// # Returns
    /// The processed output data, or an error if processing fails.
    fn process(&self, input: Self::Input, ctx: &StageContext) -> Result<Self::Output, StageError>;

    /// Human-readable stage name for logging and profiling.
    fn name(&self) -> &'static str;

    /// Optional validation of input data before processing.
    ///
    /// Override this to add input validation. Default implementation accepts all input.
    fn validate(&self, _input: &Self::Input) -> Result<(), StageError> {
        Ok(())
    }
}

// ============================================================================
// STAGE CONTEXT
// ============================================================================

/// Shared context passed to each pipeline stage.
///
/// Contains read-only references to shared resources needed by all stages.
/// This is passed by reference to each stage's `process` method.
pub struct StageContext<'a> {
    /// BWA index (BWT, suffix array, reference sequences)
    pub index: &'a BwaIndex,

    /// Alignment options (scoring, thresholds, etc.)
    pub options: &'a MemOpt,

    /// Compute backend context (SIMD engine selection)
    pub compute_ctx: &'a ComputeContext,

    /// Current batch identifier (for logging/debugging)
    pub batch_id: u64,
}

impl<'a> StageContext<'a> {
    /// Create a new stage context.
    pub fn new(
        index: &'a BwaIndex,
        options: &'a MemOpt,
        compute_ctx: &'a ComputeContext,
        batch_id: u64,
    ) -> Self {
        Self {
            index,
            options,
            compute_ctx,
            batch_id,
        }
    }
}

// ============================================================================
// STAGE ERROR
// ============================================================================

/// Errors that can occur during pipeline stage processing.
///
/// Each variant corresponds to a specific stage or error category,
/// allowing for targeted error handling and reporting.
#[derive(Debug)]
pub enum StageError {
    /// Error during read loading
    Loading(String),

    /// Error during SMEM extraction (seeding)
    Seeding(String),

    /// Error during seed chaining
    Chaining(String),

    /// Error during Smith-Waterman extension
    Extension(String),

    /// Error during alignment finalization
    Finalization(String),

    /// I/O error (file reading, writing)
    Io(std::io::Error),

    /// Empty batch (not necessarily an error, signals EOF)
    EmptyBatch,

    /// Input validation failed
    ValidationFailed(String),
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageError::Loading(msg) => write!(f, "Loading error: {}", msg),
            StageError::Seeding(msg) => write!(f, "Seeding error: {}", msg),
            StageError::Chaining(msg) => write!(f, "Chaining error: {}", msg),
            StageError::Extension(msg) => write!(f, "Extension error: {}", msg),
            StageError::Finalization(msg) => write!(f, "Finalization error: {}", msg),
            StageError::Io(err) => write!(f, "I/O error: {}", err),
            StageError::EmptyBatch => write!(f, "Empty batch"),
            StageError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
        }
    }
}

impl std::error::Error for StageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StageError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for StageError {
    fn from(err: std::io::Error) -> Self {
        StageError::Io(err)
    }
}

// ============================================================================
// SUBMODULES (stubs for now, will be populated in later phases)
// ============================================================================

pub mod chaining;
pub mod extension;
pub mod finalization;
pub mod loading;
pub mod seeding;

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the PipelineStage trait is object-safe (can be used with dyn)
    #[test]
    fn test_stage_trait_object_safety() {
        // This test verifies the trait can be used as a trait object
        // The trait IS object-safe because:
        // - It uses associated types (not generic parameters on methods)
        // - All methods have &self receiver
        // - No methods return Self

        // We can't actually create a Box<dyn PipelineStage> because
        // associated types aren't determined, but we can verify the
        // trait compiles with these constraints.

        struct DummyStage;

        impl PipelineStage for DummyStage {
            type Input = ();
            type Output = ();

            fn process(
                &self,
                _input: Self::Input,
                _ctx: &StageContext,
            ) -> Result<Self::Output, StageError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "Dummy"
            }
        }

        let stage = DummyStage;
        assert_eq!(stage.name(), "Dummy");
    }

    #[test]
    fn test_stage_error_display() {
        let err = StageError::Seeding("test error".to_string());
        assert_eq!(format!("{}", err), "Seeding error: test error");

        let err = StageError::EmptyBatch;
        assert_eq!(format!("{}", err), "Empty batch");
    }

    #[test]
    fn test_stage_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let stage_err: StageError = io_err.into();

        match stage_err {
            StageError::Io(_) => {} // Expected
            _ => panic!("Expected Io variant"),
        }
    }
}
