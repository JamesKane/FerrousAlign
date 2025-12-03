pub mod collect_soa;
pub mod dispatch;
pub mod finalize_soa;
pub mod orchestration_soa;
pub mod soa;
pub mod types;

pub use collect_soa::collect_extension_jobs_batch_soa;
pub use dispatch::*;
pub use finalize_soa::finalize_alignments_soa;
pub use orchestration_soa::process_sub_batch_internal_soa;
pub use types::*; // keep external API stable (includes SoAAlignmentResult from PR4)
