pub mod shared;
pub mod kernel;
pub mod scalar;
pub mod dispatch;
pub mod types;

pub mod isa_sse_neon;
#[cfg(target_arch = "x86_64")]
pub mod isa_avx2;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub mod isa_avx512;

// Re-export public items from submodules
pub use shared::*;
pub use kernel::*;
pub use scalar::*;
pub use dispatch::*;
pub use types::*;
pub use isa_sse_neon::*;
#[cfg(target_arch = "x86_64")]
pub use isa_avx2::*;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use isa_avx512::*;