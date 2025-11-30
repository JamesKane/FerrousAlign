//! Shared alignment types used across core and pipelines.

/// Direction of extension from a seed or anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionDirection {
    Left,
    Right,
}
