//! Core reusable components for alignment operations.
//!
//! This module contains components that are agnostic to the reference structure
//! and can be reused across different alignment pipelines (linear, graph, etc.).

pub mod alignment;
pub mod compute;
pub mod io;
pub mod kbtree;
pub mod utils;
