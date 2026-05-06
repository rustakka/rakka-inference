//! # inference-cli
//!
//! `atomr-infer serve` binary plus the supporting config-loading machinery.
//! Doc §11.3.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod serve;

pub use config::{ClusterConfig, ProjectFile};
pub use serve::run_server;
