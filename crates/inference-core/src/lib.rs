//! # inference-core
//!
//! Foundation types for the atomr-infer workspace. Per architecture
//! doc v4 §10.4 this crate has no actor-system dependencies — only
//! serde / thiserror / bytes / secrecy (plus the documented `async-trait`
//! exception for the `ModelRunner` trait).
//!
//! Everything in here is consumed by `inference-runtime` (actor
//! implementations) and the per-runtime crates. Authors of new runtime
//! backends only need to depend on this crate to satisfy the
//! [`ModelRunner`] contract.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod batch;
pub mod cost;
pub mod deployment;
pub mod error;
pub mod registry;
pub mod runner;
pub mod runtime;
pub mod tokens;

pub use batch::{ExecuteBatch, Message, MessageContent, Role, SamplingParams};
pub use cost::{CostEstimate, EstimateCost};
pub use deployment::{
    Budget, BudgetAction, CapacityPolicy, Deployment, RateLimits, Replica, RetryPolicy, Serving, Timeouts,
};
pub use error::{InferenceError, InferenceResult};
pub use registry::infer_runtime;
pub use runner::{ModelRunner, RunHandle, SessionRebuildCause, WeightSource};
pub use runtime::{
    CircuitBreakerConfig, JitterKind, ProviderKind, RuntimeConfig, RuntimeKind, TransportKind,
};
pub use tokens::{FinishReason, TokenChunk, TokenUsage, Tokens};

/// Re-export of [`secrecy::SecretString`] so consumer crates do not need
/// to take a direct dependency on `secrecy`. Architecturally significant:
/// credentials are part of the type system from the bottom up (doc §12.5).
pub type SecretString = secrecy::SecretString;
pub use secrecy::{ExposeSecret, SecretBox};
