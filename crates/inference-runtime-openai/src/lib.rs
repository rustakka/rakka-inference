//! # inference-runtime-openai
//!
//! OpenAI Chat Completions runtime + Azure OpenAI variant. Doc §10.3.
//!
//! Implements the [`inference_core::ModelRunner`] contract over
//! HTTP/2 (via `reqwest`). SSE chunks are parsed by
//! `inference-remote-core::sse` and lifted into provider-typed deltas
//! by [`wire`].

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
pub mod runner;
pub mod wire;

pub use config::{OpenAiConfig, OpenAiVariant};
pub use cost::OpenAiPricing;
pub use error::classify_openai_error;
pub use runner::OpenAiRunner;
