//! # inference-runtime-anthropic
//!
//! Anthropic Messages API runtime. Doc §10.3. Same shape as
//! `inference-runtime-openai`; per-provider differences live in
//! `wire` and `error`.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
pub mod runner;
pub mod wire;

pub use config::AnthropicConfig;
pub use cost::AnthropicPricing;
pub use error::classify_anthropic_error;
pub use runner::AnthropicRunner;
