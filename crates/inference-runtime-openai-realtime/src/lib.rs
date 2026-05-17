//! # atomr-infer-runtime-openai-realtime
//!
//! OpenAI Realtime API provider for `atomr-infer`. Implements
//! [`atomr_infer_core::RealtimeRunner`] against the OpenAI Realtime
//! WebSocket API (FR-TTS-001, M9-A).
//!
//! ## Build profiles
//!
//! | Profile | Feature | Behaviour |
//! |---|---|---|
//! | Default | *(none)* | Stub — returns `InferenceError::Internal` |
//! | Full | `tts-openai-realtime` | Real WSS bidirectional adapter |
//!
//! ## WebSocket endpoint
//!
//! `wss://api.openai.com/v1/realtime?model=<model>`
//!
//! Auth: `Authorization: Bearer <key>` and `OpenAI-Beta: realtime=v1`.
//!
//! ## Reference
//!
//! <https://platform.openai.com/docs/guides/realtime>

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
pub mod runner;
#[cfg(feature = "tts-openai-realtime")]
pub mod wire;

pub use config::OpenAiRealtimeConfig;
pub use error::classify_realtime_error;
pub use runner::OpenAiRealtimeRunner;
