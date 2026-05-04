//! # inference-runtime-gemini
//!
//! Google Gemini runtime for both AI Studio (`api_key`) and Vertex AI
//! (OAuth2 over `project + region`). Doc §10.3.
//!
//! OAuth2 token refresh is exposed as a pluggable
//! [`CredentialProvider`](atomr_infer_remote_core::session::CredentialProvider)
//! trait so we don't pull a full `oauth2` stack into the workspace
//! root. Operators wire `StaticApiKey` for AI Studio or supply their
//! own provider for Vertex (typically `gcloud auth print-access-token`
//! refreshed on a timer).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
pub mod runner;
pub mod wire;

pub use config::{GeminiConfig, GeminiVariant, SafetySetting};
pub use cost::GeminiPricing;
pub use error::classify_gemini_error;
pub use runner::GeminiRunner;
