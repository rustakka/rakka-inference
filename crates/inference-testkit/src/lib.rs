//! # inference-testkit
//!
//! Mocks and harnesses for testing the atomr-infer workspace.
//! Doc §10.3.
//!
//! Provides:
//! - [`MockRunner`] — `ModelRunner` impl that streams a fixed list of
//!   chunks at a configurable cadence; lets actor-system tests exercise
//!   the gateway → request → engine path without a real backend.
//! - [`mock_openai`] — `wiremock::MockServer` factory pre-loaded with
//!   the OpenAI Chat Completions endpoint that emits a deterministic
//!   SSE response. Test code controls injection of 429 / 5xx /
//!   timeout via the returned helper handle.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod mock_openai;
pub mod mock_runner;

pub use mock_openai::{inject_429_once, inject_5xx_once, mount_chat_happy_path, MockOpenAi};
pub use mock_runner::{MockRunner, MockScript};
