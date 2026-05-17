//! # inference-testkit
//!
//! Mocks and harnesses for testing the atomr-infer workspace.
//! Doc §10.3.
//!
//! Provides:
//! - [`MockRunner`] — `ModelRunner` impl that streams a fixed list of
//!   chunks at a configurable cadence; lets actor-system tests exercise
//!   the gateway → request → engine path without a real backend.
//! - [`MockTtsRunner`], [`MockSttRunner`], [`MockA2FRunner`],
//!   [`MockRealtimeRunner`] — sibling mocks for each audio modality
//!   (`FR-TTS-001`, `FR-STT-001`, `FR-A2F-001`).
//! - [`mock_openai`] — `wiremock::MockServer` factory pre-loaded with
//!   the OpenAI Chat Completions endpoint that emits a deterministic
//!   SSE response. Test code controls injection of 429 / 5xx /
//!   timeout via the returned helper handle. Audio additions:
//!   [`mount_audio_speech_happy_path`],
//!   [`mount_audio_transcriptions_happy_path`], [`inject_audio_429`].
//! - [`MockWsServer`] — in-process WebSocket server used by provider
//!   WS client tests (Deepgram, AssemblyAI, ElevenLabs, OpenAI
//!   Realtime, Gemini Live).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod mock_a2f_runner;
pub mod mock_openai;
pub mod mock_realtime_runner;
pub mod mock_runner;
pub mod mock_stt_runner;
pub mod mock_tts_runner;
pub mod mock_ws;

pub use mock_a2f_runner::{MockA2FRunner, MockA2FScript};
pub use mock_openai::{
    inject_429_once, inject_5xx_once, inject_audio_429, mount_audio_speech_happy_path,
    mount_audio_transcriptions_happy_path, mount_chat_happy_path, MockOpenAi,
};
pub use mock_realtime_runner::{MockRealtimeRunner, MockRealtimeScript};
pub use mock_runner::{MockRunner, MockScript};
pub use mock_stt_runner::{MockSttRunner, MockSttScript};
pub use mock_tts_runner::{MockTtsRunner, MockTtsScript};
pub use mock_ws::MockWsServer;
