//! `atomr-infer-runtime-ws-core` — shared WebSocket transport for
//! the audio program of work (FR-TTS-001, FR-STT-001, FR-A2F-001).
//!
//! Five provider crates consume this transport:
//!
//! - `atomr-infer-runtime-deepgram`     (STT, WSS streaming)
//! - `atomr-infer-runtime-assemblyai`   (STT, WSS streaming)
//! - `atomr-infer-runtime-elevenlabs`   (TTS, HTTPS + WSS alignment)
//! - `atomr-infer-runtime-openai-realtime` (bidirectional WSS)
//! - `atomr-infer-runtime-gemini-live`     (bidirectional WSS)
//!
//! What this crate provides
//! ------------------------
//!
//! - [`WsClient::connect`] — TLS-aware [`url`]-parsed connect with
//!   a deadline; returns split [`WsSender`] / [`WsReceiver`] halves.
//! - [`Frame`] — provider-agnostic frame variants (binary, text,
//!   ping, pong, close). Hides `tungstenite::Message`.
//! - [`ReconnectEngine`] — exponential-backoff state machine that
//!   reuses [`atomr_infer_remote_core::backoff::BackoffPolicy`] for
//!   policy and a `max_attempts` ceiling for termination.
//! - [`Keepalive`] — ping/pong + idle-timeout tracker. Time is
//!   passed in explicitly so tests stay deterministic.
//! - [`coalesce_binary`] — drop-oldest binary coalescing used by
//!   providers under upstream backpressure.
//!
//! What this crate intentionally does NOT provide
//! ----------------------------------------------
//!
//! - JSON envelopes (each provider's wire format lives in its crate).
//! - Auth header injection (provider crates own credentials).
//! - Session-level orchestration ([`crate::ReconnectEngine`] only
//!   sequences delays — the *what to send after reconnect* is
//!   provider state).
//!
//! See `docs/audio-modalities.md` for the architectural decision
//! record.

#![doc(html_root_url = "https://docs.rs/atomr-infer-runtime-ws-core")]

pub mod client;
pub mod error;
pub mod frame;
pub mod keepalive;
pub mod reconnect;

pub use client::{WsClient, WsReceiver, WsSender};
pub use error::{WsError, WsResult};
pub use frame::{coalesce_binary, Frame};
pub use keepalive::{Keepalive, KeepaliveAction, KeepaliveConfig};
pub use reconnect::ReconnectEngine;
