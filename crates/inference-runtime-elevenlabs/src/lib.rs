//! # inference-runtime-elevenlabs
//!
//! ElevenLabs text-to-speech runtime for `atomr-infer`. Implements
//! [`atomr_infer_core::runner::SpeechRunner`] against both
//! `POST /v1/text-to-speech/{voice_id}` (one-shot HTTPS) and
//! `WSS /v1/text-to-speech/{voice_id}/stream-input` (bidirectional
//! WebSocket with per-character alignment frames). Shares WebSocket
//! transport with sibling runtimes via
//! `atomr_infer_runtime_ws_core` (available only when the
//! `tts-elevenlabs` feature is on).
//!
//! ## Build profiles
//!
//! | Build                                                                          | Result                                                |
//! |--------------------------------------------------------------------------------|-------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-elevenlabs`                                | Stub — [`SpeechRunner::speak`] returns `Internal("tts-elevenlabs feature disabled at build time")`. |
//! | `cargo build -p atomr-infer-runtime-elevenlabs --features tts-elevenlabs`      | Real path — HTTPS one-shot + WSS streaming with alignment. |
//!
//! ## Voice + model identifiers
//!
//! ElevenLabs uses opaque ids rather than named buckets:
//!
//! - [`SpeechBatch::model`] — e.g. `eleven_multilingual_v2`,
//!   `eleven_turbo_v2_5`, `eleven_flash_v2_5`.
//! - [`SpeechBatch::voice`] — a [`VoiceRef::Id`] carrying the
//!   21-character ElevenLabs voice id (e.g. `21m00Tcm4TlvDq8ikWAM` for
//!   "Rachel"). [`VoiceRef::Named`] is forwarded verbatim. A
//!   [`VoiceRef::ClonedFrom`] payload routes through
//!   `ElevenLabsTtsRunner::clone_voice` — the cloning multipart
//!   upload path against `/v1/voices/add`.
//!
//! ## Output shape
//!
//! The HTTPS path materialises the full audio body and re-chunks it at
//! `ElevenLabsTtsConfig::chunk_bytes` boundaries before emitting
//! [`SpeechChunk`]s. The terminal chunk carries `is_final = true`.
//!
//! The WS streaming path emits one [`SpeechChunk`] per inbound JSON
//! frame, attaching an [`AlignmentDelta`] (with per-character
//! [`WordTiming`]s) to the chunk when the provider includes one.
//!
//! ## Source
//!
//! `FR-TTS-001`. See [`docs/audio-modalities.md`](../../docs/audio-modalities.md).
//!
//! [`SpeechRunner::speak`]: atomr_infer_core::runner::SpeechRunner::speak
//! [`SpeechBatch::model`]: atomr_infer_core::audio::SpeechBatch::model
//! [`SpeechBatch::voice`]: atomr_infer_core::audio::SpeechBatch::voice
//! [`VoiceRef::Id`]: atomr_infer_core::audio::VoiceRef::Id
//! [`VoiceRef::Named`]: atomr_infer_core::audio::VoiceRef::Named
//! [`VoiceRef::ClonedFrom`]: atomr_infer_core::audio::VoiceRef::ClonedFrom
//! [`SpeechChunk`]: atomr_infer_core::audio::SpeechChunk
//! [`AlignmentDelta`]: atomr_infer_core::audio::AlignmentDelta
//! [`WordTiming`]: atomr_infer_core::audio::WordTiming

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
mod runner;
#[cfg(feature = "tts-elevenlabs")]
mod wire;

pub use config::{ElevenLabsSecret, ElevenLabsTtsConfig};
pub use cost::{estimate_usd, per_million_chars_usd};
pub use error::ElevenLabsError;
pub use runner::ElevenLabsTtsRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::SpeechRunner;
    use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

    #[cfg(not(feature = "tts-elevenlabs"))]
    fn runner() -> ElevenLabsTtsRunner {
        ElevenLabsTtsRunner::new_stub()
    }

    #[cfg(feature = "tts-elevenlabs")]
    fn runner() -> ElevenLabsTtsRunner {
        use arc_swap::ArcSwap;
        use atomr_infer_remote_core::http::build_client;
        use atomr_infer_remote_core::session::SessionSnapshot;
        use secrecy::SecretString;
        use std::sync::Arc;

        let cfg = ElevenLabsTtsConfig::defaults_for_elevenlabs(ElevenLabsSecret::Env {
            name: "ELEVEN_API_KEY".into(),
        });
        let client = build_client(&Default::default(), "test/0").expect("build client");
        let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
            client,
            credential: SecretString::from("sk-fake".to_string()),
        }));
        ElevenLabsTtsRunner::new(cfg, snap).expect("construct runner")
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::TextToSpeech);
        assert_eq!(
            r.transport_kind(),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::ElevenLabs,
            }
        );
    }

    #[cfg(not(feature = "tts-elevenlabs"))]
    #[tokio::test]
    async fn speak_without_feature_returns_internal_error() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "eleven_turbo_v2_5".into(),
            text: "hi".into(),
            voice: VoiceRef::Id("21m00Tcm4TlvDq8ikWAM".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 2,
        };
        match r.speak(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("tts-elevenlabs feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }
}
