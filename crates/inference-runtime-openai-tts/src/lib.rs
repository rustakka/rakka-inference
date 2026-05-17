//! # inference-runtime-openai-tts
//!
//! OpenAI text-to-speech runtime for `atomr-infer`. Implements
//! [`atomr_infer_core::runner::SpeechRunner`] against
//! `POST /v1/audio/speech`.
//!
//! ## Build profiles
//!
//! | Build                                                                  | Result                                                |
//! |------------------------------------------------------------------------|-------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-openai-tts`                        | Stub — `speak` returns `Internal("tts-openai feature disabled at build time")`. |
//! | `cargo build -p atomr-infer-runtime-openai-tts --features tts-openai`  | Real path — HTTPS POST + chunked PCM streaming.       |
//!
//! ## Output shape
//!
//! [`OpenAiTtsRunner`]'s [`SpeechRunner::speak`] implementation
//! streams a sequence of
//! [`atomr_infer_core::audio::SpeechChunk`]s. With the default
//! `response_format=pcm` each chunk carries 24 kHz mono signed
//! 16-bit LE PCM bytes. With `mp3` / `wav` / `opus` / `flac` the
//! container header rides at the head of the first chunk; consumers
//! must reassemble the container before decoding.
//!
//! The terminal chunk has `is_final = true`.
//!
//! ## Source
//!
//! `FR-TTS-001`. See [`docs/audio-modalities.md`](../../docs/audio-modalities.md).
//!
//! [`SpeechRunner::speak`]: atomr_infer_core::runner::SpeechRunner::speak

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
mod runner;
#[cfg(feature = "tts-openai")]
mod wire;

#[cfg(feature = "tts-openai")]
pub use config::OpenAiTtsConfig;
#[cfg(not(feature = "tts-openai"))]
pub use config::OpenAiTtsConfig;
pub use cost::{estimate_usd, per_million_chars_usd};
pub use error::OpenAiTtsError;
pub use runner::OpenAiTtsRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::SpeechRunner;
    use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

    #[cfg(not(feature = "tts-openai"))]
    fn runner() -> OpenAiTtsRunner {
        OpenAiTtsRunner::new_stub()
    }

    #[cfg(feature = "tts-openai")]
    fn runner() -> OpenAiTtsRunner {
        use arc_swap::ArcSwap;
        use atomr_infer_remote_core::http::build_client;
        use atomr_infer_remote_core::session::SessionSnapshot;
        use atomr_infer_runtime_openai::config::SecretRef;
        use secrecy::SecretString;
        use std::sync::Arc;

        let cfg = OpenAiTtsConfig::defaults_for_openai(SecretRef::Env {
            name: "OPENAI_API_KEY".into(),
        });
        let client = build_client(&Default::default(), "test/0").expect("build client");
        let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
            client,
            credential: SecretString::from("sk-test".to_string()),
        }));
        OpenAiTtsRunner::new(cfg, snap).expect("construct runner")
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::TextToSpeech);
        assert_eq!(
            r.transport_kind(),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::OpenAi,
            }
        );
    }

    #[cfg(not(feature = "tts-openai"))]
    #[tokio::test]
    async fn speak_without_feature_returns_internal_error() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "tts-1".into(),
            text: "hi".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 2,
        };
        match r.speak(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("tts-openai feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }
}
