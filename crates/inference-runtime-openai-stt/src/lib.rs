//! # inference-runtime-openai-stt
//!
//! OpenAI speech-to-text runtime for `atomr-infer`. Implements
//! [`atomr_infer_core::runner::AudioRunner`] against
//! `POST /v1/audio/transcriptions`.
//!
//! ## Build profiles
//!
//! | Build                                                                  | Result                                                |
//! |------------------------------------------------------------------------|-------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-openai-stt`                        | Stub — `execute_audio` returns `Internal("stt-openai feature disabled at build time")`. |
//! | `cargo build -p atomr-infer-runtime-openai-stt --features stt-openai`  | Real path — multipart upload + JSON/verbose-JSON.      |
//!
//! ## Response shape
//!
//! [`OpenAiSttRunner`]'s [`AudioRunner::execute_audio`] implementation
//! emits a stream of
//! [`atomr_infer_core::audio::TranscriptChunk`]s. By default the
//! runner asks the API for `response_format=json` and emits one
//! terminal chunk for the full transcript. When the caller sets
//! [`word_timestamps`] or [`interim_results`] on
//! [`atomr_infer_core::audio::TranscribeOptions`] (either implies the
//! verbose envelope), the runner asks for `verbose_json` and emits one
//! chunk per OpenAI segment, with word timings attached when requested.
//!
//! The terminal chunk carries `is_final = true`.
//!
//! ## Source
//!
//! `FR-STT-001`. See [`docs/audio-modalities.md`](../../docs/audio-modalities.md).
//!
//! [`AudioRunner::execute_audio`]: atomr_infer_core::runner::AudioRunner::execute_audio
//! [`word_timestamps`]: atomr_infer_core::audio::TranscribeOptions::word_timestamps
//! [`interim_results`]: atomr_infer_core::audio::TranscribeOptions::interim_results

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
mod runner;
#[cfg(feature = "stt-openai")]
mod wire;

pub use config::OpenAiSttConfig;
pub use cost::{estimate_usd, per_minute_usd};
pub use error::OpenAiSttError;
pub use runner::OpenAiSttRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::AudioRunner;
    use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

    #[cfg(not(feature = "stt-openai"))]
    fn runner() -> OpenAiSttRunner {
        OpenAiSttRunner::new_stub()
    }

    #[cfg(feature = "stt-openai")]
    fn runner() -> OpenAiSttRunner {
        use arc_swap::ArcSwap;
        use atomr_infer_remote_core::http::build_client;
        use atomr_infer_remote_core::session::SessionSnapshot;
        use atomr_infer_runtime_openai::config::SecretRef;
        use secrecy::SecretString;
        use std::sync::Arc;

        let cfg = OpenAiSttConfig::defaults_for_openai(SecretRef::Env {
            name: "OPENAI_API_KEY".into(),
        });
        let client = build_client(&Default::default(), "test/0").expect("build client");
        let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
            client,
            credential: SecretString::from("sk-test".to_string()),
        }));
        OpenAiSttRunner::new(cfg, snap).expect("construct runner")
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::SpeechToText);
        assert_eq!(
            r.transport_kind(),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::OpenAi,
            }
        );
    }

    #[cfg(not(feature = "stt-openai"))]
    #[tokio::test]
    async fn execute_audio_without_feature_returns_internal_error() {
        use atomr_infer_core::audio::{
            AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscribeOptions,
        };
        use atomr_infer_core::error::InferenceError;
        use bytes::Bytes;

        let mut r = runner();
        let batch = AudioBatch {
            request_id: "t".into(),
            model: "whisper-1".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from_static(&[0u8; 4]),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: false,
            options: AudioOptions::Transcribe(TranscribeOptions::default()),
            estimated_units: 1,
        };
        match r.execute_audio(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("stt-openai feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }
}
