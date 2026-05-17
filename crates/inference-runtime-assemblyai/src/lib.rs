//! # inference-runtime-assemblyai
//!
//! AssemblyAI speech-to-text runtime for `atomr-infer`. Implements
//! [`atomr_infer_core::runner::AudioRunner`] against
//! `WSS /v3/ws` (the Universal-Streaming v3 protocol) via the shared
//! `atomr_infer_runtime_ws_core` transport (available only when the
//! `stt-assemblyai` feature is on).
//!
//! ## Build profiles
//!
//! | Build                                                                       | Result                                                |
//! |-----------------------------------------------------------------------------|-------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-assemblyai`                             | Stub — `execute_audio` returns `Internal("stt-assemblyai feature disabled at build time")`. |
//! | `cargo build -p atomr-infer-runtime-assemblyai --features stt-assemblyai`   | Real path — WSS streaming + per-turn partial / final progression. |
//!
//! ## Partial vs final transcripts
//!
//! AssemblyAI v3 emits `Turn` envelopes on the downlink:
//!
//! - **Partial** (`end_of_turn = false`): rolling text-so-far for the
//!   in-flight turn. Per-token `word_is_final` separates the stable
//!   prefix from the unstable suffix.
//! - **Final** (`end_of_turn = true`): the turn-final update —
//!   exactly one per spoken turn.
//!
//! The runner emits one
//! [`atomr_infer_core::audio::TranscriptChunk`] per inbound `Turn`
//! envelope. Partial chunks are dropped when
//! [`atomr_infer_core::audio::TranscribeOptions::interim_results`] is
//! `false`. The chunk's `is_final` field follows `end_of_turn` — turn-
//! final, with no Deepgram-style segment-final vs utterance-final
//! distinction.
//!
//! Compare with the sibling Deepgram runtime: both emit turn-shaped
//! chunks here, but their cadence differs — AssemblyAI delivers many
//! partials and exactly one final per turn; Deepgram delivers
//! interims and may finalise multiple segments before `speech_final`.
//!
//! ## Audio format constraints
//!
//! AssemblyAI v3 only accepts 16 kHz / 16-bit / mono PCM on the
//! uplink. The runner rejects any other [`atomr_infer_core::audio::AudioFormat`]
//! up front with [`atomr_infer_core::error::InferenceError::UnsupportedAudioFormat`]
//! so callers don't waste a connect. Resample upstream if your
//! source audio is anything else.
//!
//! ## Punctuation + formatting
//!
//! Set `AssemblyAiSttConfig::format_turns` to `true` to ask the
//! provider for Punctuated & Formatted text on the turn-final
//! update. Off by default because it adds latency to the turn-final
//! emission.
//!
//! ## Source
//!
//! `FR-STT-001`. See [`docs/audio-modalities.md`](../../docs/audio-modalities.md).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
mod runner;
#[cfg(feature = "stt-assemblyai")]
mod wire;

pub use config::{AssemblyAiSecret, AssemblyAiSttConfig};
pub use cost::{estimate_usd, per_hour_usd};
pub use error::AssemblyAiError;
pub use runner::AssemblyAiSttRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::AudioRunner;
    use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

    #[cfg(not(feature = "stt-assemblyai"))]
    fn runner() -> AssemblyAiSttRunner {
        AssemblyAiSttRunner::new_stub()
    }

    #[cfg(feature = "stt-assemblyai")]
    fn runner() -> AssemblyAiSttRunner {
        use arc_swap::ArcSwap;
        use atomr_infer_remote_core::http::build_client;
        use atomr_infer_remote_core::session::SessionSnapshot;
        use secrecy::SecretString;
        use std::sync::Arc;

        let cfg = AssemblyAiSttConfig::defaults_for_assemblyai(AssemblyAiSecret::Env {
            name: "ASSEMBLYAI_API_KEY".into(),
        });
        let client = build_client(&Default::default(), "test/0").expect("build client");
        let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
            client,
            credential: SecretString::from("aa-fake".to_string()),
        }));
        AssemblyAiSttRunner::new(cfg, snap).expect("construct runner")
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::SpeechToText);
        assert_eq!(
            r.transport_kind(),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::AssemblyAi,
            }
        );
    }

    #[cfg(not(feature = "stt-assemblyai"))]
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
            model: "universal".into(),
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
                assert!(msg.contains("stt-assemblyai feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }
}
