//! # inference-runtime-deepgram
//!
//! Deepgram speech-to-text runtime for `atomr-infer`. Implements
//! [`atomr_infer_core::runner::AudioRunner`] against
//! `WSS /v1/listen` via the shared
//! `atomr_infer_runtime_ws_core` transport (available only when the
//! `stt-deepgram` feature is on).
//!
//! ## Build profiles
//!
//! | Build                                                                       | Result                                                |
//! |-----------------------------------------------------------------------------|-------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-deepgram`                               | Stub — `execute_audio` returns `Internal("stt-deepgram feature disabled at build time")`. |
//! | `cargo build -p atomr-infer-runtime-deepgram --features stt-deepgram`       | Real path — WSS streaming + interim/final progression. |
//!
//! ## Interim vs final transcripts
//!
//! Deepgram emits two granularities of transcript on the downlink:
//!
//! - **Interim** (`is_final = false`): mid-utterance edits — the
//!   provider may revise an interim transcript before finalising it.
//! - **Final** (`is_final = true`): a finalised *segment*. Within one
//!   utterance Deepgram may finalise several segments before the
//!   speaker stops.
//! - **Speech-final** (`speech_final = true`): the provider's VAD has
//!   detected end-of-utterance.
//!
//! The runner emits one
//! [`atomr_infer_core::audio::TranscriptChunk`] per inbound `Results`
//! envelope. Interim chunks are dropped when
//! [`atomr_infer_core::audio::TranscribeOptions::interim_results`] is
//! `false`. The chunk's `is_final` field follows `speech_final` —
//! turn-final, not segment-final.
//!
//! Compare with the sibling AssemblyAI runtime: both emit
//! turn-shaped chunks here, but their interim cadence differs —
//! AssemblyAI delivers many partials and exactly one final per
//! turn; Deepgram delivers interims and may finalise multiple
//! segments before `speech_final`.
//!
//! ## Voice activity / endpointing
//!
//! The runner appends `endpointing=300` on every connect so the
//! provider's VAD declares end-of-utterance after ~300 ms of
//! silence. `speech_final` will not arrive without endpointing on.
//!
//! ## Diarization + word timestamps
//!
//! Set [`atomr_infer_core::audio::TranscribeOptions::diarize`] for
//! per-word speaker labels and
//! [`atomr_infer_core::audio::TranscribeOptions::word_timestamps`] to
//! receive per-word timing in
//! [`atomr_infer_core::audio::TranscriptChunk::words`].
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
#[cfg(feature = "stt-deepgram")]
mod wire;

pub use config::{DeepgramSecret, DeepgramSttConfig};
pub use cost::{estimate_usd, per_minute_usd};
pub use error::DeepgramError;
pub use runner::DeepgramSttRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::AudioRunner;
    use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

    #[cfg(not(feature = "stt-deepgram"))]
    fn runner() -> DeepgramSttRunner {
        DeepgramSttRunner::new_stub()
    }

    #[cfg(feature = "stt-deepgram")]
    fn runner() -> DeepgramSttRunner {
        use arc_swap::ArcSwap;
        use atomr_infer_remote_core::http::build_client;
        use atomr_infer_remote_core::session::SessionSnapshot;
        use secrecy::SecretString;
        use std::sync::Arc;

        let cfg = DeepgramSttConfig::defaults_for_deepgram(DeepgramSecret::Env {
            name: "DEEPGRAM_API_KEY".into(),
        });
        let client = build_client(&Default::default(), "test/0").expect("build client");
        let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
            client,
            credential: SecretString::from("dg-fake".to_string()),
        }));
        DeepgramSttRunner::new(cfg, snap).expect("construct runner")
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::SpeechToText);
        assert_eq!(
            r.transport_kind(),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::Deepgram,
            }
        );
    }

    #[cfg(not(feature = "stt-deepgram"))]
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
            model: "nova-2".into(),
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
                assert!(msg.contains("stt-deepgram feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }
}
