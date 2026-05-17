//! # inference-runtime-moss
//!
//! Local [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD) TTS runtime for
//! `atomr-infer`. Implements [`atomr_infer_core::runner::SpeechRunner`]
//! against a MOSS transformer-based TTS model and emits PCM16-LE
//! [`atomr_infer_core::audio::SpeechChunk`]s.
//!
//! **Linux-only.** On non-Linux platforms the feature compiles cleanly but
//! `speak` returns `InferenceError::Internal("tts-moss requires Linux")`.
//!
//! ## Build profiles
//!
//! | Build                                                              | Result                                                  |
//! |--------------------------------------------------------------------|---------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-moss`                          | Stub — feature disabled.                                |
//! | `cargo build -p atomr-infer-runtime-moss --features tts-moss`      | Linux: real path. Other OS: `requires Linux` error.     |
//!
//! ## Source
//!
//! `FR-TTS-001`. See `docs/audio-modalities.md`.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
mod runner;
pub mod wire;

pub use config::MossConfig;
pub use error::MossError;
pub use runner::MossRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::SpeechRunner;
    use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

    fn runner() -> MossRunner {
        MossRunner::new(MossConfig {
            model_dir: "/tmp/moss-does-not-exist".into(),
            default_voice: "default".into(),
            chunk_samples: 4096,
        })
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::TextToSpeech);
        assert_eq!(r.transport_kind(), TransportKind::LocalCpu);
    }

    #[cfg(not(feature = "tts-moss"))]
    #[tokio::test]
    async fn speak_without_feature_returns_internal_error() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "moss-tts".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("default".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        };
        match r.speak(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("tts-moss feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    // On non-Linux platforms with the feature on, speak must return the
    // Linux-required error.
    #[cfg(all(feature = "tts-moss", not(target_os = "linux")))]
    #[tokio::test]
    async fn non_linux_returns_unsupported() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "moss-tts".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("default".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        };
        match r.speak(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("requires Linux"), "{msg}");
            }
            other => panic!("expected Internal(requires Linux), got {other:?}"),
        }
    }

    #[cfg(all(feature = "tts-moss", target_os = "linux"))]
    #[tokio::test]
    async fn speak_on_linux_with_missing_model_returns_bad_request() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner(); // model_dir points to /tmp/moss-does-not-exist
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "moss-tts".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("default".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        };
        match r.speak(batch).await {
            Err(InferenceError::BadRequest { message }) => {
                assert!(message.contains("model directory not found"), "{message}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }
}
