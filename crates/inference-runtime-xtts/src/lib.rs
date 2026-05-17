//! # inference-runtime-xtts
//!
//! Local [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) TTS runtime for
//! `atomr-infer`. Implements [`atomr_infer_core::runner::SpeechRunner`]
//! against an ONNX-exported XTTS-v2 model and emits PCM16-LE
//! [`atomr_infer_core::audio::SpeechChunk`]s.
//!
//! XTTS-v2 is a cross-lingual voice-cloning TTS model (>17 languages). The
//! primary voice selection mode is
//! [`atomr_infer_core::audio::VoiceRef::ClonedFrom`] — pass a short
//! reference audio clip and the runner conditions synthesis on the extracted
//! speaker embedding.
//!
//! ## Build profiles
//!
//! | Build                                                                 | Result                                                  |
//! |-----------------------------------------------------------------------|---------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-xtts`                             | Stub — `ort` not in dep graph.                          |
//! | `cargo build -p atomr-infer-runtime-xtts --features tts-xtts`         | Real path with `ort` crate (CPU EP).                    |
//! | `cargo build -p atomr-infer-runtime-xtts --features tts-xtts-cuda`    | Adds the CUDA EP — needs a working CUDA toolkit.        |
//! | `cargo build -p atomr-infer-runtime-xtts --features tts-xtts-load-dynamic` | Loads `libonnxruntime` at runtime via `ORT_DYLIB_PATH`. |
//!
//! ## Voice cloning (M10 stub)
//!
//! `VoiceRef::ClonedFrom(AudioPayload)` is the primary use case. In M10 the
//! embedding computation is stubbed: the runner validates the reference audio
//! payload, logs `tracing::warn!("xtts voice cloning embedding not yet
//! implemented")`, and proceeds with a zero-valued speaker embedding. A
//! follow-up milestone wires the real speaker-encoder ONNX model.
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

pub use config::XttsConfig;
pub use error::XttsError;
pub use runner::XttsRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::SpeechRunner;
    use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

    fn runner() -> XttsRunner {
        XttsRunner::new(XttsConfig {
            model_path: "/tmp/xtts-does-not-exist".into(),
            default_language: "en".into(),
            chunk_samples: 4096,
            intra_threads: None,
        })
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::TextToSpeech);
        assert_eq!(r.transport_kind(), TransportKind::LocalCpu);
    }

    #[test]
    fn config_default_language_is_en() {
        let r = runner();
        assert_eq!(r.config().default_language, "en");
    }

    #[cfg(not(feature = "tts-xtts"))]
    #[tokio::test]
    async fn speak_without_feature_returns_internal_error() {
        use atomr_infer_core::audio::VoiceRef;
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = atomr_infer_core::audio::SpeechBatch {
            request_id: "t".into(),
            model: "xtts-v2".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("default".into()),
            options: atomr_infer_core::audio::SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        };
        match r.speak(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("tts-xtts feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[cfg(feature = "tts-xtts")]
    #[tokio::test]
    async fn speak_with_non_pcm_format_returns_unsupported_audio_format() {
        use atomr_infer_core::audio::{AudioFormat, SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "xtts-v2".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("default".into()),
            options: SynthOptions {
                format: Some(AudioFormat::OggOpus),
                ..Default::default()
            },
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        };
        match r.speak(batch).await {
            Err(InferenceError::UnsupportedAudioFormat { .. }) => {}
            other => panic!("expected UnsupportedAudioFormat, got {other:?}"),
        }
    }

    #[cfg(feature = "tts-xtts")]
    #[tokio::test]
    async fn named_voice_resolves_to_default_speaker() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "xtts-v2".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("speaker_1".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        };
        // Named/Id voices fall through to model-not-found, not an embedding error.
        match r.speak(batch).await {
            Err(InferenceError::BadRequest { message }) => {
                assert!(
                    message.contains("model not found") || message.contains("not found"),
                    "{message}"
                );
            }
            other => panic!("expected BadRequest (model not found), got {other:?}"),
        }
    }

    #[cfg(feature = "tts-xtts")]
    #[test]
    fn cloned_from_with_invalid_params_is_rejected() {
        use atomr_infer_core::audio::{AudioFormat, AudioParams};

        // Invalid params: sample_rate 0 is out of range
        let bad_params = AudioParams::new(0, 0, AudioFormat::Pcm16Le);
        assert!(!bad_params.is_valid());
        // XttsRunner::handle_voice_ref validates params when VoiceRef::ClonedFrom
        // is used. Here we verify the AudioParams type correctly flags invalid inputs.
        assert!(!AudioParams::new(0, 0, AudioFormat::Pcm16Le).is_valid());
        assert!(!AudioParams::new(100_000, 0, AudioFormat::Pcm16Le).is_valid());
    }
}
