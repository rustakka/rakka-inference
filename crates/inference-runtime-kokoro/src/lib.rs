//! # inference-runtime-kokoro
//!
//! Local [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS runtime
//! for `atomr-infer`. Implements
//! [`atomr_infer_core::runner::SpeechRunner`] against a pre-converted Kokoro
//! ONNX voice pack (`.onnx`).
//!
//! ## Build profiles
//!
//! | Build                                                                      | Result                                                  |
//! |----------------------------------------------------------------------------|---------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-kokoro`                                | Stub — `ort` not in dep graph.                          |
//! | `cargo build -p atomr-infer-runtime-kokoro --features tts-kokoro`          | Real path with `ort` crate (CPU EP).                    |
//! | `cargo build -p atomr-infer-runtime-kokoro --features tts-kokoro-cuda`     | Adds the CUDA EP — needs a working CUDA toolkit.        |
//! | `cargo build -p atomr-infer-runtime-kokoro --features tts-kokoro-load-dynamic` | Loads `libonnxruntime` at runtime via `ORT_DYLIB_PATH`. |
//!
//! ## Voice packs
//!
//! Kokoro-82M ships weights as PyTorch `.pt` files. Pre-convert them to ONNX
//! and place the resulting `.onnx` files in `KokoroConfig::voice_pack_dir`.
//! Set `KokoroConfig::default_voice` to the filename stem (without `.onnx`).
//!
//! ## Output shape
//!
//! `KokoroRunner::speak` streams [`atomr_infer_core::audio::SpeechChunk`]s
//! carrying PCM16-LE mono audio at 24 000 Hz (Kokoro's native sample rate).
//! Chunk size is configurable via [`KokoroConfig::chunk_samples`] (default
//! 4096 — about 170 ms at 24 000 Hz).
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

pub use config::KokoroConfig;
pub use error::KokoroError;
pub use runner::KokoroRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::SpeechRunner;
    use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

    fn runner() -> KokoroRunner {
        KokoroRunner::new(KokoroConfig {
            voice_pack_dir: "/tmp/kokoro-voices".into(),
            default_voice: "af_heart".into(),
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
    fn empty_voice_name_is_caught_at_validation() {
        let r = KokoroRunner::new(KokoroConfig {
            voice_pack_dir: "/tmp".into(),
            default_voice: String::new(),
            chunk_samples: 4096,
            intra_threads: None,
        });
        // We can inspect the config but validation only fires inside speak().
        assert!(r.config().default_voice.is_empty());
    }

    #[cfg(not(feature = "tts-kokoro"))]
    #[tokio::test]
    async fn speak_without_feature_returns_internal_error() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "kokoro-82m".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("af_heart".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        };
        match r.speak(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("tts-kokoro feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[cfg(feature = "tts-kokoro")]
    #[tokio::test]
    async fn speak_with_non_pcm_format_returns_unsupported_audio_format() {
        use atomr_infer_core::audio::{AudioFormat, SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "kokoro-82m".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("af_heart".into()),
            options: SynthOptions {
                format: Some(AudioFormat::Mp3),
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

    #[cfg(feature = "tts-kokoro")]
    #[tokio::test]
    async fn speak_with_missing_voice_returns_bad_request() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "kokoro-82m".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("af_heart".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        };
        match r.speak(batch).await {
            Err(InferenceError::BadRequest { message }) => {
                assert!(message.contains("voice file not found"), "{message}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }
}
