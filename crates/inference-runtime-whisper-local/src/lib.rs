//! Local speech-to-text via [whisper.cpp](https://github.com/ggerganov/whisper.cpp).
//!
//! Implements [`AudioRunner`](atomr_infer_core::runner::AudioRunner)
//! against a ggml-format whisper model file. Reports
//! [`TransportKind::LocalCpu`](atomr_infer_core::runtime::TransportKind::LocalCpu)
//! so the placement layer never asks for a GPU ordinal — accelerator
//! backends (CUDA, Metal, Vulkan, CoreML) are opt-in cargo features
//! that toggle the underlying `whisper-rs` flags transparently.
//!
//! # Build profile
//!
//! | feature                  | enables                                                    |
//! |--------------------------|------------------------------------------------------------|
//! | (none)                   | crate compiles; runner returns [`InferenceError::Internal`] |
//! | `stt-whisper`            | real whisper.cpp inference on `x86_64` / `aarch64` hosts   |
//! | `stt-whisper-cuda`       | forwards `whisper-rs/cuda`                                 |
//! | `stt-whisper-metal`      | forwards `whisper-rs/metal`                                |
//! | `stt-whisper-coreml`     | forwards `whisper-rs/coreml`                               |
//! | `stt-whisper-vulkan`     | forwards `whisper-rs/vulkan`                               |
//! | `stt-whisper-openblas`   | forwards `whisper-rs/openblas`                             |
//!
//! On hosts whose `target_arch` is neither `x86_64` nor `aarch64`, the
//! runner returns [`InferenceError::Unsupported`] from
//! [`AudioRunner::execute_audio`]. The rest of the workspace still
//! type-checks against this crate on every arch — important for the
//! cross-arch CI matrix.
//!
//! [`InferenceError::Internal`]: atomr_infer_core::error::InferenceError::Internal
//! [`InferenceError::Unsupported`]: atomr_infer_core::error::InferenceError::Unsupported
//! [`AudioRunner::execute_audio`]: atomr_infer_core::runner::AudioRunner::execute_audio
//!
//! Source: `FR-STT-001`.

pub mod audio_decode;
pub mod config;
pub mod error;
pub mod runner;

#[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod session;

pub use config::WhisperConfig;
pub use error::WhisperError;
pub use runner::WhisperRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::audio::{
        AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscribeOptions,
    };
    use atomr_infer_core::error::InferenceError;
    use atomr_infer_core::runner::AudioRunner;
    use atomr_infer_core::runtime::{RuntimeKind, TransportKind};
    use bytes::Bytes;
    use std::path::PathBuf;

    fn batch_with_pcm16(samples: Vec<u8>) -> AudioBatch {
        AudioBatch {
            request_id: "req-1".into(),
            model: "whisper-1".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from(samples),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: false,
            options: AudioOptions::Transcribe(TranscribeOptions::default()),
            estimated_units: 1,
        }
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let runner = WhisperRunner::new(WhisperConfig {
            model_path: PathBuf::from("/does/not/exist/ggml.bin"),
            ..Default::default()
        });
        assert_eq!(runner.runtime_kind(), RuntimeKind::SpeechToText);
        assert_eq!(runner.transport_kind(), TransportKind::LocalCpu);
    }

    #[tokio::test]
    async fn rejects_a2f_options() {
        use atomr_infer_core::audio::A2FOptions;
        let mut runner = WhisperRunner::new(WhisperConfig::default());
        let batch = AudioBatch {
            request_id: "r".into(),
            model: "whisper".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::new(),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: false,
            options: AudioOptions::Audio2Face(A2FOptions::default()),
            estimated_units: 1,
        };
        let err = runner.execute_audio(batch).await.unwrap_err();
        assert!(
            matches!(err, InferenceError::BadRequest { ref message } if message.contains("A2F")),
            "expected BadRequest with A2F message, got {err:?}"
        );
    }

    #[tokio::test]
    async fn rejects_streaming_input() {
        let mut runner = WhisperRunner::new(WhisperConfig::default());
        let (_tx, rx) = tokio::sync::mpsc::channel::<Bytes>(1);
        let batch = AudioBatch {
            request_id: "r".into(),
            model: "whisper".into(),
            input: AudioInput::Stream {
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
                rx,
            },
            stream: true,
            options: AudioOptions::Transcribe(TranscribeOptions::default()),
            estimated_units: 1,
        };
        let err = runner.execute_audio(batch).await.unwrap_err();
        match err {
            InferenceError::Unsupported { method, runtime } => {
                assert!(
                    method.contains("execute_audio"),
                    "expected execute_audio in method name, got {method:?}"
                );
                assert_eq!(runtime, RuntimeKind::SpeechToText);
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[cfg(not(feature = "stt-whisper"))]
    #[tokio::test]
    async fn execute_audio_without_feature_returns_internal() {
        let mut runner = WhisperRunner::new(WhisperConfig::default());
        let err = runner
            .execute_audio(batch_with_pcm16(vec![0u8; 4]))
            .await
            .unwrap_err();
        match err {
            InferenceError::Internal(message) => {
                assert!(message.contains("stt-whisper feature disabled"));
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    // Arch-gate test: on hosts that aren't x86_64 / aarch64, even with
    // the feature on, the runner should return Unsupported instead of
    // attempting to call whisper.cpp.
    #[cfg(all(
        feature = "stt-whisper",
        not(any(target_arch = "x86_64", target_arch = "aarch64"))
    ))]
    #[tokio::test]
    async fn execute_audio_on_unsupported_arch_returns_unsupported() {
        let mut runner = WhisperRunner::new(WhisperConfig::default());
        let err = runner
            .execute_audio(batch_with_pcm16(vec![0u8; 4]))
            .await
            .unwrap_err();
        assert!(matches!(err, InferenceError::Unsupported { .. }));
    }

    #[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[tokio::test]
    async fn execute_audio_with_feature_but_missing_model_returns_bad_request() {
        let mut runner = WhisperRunner::new(WhisperConfig {
            model_path: PathBuf::from("/does/not/exist/ggml.bin"),
            ..Default::default()
        });
        let err = runner
            .execute_audio(batch_with_pcm16(vec![0u8; 4]))
            .await
            .unwrap_err();
        assert!(
            matches!(err, InferenceError::BadRequest { .. }),
            "expected BadRequest for missing model, got {err:?}"
        );
    }
}
