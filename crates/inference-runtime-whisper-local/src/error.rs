//! `WhisperError` — the crate-local error surface. Maps to
//! [`atomr_infer_core::error::InferenceError`] at the trait boundary.

use std::path::PathBuf;

use atomr_infer_core::error::InferenceError;

/// Crate-local error type. Each variant maps to a specific
/// [`InferenceError`] arm at the [`AudioRunner`](atomr_infer_core::runner::AudioRunner)
/// boundary via [`From<WhisperError> for InferenceError`].
#[derive(Debug, thiserror::Error)]
pub enum WhisperError {
    /// The configured model file does not exist.
    #[error("whisper: model file not found at {path}")]
    ModelNotFound { path: PathBuf },

    /// The configured model file exists but cannot be opened/read.
    #[error("whisper: model io error at {path}: {source}")]
    ModelIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Audio is not in a format the runner can process directly.
    #[error("whisper: unsupported audio format ({message})")]
    UnsupportedAudio { message: String },

    /// `AudioInput::Stream` was passed to `execute_audio`. Streaming
    /// STT against whisper.cpp requires a chunking VAD layer that lives
    /// upstream of this crate; M5 ships static-only.
    #[error("whisper: streaming AudioInput not supported by this runtime (use AudioInput::Static)")]
    StreamingNotSupported,

    /// Underlying `whisper-rs` call failed.
    #[error("whisper: backend error: {0}")]
    Backend(String),

    /// The crate was compiled without the `stt-whisper` feature.
    #[error("stt-whisper feature disabled at build time — rebuild with --features stt-whisper")]
    FeatureDisabled,

    /// The host architecture isn't in the supported set.
    #[error("whisper: unsupported host architecture ({arch})")]
    UnsupportedArch { arch: &'static str },
}

impl From<WhisperError> for InferenceError {
    fn from(e: WhisperError) -> Self {
        use atomr_infer_core::runtime::RuntimeKind;
        match e {
            WhisperError::ModelNotFound { .. } | WhisperError::ModelIo { .. } => InferenceError::BadRequest {
                message: e.to_string(),
            },
            WhisperError::UnsupportedAudio { message } => InferenceError::UnsupportedAudioFormat { message },
            WhisperError::StreamingNotSupported => InferenceError::Unsupported {
                method: "execute_audio".into(),
                runtime: RuntimeKind::SpeechToText,
            },
            WhisperError::UnsupportedArch { arch } => InferenceError::Unsupported {
                method: "execute_audio".into(),
                runtime: RuntimeKind::SpeechToText,
            }
            .with_arch_hint(arch),
            WhisperError::Backend(_) | WhisperError::FeatureDisabled => {
                InferenceError::Internal(e.to_string())
            }
        }
    }
}

/// Helper extension so we can decorate `Unsupported` with the offending
/// arch name without changing the public [`InferenceError`] surface.
trait UnsupportedArchHint {
    fn with_arch_hint(self, arch: &str) -> Self;
}

impl UnsupportedArchHint for InferenceError {
    fn with_arch_hint(self, arch: &str) -> Self {
        match self {
            InferenceError::Unsupported { method, runtime } => InferenceError::Unsupported {
                method: format!(
                    "{method} (host arch `{arch}` is not in the supported set: linux-x86_64, linux-aarch64)"
                ),
                runtime,
            },
            other => other,
        }
    }
}
