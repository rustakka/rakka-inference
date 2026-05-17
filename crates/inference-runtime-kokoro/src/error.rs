//! Crate-level error type. Provider crates surface these via
//! `InferenceError::Internal` / `InferenceError::BadRequest` from
//! the [`crate::KokoroRunner`] entry points.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KokoroError {
    #[error("kokoro voice file not found: {path}")]
    VoiceNotFound { path: PathBuf },

    #[error("kokoro voice name is empty")]
    EmptyVoiceName,

    #[error("kokoro io: {0}")]
    Io(#[from] io::Error),

    #[error("kokoro ort error: {0}")]
    Ort(String),

    #[error("kokoro unsupported format: {message}")]
    UnsupportedFormat { message: String },

    #[error("kokoro feature disabled at build time — rebuild with --features tts-kokoro")]
    FeatureDisabled,
}

impl From<KokoroError> for atomr_infer_core::error::InferenceError {
    fn from(e: KokoroError) -> Self {
        use atomr_infer_core::error::InferenceError;
        match e {
            KokoroError::VoiceNotFound { .. } | KokoroError::EmptyVoiceName => InferenceError::BadRequest {
                message: e.to_string(),
            },
            KokoroError::UnsupportedFormat { message } => InferenceError::UnsupportedAudioFormat { message },
            KokoroError::Io(_) | KokoroError::Ort(_) => InferenceError::Internal(e.to_string()),
            KokoroError::FeatureDisabled => InferenceError::Internal(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::error::InferenceError;

    #[test]
    fn feature_disabled_maps_to_internal() {
        let e: InferenceError = KokoroError::FeatureDisabled.into();
        assert!(matches!(e, InferenceError::Internal(_)));
    }

    #[test]
    fn voice_not_found_maps_to_bad_request() {
        let e: InferenceError = KokoroError::VoiceNotFound {
            path: "/tmp/x.onnx".into(),
        }
        .into();
        assert!(matches!(e, InferenceError::BadRequest { .. }));
    }

    #[test]
    fn unsupported_format_maps_to_unsupported_audio_format() {
        let e: InferenceError = KokoroError::UnsupportedFormat {
            message: "only pcm supported".into(),
        }
        .into();
        assert!(matches!(e, InferenceError::UnsupportedAudioFormat { .. }));
    }

    #[test]
    fn ort_error_maps_to_internal() {
        let e: InferenceError = KokoroError::Ort("session create failed".into()).into();
        assert!(matches!(e, InferenceError::Internal(_)));
    }
}
