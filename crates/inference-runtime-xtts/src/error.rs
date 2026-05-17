//! Crate-level error type for the XTTS runtime.

use std::io;

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum XttsError {
    #[error("xtts model not found: {path}")]
    ModelNotFound { path: std::path::PathBuf },

    #[error("xtts reference audio could not be materialized: {reason}")]
    ReferenceAudioError { reason: String },

    #[error("xtts unsupported language: {lang}")]
    UnsupportedLanguage { lang: String },

    #[error("xtts unsupported format: {message}")]
    UnsupportedFormat { message: String },

    #[error("xtts io: {0}")]
    Io(#[from] io::Error),

    #[error("xtts ort error: {0}")]
    Ort(String),

    #[error("xtts feature disabled at build time — rebuild with --features tts-xtts")]
    FeatureDisabled,
}

impl From<XttsError> for atomr_infer_core::error::InferenceError {
    fn from(e: XttsError) -> Self {
        use atomr_infer_core::error::InferenceError;
        match e {
            XttsError::ModelNotFound { .. }
            | XttsError::ReferenceAudioError { .. }
            | XttsError::UnsupportedLanguage { .. } => InferenceError::BadRequest {
                message: e.to_string(),
            },
            XttsError::UnsupportedFormat { message } => InferenceError::UnsupportedAudioFormat { message },
            XttsError::Io(_) | XttsError::Ort(_) | XttsError::FeatureDisabled => {
                InferenceError::Internal(e.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::error::InferenceError;

    #[test]
    fn feature_disabled_maps_to_internal() {
        let e: InferenceError = XttsError::FeatureDisabled.into();
        assert!(matches!(e, InferenceError::Internal(_)));
    }

    #[test]
    fn unsupported_format_maps_to_unsupported_audio_format() {
        let e: InferenceError = XttsError::UnsupportedFormat {
            message: "mp3 not supported".into(),
        }
        .into();
        assert!(matches!(e, InferenceError::UnsupportedAudioFormat { .. }));
    }

    #[test]
    fn reference_audio_error_maps_to_bad_request() {
        let e: InferenceError = XttsError::ReferenceAudioError {
            reason: "file missing".into(),
        }
        .into();
        assert!(matches!(e, InferenceError::BadRequest { .. }));
    }

    #[test]
    fn ort_error_maps_to_internal() {
        let e: InferenceError = XttsError::Ort("ort init failed".into()).into();
        assert!(matches!(e, InferenceError::Internal(_)));
    }
}
