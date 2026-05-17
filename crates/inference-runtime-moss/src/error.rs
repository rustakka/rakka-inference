//! Crate-level error type for the MOSS TTS runtime.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MossError {
    #[error("moss model directory not found: {path}")]
    ModelNotFound { path: PathBuf },

    #[error("moss unsupported format: {message}")]
    UnsupportedFormat { message: String },

    #[error("moss requires Linux — this binary was built for a different target OS")]
    RequiresLinux,

    #[error("moss io: {0}")]
    Io(#[from] io::Error),

    #[error("moss internal: {0}")]
    Internal(String),

    #[error("moss feature disabled at build time — rebuild with --features tts-moss")]
    FeatureDisabled,
}

impl From<MossError> for atomr_infer_core::error::InferenceError {
    fn from(e: MossError) -> Self {
        use atomr_infer_core::error::InferenceError;
        match e {
            MossError::ModelNotFound { .. } => InferenceError::BadRequest {
                message: e.to_string(),
            },
            MossError::UnsupportedFormat { message } => InferenceError::UnsupportedAudioFormat { message },
            MossError::RequiresLinux
            | MossError::Io(_)
            | MossError::Internal(_)
            | MossError::FeatureDisabled => InferenceError::Internal(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::error::InferenceError;

    #[test]
    fn feature_disabled_maps_to_internal() {
        let e: InferenceError = MossError::FeatureDisabled.into();
        assert!(matches!(e, InferenceError::Internal(_)));
    }

    #[test]
    fn requires_linux_maps_to_internal() {
        let e: InferenceError = MossError::RequiresLinux.into();
        let InferenceError::Internal(msg) = e else {
            panic!("expected Internal");
        };
        assert!(msg.contains("Linux"), "{msg}");
    }

    #[test]
    fn unsupported_format_maps_to_unsupported_audio_format() {
        let e: InferenceError = MossError::UnsupportedFormat {
            message: "mp3 not supported".into(),
        }
        .into();
        assert!(matches!(e, InferenceError::UnsupportedAudioFormat { .. }));
    }

    #[test]
    fn model_not_found_maps_to_bad_request() {
        let e: InferenceError = MossError::ModelNotFound {
            path: "/tmp/model".into(),
        }
        .into();
        assert!(matches!(e, InferenceError::BadRequest { .. }));
    }
}
