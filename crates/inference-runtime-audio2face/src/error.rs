//! `Audio2FaceError` — typed errors for the Audio2Face-3D runtime.

use atomr_infer_core::error::InferenceError;

/// Errors produced by the Audio2Face-3D runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Audio2FaceError {
    /// The `audio2face` Cargo feature was not enabled at build time.
    #[error("audio2face feature is disabled; rebuild with --features audio2face")]
    FeatureDisabled,

    /// The current host architecture is not supported (linux x86_64 required).
    #[error("audio2face requires Linux x86_64; current platform is unsupported")]
    UnsupportedArch,

    /// The operation or configuration is not supported by this runner.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// The [`AudioBatch`][atomr_infer_core::AudioBatch] was malformed or
    /// carried wrong `AudioOptions` variant.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// The audio format in the batch is not accepted by this runner.
    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),
}

impl From<Audio2FaceError> for InferenceError {
    fn from(e: Audio2FaceError) -> Self {
        match e {
            Audio2FaceError::FeatureDisabled | Audio2FaceError::UnsupportedArch => {
                InferenceError::Internal(e.to_string())
            }
            Audio2FaceError::Unsupported(msg) => InferenceError::BadRequest { message: msg },
            Audio2FaceError::BadRequest(msg) => InferenceError::BadRequest { message: msg },
            Audio2FaceError::UnsupportedFormat(msg) => InferenceError::BadRequest {
                message: format!("audio format: {msg}"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_disabled_converts_to_internal() {
        let ie: InferenceError = Audio2FaceError::FeatureDisabled.into();
        assert!(matches!(ie, InferenceError::Internal(_)));
    }

    #[test]
    fn unsupported_arch_converts_to_internal() {
        let ie: InferenceError = Audio2FaceError::UnsupportedArch.into();
        assert!(matches!(ie, InferenceError::Internal(_)));
    }

    #[test]
    fn bad_request_converts() {
        let ie: InferenceError = Audio2FaceError::BadRequest("nope".into()).into();
        assert!(matches!(ie, InferenceError::BadRequest { .. }));
    }

    #[test]
    fn unsupported_converts() {
        let ie: InferenceError = Audio2FaceError::Unsupported("xyz".into()).into();
        assert!(matches!(ie, InferenceError::BadRequest { .. }));
    }

    #[test]
    fn unsupported_format_converts() {
        let ie: InferenceError = Audio2FaceError::UnsupportedFormat("mp3".into()).into();
        assert!(matches!(ie, InferenceError::BadRequest { .. }));
    }
}
