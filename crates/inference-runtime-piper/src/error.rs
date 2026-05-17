//! Crate-level error type. Provider crates surface these via
//! `InferenceError::Internal` / `InferenceError::BadRequest` from
//! the [`crate::PiperRunner`] entry points.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PiperError {
    #[error("piper voice file not found: {path}")]
    VoiceNotFound { path: PathBuf },

    #[error("piper voice manifest not found: {path}")]
    ManifestNotFound { path: PathBuf },

    #[error("piper manifest io: {0}")]
    ManifestIo(#[from] io::Error),

    #[error("piper manifest parse: {0}")]
    ManifestParse(String),

    #[error("piper unknown phoneme: {phoneme:?}")]
    UnknownPhoneme { phoneme: String },

    #[error("piper speaker id {requested} out of range (voice has {num_speakers})")]
    SpeakerOutOfRange { requested: i64, num_speakers: u32 },

    #[error("piper ort error: {0}")]
    Ort(String),

    #[error("piper feature disabled at build time — rebuild with --features piper")]
    FeatureDisabled,
}

impl From<PiperError> for atomr_infer_core::error::InferenceError {
    fn from(e: PiperError) -> Self {
        use atomr_infer_core::error::InferenceError;
        match e {
            PiperError::VoiceNotFound { .. }
            | PiperError::ManifestNotFound { .. }
            | PiperError::UnknownPhoneme { .. }
            | PiperError::SpeakerOutOfRange { .. } => InferenceError::BadRequest {
                message: e.to_string(),
            },
            PiperError::ManifestIo(_)
            | PiperError::ManifestParse(_)
            | PiperError::Ort(_)
            | PiperError::FeatureDisabled => InferenceError::Internal(e.to_string()),
        }
    }
}

impl From<serde_json::Error> for PiperError {
    fn from(e: serde_json::Error) -> Self {
        PiperError::ManifestParse(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::error::InferenceError;

    #[test]
    fn unknown_phoneme_maps_to_bad_request() {
        let e: InferenceError = PiperError::UnknownPhoneme { phoneme: "x".into() }.into();
        assert!(matches!(e, InferenceError::BadRequest { .. }));
    }

    #[test]
    fn ort_error_maps_to_internal() {
        let e: InferenceError = PiperError::Ort("session create failed".into()).into();
        assert!(matches!(e, InferenceError::Internal(_)));
    }
}
