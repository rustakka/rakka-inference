//! Error types for the Deepgram STT runtime.

use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runtime::RuntimeKind;
use thiserror::Error;

/// Errors raised by the Deepgram STT runner before delegating to the
/// shared `InferenceError` mapper.
#[derive(Debug, Error)]
pub enum DeepgramError {
    /// The feature flag is off — the crate compiled to a stub.
    #[error("stt-deepgram feature disabled at build time")]
    FeatureDisabled,

    /// The caller asked for a method this runner does not implement.
    #[error("{method} is not supported by the Deepgram runtime")]
    Unsupported { method: &'static str },

    /// The caller's `AudioBatch` is malformed for Deepgram (e.g.
    /// audio sample rate is missing or zero).
    #[error("bad request: {message}")]
    BadRequest { message: String },

    /// The caller asked for an audio format Deepgram does not accept.
    #[error("unsupported audio format: {message}")]
    UnsupportedFormat { message: String },
}

impl From<DeepgramError> for InferenceError {
    fn from(err: DeepgramError) -> Self {
        match err {
            DeepgramError::FeatureDisabled => {
                InferenceError::Internal("stt-deepgram feature disabled at build time".into())
            }
            DeepgramError::Unsupported { method } => InferenceError::Unsupported {
                method: method.to_string(),
                runtime: RuntimeKind::SpeechToText,
            },
            DeepgramError::BadRequest { message } => InferenceError::BadRequest { message },
            DeepgramError::UnsupportedFormat { message } => {
                InferenceError::UnsupportedAudioFormat { message }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_disabled_maps_to_internal() {
        let e: InferenceError = DeepgramError::FeatureDisabled.into();
        match e {
            InferenceError::Internal(msg) => assert!(msg.contains("stt-deepgram feature")),
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_carries_runtime_kind() {
        let e: InferenceError = DeepgramError::Unsupported { method: "x" }.into();
        match e {
            InferenceError::Unsupported { method, runtime } => {
                assert_eq!(method, "x");
                assert_eq!(runtime, RuntimeKind::SpeechToText);
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
