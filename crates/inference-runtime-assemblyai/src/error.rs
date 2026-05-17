//! Error types for the AssemblyAI STT runtime.

use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runtime::RuntimeKind;
use thiserror::Error;

/// Errors raised by the AssemblyAI STT runner before delegating to the
/// shared `InferenceError` mapper.
#[derive(Debug, Error)]
pub enum AssemblyAiError {
    /// The feature flag is off — the crate compiled to a stub.
    #[error("stt-assemblyai feature disabled at build time")]
    FeatureDisabled,

    /// The caller asked for a method this runner does not implement.
    #[error("{method} is not supported by the AssemblyAI runtime")]
    Unsupported { method: &'static str },

    /// The caller's `AudioBatch` is malformed for AssemblyAI (e.g.
    /// audio sample rate is missing or zero).
    #[error("bad request: {message}")]
    BadRequest { message: String },

    /// The caller asked for an audio format AssemblyAI does not accept.
    #[error("unsupported audio format: {message}")]
    UnsupportedFormat { message: String },
}

impl From<AssemblyAiError> for InferenceError {
    fn from(err: AssemblyAiError) -> Self {
        match err {
            AssemblyAiError::FeatureDisabled => {
                InferenceError::Internal("stt-assemblyai feature disabled at build time".into())
            }
            AssemblyAiError::Unsupported { method } => InferenceError::Unsupported {
                method: method.to_string(),
                runtime: RuntimeKind::SpeechToText,
            },
            AssemblyAiError::BadRequest { message } => InferenceError::BadRequest { message },
            AssemblyAiError::UnsupportedFormat { message } => {
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
        let e: InferenceError = AssemblyAiError::FeatureDisabled.into();
        match e {
            InferenceError::Internal(msg) => assert!(msg.contains("stt-assemblyai feature")),
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_carries_runtime_kind() {
        let e: InferenceError = AssemblyAiError::Unsupported { method: "x" }.into();
        match e {
            InferenceError::Unsupported { method, runtime } => {
                assert_eq!(method, "x");
                assert_eq!(runtime, RuntimeKind::SpeechToText);
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
