//! Error types for the ElevenLabs TTS runtime.

use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runtime::RuntimeKind;
use thiserror::Error;

/// Errors raised by the ElevenLabs TTS runner before delegating to the
/// shared `InferenceError` mapper.
#[derive(Debug, Error)]
pub enum ElevenLabsError {
    /// The feature flag is off — the crate compiled to a stub.
    #[error("tts-elevenlabs feature disabled at build time")]
    FeatureDisabled,

    /// The caller asked for a method this runner does not implement.
    #[error("{method} is not supported by the ElevenLabs runtime")]
    Unsupported { method: &'static str },

    /// The caller's `SpeechBatch` is malformed for ElevenLabs (e.g.
    /// `VoiceRef` carries no id, or the model name is empty).
    #[error("bad request: {message}")]
    BadRequest { message: String },

    /// The caller asked for an output format ElevenLabs does not accept.
    #[error("unsupported audio format: {message}")]
    UnsupportedFormat { message: String },
}

impl From<ElevenLabsError> for InferenceError {
    fn from(err: ElevenLabsError) -> Self {
        match err {
            ElevenLabsError::FeatureDisabled => {
                InferenceError::Internal("tts-elevenlabs feature disabled at build time".into())
            }
            ElevenLabsError::Unsupported { method } => InferenceError::Unsupported {
                method: method.to_string(),
                runtime: RuntimeKind::TextToSpeech,
            },
            ElevenLabsError::BadRequest { message } => InferenceError::BadRequest { message },
            ElevenLabsError::UnsupportedFormat { message } => {
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
        let e: InferenceError = ElevenLabsError::FeatureDisabled.into();
        match e {
            InferenceError::Internal(msg) => assert!(msg.contains("tts-elevenlabs feature")),
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_carries_runtime_kind() {
        let e: InferenceError = ElevenLabsError::Unsupported { method: "x" }.into();
        match e {
            InferenceError::Unsupported { method, runtime } => {
                assert_eq!(method, "x");
                assert_eq!(runtime, RuntimeKind::TextToSpeech);
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
