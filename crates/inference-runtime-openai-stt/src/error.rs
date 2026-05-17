//! OpenAI-STT error surface.

use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runtime::RuntimeKind;
use thiserror::Error;

/// Errors that surface from this crate before they are funneled into
/// `InferenceError`. Used only inside the runner body; not part of the
/// public API.
#[derive(Debug, Error)]
pub enum OpenAiSttError {
    /// The crate was built without the `stt-openai` cargo feature.
    #[error("openai-stt: stt-openai feature disabled at build time")]
    FeatureDisabled,

    /// The audio input is not yet supported (currently: streaming).
    #[error("openai-stt: {method}")]
    Unsupported { method: &'static str },

    /// The supplied `AudioOptions` variant is not for transcription.
    #[error("openai-stt: {message}")]
    BadRequest { message: String },
}

impl From<OpenAiSttError> for InferenceError {
    fn from(err: OpenAiSttError) -> Self {
        match err {
            OpenAiSttError::FeatureDisabled => {
                InferenceError::Internal("stt-openai feature disabled at build time".into())
            }
            OpenAiSttError::Unsupported { method } => InferenceError::Unsupported {
                method: method.to_string(),
                runtime: RuntimeKind::SpeechToText,
            },
            OpenAiSttError::BadRequest { message } => InferenceError::BadRequest { message },
        }
    }
}
