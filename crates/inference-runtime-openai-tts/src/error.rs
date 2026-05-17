//! OpenAI-TTS error surface.
//!
//! Thin wrapper around `atomr_infer_runtime_openai::classify_openai_error`
//! when the feature is on. Without the feature we expose a single
//! local error type so the stub `speak` body can still call into
//! something typed.

use atomr_infer_core::error::InferenceError;
use thiserror::Error;

/// Errors that surface from this crate before they are funneled into
/// `InferenceError`. Used only inside the runner body; not part of the
/// public API.
#[derive(Debug, Error)]
pub enum OpenAiTtsError {
    /// The chosen [`atomr_infer_core::audio::AudioFormat`] is not one
    /// that `/v1/audio/speech` accepts as `response_format`.
    #[error("openai-tts: unsupported response format ({message})")]
    UnsupportedFormat { message: String },

    /// The cargo `tts-openai` feature is disabled and the real HTTP
    /// path is therefore inert.
    #[error("openai-tts: tts-openai feature disabled at build time")]
    FeatureDisabled,
}

impl From<OpenAiTtsError> for InferenceError {
    fn from(err: OpenAiTtsError) -> Self {
        match err {
            OpenAiTtsError::UnsupportedFormat { message } => {
                InferenceError::UnsupportedAudioFormat { message }
            }
            OpenAiTtsError::FeatureDisabled => {
                InferenceError::Internal("tts-openai feature disabled at build time".into())
            }
        }
    }
}
