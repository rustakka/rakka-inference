//! Error types for the Gemini Live runtime.

use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runtime::RuntimeKind;
use thiserror::Error;

/// Errors raised by the Gemini Live runner before delegating to the
/// shared `InferenceError` mapper.
#[derive(Debug, Error)]
pub enum GeminiLiveError {
    /// The feature flag is off — the crate compiled to a stub.
    #[error("tts-gemini-live feature disabled at build time")]
    FeatureDisabled,

    /// The caller asked for a method this runner does not implement
    /// (e.g. `interrupt` is not supported by Gemini Live).
    #[error("{method} is not supported by the Gemini Live runtime")]
    Unsupported { method: &'static str },

    /// The caller's `RealtimeBatch` is malformed for Gemini Live.
    #[error("bad request: {message}")]
    BadRequest { message: String },

    /// The caller asked for an audio format Gemini Live does not accept.
    #[error("unsupported audio format: {message}")]
    UnsupportedFormat { message: String },

    /// The server responded with an error envelope.
    #[error("gemini live server error: {body}")]
    ServerError { body: String },

    /// Setup did not complete before the deadline or session closed early.
    #[error("gemini live session closed: {reason}")]
    SessionClosed { reason: String },
}

impl From<GeminiLiveError> for InferenceError {
    fn from(err: GeminiLiveError) -> Self {
        match err {
            GeminiLiveError::FeatureDisabled => {
                InferenceError::Internal("tts-gemini-live feature disabled at build time".into())
            }
            GeminiLiveError::Unsupported { method } => InferenceError::Unsupported {
                method: method.to_string(),
                runtime: RuntimeKind::RealtimeSpeech,
            },
            GeminiLiveError::BadRequest { message } => InferenceError::BadRequest { message },
            GeminiLiveError::UnsupportedFormat { message } => {
                InferenceError::UnsupportedAudioFormat { message }
            }
            GeminiLiveError::ServerError { body } => InferenceError::ServerError {
                status: 500,
                body: Some(body),
            },
            GeminiLiveError::SessionClosed { reason } => InferenceError::RealtimeClosed { reason },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_disabled_maps_to_internal() {
        let e: InferenceError = GeminiLiveError::FeatureDisabled.into();
        match e {
            InferenceError::Internal(msg) => assert!(msg.contains("tts-gemini-live feature")),
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_carries_runtime_kind() {
        let e: InferenceError = GeminiLiveError::Unsupported { method: "interrupt" }.into();
        match e {
            InferenceError::Unsupported { method, runtime } => {
                assert_eq!(method, "interrupt");
                assert_eq!(runtime, RuntimeKind::RealtimeSpeech);
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn bad_request_maps_to_bad_request() {
        let e: InferenceError = GeminiLiveError::BadRequest {
            message: "ClonedFrom voice not supported".into(),
        }
        .into();
        assert!(matches!(e, InferenceError::BadRequest { .. }));
    }

    #[test]
    fn unsupported_format_maps_correctly() {
        let e: InferenceError = GeminiLiveError::UnsupportedFormat {
            message: "expected Pcm16Le".into(),
        }
        .into();
        assert!(matches!(e, InferenceError::UnsupportedAudioFormat { .. }));
    }

    #[test]
    fn server_error_maps_with_status_500() {
        let e: InferenceError = GeminiLiveError::ServerError {
            body: "{\"error\":\"quota exceeded\"}".into(),
        }
        .into();
        match e {
            InferenceError::ServerError { status, body } => {
                assert_eq!(status, 500);
                assert!(body.as_deref().unwrap_or("").contains("quota"));
            }
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[test]
    fn session_closed_maps_to_realtime_closed() {
        let e: InferenceError = GeminiLiveError::SessionClosed {
            reason: "peer reset".into(),
        }
        .into();
        match e {
            InferenceError::RealtimeClosed { reason } => {
                assert!(reason.contains("peer reset"));
            }
            other => panic!("expected RealtimeClosed, got {other:?}"),
        }
    }
}
