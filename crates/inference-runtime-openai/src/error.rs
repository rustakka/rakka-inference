//! OpenAI-specific error refinement. Layers on top of the generic
//! `classify_http_status` from `inference-remote-core` to recognise
//! provider-specific shapes that should *not* be retried (content
//! filter refusals, context-length-exceeded).

use serde::Deserialize;

use inference_core::error::InferenceError;
use inference_core::runtime::ProviderKind;
use inference_remote_core::classify::{classify_http_status, parse_retry_after};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(default)]
    message: String,
    #[serde(default)]
    code: Option<String>,
    #[serde(rename = "type", default)]
    kind: Option<String>,
}

pub fn classify_openai_error(
    status: u16,
    retry_after_header: Option<&str>,
    body: Option<String>,
) -> InferenceError {
    let retry_after = parse_retry_after(retry_after_header);

    // Try to upgrade common 400s into typed errors before the generic
    // mapper takes over.
    if status == 400 {
        if let Some(body_str) = body.as_deref() {
            if let Ok(env) = serde_json::from_str::<ErrorEnvelope>(body_str) {
                if env.error.code.as_deref() == Some("context_length_exceeded") {
                    return InferenceError::ContextLengthExceeded { tokens: 0, max_tokens: 0 };
                }
                if env.error.kind.as_deref() == Some("content_filter")
                    || env.error.kind.as_deref() == Some("content_filter_results_violation")
                {
                    return InferenceError::ContentFiltered { reason: env.error.message };
                }
            }
        }
    }

    classify_http_status(ProviderKind::OpenAi, status, retry_after, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upgrades_context_length_exceeded() {
        let body = r#"{"error":{"message":"too long","code":"context_length_exceeded","type":"invalid_request_error"}}"#;
        let e = classify_openai_error(400, None, Some(body.into()));
        assert!(matches!(e, InferenceError::ContextLengthExceeded { .. }));
    }

    #[test]
    fn detects_content_filter() {
        let body = r#"{"error":{"message":"blocked","type":"content_filter"}}"#;
        let e = classify_openai_error(400, None, Some(body.into()));
        assert!(matches!(e, InferenceError::ContentFiltered { .. }));
    }
}
