use serde::Deserialize;

use inference_core::error::InferenceError;
use inference_core::runtime::ProviderKind;
use inference_remote_core::classify::{classify_http_status, parse_retry_after};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    #[serde(rename = "type", default)]
    kind: String,
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    message: String,
}

pub fn classify_anthropic_error(
    status: u16,
    retry_after_header: Option<&str>,
    body: Option<String>,
) -> InferenceError {
    let retry_after = parse_retry_after(retry_after_header);

    if let Some(body_str) = body.as_deref() {
        if let Ok(env) = serde_json::from_str::<ErrorEnvelope>(body_str) {
            // Anthropic uses `invalid_request_error` for length/filter
            // shapes too; differentiate by the inner type.
            match env.error.kind.as_str() {
                "invalid_request_error"
                    if env.error.message.to_lowercase().contains("context length")
                        || env.error.message.to_lowercase().contains("too long") =>
                {
                    return InferenceError::ContextLengthExceeded { tokens: 0, max_tokens: 0 };
                }
                "permission_error" => {
                    return InferenceError::Forbidden { message: env.error.message };
                }
                _ => {}
            }
            // Some refusals come back as `error.type = "overloaded_error"` with
            // 529; mapped via status. Content filter is per-message in stream
            // events, not in the envelope.
            let _ = env.kind;
        }
    }

    classify_http_status(ProviderKind::Anthropic, status, retry_after, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_context_length() {
        let body =
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt too long"}}"#;
        let e = classify_anthropic_error(400, None, Some(body.into()));
        assert!(matches!(e, InferenceError::ContextLengthExceeded { .. }));
    }
}
