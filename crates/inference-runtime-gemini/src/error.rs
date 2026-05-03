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
    status: String,
}

pub fn classify_gemini_error(
    status: u16,
    retry_after_header: Option<&str>,
    body: Option<String>,
) -> InferenceError {
    let retry_after = parse_retry_after(retry_after_header);

    if let Some(body_str) = body.as_deref() {
        if let Ok(env) = serde_json::from_str::<ErrorEnvelope>(body_str) {
            // Gemini returns `RESOURCE_EXHAUSTED` for quota.
            if env.error.status == "RESOURCE_EXHAUSTED" {
                return InferenceError::RateLimited { provider: ProviderKind::Gemini, retry_after };
            }
            // `FAILED_PRECONDITION` covers safety/policy and a few
            // other unrelated cases — treat the safety subset as
            // ContentFiltered when the message hints at it.
            let lower = env.error.message.to_lowercase();
            if lower.contains("safety") || lower.contains("blocked") {
                return InferenceError::ContentFiltered { reason: env.error.message };
            }
        }
    }

    classify_http_status(ProviderKind::Gemini, status, retry_after, body)
}
