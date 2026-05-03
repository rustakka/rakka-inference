//! HTTP-status → typed `InferenceError` classification.
//!
//! Doc §5.8 ("classify_error is provider-specific and produces typed
//! errors"). This crate provides the *generic* table; per-provider
//! crates layer on body-shape recognition (e.g. OpenAI's
//! `error.code == "context_length_exceeded"` upgrade from a 400 to a
//! `ContextLengthExceeded`).

use std::time::Duration;

use inference_core::error::InferenceError;
use inference_core::runtime::ProviderKind;

/// Map a status code (and optional `Retry-After` header) to a typed
/// error. The body is captured for diagnostics. Per-provider crates
/// post-process to refine specific shapes.
pub fn classify_http_status(
    provider: ProviderKind,
    status: u16,
    retry_after: Option<Duration>,
    body: Option<String>,
) -> InferenceError {
    match status {
        429 => InferenceError::RateLimited { provider, retry_after },
        400 => InferenceError::BadRequest { message: body.unwrap_or_else(|| "bad request".into()) },
        401 => InferenceError::Unauthorized { message: body.unwrap_or_else(|| "unauthorized".into()) },
        403 => InferenceError::Forbidden { message: body.unwrap_or_else(|| "forbidden".into()) },
        s if (500..600).contains(&s) => InferenceError::ServerError { status: s, body },
        s => InferenceError::Internal(format!("unexpected status {s}: {body:?}")),
    }
}

/// Parse a `Retry-After` header. Accepts either delta-seconds or an
/// HTTP-date; on parse failure returns `None`.
pub fn parse_retry_after(value: Option<&str>) -> Option<Duration> {
    let v = value?;
    if let Ok(secs) = v.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    // HTTP-date parsing — keep it light, fall back to None on miss.
    chrono::DateTime::parse_from_rfc2822(v.trim())
        .ok()
        .and_then(|t| {
            let now = chrono::Utc::now().timestamp();
            let then = t.timestamp();
            (then > now).then(|| Duration::from_secs((then - now) as u64))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_seconds_retry_after() {
        assert_eq!(parse_retry_after(Some("12")), Some(Duration::from_secs(12)));
        assert_eq!(parse_retry_after(Some("  3 ")), Some(Duration::from_secs(3)));
        assert_eq!(parse_retry_after(None), None);
        assert_eq!(parse_retry_after(Some("garbage")), None);
    }

    #[test]
    fn classify_known_codes() {
        let e = classify_http_status(ProviderKind::OpenAi, 429, Some(Duration::from_secs(2)), None);
        assert!(matches!(e, InferenceError::RateLimited { .. }));
        let e = classify_http_status(ProviderKind::Anthropic, 503, None, Some("oops".into()));
        assert!(matches!(e, InferenceError::ServerError { status: 503, .. }));
        let e = classify_http_status(ProviderKind::Gemini, 401, None, None);
        assert!(matches!(e, InferenceError::Unauthorized { .. }));
    }
}
