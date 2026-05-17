//! Error classification for the OpenAI Realtime WebSocket API.

use atomr_infer_core::error::InferenceError;

/// Map an OpenAI Realtime `error` envelope body to an [`InferenceError`].
///
/// The Realtime API delivers errors as JSON frames with
/// `{"type":"error","error":{"type":"...","message":"..."}}`.
pub fn classify_realtime_error(error_body: Option<String>) -> InferenceError {
    InferenceError::ServerError {
        status: 500,
        body: error_body,
    }
}

/// Map a WebSocket close with an OpenAI Realtime error body.
pub fn classify_ws_close(code: u16, reason: Option<String>) -> InferenceError {
    match code {
        4001 => InferenceError::Unauthorized {
            message: reason.unwrap_or_else(|| "unauthorized".into()),
        },
        4003 => InferenceError::Forbidden {
            message: reason.unwrap_or_else(|| "forbidden".into()),
        },
        4004 => InferenceError::BadRequest {
            message: reason.unwrap_or_else(|| "bad request".into()),
        },
        4029 => InferenceError::RateLimited {
            provider: atomr_infer_core::runtime::ProviderKind::OpenAi,
            retry_after: None,
        },
        _ => InferenceError::ServerError {
            status: code,
            body: reason,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_realtime_error_is_server_error() {
        let e = classify_realtime_error(Some("oops".into()));
        assert!(matches!(e, InferenceError::ServerError { status: 500, .. }));
    }

    #[test]
    fn classify_4001_is_unauthorized() {
        let e = classify_ws_close(4001, None);
        assert!(matches!(e, InferenceError::Unauthorized { .. }));
    }

    #[test]
    fn classify_4029_is_rate_limited() {
        let e = classify_ws_close(4029, None);
        assert!(matches!(e, InferenceError::RateLimited { .. }));
    }

    #[test]
    fn classify_unknown_close_is_server_error() {
        let e = classify_ws_close(1006, Some("connection closed abnormally".into()));
        assert!(matches!(e, InferenceError::ServerError { .. }));
    }
}
