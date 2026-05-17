//! Domain-level WebSocket transport errors.
//!
//! Wraps the lower-level `tokio_tungstenite::tungstenite::Error` so
//! provider crates can match on intent (`Closed`, `Timeout`, `Tls`,
//! `Protocol`) without depending on `tungstenite`'s private types.

use std::io;

use thiserror::Error;
use tokio_tungstenite::tungstenite;

/// Errors returned by [`crate::WsClient`] and friends.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WsError {
    /// The remote sent a Close frame, or the underlying socket
    /// terminated cleanly. Reconnect engines treat this as eligible
    /// for retry depending on the close code.
    #[error("websocket closed: code={code} reason={reason}")]
    Closed { code: u16, reason: String },

    /// Connection attempt could not complete in time.
    #[error("websocket connect timed out after {seconds}s")]
    ConnectTimeout { seconds: u64 },

    /// Connection reached the inactivity deadline (no traffic, ping
    /// unanswered). Distinct from `ConnectTimeout` so callers can
    /// treat dead-link recovery differently from initial-handshake
    /// failure.
    #[error("websocket idle timeout after {seconds}s with no pong")]
    IdleTimeout { seconds: u64 },

    /// TLS negotiation failed. Usually fatal for the call site.
    #[error("websocket tls error: {0}")]
    Tls(String),

    /// Underlying I/O failure on the TCP socket.
    #[error("websocket io error: {0}")]
    Io(#[from] io::Error),

    /// `tungstenite` framing or protocol violation. The provider
    /// usually wants to surface this as
    /// `InferenceError::RealtimeClosed`.
    #[error("websocket protocol error: {0}")]
    Protocol(String),

    /// The connect URL was malformed.
    #[error("websocket bad url: {0}")]
    BadUrl(String),

    /// Reconnect attempts exhausted without success.
    #[error("websocket reconnect exhausted after {attempts} attempts")]
    ReconnectExhausted { attempts: u32 },
}

impl WsError {
    /// Returns true if a [`crate::ReconnectEngine`] should keep
    /// trying. `BadUrl` and `Tls` are terminal; everything else
    /// (network blips, idle timeouts, clean closes) is retryable.
    pub fn is_retryable(&self) -> bool {
        !matches!(
            self,
            Self::BadUrl(_) | Self::Tls(_) | Self::ReconnectExhausted { .. }
        )
    }
}

impl From<tungstenite::Error> for WsError {
    fn from(e: tungstenite::Error) -> Self {
        match e {
            tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed => Self::Closed {
                code: 1000,
                reason: String::new(),
            },
            tungstenite::Error::Io(io) => Self::Io(io),
            tungstenite::Error::Tls(t) => Self::Tls(t.to_string()),
            tungstenite::Error::Url(u) => Self::BadUrl(u.to_string()),
            tungstenite::Error::Protocol(p) => Self::Protocol(p.to_string()),
            other => Self::Protocol(other.to_string()),
        }
    }
}

/// Convenience alias matching the rest of the workspace style.
pub type WsResult<T> = std::result::Result<T, WsError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        assert!(WsError::Closed {
            code: 1011,
            reason: String::new()
        }
        .is_retryable());
        assert!(WsError::ConnectTimeout { seconds: 5 }.is_retryable());
        assert!(WsError::IdleTimeout { seconds: 30 }.is_retryable());
        assert!(!WsError::BadUrl("ftp://nope".into()).is_retryable());
        assert!(!WsError::Tls("handshake failure".into()).is_retryable());
        assert!(!WsError::ReconnectExhausted { attempts: 5 }.is_retryable());
    }

    #[test]
    fn closed_renders_with_code_and_reason() {
        let s = WsError::Closed {
            code: 1006,
            reason: "abrupt".into(),
        }
        .to_string();
        assert!(s.contains("1006"), "{s}");
        assert!(s.contains("abrupt"), "{s}");
    }
}
