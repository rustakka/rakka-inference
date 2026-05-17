//! `InferenceError` — the typed error surface that flows up to the
//! `RequestActor` regardless of whether the bottleneck was GPU memory,
//! GIL contention, or remote provider quota (doc §6.2).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::runtime::{ProviderKind, RuntimeKind};

pub type InferenceResult<T> = Result<T, InferenceError>;

#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum InferenceError {
    /// 429 from a remote provider. Worker backs off and retries unless
    /// `max_retries` is exhausted; then this surfaces to the request.
    #[error("rate-limited (retry after {retry_after:?})")]
    RateLimited {
        provider: ProviderKind,
        #[serde(with = "duration_opt_ms")]
        retry_after: Option<Duration>,
    },

    /// Circuit breaker is open for `(provider, endpoint)`. Fail-fast.
    #[error("circuit open for {provider:?} until {retry_at_unix_ms} (opened at {opened_at_unix_ms})")]
    CircuitOpen {
        provider: ProviderKind,
        opened_at_unix_ms: u64,
        retry_at_unix_ms: u64,
    },

    /// Provider safety filter rejected the input/output. Not retryable.
    #[error("content filtered: {reason}")]
    ContentFiltered { reason: String },

    /// Input exceeded the model's context window. Not retryable.
    #[error("context length exceeded ({tokens} > {max_tokens})")]
    ContextLengthExceeded { tokens: u32, max_tokens: u32 },

    /// 400 from the provider — caller-side bug.
    #[error("bad request: {message}")]
    BadRequest { message: String },

    /// 401 — triggers `RemoteSessionActor::rebuild`.
    #[error("unauthorized: {message}")]
    Unauthorized { message: String },

    /// 403 — model/feature access denied.
    #[error("forbidden: {message}")]
    Forbidden { message: String },

    /// Mailbox / engine queue full. Upstream decides fallback / 429.
    #[error("backpressure: {0}")]
    Backpressure(String),

    /// Spend ceiling reached (doc §12.4).
    #[error("budget exceeded for `{deployment}`")]
    BudgetExceeded { deployment: String },

    /// Network blip below the HTTP layer.
    #[error("network error: {0}")]
    NetworkError(String),

    /// 5xx from provider. Counts toward circuit breaker.
    #[error("server error: {status}")]
    ServerError { status: u16, body: Option<String> },

    /// Request or read timeout.
    #[error("timeout after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },

    /// Local CUDA context poisoned (sticky failure). Triggers two-tier
    /// rebuild on the local `WorkerActor` → `ContextActor` boundary.
    #[error("CUDA context poisoned: {0}")]
    CudaContextPoisoned(String),

    /// A runner does not support the called method. Surfaces from
    /// adapter stubs that exist only to satisfy the trait surface (e.g.
    /// a remote-only adapter rejecting a local-only call). Not
    /// retryable.
    ///
    /// `method` is the trait-method name (e.g. `"execute_audio"`); kept
    /// as an owned `String` because [`InferenceError`] is serialized
    /// across the wire and a borrowed `'static str` cannot round-trip.
    #[error("{method} is unsupported by runtime {runtime:?}")]
    Unsupported {
        /// Name of the trait method that was called.
        method: String,
        /// The runtime that rejected the call.
        runtime: RuntimeKind,
    },

    /// Audio input format / sample rate / channel count is incompatible
    /// with this runtime. Source: `FR-STT-001`, `FR-A2F-001`. Not
    /// retryable.
    #[error("unsupported audio format: {message}")]
    UnsupportedAudioFormat { message: String },

    /// A realtime bidirectional session closed before completion.
    /// Surfaces from [`crate::runner::RealtimeRunner`] adapters when the
    /// underlying WebSocket / transport tears down. Source: `FR-TTS-001`.
    /// Not retryable at the adapter layer; the session must be reopened.
    #[error("realtime session closed: {reason}")]
    RealtimeClosed { reason: String },

    /// Catch-all for runtime-internal bugs. Not retryable.
    #[error("internal: {0}")]
    Internal(String),
}

impl InferenceError {
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            InferenceError::RateLimited { .. }
                | InferenceError::ServerError { .. }
                | InferenceError::Timeout { .. }
                | InferenceError::NetworkError(_)
        )
    }

    /// Whether this error counts toward the circuit-breaker failure
    /// budget. 429s and content-filter refusals do not (doc §12.2).
    pub fn counts_as_circuit_failure(&self) -> bool {
        matches!(
            self,
            InferenceError::ServerError { .. }
                | InferenceError::Timeout { .. }
                | InferenceError::NetworkError(_)
        )
    }
}

mod duration_opt_ms {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(d: &Option<Duration>, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        d.map(|x| x.as_millis() as u64).serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<u64>::deserialize(d).map(|o| o.map(Duration::from_millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_formatting_contains_method_and_runtime() {
        let err = InferenceError::Unsupported {
            method: "execute_audio".into(),
            runtime: RuntimeKind::Audio2Face,
        };
        let s = format!("{err}");
        assert!(s.contains("execute_audio"));
        assert!(s.contains("Audio2Face"));
    }

    #[test]
    fn unsupported_is_not_retryable() {
        let err = InferenceError::Unsupported {
            method: "speak".into(),
            runtime: RuntimeKind::TextToSpeech,
        };
        assert!(!err.is_retryable());
        assert!(!err.counts_as_circuit_failure());
    }

    #[test]
    fn unsupported_audio_format_is_not_retryable() {
        let err = InferenceError::UnsupportedAudioFormat {
            message: "expected 16 kHz mono Pcm16Le".into(),
        };
        assert!(!err.is_retryable());
        assert!(!err.counts_as_circuit_failure());
        assert!(format!("{err}").contains("16 kHz"));
    }

    #[test]
    fn realtime_closed_is_not_retryable() {
        let err = InferenceError::RealtimeClosed {
            reason: "peer reset".into(),
        };
        assert!(!err.is_retryable());
        assert!(!err.counts_as_circuit_failure());
        assert!(format!("{err}").contains("peer reset"));
    }

    #[test]
    fn new_error_variants_serde_round_trip() {
        let unsupported = InferenceError::Unsupported {
            method: "speak".into(),
            runtime: RuntimeKind::TextToSpeech,
        };
        let json = serde_json::to_string(&unsupported).unwrap();
        let back: InferenceError = serde_json::from_str(&json).unwrap();
        match back {
            InferenceError::Unsupported { method, runtime } => {
                assert_eq!(method, "speak");
                assert_eq!(runtime, RuntimeKind::TextToSpeech);
            }
            other => panic!("variant changed across round-trip: {other:?}"),
        }

        let format_err = InferenceError::UnsupportedAudioFormat {
            message: "bad rate".into(),
        };
        let json = serde_json::to_string(&format_err).unwrap();
        let back: InferenceError = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, InferenceError::UnsupportedAudioFormat { .. }));

        let rt = InferenceError::RealtimeClosed {
            reason: "ws drop".into(),
        };
        let json = serde_json::to_string(&rt).unwrap();
        let back: InferenceError = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, InferenceError::RealtimeClosed { .. }));
    }
}
