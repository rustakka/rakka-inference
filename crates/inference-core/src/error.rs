//! `InferenceError` — the typed error surface that flows up to the
//! `RequestActor` regardless of whether the bottleneck was GPU memory,
//! GIL contention, or remote provider quota (doc §6.2).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::runtime::ProviderKind;

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
