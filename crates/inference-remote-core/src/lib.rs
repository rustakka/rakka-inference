//! # inference-remote-core
//!
//! Shared remote-runtime infrastructure (doc §5, §10.3, §12). Provides
//! the HTTP-shaped analog of the local-GPU `WorkerActor` /
//! `EngineCoreActor` pair, plus the cross-cutting concerns that the
//! GPU side doesn't need (rate limiting, circuit breaking, credential
//! refresh, SSE parsing, retry/backoff, error classification, cost
//! aggregation).
//!
//! Per-provider crates (`inference-runtime-openai`, `-anthropic`,
//! `-gemini`, `-litellm`) depend on this crate and contribute one
//! `ModelRunner` impl plus a `RuntimeConfig` shape.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod backoff;
pub mod circuit_breaker;
pub mod classify;
pub mod engine;
pub mod http;
pub mod queue;
pub mod rate_limit;
pub mod retry;
pub mod session;
pub mod sse;
pub mod worker;

pub use backoff::{compute_backoff, BackoffPolicy};
pub use circuit_breaker::{CircuitBreakerActor, CircuitBreakerHandle, CircuitState};
pub use classify::{classify_http_status, parse_retry_after};
pub use engine::{AddRequest, EngineMetrics, EngineMsg, RemoteEngineConfig, RemoteEngineCoreActor};
pub use http::{build_client, HttpClient};
pub use queue::{Priority, PriorityRequest, RequestQueue};
pub use rate_limit::{
    AcquirePermit, Permit, RateLimiterActor, RateLimiterHandle, StrictRateLimiterActor,
};
pub use retry::{Attempt, RetryDecision, RetryEngine};
pub use session::{
    CredentialProvider, RemoteSessionActor, SessionConfig, SessionRebuildRequest, SessionSnapshot,
    StaticApiKey,
};
pub use sse::{decode_sse_stream, SseChunk};
pub use worker::{RemoteWorkerActor, WorkerMsg, WorkerSlot};
