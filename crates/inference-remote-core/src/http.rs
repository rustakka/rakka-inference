//! HTTP/2 client construction and shared types. Doc §3.5, §5.8.
//!
//! `RemoteWorkerActor`s share one `reqwest::Client` per
//! `RemoteSessionActor` so that connection pooling, HTTP/2 multiplexing
//! and TLS session caching are reused across the worker pool.

use std::time::Duration;

use inference_core::deployment::Timeouts;

pub type HttpClient = reqwest::Client;

/// Build a `reqwest::Client` configured for streaming SSE workloads.
pub fn build_client(timeouts: &Timeouts, user_agent: &str) -> reqwest::Result<HttpClient> {
    reqwest::Client::builder()
        .user_agent(user_agent)
        .http2_prior_knowledge_disabled()
        .pool_idle_timeout(Some(Duration::from_secs(90)))
        .pool_max_idle_per_host(64)
        // Connect timeout is short; the doc-specified `request_timeout`
        // covers send + first-byte at the call site.
        .connect_timeout(Duration::from_secs(5))
        .timeout(timeouts.request_timeout)
        .read_timeout(timeouts.read_timeout)
        .build()
}

/// Shadow of `reqwest::ClientBuilder::http2_prior_knowledge` that does
/// **not** force prior-knowledge — keeps HTTP/2-via-ALPN as the default
/// for TLS, while still allowing HTTP/1.1 fallback for plaintext mock
/// endpoints (`wiremock`'s server is HTTP/1.1). A no-op here is the
/// right behaviour; the helper exists so the call-site reads as intent.
trait DisablePriorKnowledge {
    fn http2_prior_knowledge_disabled(self) -> Self;
}

impl DisablePriorKnowledge for reqwest::ClientBuilder {
    fn http2_prior_knowledge_disabled(self) -> Self {
        // Intentional no-op: we want ALPN-negotiated H2 over TLS and
        // graceful HTTP/1.1 over plaintext. `reqwest`'s default is
        // exactly that.
        self
    }
}
