//! # inference-runtime-litellm
//!
//! Thin LiteLLM-proxy adapter on top of `inference-runtime-openai`.
//! LiteLLM exposes an OpenAI-compatible HTTP surface fronting any
//! backend (OpenAI, Anthropic, Bedrock, Azure, …) and applies its own
//! caching / fallback / retry policies. Doc §10.3.
//!
//! The `LiteLlmRunner` is a newtype around `OpenAiRunner` that:
//! - points at the LiteLLM proxy URL instead of `api.openai.com`,
//! - lowers `max_retries` (LiteLLM does its own retries; we want fast
//!   fail-through),
//! - preserves `runtime_kind() == LiteLlm` so dashboards and routing
//!   can distinguish "via LiteLLM" from "direct to OpenAI".

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use url::Url;

use inference_core::batch::ExecuteBatch;
use inference_core::deployment::{RateLimits, RetryPolicy, Timeouts};
use inference_core::error::InferenceResult;
use inference_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use inference_core::runtime::{CircuitBreakerConfig, ProviderKind, RuntimeKind, TransportKind};

use inference_remote_core::session::SessionSnapshot;
use inference_runtime_openai::{OpenAiConfig, OpenAiRunner, OpenAiVariant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteLlmConfig {
    pub endpoint: Url,
    pub api_key: SecretRef,
    #[serde(default)]
    pub rate_limits: RateLimits,
    #[serde(default = "default_retry")]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub timeouts: Timeouts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum SecretRef {
    Env { name: String },
    File { path: std::path::PathBuf },
    Inline { value: String },
}

fn default_retry() -> RetryPolicy {
    // LiteLLM has its own retries; drive client-side retries low so we
    // don't compound. Doc §10.3.
    RetryPolicy { max_retries: 1, ..RetryPolicy::default() }
}

impl LiteLlmConfig {
    pub fn into_openai(self, openai_secret: inference_runtime_openai::config::SecretRef) -> OpenAiConfig {
        OpenAiConfig {
            variant: OpenAiVariant::Direct { endpoint: self.endpoint },
            api_key: openai_secret,
            organization: None,
            project: None,
            rate_limits: self.rate_limits,
            retry: self.retry,
            circuit_breaker: self.circuit_breaker,
            timeouts: self.timeouts,
        }
    }
}

/// Newtype wrapper. Delegates to the inner `OpenAiRunner` for all
/// `ModelRunner` ops; only `runtime_kind` and `transport_kind` differ
/// so observability can distinguish LiteLLM from direct OpenAI.
pub struct LiteLlmRunner {
    inner: OpenAiRunner,
}

impl LiteLlmRunner {
    pub fn new(config: OpenAiConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> InferenceResult<Self> {
        Ok(Self { inner: OpenAiRunner::new(config, session)? })
    }
}

#[async_trait]
impl ModelRunner for LiteLlmRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        self.inner.execute(batch).await
    }

    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()> {
        self.inner.rebuild_session(cause).await
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::LiteLlm
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork { provider: ProviderKind::LiteLlm }
    }
    fn rate_limits(&self) -> Option<&RateLimits> {
        self.inner.rate_limits()
    }
    fn estimate_cost_usd(&self, batch: &ExecuteBatch) -> f64 {
        // LiteLLM proxies many backends; we don't try to recover the
        // actual price here. Operators set per-deployment pricing
        // explicitly in `inference-cli`.
        self.inner.estimate_cost_usd(batch)
    }
}
