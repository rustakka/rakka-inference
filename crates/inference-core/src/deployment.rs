//! `Deployment` value object — the shared declarative surface for every
//! local-GPU and remote-network backend (doc §11.1, §11.3).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::runtime::{humantime_serde_ms, CircuitBreakerConfig, JitterKind, RuntimeConfig, RuntimeKind};

/// A model deployment. The `runtime` field selects the backend; every
/// other field has a runtime-agnostic interpretation. Local deployments
/// fill `gpus`; remote deployments leave it `None` and use `serving`'s
/// `max_concurrent` instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deployment {
    pub name: String,
    pub model: String,
    /// Optional explicit runtime. When omitted, `infer_runtime` picks
    /// based on the model name (doc §3.2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<RuntimeKind>,
    /// Backend-specific configuration. When omitted, defaults are used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_config: Option<RuntimeConfig>,
    /// Local-only: number of GPUs per replica.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<u32>,
    /// Number of replicas (local: HA + scale-out; remote: independent
    /// worker pools, possibly different API keys).
    #[serde(default = "default_replicas")]
    pub replicas: u32,
    #[serde(default)]
    pub serving: Serving,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<Budget>,
    /// True for normal LLM inference; false to disable retries on
    /// non-idempotent stateful APIs (doc §12.3).
    #[serde(default = "default_idempotent")]
    pub idempotent: bool,
}

fn default_replicas() -> u32 {
    1
}
fn default_idempotent() -> bool {
    true
}

impl Deployment {
    /// Effective runtime kind: explicit override wins, otherwise infer
    /// from the model name (doc §3.2).
    pub fn effective_runtime(&self) -> RuntimeKind {
        self.runtime
            .clone()
            .or_else(|| self.runtime_config.as_ref().map(RuntimeConfig::runtime_kind))
            .unwrap_or_else(|| crate::registry::infer_runtime(&self.model))
    }

    /// Cheap structural validation done at deploy time. Heavier checks
    /// (provider tier limits, network egress) live in `inference-runtime`
    /// where we can perform IO.
    pub fn validate(&self) -> Result<(), DeploymentValidationError> {
        if self.name.is_empty() {
            return Err(DeploymentValidationError::EmptyName);
        }
        if self.model.is_empty() {
            return Err(DeploymentValidationError::EmptyModel);
        }
        if self.replicas == 0 {
            return Err(DeploymentValidationError::ZeroReplicas);
        }
        Ok(())
    }
}

/// Per-deployment serving capacity (doc §11.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Serving {
    /// For remote: worker pool size; for local: maximum in-flight
    /// requests on the engine. Doc §3.5 (capacity bottleneck).
    pub max_concurrent: u32,
    pub on_capacity_exhausted: CapacityPolicy,
}

impl Default for Serving {
    fn default() -> Self {
        Self {
            max_concurrent: 32,
            on_capacity_exhausted: CapacityPolicy::Queue,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapacityPolicy {
    Reject,
    Queue,
    Fallback,
}

/// Replica metadata (doc §7.2). Owned by the `DeploymentManagerActor`;
/// the `Deployment` itself doesn't carry placements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replica {
    pub deployment: String,
    pub replica_index: u32,
    pub node: Option<String>,
    pub gpu_indices: Vec<u32>,
}

/// Provider-imposed rate limits (doc §3.5). Per `(provider, api_key,
/// model)`. Cluster-distributed via `RateLimiterActor`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_per_minute: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_minute: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrent_requests: Option<u32>,
    /// True selects `StrictRateLimiterActor` (cluster singleton, exact
    /// accounting). False selects the approximate CRDT-backed variant.
    /// Doc §12.1.
    #[serde(default)]
    pub strict: bool,
}

/// Retry policy applied inside `RemoteWorkerActor` (doc §3.5, §12.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    #[serde(with = "humantime_serde_ms")]
    pub initial_backoff: Duration,
    #[serde(with = "humantime_serde_ms")]
    pub max_backoff: Duration,
    pub backoff_multiplier: f64,
    pub jitter: JitterKind,
    pub respect_retry_after: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(1_000),
            max_backoff: Duration::from_millis(60_000),
            backoff_multiplier: 2.0,
            jitter: JitterKind::Equal,
            respect_retry_after: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeouts {
    /// Time from send to first byte received.
    #[serde(with = "humantime_serde_ms")]
    pub request_timeout: Duration,
    /// For streaming responses, time between consecutive bytes.
    #[serde(with = "humantime_serde_ms")]
    pub read_timeout: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_millis(30_000),
            read_timeout: Duration::from_millis(10_000),
        }
    }
}

/// Spend ceilings; enforced by `MetricsActor` + `RemoteEngineCoreActor`
/// (doc §11.6, §12.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_spend_per_hour_usd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_spend_per_day_usd: Option<f64>,
    pub on_exceeded: BudgetAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetAction {
    Reject,
    Warn,
    Throttle,
}

/// Re-export so callers that only depend on `inference-core` can also
/// see the circuit-breaker config without reaching into `runtime`.
pub use crate::runtime::CircuitBreakerConfig as CircuitBreakerConfigAlias;

#[derive(Debug, thiserror::Error)]
pub enum DeploymentValidationError {
    #[error("deployment name must not be empty")]
    EmptyName,
    #[error("deployment model must not be empty")]
    EmptyModel,
    #[error("deployment must have at least one replica")]
    ZeroReplicas,
    #[error("rate limits exceed known provider tier: {0}")]
    RateLimitTooHigh(String),
}

// keep CircuitBreakerConfig importable from this module too — the doc's
// examples (e.g. §11.2) put `CircuitBreakerConfig` next to `RetryPolicy`.
#[allow(dead_code)]
fn _ensure_cb_visible(_c: CircuitBreakerConfig) {}
