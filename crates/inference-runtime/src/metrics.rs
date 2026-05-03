//! `MetricsActor` — in-process aggregation of per-deployment counters.
//! Doc §7.7, §12.4.
//!
//! Sinks observability via `tracing` events; a future
//! `inference-telemetry` integration plugs Prometheus / OTel exporters
//! over the same actor.

use std::collections::HashMap;

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context};
use tokio::sync::oneshot;

use inference_core::tokens::TokenUsage;

#[derive(Debug, Clone, Default)]
pub struct DeploymentMetrics {
    pub requests_succeeded: u64,
    pub requests_failed: u64,
    pub rate_limited: u64,
    pub circuit_open: u64,
    pub timed_out: u64,
    pub content_filtered: u64,
    pub usage: TokenUsage,
    pub cost_usd: f64,
}

#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    pub per_deployment: HashMap<String, DeploymentMetrics>,
}

pub enum MetricsMsg {
    RecordSuccess {
        deployment: String,
        usage: TokenUsage,
        cost_usd: f64,
    },
    RecordFailure {
        deployment: String,
        kind: FailureKind,
    },
    Snapshot {
        reply: oneshot::Sender<MetricsSnapshot>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum FailureKind {
    RateLimited,
    CircuitOpen,
    Timeout,
    ContentFiltered,
    Other,
}

#[derive(Default)]
pub struct MetricsActor {
    state: MetricsSnapshot,
}

impl MetricsActor {
    pub fn new() -> Self {
        Self::default()
    }

    fn entry(&mut self, name: &str) -> &mut DeploymentMetrics {
        self.state.per_deployment.entry(name.to_string()).or_default()
    }
}

#[async_trait]
impl Actor for MetricsActor {
    type Msg = MetricsMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            MetricsMsg::RecordSuccess { deployment, usage, cost_usd } => {
                let e = self.entry(&deployment);
                e.requests_succeeded += 1;
                e.usage.add(usage);
                e.cost_usd += cost_usd;
                tracing::trace!(deployment, ?usage, cost_usd, "metrics: success");
            }
            MetricsMsg::RecordFailure { deployment, kind } => {
                let e = self.entry(&deployment);
                e.requests_failed += 1;
                match kind {
                    FailureKind::RateLimited => e.rate_limited += 1,
                    FailureKind::CircuitOpen => e.circuit_open += 1,
                    FailureKind::Timeout => e.timed_out += 1,
                    FailureKind::ContentFiltered => e.content_filtered += 1,
                    FailureKind::Other => {}
                }
                tracing::debug!(deployment, ?kind, "metrics: failure");
            }
            MetricsMsg::Snapshot { reply } => {
                let _ = reply.send(self.state.clone());
            }
        }
    }
}
