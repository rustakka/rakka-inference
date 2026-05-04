//! # inference-pipeline
//!
//! `rakka-streams` integration for inference graphs (doc §9), plus a
//! re-export shim over `atomr-accel-patterns` so callers get the
//! upstream universal-GPU blueprints (batching, cascade, replica pool,
//! fair-share scheduler, hot-swap, MoE router, speculative decoder)
//! without taking a second dependency.
//!
//! The patterns are runtime-agnostic: they accept user-supplied
//! closures / trait impls as the backend, so an inference deployment
//! plugs in by handing them a closure that calls into a
//! `Box<dyn ModelRunner>`. That avoids reimplementing any of the
//! patterns locally — they're the §9 building blocks the doc names.
//!
//! Re-exports are gated behind the `cuda-patterns` feature so
//! `inference --features remote-only` builds don't pull `cudarc`.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use atomr_streams::Source;
use tokio::sync::mpsc;

use atomr_infer_core::batch::ExecuteBatch;

/// Adapter — accept a `tokio::mpsc` receiver and emit it as a stream
/// `Source`. The caller owns the sender and is responsible for closing
/// it to terminate the stream.
pub fn request_source(rx: mpsc::UnboundedReceiver<ExecuteBatch>) -> Source<ExecuteBatch> {
    Source::from_receiver(rx)
}

/// Re-export of the upstream `atomr-accel-patterns` crate so callers
/// can write `atomr_infer_pipeline::patterns::DynamicBatchingServer`
/// without separately adding it to their workspace deps.
///
/// Use these directly to compose §9-shaped graphs:
/// - `patterns::batching::DynamicBatchingServer` — accumulate
///   `ExecuteBatch`es up to a size/time bound, then dispatch as one
///   `ModelRunner::execute` call.
/// - `patterns::cascade::InferenceCascade` — early-exit routing with
///   a confidence gate (cheap classifier → escalation, doc §9.1).
/// - `patterns::replica_pool::ModelReplicaPool` — round-robin /
///   least-loaded routing across N replicas.
/// - `patterns::scheduler::FairShareScheduler` — WFQ tenant
///   scheduling.
/// - `patterns::hot_swap::ModelHotSwapServer` — live model
///   replacement (doc §7.5 canary / hot-swap).
/// - `patterns::speculative::SpeculativeDecoder` — draft + verifier
///   pair.
/// - `patterns::moe::MoeRouter` — mixture-of-experts gating.
#[cfg(feature = "cuda-patterns")]
pub mod patterns {
    pub use atomr_accel_patterns::*;
}

/// Reference hybrid-graph descriptor. Pure metadata; the
/// instantiation lives in caller code (the `examples/remote_only_demo`
/// crate exercises one path). When the `cuda-patterns` feature is on,
/// callers turn the descriptor into an `InferenceCascade` by handing
/// each deployment name to a `CascadeStage` whose closure looks the
/// `ActorRef` up in the cluster.
pub struct HybridGraph {
    pub local_classify_deployment: String,
    pub local_executor_deployment: String,
    pub remote_planner_deployment: String,
    pub remote_fallback_deployment: String,
}

impl HybridGraph {
    pub fn new(
        local_classify: impl Into<String>,
        local_executor: impl Into<String>,
        remote_planner: impl Into<String>,
        remote_fallback: impl Into<String>,
    ) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            local_classify_deployment: local_classify.into(),
            local_executor_deployment: local_executor.into(),
            remote_planner_deployment: remote_planner.into(),
            remote_fallback_deployment: remote_fallback.into(),
        })
    }
}
