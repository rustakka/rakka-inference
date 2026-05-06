---
name: atomr-infer-pipelines
description: Use when composing multi-runtime pipelines in atomr-infer — hybrid local→remote escalation, fallback on `RateLimitExceeded` / `CircuitOpen`, dynamic batching, cascade routing, replica pools, hot-swap, speculative decoding, MoE. Triggers on writing an actor that calls multiple `Deployment`s, using `inference::accel_patterns::*`, or asking "how do I escalate from local to OpenAI when confidence is low".
---

# Composing multi-runtime pipelines

`atomr-infer` doesn't reinvent batching, cascading, or replica
pools — those are generic actor blueprints from
[`atomr-accel-patterns`](https://github.com/rustakka/atomr-accel/tree/main/crates/atomr-accel-patterns).
The `inference --features accel-patterns` rollup re-exports them as
`inference::accel_patterns::*`. You plug a closure that calls
`ModelRunner::execute` into each blueprint and you've composed a §9
pipeline.

## The §9 building blocks

| Pattern | What it does | When to use |
|---|---|---|
| `DynamicBatchingServer<Req, Resp>` | Accumulate requests up to `(max_batch, max_wait)`; dispatch as one batched call. | Local LLM throughput; embedding services. |
| `InferenceCascade<Req, Resp>` | Cheap classifier → escalation when confidence < threshold. | Cost optimization (small local first, big remote on hard cases). |
| `ModelReplicaPool<Msg>` | Round-robin / least-loaded routing across N replicas. | Saturate multi-GPU node; load-balance across cluster nodes. |
| `FairShareScheduler<Req, Resp>` | WFQ tenant scheduling. | Multi-tenant SaaS where one customer can't starve another. |
| `ModelHotSwapServer<P>` | Live model replacement; in-flight requests drain on old, new ones use new. | Canary deployments, no-downtime upgrades. |
| `SpeculativeDecoder` | Draft model proposes; verifier model accepts/rejects. | LLM throughput at fixed quality. |
| `MoeRouter<P>` | Mixture-of-experts gating. | Routing across specialist sub-models. |

## The hybrid agent (doc §9.1)

The doc-canonical example: cheap local classifier → escalate to
GPT-4o for hard queries → fall back to Claude on saturation.

```rust
use inference::prelude::*;
use inference_core::error::InferenceError;

#[async_trait::async_trait]
trait HybridAgent {
    async fn handle(&self, query: String) -> Result<String, InferenceError>;
}

struct Agent {
    local_router:    DeploymentRef,    // small classifier
    local_executor:  DeploymentRef,    // simple-case path
    remote_planner:  DeploymentRef,    // GPT-4o
    remote_fallback: DeploymentRef,    // Claude
}

#[async_trait::async_trait]
impl HybridAgent for Agent {
    async fn handle(&self, query: String) -> Result<String, InferenceError> {
        let intent = self.local_router.ask(Classify(query.clone())).await?;

        if intent.complexity == Simple {
            return self.local_executor.ask(query).await;
        }

        // Complex query — escalate to remote.
        match self.remote_planner.ask(Plan(query.clone())).await {
            Ok(plan) => self.local_executor.ask(Execute(plan)).await,
            Err(InferenceError::RateLimited { .. })
            | Err(InferenceError::CircuitOpen { .. }) => {
                // OpenAI saturated/down → Anthropic fallback.
                let plan = self.remote_fallback.ask(Plan(query)).await?;
                self.local_executor.ask(Execute(plan)).await
            }
            Err(e @ InferenceError::ContentFiltered { .. }) => Err(e),  // Don't retry — surface.
            Err(e) => Err(e),
        }
    }
}
```

The fallback chain is **explicit** in actor logic, not buried in HTTP
retry interceptors. That makes traces, debugging, and budget
attribution work correctly.

## Tiered routing by quality + budget

```rust
struct TieredRouter {
    tiers: HashMap<Tier, DeploymentRef>,    // premium / standard / fast / cheap
}

impl TieredRouter {
    async fn handle(&self, q: String, tier: Tier, budget_usd: f64)
        -> Result<String, InferenceError>
    {
        let dep = &self.tiers[&tier];
        let est = dep.ask(EstimateCost(q.clone())).await?;
        let chosen = if est.usd > budget_usd { &self.tiers[&Tier::Fast] } else { dep };
        chosen.ask(q).await
    }
}
```

`EstimateCost` is built into the per-provider crates;
`from_rates(input_per_mtok, output_per_mtok, batch)` lifts a price
table into a `CostEstimate`.

## Using the patterns directly

```toml
[dependencies]
inference = { version = "0.2", features = ["openai", "anthropic", "candle", "accel-patterns"] }
```

```rust
use inference::accel_patterns::{
    DynamicBatchingServer,         // batch ExecuteBatches
    InferenceCascade,              // confidence-gated escalation
    ModelReplicaPool,              // N-replica round-robin
    FairShareScheduler,            // WFQ multi-tenant
    ModelHotSwapServer,            // live replacement
    SpeculativeDecoder,            // draft + verifier
    MoeRouter,                     // expert gating
};
```

Each pattern accepts a user-supplied closure / trait impl as the
backend, so you plug in something that calls `ModelRunner::execute`:

```rust
use inference_pipeline::patterns::{BatchFn, BatchingConfig, DynamicBatchingServer};

struct ModelRunnerBatchFn { runner: Arc<Mutex<Box<dyn ModelRunner>>> }
#[async_trait::async_trait]
impl BatchFn<ExecuteBatch, RunHandle> for ModelRunnerBatchFn {
    async fn invoke(&self, batch: Vec<ExecuteBatch>) -> Vec<Result<RunHandle, _>> {
        // dispatch each batch element through the runner; the actual
        // micro-batching happens inside the runner (vLLM's continuous
        // batcher, etc.). DynamicBatchingServer's job is to aggregate
        // *across requests* before they hit the runner.
        todo!()
    }
}

let server = DynamicBatchingServer::new(
    BatchingConfig { max_batch: 32, max_wait_ms: 50, /* ... */ },
    Arc::new(ModelRunnerBatchFn { /* ... */ }),
);
```

## What `inference-pipeline` ships itself

Even without the `accel-patterns` feature, `inference-pipeline` gives
you:

- `request_source(rx)` — bridge a `tokio::mpsc::UnboundedReceiver<ExecuteBatch>`
  into a `atomr_streams::Source<ExecuteBatch>` so you can compose with
  the atomr-streams DSL.
- `HybridGraph { local_classify_deployment, local_executor_deployment,
  remote_planner_deployment, remote_fallback_deployment }` — pure
  metadata describing the §9.1 shape; turn into a real
  `InferenceCascade` when the patterns feature is on.

## Canonical references

- [`inference-pipeline` README](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-pipeline/README.md)
- [Architecture doc §9](https://github.com/rustakka/atomr-infer/blob/main/docs/architecture.md) — pipeline composition
- [`atomr-accel-patterns`](https://github.com/rustakka/atomr-accel/tree/main/crates/atomr-accel-patterns) — the upstream blueprints

## Common mistakes

- **Hard-coding fallback in HTTP retry interceptors.** Express
  fallback in actor logic so it shows up in traces.
- **Using `tokio::time::sleep` in a fallback path instead of
  `RetryEngine::decide`.** The retry engine honors `Retry-After`
  headers and applies jitter; ad-hoc sleeps don't.
- **Mixing batching server + per-runtime batcher.** `vllm` batches
  internally already; `DynamicBatchingServer` in front of it adds
  *cross-request* aggregation, not duplicate batching. For
  `inference-runtime-tensorrt` (no internal batcher), the server is
  load-bearing.
- **Forgetting to enable `pipeline` when enabling `accel-patterns`.**
  The rollup's `accel-patterns` feature implies `pipeline`, so this
  resolves automatically.
