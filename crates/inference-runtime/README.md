# inference-runtime

> Runtime-agnostic actors on top of `rakka-core`. Gateway, per-request
> lifecycle, coordinator, deployment manager, two-tier supervision —
> none of which knows or cares whether the underlying backend is a
> GPU or a remote network call.

## Actors

| Actor                          | Doc §        | Purpose                                                         |
|--------------------------------|--------------|-----------------------------------------------------------------|
| `ApiGatewayActor`              | §4, §6.1     | OpenAI-compatible HTTP endpoint; spawns a `RequestActor` per request. |
| `RequestActor`                 | §6.1         | Per-request lifecycle; aggregates `TokenChunk`s into `Tokens`.  |
| `DpCoordinatorActor`           | §4           | Cluster-singleton routing CRDT — picks an engine for a deployment. |
| `EngineCoreActor` (local)      | §5.1         | Per-replica local-GPU orchestrator; owns a `Box<dyn ModelRunner>`. |
| `WorkerActor` + `ContextActor` | §5.3, §5.11  | Two-tier supervision; restarts on `ContextPoisoned`.             |
| `DeploymentPlacementActor`     | §7.2         | Picks nodes for new deployments; delegates GPU choice to `rakka_cuda::placement::PlacementActor`. |
| `DeploymentManagerActor`       | §4           | Cluster-singleton catalog of deployments.                       |
| `MetricsActor`                 | §7.7, §12.4  | Per-deployment counters and budget tracking.                    |

Remote-network engine cores live in
[`inference-remote-core`](../inference-remote-core/) — same actor
*shapes*, different *internals* (HTTP/2 worker pool instead of CUDA
streams).

## Two-tier supervision — adopted, not reinvented

```toml
[dependencies]
inference-runtime = { workspace = true, features = ["local-gpu"] }
```

With the `local-gpu` feature, `WorkerActor::supervisor_strategy()`
returns
[`rakka_cuda::error::device_supervisor_strategy()`](../../../rakka-cuda/crates/rakka-cuda/src/error.rs)
verbatim — three retries inside a 60-second window with the upstream
`ContextPoisoned` / `OutOfMemory` / `Unrecoverable` decider. When a
`ModelRunner::execute` returns `InferenceError::CudaContextPoisoned`,
the `ContextActor` panics with the
[`rakka_cuda::error::CONTEXT_POISONED_TAG`](../../../rakka-cuda/crates/rakka-cuda/src/error.rs)
marker so the upstream supervisor routes the failure to `Restart`.

Without the feature, the same shape is preserved with an in-crate
fallback strategy — useful when you embed the runtime-agnostic actors
into a remote-only build that doesn't want `rakka-cuda` in its
dependency graph.

## Feature flags

| Feature      | Adds                                            | When to enable                                  |
|--------------|-------------------------------------------------|-------------------------------------------------|
| (default)    | runtime-agnostic actors only                    | Remote-only deployments                         |
| `local-gpu`  | `rakka-cuda` dep; upstream supervisor strategy  | Any deployment with local GPU runtimes          |

## A canonical wiring

```rust
use inference_runtime::{
    ApiGatewayActor, DeploymentManagerActor, DpCoordinatorActor, GatewayConfig,
    MetricsActor, spawn_gateway,
};
use rakka_core::actor::{ActorSystem, Props};
use rakka_config::Config;

# async fn run() -> anyhow::Result<()> {
let sys = ActorSystem::create("inference", Config::reference()).await?;

let dp = sys.actor_of(Props::create(|| DpCoordinatorActor::new()), "dp")?;
let _mgr = sys.actor_of(Props::create(|| DeploymentManagerActor::new()), "mgr")?;
let _metrics = sys.actor_of(Props::create(|| MetricsActor::new()), "metrics")?;
let _gateway = spawn_gateway(&sys, GatewayConfig::default(), dp)?;
# Ok(())
# }
```
