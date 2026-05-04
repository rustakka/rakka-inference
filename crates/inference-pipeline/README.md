# atomr-infer-pipeline

> Streams DSL adapter + a re-export shim over
> [`rakka-accel-patterns`](../../../rakka-accel/crates/rakka-accel-patterns/).
> If you're composing inference graphs (§9 of the architecture doc),
> this is the crate you reach for.

## Why this is *small*

The doc's §9 building blocks — dynamic batching, cascade routing,
replica pools, fair-share scheduling, hot-swap, speculative decoding,
MoE gating — are all *generic actor blueprints* that already live in
`rakka-accel-patterns`. They take user-supplied closures / trait impls
as the backend, so we don't reimplement them — we just plug a
`Box<dyn ModelRunner>` into each one.

## Build profiles

| Build                                                            | Result                                            |
|------------------------------------------------------------------|---------------------------------------------------|
| `cargo build -p atomr-infer-pipeline` (default)                    | `Source` adapter + `HybridGraph` descriptor only — no `cudarc` deps. |
| `cargo build -p atomr-infer-pipeline --features cuda-patterns`     | Adds `inference_pipeline::patterns::*` re-exports of all upstream blueprints. |

## What you get with `cuda-patterns`

```rust
use inference_pipeline::patterns::{
    DynamicBatchingServer,         // accumulate ExecuteBatches → one execute() call
    InferenceCascade,              // cheap classifier → escalation (doc §9.1)
    ModelReplicaPool,              // round-robin / least-loaded across N replicas
    FairShareScheduler,            // WFQ tenant scheduling
    ModelHotSwapServer,            // canary / hot-swap (doc §7.5)
    SpeculativeDecoder,            // draft + verifier
    MoeRouter,                     // mixture-of-experts gating
};
```

## Reference: hybrid local + remote

```rust
use inference_pipeline::HybridGraph;

let graph = HybridGraph::new(
    "llama-3.1-8b-router",     // local, cheap classifier
    "mistral-7b",              // local executor
    "gpt-4o",                  // remote planner — escalation target
    "claude-sonnet-4",         // remote fallback when openai is rate-limited
);
```

The descriptor is pure metadata; turn it into an `InferenceCascade`
from `cuda-patterns` to make it executable.

## `request_source`

```rust
use inference_pipeline::request_source;
use tokio::sync::mpsc;

let (tx, rx) = mpsc::unbounded_channel();
let source = request_source(rx);
// Compose with rakka-streams operators: filter, map, throttle, ...
```

This is the bridge between the actor system (which talks `tokio::mpsc`
under the hood) and the rakka-streams DSL (`Source` / `Flow` / `Sink`
operator graphs).
