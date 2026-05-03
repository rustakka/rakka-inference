# inference-remote-core

> Shared infrastructure for *every* remote-network runtime. The seam
> where HTTP/2 connection pooling meets the actor system; where rate
> limits become a cluster-wide CRDT; where 5xx storms get isolated
> behind a circuit breaker.

## What's in here

| Component                       | What it does                                                            |
|---------------------------------|-------------------------------------------------------------------------|
| `RemoteEngineCoreActor`         | Per-replica HTTP orchestrator: bounded priority queue + worker pool.   |
| `RemoteWorkerActor`             | One per concurrent slot; pulls from the queue, retries with backoff, emits `TokenChunk`s. |
| `RemoteSessionActor`            | Credential lifecycle — analog of `ContextActor` for the network world. |
| `RateLimiterActor`              | Approximate distributed token bucket via `rakka_distributed_data::GCounter`. |
| `StrictRateLimiterActor`        | Cluster-singleton variant for premium API keys with hard caps.         |
| `CircuitBreakerActor` + `CircuitBreakerHandle` | State machine (Closed → Open → Half-open) per `(provider, endpoint)`. |
| `RetryEngine`                   | Per-call retry decisions with `Retry-After` parsing and jitter.        |
| `decode_sse_stream`             | Provider-agnostic SSE byte-stream → `SseChunk` decoder.                |
| `classify_http_status`          | Status code + `Retry-After` → typed `InferenceError`.                  |

## Why all this lives in one crate

Every remote provider — OpenAI, Anthropic, Gemini, LiteLLM, Bedrock,
Cohere, an internal LLM gateway — needs the same scaffolding:
HTTP/2 client, rate limiting, retry-with-backoff, circuit breaker, SSE
parsing, credential refresh. Dropping that scaffolding into one shared
crate means each per-provider crate ships as a thin
`ModelRunner` impl plus wire types — typically <300 LOC.

## Distributed rate limiting in 30 seconds

```rust
use inference_remote_core::{RateLimiterActor, AcquirePermit};
use inference_core::deployment::RateLimits;

let mut rl = RateLimiterActor::new(
    "node-a",
    RateLimits {
        requests_per_minute: Some(10_000),
        tokens_per_minute:   Some(10_000_000),
        ..Default::default()
    },
);
```

`RateLimiterActor` keeps a per-node `GCounter` of tokens spent in the
current window. The replicator (when wired up via `rakka-cluster`)
syncs deltas across nodes so the cluster collectively respects the
provider's RPM/TPM budget. For premium API keys with hard caps, swap
in `StrictRateLimiterActor` and register it as a cluster singleton —
exact accounting at the cost of an extra mailbox hop per request.

## Circuit breakers that don't lie

```rust
use inference_remote_core::CircuitBreakerHandle;
use inference_core::runtime::{CircuitBreakerConfig, ProviderKind};

let breaker = CircuitBreakerHandle::new(ProviderKind::OpenAi, CircuitBreakerConfig::default());
breaker.run(|| async { /* HTTP call */ Ok(()) }).await
```

When sustained failures (5xx, network errors, timeouts) trip the
threshold, the breaker opens and `breaker.check()` returns
`InferenceError::CircuitOpen { provider, opened_at_unix_ms, retry_at_unix_ms }`.
Upstream actors decide: fall back to a different deployment, surface a
429 to the caller, or queue. 429s and content-filter refusals
deliberately *don't* count toward the circuit — those are handled by
the rate limiter and the per-provider error classifier.

## Building a new provider on top

```rust
// crates/inference-runtime-bedrock/src/runner.rs (sketch)
use inference_core::runner::ModelRunner;
use inference_remote_core::sse::decode_sse_stream;
use inference_remote_core::session::SessionSnapshot;

pub struct BedrockRunner { /* SessionSnapshot + endpoint + region */ }

#[async_trait::async_trait]
impl ModelRunner for BedrockRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        // 1. Build the request.
        // 2. POST it via the session's reqwest::Client.
        // 3. Wrap response.bytes_stream() with decode_sse_stream.
        // 4. Convert per-chunk JSON into TokenChunk.
    }
    /* ... */
}
```

The crate carries its own wire types and pricing tables; everything
else (rate limit, circuit breaker, retry, session) is reusable from
this crate.

## Dependencies

Pure HTTP / async — no GPU, no Python:

- `reqwest` (rustls + http2 + json + stream) for HTTP/2
- `eventsource-stream` for SSE parsing
- `tower` for middleware composition
- `rakka-core` + `rakka-distributed-data` for actor + CRDT

Importantly, `rakka-accel` is **not** a dependency. This crate is the
linchpin of the remote-only invariant: a build with `--features
remote-only` reaches `inference-remote-core` and stops — no `cudarc`,
no `pyo3`, no GPU code anywhere in the dep graph.
