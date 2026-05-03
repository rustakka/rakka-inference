---
name: rakka-inference-extending
description: Use when adding a new backend to rakka-inference — implementing `ModelRunner`, plugging into the rollup, slotting a new crate into the publish dep-order. Triggers on writing `impl ModelRunner for ...`, asking "how do I add Bedrock / Cohere / a custom CUDA kernel package", or considering a fork.
---

# Extending rakka-inference with a new backend

The contract is small: implement `inference_core::ModelRunner`,
provide a `RuntimeConfig`-shaped struct, add a feature flag in the
rollup. The 18-crate layout is *additive* — third-party runtimes ship
as sibling crates that depend on `inference-core` (and
`inference-remote-core` for HTTP-shaped backends), without forking
the workspace.

## The trait

```rust
use async_trait::async_trait;
use inference_core::{
    ExecuteBatch, InferenceResult, ModelRunner, RunHandle,
    SessionRebuildCause, RuntimeKind, TransportKind,
};

pub struct MyBackendRunner { /* config + session state */ }

#[async_trait]
impl ModelRunner for MyBackendRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        // 1. Translate ExecuteBatch into your wire format.
        // 2. Dispatch (HTTP for remote, kernel for local).
        // 3. Wrap the response in RunHandle::streaming(BoxStream<Result<TokenChunk, _>>).
        todo!()
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause)
        -> InferenceResult<()> { Ok(()) }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Custom("my-backend".into())
    }

    fn transport_kind(&self) -> TransportKind {
        // LocalGpu | RemoteNetwork { provider: ProviderKind::Custom("...".into()) }
        TransportKind::RemoteNetwork {
            provider: inference_core::runtime::ProviderKind::Custom("my-backend".into()),
        }
    }

    fn rate_limits(&self)
        -> Option<&inference_core::deployment::RateLimits> { None }

    fn estimate_cost_usd(&self, _batch: &ExecuteBatch) -> f64 { 0.0 }
}
```

That's the entire contract. Plug it into the runner pool and the
`RequestActor` / engine-core / supervision / metrics all work for
free.

## Adding a remote provider (Bedrock / Cohere / internal proxy)

```toml
# inference-runtime-bedrock/Cargo.toml
[package]
name    = "inference-runtime-bedrock"
version = "0.1.0"
# ...

[dependencies]
inference-core        = "0.2"
inference-remote-core = "0.2"          # the seam: rate limiter, circuit breaker, retry, SSE
inference-runtime     = "0.2"
reqwest      = { version = "0.12", features = ["rustls-tls", "http2", "json", "stream"] }
async-trait  = "0.1"
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
secrecy      = "0.10"
arc-swap     = "1.7"
url          = { version = "2", features = ["serde"] }
```

```rust
// crates/inference-runtime-bedrock/src/runner.rs
use std::sync::Arc;
use arc_swap::ArcSwap;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header;

use inference_core::{ExecuteBatch, InferenceResult, ModelRunner, RunHandle, /* ... */};
use inference_remote_core::session::SessionSnapshot;
use inference_remote_core::sse::decode_sse_stream;
use inference_remote_core::classify::classify_http_status;

pub struct BedrockRunner {
    config: BedrockConfig,
    session: Arc<ArcSwap<SessionSnapshot>>,
}

#[async_trait]
impl ModelRunner for BedrockRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        let snap = self.session.load_full();
        let body = wire::InvokeRequest::from_batch(&batch);
        let resp = snap.client.post(self.endpoint()?)
            .headers(self.sigv4_headers(&body)?)
            .json(&body)
            .send().await
            .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let retry_after = resp.headers().get("retry-after")
                .and_then(|v| v.to_str().ok()).map(String::from);
            let body = resp.text().await.ok();
            return Err(classify_http_status(/* ProviderKind::Custom("bedrock") */,
                                            status, retry_after.as_deref(), body));
        }

        let stream = decode_sse_stream(resp.bytes_stream());
        let request_id = batch.request_id.clone();
        Ok(RunHandle::streaming(stream.filter_map(move |item| {
            let id = request_id.clone();
            async move {
                match item {
                    Ok(chunk) => lift_chunk(&id, chunk),    // your wire-type decoder
                    Err(e) => Some(Err(e)),
                }
            }
        }).boxed()))
    }
    /* rebuild_session, runtime_kind, transport_kind, rate_limits, estimate_cost_usd */
}
```

The `inference-remote-core` crate gives you everything except wire
format and pricing — typically <300 LOC for a new provider.

## Adding a local-GPU backend

For a custom CUDA kernel package, an exotic ML framework, or
research code:

```toml
[dependencies]
inference-core    = "0.2"
inference-runtime = "0.2"
rakka-accel       = { version = "0.2", features = ["cuda"] }   # for GpuDispatcher etc.
async-trait       = "0.1"
```

```rust
use rakka_accel::cuda::{
    dispatcher::GpuDispatcher,
    stream::PerActorAllocator,
    device::DeviceActor,           // two-tier supervision substrate
    kernel::BlasActor,
};

pub struct MyKernelRunner { /* device handle, kernel package, ... */ }

#[async_trait]
impl ModelRunner for MyKernelRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        // 1. Pin to a thread via GpuDispatcher.
        // 2. Allocate a stream from PerActorAllocator.
        // 3. Launch your kernel via cudarc / NVRTC.
        // 4. Sync via rakka_accel::cuda::completion::HostFnCompletion.
        // 5. Stream the de-tokenized output as TokenChunks.
        todo!()
    }
    /* ... */
}
```

The two-tier supervision (parent `WorkerActor` + restartable
`ContextActor`) is shared infrastructure. When your kernel hits a
sticky CUDA error, panic with `format!("{}: {e}", CONTEXT_POISONED_TAG)`
and the upstream supervisor restarts the context for you.

## Plugging into the rollup

```toml
# inference/Cargo.toml — add the feature
[features]
bedrock = ["dep:inference-runtime-bedrock"]

# all-remote aggregate updated:
all-remote = ["openai", "anthropic", "gemini", "litellm", "bedrock"]

[dependencies]
inference-runtime-bedrock = { workspace = true, optional = true }
```

```rust
// crates/inference/src/lib.rs
#[cfg(feature = "bedrock")]
pub use inference_runtime_bedrock as runtime_bedrock;
```

That's it. Now `inference = { features = ["bedrock"] }` in any
consumer's `Cargo.toml` picks up the new backend, and the gateway,
routing CRDT, rate limiter, circuit breaker, retry, and metrics
all work automatically.

## Slotting into publish dep-order

[`RELEASING.md`](https://github.com/rustakka/rakka-inference/blob/main/RELEASING.md)
documents the publish loop's strict dep-order list. Add your crate at
the earliest layer where all its dependencies are already listed:

```
# in release.yml's `publish-crates` job, the for-loop:
inference-core
inference-runtime
inference-python-bridge
inference-remote-core
inference-runtime-openai
...
inference-runtime-bedrock        ← slot here, after inference-remote-core
...
inference-pipeline
inference-cli
inference                        ← rollup
```

## Updating `infer_runtime`

Add a regex match in `inference-core/src/registry.rs`'s
`infer_runtime(model)` so model names auto-resolve to your runtime:

```rust
if m.starts_with("anthropic.") || m.starts_with("us.anthropic.") {
    return RuntimeKind::Custom("bedrock".into());
}
```

(Or `RuntimeKind::Bedrock` if you've added it to the enum.)

## Per-crate README convention

Every crate has a value-first README following this pattern:
- One-sentence value prop.
- "What's different from <similar runtime>" table.
- Build profiles table (default vs feature-on).
- Quick-start config snippet.
- Notes on integration with the upstream substrate.
- "Common mistakes" if any.

See [`inference-runtime-openai/README.md`](https://github.com/rustakka/rakka-inference/blob/main/crates/inference-runtime-openai/README.md)
for the canonical example.

## Canonical references

- [`inference-core::ModelRunner`](https://github.com/rustakka/rakka-inference/blob/main/crates/inference-core/src/runner.rs) — the trait
- [`inference-remote-core` README](https://github.com/rustakka/rakka-inference/blob/main/crates/inference-remote-core/README.md) — the seam for remote providers
- [`CONTRIBUTING.md`](https://github.com/rustakka/rakka-inference/blob/main/CONTRIBUTING.md) — full contributor guide
- [`RELEASING.md`](https://github.com/rustakka/rakka-inference/blob/main/RELEASING.md) — publish dep-order, allowlist mechanism

## Common mistakes

- **Implementing `ModelRunner` directly on a remote provider without
  using `inference-remote-core`.** You'll re-implement HTTP pooling,
  rate limiting, circuit breaking, retry, SSE parsing, and credential
  refresh. The seam exists — use it.
- **Forking the workspace to add a backend.** Don't. Sibling crate +
  trait impl + rollup feature flag is the additive path.
- **Skipping the `feature-disabled` stub pattern.** When your feature
  is *off*, your runner should still compile (returning a typed
  `InferenceError::Internal("<runtime> feature disabled at build time")`)
  so the workspace builds without your system deps.
- **Putting the runner in `inference-core`.** Core has zero `tokio` /
  `rakka` / GPU / HTTP deps by design. New runtimes go in their own
  sibling crate.
