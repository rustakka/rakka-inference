# atomr-infer-core

> Foundation types for the atomr-infer workspace. Zero actor / GPU
> / HTTP dependencies — pure types that every other layer plugs into.

## What it gives you

| Type                          | Purpose                                                             |
|-------------------------------|---------------------------------------------------------------------|
| `ModelRunner`                 | The trait every backend implements (local GPU + remote network).    |
| `Deployment`                  | The shared declarative surface — what runs, where, with what limits. |
| `ExecuteBatch` / `RunHandle`  | Request input + streaming output handle.                            |
| `TokenChunk` / `Tokens`       | Per-chunk and aggregate output, with usage + cost.                  |
| `InferenceError`              | Typed error surface (rate-limited, circuit-open, content-filtered, context-length, …). |
| `RuntimeKind` / `TransportKind` / `ProviderKind` | Backend identity used by placement, observability, dispatcher choice. |
| `RateLimits` / `RetryPolicy` / `Timeouts` / `CircuitBreakerConfig` | Per-deployment policy primitives. |
| `Secret<T>` / `SecretString`  | Typed credentials — won't `Debug`, won't `Display`.                 |
| `infer_runtime(model)`        | Default backend selection from a model name (`gpt-*` → OpenAI, etc). |

## Why depend on this directly

If you're building a third-party runtime, this is the **only** crate
you need. Implement `ModelRunner` and you slot into the rest of the
workspace — gateway, supervision, distributed rate limiting, circuit
breaking, streaming — for free.

```rust
use inference_core::{
    ExecuteBatch, InferenceResult, ModelRunner, RunHandle, RuntimeKind,
    SessionRebuildCause, TransportKind,
};

pub struct MyCustomRunner { /* ... */ }

#[async_trait::async_trait]
impl ModelRunner for MyCustomRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> { /* ... */ }
    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> { Ok(()) }
    fn runtime_kind(&self) -> RuntimeKind { RuntimeKind::Custom("my-backend".into()) }
    fn transport_kind(&self) -> TransportKind { TransportKind::LocalGpu }
}
```

## Dependency budget

`atomr-infer-core` depends only on:

- `serde` / `serde_json` / `thiserror` / `bytes` — wire types
- `secrecy` — typed secrets
- `async-trait` — for the `ModelRunner` trait (documented exception)
- `futures` — `BoxStream` for `RunHandle`
- `chrono` (no-default-features, `clock` only) — for `Tokens` timestamps
- `url` — for endpoint validation

**No** `tokio`, **no** `rakka`, **no** `rakka-accel`, **no** `pyo3`,
**no** `reqwest`. This is what makes `cargo build -p atomr-infer
--features remote-only` produce a binary with zero GPU deps — the
foundation layer simply doesn't carry any.
