# inference

> The **rollup**. One dependency, feature-flag-driven, picks any subset
> of the 18-crate workspace.

## Why a rollup

Adding `inference = { features = [...] }` is a single statement of
intent. The Cargo feature graph computes the actual dependency graph
for you: enable `openai` and you pull `atomr-infer-runtime-openai` plus
its `reqwest` / `eventsource-stream` deps; enable `candle` and you
additionally pull `atomr-accel`, `cudarc`, `candle-*`. Disable
everything and you compile only `atomr-infer-core` + `atomr-infer-runtime`.

## The shape that matters: `remote-only`

```toml
inference = { workspace = true, default-features = false, features = ["remote-only"] }
```

→ a binary with **zero** GPU dependencies. Verified end-to-end:

```sh
$ cargo tree -p inference --no-default-features --features remote-only \
    | grep -Ec 'cudarc|atomr-accel|candle|pyo3'
0
```

The dep graph stops at HTTP / actor primitives — no CUDA toolchain,
no candle-* tree, no PyO3, no Python at link time. Perfect for
container images that should weigh megabytes, not gigabytes.

## All features in one table

See [`docs/feature-matrix.md`](../../docs/feature-matrix.md) for the
full grid. Headlines:

- `openai`, `anthropic`, `gemini`, `litellm` — remote providers, no GPU.
- `vllm`, `tensorrt`, `ort`, `candle`, `cudarc`, `mistralrs` — local
  runtimes; each gates its own system deps.
- `pipeline` — `atomr-streams` adapter (no GPU).
- `cuda-patterns` — `atomr-accel-patterns` re-export (DynamicBatching,
  Cascade, ReplicaPool, FairShare, HotSwap, Speculative, MoE).
- `cuda` — direct `atomr-accel` re-export, reachable as
  `inference::cuda::*`.
- `testkit` — `atomr-infer-testkit` mocks.

Aggregates: `all-native`, `all-python`, `all-local`, `all-remote`,
`all-runtimes`, `default-prod`, `remote-only`.

## Prelude

```rust
use inference::prelude::*;

// In scope:
//   Deployment, Serving, ExecuteBatch, ModelRunner, RuntimeKind,
//   TransportKind, ProviderKind, RateLimits, RetryPolicy, Timeouts,
//   InferenceError, InferenceResult, TokenChunk, Tokens, SecretString,
//   RuntimeConfig.
```

## Re-exports

| `inference::core`            | All of `atomr-infer-core`                                       |
| `inference::runtime`         | All of `atomr-infer-runtime`                                    |
| `inference::runtime_openai`  | …if `features = ["openai"]`                                   |
| `inference::runtime_anthropic` | …if `features = ["anthropic"]`                              |
| `inference::runtime_gemini`  | …if `features = ["gemini"]`                                   |
| `inference::runtime_litellm` | …if `features = ["litellm"]`                                  |
| `inference::runtime_candle`  | …if `features = ["candle"]`                                   |
| `inference::runtime_cudarc`  | …if `features = ["cudarc"]`                                   |
| `inference::runtime_vllm`    | …if `features = ["vllm"]`                                     |
| `inference::runtime_ort`     | …if `features = ["ort"]`                                      |
| `inference::runtime_tensorrt`| …if `features = ["tensorrt"]`                                 |
| `inference::runtime_mistralrs` | …if `features = ["mistralrs"]`                              |
| `inference::pipeline`        | …if `features = ["pipeline"]`                                 |
| `inference::testkit`         | …if `features = ["testkit"]`                                  |
| `inference::cuda`            | re-export of `atomr_accel` if `features = ["cuda"]`            |
| `inference::cuda_patterns`   | re-export of `atomr_accel_patterns` if `features = ["cuda-patterns"]` |
