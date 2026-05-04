# Feature matrix

`atomr-infer` is layered so you can opt into exactly the runtimes
and infrastructure pieces you need. This page tells you *which feature
to flip* and *what it pulls in*.

The principle: **declaring `inference = { features = [...] }` in your
`Cargo.toml` is a statement of intent**. The feature graph computes the
actual dependency graph for you.

---

## Quick recipes

| You want…                                          | Feature flags                                  |
|----------------------------------------------------|------------------------------------------------|
| Pure-remote router (no GPU, no Python)             | `remote-only`                                  |
| Just OpenAI                                        | `openai`                                       |
| OpenAI + Anthropic, with hybrid pipeline           | `openai`, `anthropic`, `pipeline`              |
| Local Candle GPU + remote OpenAI                   | `candle`, `openai`, `pipeline`                 |
| The full v3-ish production preset                  | `default-prod`                                 |
| Everything                                         | `all-runtimes`                                 |
| Mocking + wiremock for tests                       | `testkit` (alongside the runtimes you mock)    |
| Reach into `rakka_accel::*` directly                | `cuda` (re-exports as `inference::cuda`)       |
| Use `DynamicBatchingServer` / `InferenceCascade`   | `cuda-patterns` (re-exports as `inference::cuda_patterns`) |
| Embed in Python                                    | `inference-py-bindings/python` on the bindings crate |

---

## What each feature pulls in

| Feature              | Adds crate(s)                                  | System / heavy deps   | Notes |
|----------------------|------------------------------------------------|-----------------------|-------|
| `openai`             | `inference-runtime-openai`                     | `reqwest`, `hyper`    | Includes the Azure variant. |
| `anthropic`          | `inference-runtime-anthropic`                  | `reqwest`, `hyper`    | Tool-use + base64 vision. |
| `gemini`             | `inference-runtime-gemini`                     | `reqwest`, `hyper`    | AI Studio + Vertex; OAuth2 via pluggable `CredentialProvider`. |
| `litellm`            | `inference-runtime-litellm`                    | (re-uses `openai`)    | LiteLLM proxy with proxy-friendly defaults. |
| `vllm`               | `inference-runtime-vllm`                       | **`pyo3`**, `python`  | Pulls `inference-python-bridge/python`. |
| `tensorrt`           | `inference-runtime-tensorrt`                   | `libnvinfer.so` (link-time) | Default-features-off compiles a stub. |
| `ort`                | `inference-runtime-ort`                        | `ort`                 | ONNX Runtime via the `ort` crate. |
| `candle`             | `inference-runtime-candle` + `cuda`            | `candle-*`, `cudarc`  | Pure-Rust transformer inference. |
| `cudarc`             | `inference-runtime-cudarc` + `cuda`            | `cudarc`              | Direct kernel dispatch via `rakka_accel::cuda::kernel::*`. |
| `mistralrs`          | `inference-runtime-mistralrs`                  | `mistralrs`           | Rust-native LLM runtime. |
| `pipeline`           | `inference-pipeline`                           | `rakka-streams`       | Streams DSL adapter. |
| `cuda`               | `rakka-accel` re-export, `inference-runtime/local-gpu` | `cudarc`         | Use only if you want `inference::cuda::*` reachable. |
| `cuda-patterns`      | `rakka-accel-patterns` re-export, `pipeline`    | `cudarc`              | `DynamicBatchingServer`, `InferenceCascade`, `ModelReplicaPool`, `FairShareScheduler`, `ModelHotSwapServer`, `SpeculativeDecoder`, `MoeRouter`. |
| `testkit`            | `inference-testkit`                            | `wiremock`            | `MockRunner`, OpenAI/Anthropic/Gemini wiremock fixtures. |

The `candle` and `cudarc` features automatically imply `cuda` because
their bodies use `rakka_accel::cuda::dispatcher::GpuDispatcher` and
`rakka_accel::cuda::kernel::*` for thread pinning and kernel dispatch.

---

## Aggregates

| Aggregate            | Expands to                                                              |
|----------------------|--------------------------------------------------------------------------|
| `all-native`         | `tensorrt`, `ort`, `candle`, `cudarc`, `mistralrs`                       |
| `all-python`         | `vllm`                                                                   |
| `all-local`          | `all-native` + `all-python`                                              |
| `all-remote`         | `openai`, `anthropic`, `gemini`, `litellm`                               |
| `all-runtimes`       | `all-local` + `all-remote` + `cuda-patterns`                             |
| `default-prod`       | `vllm`, `tensorrt`, `ort`, `openai`, `anthropic`, `pipeline`             |
| `remote-only`        | `all-remote` + `pipeline` *(deliberately excludes `cuda` / `cuda-patterns`)* |

---

## The remote-only invariant

> `cargo build -p inference --no-default-features --features remote-only`
> compiles **zero** GPU dependencies.

This is enforced by the feature graph:

```sh
$ cargo tree -p inference --no-default-features --features remote-only \
    | grep -Ec 'cudarc|rakka-accel|candle|pyo3'
0
```

Why this matters: a remote-only deployment (a fleet that fronts OpenAI
/ Anthropic / Gemini with rate limiting, fallback chains, and
observability) doesn't need to drag CUDA toolchains, candle's
ML stack, or a Python interpreter into its container image. The
feature gate guarantees the dep graph reflects intent.

---

## Per-crate features

Some crates expose their own gates so they can be consumed
**independently** without going through the rollup:

### `inference-runtime`

| Feature      | Adds                                            |
|--------------|-------------------------------------------------|
| `local-gpu`  | `rakka-accel` dep; `WorkerActor` adopts upstream `device_supervisor_strategy()` |

Default builds compile without rakka-accel; useful when you're embedding
the runtime-agnostic actors into a remote-only service.

### `inference-pipeline`

| Feature           | Adds                            |
|-------------------|---------------------------------|
| `cuda-patterns`   | `rakka-accel-patterns` re-export |

Without the feature you still get `request_source`, `HybridGraph`, and
the `rakka-streams` `Source` adapter — useful for remote-only
pipelines.

### `inference-python-bridge`

| Feature   | Adds                              |
|-----------|-----------------------------------|
| `python`  | `pyo3` + `tokio` + `parking_lot`; real `PythonGpuBridge` |

Off by default so the workspace builds without a Python venv.

### `inference-py-bindings`

| Feature   | Adds                                       |
|-----------|--------------------------------------------|
| `python`  | `pyo3` + `tracing`; builds the `cdylib`    |

### Per-runtime crates (`inference-runtime-*`)

Each carries one feature whose name matches the runtime
(`vllm`, `tensorrt`, `ort`, `candle`, `cudarc`, `mistralrs`). Without
the feature, the runner returns
`InferenceError::Internal("<runtime> feature disabled at build time")`
so dependent code links cleanly. With the feature, real bodies pull
their respective system / Rust crates.

---

## Choosing a slice

A few common shapes:

**1. The OpenAI-compatible router.** No hardware. Sits in front of
managed APIs. Adds rate limiting and fallback chains.

```toml
inference = { workspace = true, features = ["remote-only"] }
```

**2. The Rust-native LLM box.** Owns one box of GPUs, runs Candle (or
mistral.rs), no Python.

```toml
inference = { workspace = true, features = ["candle", "mistralrs", "pipeline"] }
```

**3. The hybrid agent.** Local Mistral classifier escalates to GPT-4o
on hard queries; falls back to Claude on saturation.

```toml
inference = { workspace = true, features = ["mistralrs", "openai", "anthropic", "cuda-patterns"] }
```

**4. The vLLM cluster.** Production LLM inference on owned hardware.

```toml
inference = { workspace = true, features = ["vllm", "tensorrt", "openai", "pipeline"] }
```

Each shape uses **only** the layers it needs. No dead weight in your
binary.

---

## Adding a new backend

The contract is small: implement `inference_core::ModelRunner`,
provide a `RuntimeConfig`-shaped struct, and add a feature flag in the
rollup to wire it in. The 18-crate layout is *additive*: a third-party
runtime (Bedrock, Cohere, internal proxy, custom CUDA kernel package)
ships as a sibling crate that depends on `inference-core` and
`inference-remote-core` (for remote) or `inference-core` +
`rakka-accel` (for local), without forking the workspace.
