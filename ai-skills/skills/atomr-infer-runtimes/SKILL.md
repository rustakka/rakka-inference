---
name: atomr-infer-runtimes
description: Use when choosing a backend for a model deployment in atomr-infer, configuring `RuntimeConfig`, deciding between local Rust-native (Candle / cudarc / mistralrs) vs Python (vLLM) vs FFI (TensorRT / ORT) vs remote (OpenAI / Anthropic / Gemini / LiteLLM). Triggers on writing a `Deployment.runtime = ...` field, choosing a feature flag, asking "what's the right backend for X model".
---

# Choosing and configuring runtimes

`atomr-infer` ships ten runtime backends. The trait that unifies
them is `inference_core::ModelRunner`; the difference between them is
*what they do* in `execute()` — local kernels, Python interpreter
calls, or HTTP/2 to a managed API.

## The matrix

| Runtime | Crate | Best for | Feature flag | Heavy deps |
|---|---|---|---|---|
| `vllm` | `inference-runtime-vllm` | Production LLM throughput on owned GPUs | `vllm` | `pyo3`, Python venv, `vllm` package |
| `tensorrt` | `inference-runtime-tensorrt` | Stable-shape pre-compiled plans (Whisper, vision, embeddings) | `tensorrt` | `libnvinfer.so` |
| `ort` | `inference-runtime-ort` | ONNX graphs (rerankers, embeddings, vision) | `ort` | `ort` crate |
| `candle` | `inference-runtime-candle` | Pure-Rust transformers; quantized GGUF on edge | `candle` | `candle-*`, `cudarc`, `rakka-accel` |
| `cudarc` | `inference-runtime-cudarc` | Custom CUDA kernels; research code | `cudarc` | `cudarc`, `rakka-accel` |
| `mistralrs` | `inference-runtime-mistralrs` | Mistral / Llama / Gemma in pure Rust with paged attention | `mistralrs` | `mistralrs` crate |
| `openai` | `inference-runtime-openai` | OpenAI Chat Completions + Azure OpenAI | `openai` | `reqwest`, `hyper` |
| `anthropic` | `inference-runtime-anthropic` | Anthropic Messages API | `anthropic` | `reqwest`, `hyper` |
| `gemini` | `inference-runtime-gemini` | Google Gemini AI Studio + Vertex | `gemini` | `reqwest`, `hyper` |
| `litellm` | `inference-runtime-litellm` | LiteLLM proxy fronting any backend | `litellm` | `reqwest` (re-uses `openai` crate) |

## Picking the right runtime

```
Is it a remote API?
├── OpenAI / GPT / o1 / Azure OpenAI ───► openai
├── Anthropic / Claude              ───► anthropic
├── Google / Gemini / Vertex        ───► gemini
├── LiteLLM proxy in front          ───► litellm
└── Custom HTTP provider            ───► implement on inference-remote-core (see atomr-infer-extending)

Is it a local model on owned hardware?
├── Production LLM, willing to run Python  ───► vllm
├── ONNX graph (Whisper / BGE / reranker)  ───► ort
├── TensorRT plan (.plan / .engine)        ───► tensorrt
├── Mistral / Llama / Gemma family + Rust  ───► mistralrs
├── Anything else in pure Rust             ───► candle
└── Custom CUDA kernels                    ───► cudarc
```

## Default-features-off contract

Every local-runtime crate compiles to a typed-error stub when its
feature is **off**. The runner's `execute()` returns
`InferenceError::Internal("<runtime> feature disabled at build time")`.
This lets the workspace build cleanly on hosts without `libnvinfer.so`
/ Python / candle's dep tree, while `inference --features <r>` flips
the body in.

## Configuring a remote runtime

```rust
use inference_runtime_openai::{OpenAiConfig, OpenAiVariant};
use inference_runtime_openai::config::SecretRef;
use inference_core::deployment::{RateLimits, Timeouts};
use inference_core::runtime::CircuitBreakerConfig;
use std::time::Duration;

let cfg = OpenAiConfig {
    variant: OpenAiVariant::Direct {
        endpoint: "https://api.openai.com/v1/".parse().unwrap(),
    },
    api_key: SecretRef::Env { name: "OPENAI_API_KEY".into() },
    organization: None,
    project: None,
    rate_limits: RateLimits {
        requests_per_minute: Some(10_000),
        tokens_per_minute: Some(10_000_000),
        ..Default::default()
    },
    retry: Default::default(),       // 3 retries, 1s→60s exp backoff, jitter, honors Retry-After
    circuit_breaker: CircuitBreakerConfig {
        failure_threshold: 10,
        open_duration: Duration::from_secs(30),
        half_open_max_probes: 1,
    },
    timeouts: Default::default(),    // 30s request, 10s read
};
```

Azure OpenAI is the same `OpenAiConfig` with
`variant: OpenAiVariant::Azure { resource, deployment, api_version }`.

For Gemini Vertex, OAuth2 is pluggable: implement
`inference_remote_core::session::CredentialProvider` over your
preferred token source (`gcloud auth print-access-token`, ADC, your
secrets vault) and the runner refreshes via that.

## Configuring a local runtime

```rust
// Candle — pure Rust transformers
use inference_runtime_candle::{CandleConfig, CandleDevice, CandleDtype};
let cfg = CandleConfig {
    model_path: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
    device: CandleDevice::Cuda,
    dtype: CandleDtype::Q4_0,
};

// vLLM — production LLM on Python
use inference_runtime_vllm::VllmConfig;
let cfg = VllmConfig {
    model: "meta-llama/Llama-3.1-70B-Instruct".into(),
    tensor_parallel_size: 4,                  // TP across 4 GPUs
    dtype: "bfloat16".into(),
    gpu_memory_utilization: Some(0.9),
};

// TensorRT — pre-compiled plan
use inference_runtime_tensorrt::TensorRtConfig;
let cfg = TensorRtConfig {
    plan_path: "/etc/models/whisper-large-v3.plan".into(),
    max_batch_size: 8,
};
```

Local runtimes integrate with the upstream `rakka-accel` substrate:
- `rakka_accel::cuda::dispatcher::GpuDispatcher` for thread pinning.
- `rakka_accel::cuda::stream::PerActorAllocator` for per-request streams.
- `rakka_accel::cuda::device::DeviceActor` two-tier supervision (pulled
  in via `inference --features cudarc` or `--features candle`, both of
  which imply `accel`).

## TOML project-file shape

```toml
[[deployment]]
name     = "gpt-4o-mini"
model    = "gpt-4o-mini"
runtime  = "open_ai"          # serde tag — snake_case
replicas = 2

[deployment.runtime_config]
endpoint = "https://api.openai.com/v1"
api_key  = { from_env = "OPENAI_API_KEY" }

[deployment.runtime_config.rate_limits]
requests_per_minute = 10_000
tokens_per_minute   = 10_000_000

[deployment.serving]
max_concurrent        = 50
on_capacity_exhausted = "queue"
```

## Canonical references

- Per-crate READMEs:
  [openai](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-openai/README.md),
  [anthropic](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-anthropic/README.md),
  [gemini](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-gemini/README.md),
  [litellm](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-litellm/README.md),
  [candle](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-candle/README.md),
  [cudarc](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-cudarc/README.md),
  [vllm](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-vllm/README.md),
  [tensorrt](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-tensorrt/README.md),
  [ort](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-ort/README.md),
  [mistralrs](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-mistralrs/README.md)
- [Architecture doc §3](https://github.com/rustakka/atomr-infer/blob/main/docs/rustakka-inference-architecture-v4.md) — backend taxonomy
- [`inference-core::registry`](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-core/src/registry.rs) — the `infer_runtime(model)` table

## Common mistakes

- **Mixing GIL-pinned deployments.** Two `vllm` deployments on the
  same node share an interpreter only if explicitly configured;
  default placement is dedicated-interpreter-per-deployment.
- **Setting `gpus = N` for a remote deployment.** Remote runtimes
  use `serving.max_concurrent` (worker pool size) instead.
- **Building `--features all-runtimes` in CI without GPU runners.**
  Use `--features remote-only` for non-GPU CI; `--features all-remote`
  to also exercise non-GPU local runtimes (none yet).
