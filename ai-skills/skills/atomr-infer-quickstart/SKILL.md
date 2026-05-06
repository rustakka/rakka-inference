---
name: atomr-infer-quickstart
description: Use when standing up the first atomr-infer deployment in a consumer project, choosing feature flags for the `inference` rollup, or writing a `Deployment` value object. Triggers on adding `inference = ...` to Cargo.toml, writing a `Deployment {...}` literal, configuring `atomr-infer serve`, or asking "how do I get atomr-infer running".
---

# atomr-infer quickstart

Multi-runtime GPU + remote inference as a supervised actor system on
the [atomr](https://github.com/rustakka/atomr) actor runtime.

## The 30-second mental model

- **One topology hosts heterogeneous backends.** A request that hits
  `model="gpt-4o"` lands on a remote OpenAI deployment; `model="llama-3.1-70b"`
  lands on a local vLLM replica. The gateway, request actor, and
  routing CRDT don't know or care.
- **One value object describes any deployment.** `Deployment` carries
  `name`, `model`, optional `runtime`, optional `runtime_config`,
  `serving`, optional `budget`. The `runtime` field is the only thing
  that changes between an OpenAI deployment and a 4×H100 vLLM replica.
- **One feature flag picks your dep graph.** `inference =
  { features = [...] }` in your `Cargo.toml` is a statement of intent;
  the feature graph computes the actual deps. The architectural
  invariant: `--features remote-only` builds compile **zero** GPU /
  Python deps.

## The minimal consumer Cargo.toml

```toml
[dependencies]
# Pure remote — no GPU, no Python:
atomr-infer = { version = "0.4", features = ["remote-only"] }

# Mixed local Candle + remote OpenAI fallback:
# atomr-infer = { version = "0.4", features = ["candle", "openai", "pipeline"] }

# Zero-config local Gemma 4 on a workstation w/ GPU + Python + vLLM:
# atomr-infer = { version = "0.4", features = ["gemma-default", "openai", "pipeline"] }

# Production preset (vLLM + TensorRT + ORT + OpenAI + Anthropic + pipeline):
# atomr-infer = { version = "0.4", features = ["default-prod"] }

# Everything:
# atomr-infer = { version = "0.4", features = ["all-runtimes"] }
```

See [`docs/feature-matrix.md`](https://github.com/rustakka/atomr-infer/blob/main/docs/feature-matrix.md)
in the repo for every feature, what it pulls in, and the canonical
deployment shapes (router / zero-config dev / Rust-native LLM box /
hybrid agent / vLLM cluster). Zero-config Gemma 4 is documented at
[`docs/local-gemma.md`](https://github.com/rustakka/atomr-infer/blob/main/docs/local-gemma.md).

## Declaring a `Deployment`

```rust
use inference::prelude::*;

let dep = Deployment {
    name: "gpt-4o-mini".into(),
    model: "gpt-4o-mini".into(),
    runtime: None,                // inferred from model name
    runtime_config: None,         // defaults from rate_limits / retry / circuit_breaker
    gpus: None,
    replicas: 1,
    serving: Serving::default(),  // 32 concurrent, queue on overflow
    budget: None,
    idempotent: true,
};
```

**Auto-runtime inference** (when `runtime` is omitted):
- `gpt-4*`, `gpt-3.5*`, `o1-*`, `o3-*` → `OpenAi`
- `claude-*`, `anthropic/*` → `Anthropic`
- `gemini-*`, `google/gemini*` → `Gemini`
- `litellm/*`, `*via-litellm*` → `LiteLlm`
- `*mistral*` → `MistralRs`
- otherwise → `Vllm`

Override by setting `runtime` explicitly.

## Running the gateway

The `atomr-infer serve --config <path>` binary boots an `ActorSystem`,
applies every `[[deployment]]` from a TOML file, and mounts an
OpenAI-compatible HTTP endpoint:

```toml
# inference.toml
[cluster]
name = "production"
bind = "0.0.0.0:8080"

[[deployment]]
name     = "gpt-4o-mini"
model    = "gpt-4o-mini"
runtime  = "open_ai"
replicas = 2

[deployment.serving]
max_concurrent        = 50
on_capacity_exhausted = "queue"     # queue | reject | fallback
```

```sh
cargo run -p inference-cli --features all-remote -- serve --config inference.toml
```

Then `curl http://127.0.0.1:8080/v1/chat/completions` against it.

## Smoke-test without spending money

```sh
cargo run --bin remote_only_demo
```

Drives `wiremock` through three scenarios end-to-end:

1. Happy-path SSE streaming.
2. 429 → `Retry-After` honored → success on retry.
3. Three consecutive 503s → circuit breaker opens → next call
   short-circuits with `InferenceError::CircuitOpen`.

Useful as a regression test or as a code-skim of how the actors compose.

## When to reach beyond this skill

| You need to… | Reach for skill… |
|---|---|
| Choose between local backends or wire a specific runtime | `atomr-infer-runtimes` |
| Wire OpenAI / Anthropic / Gemini / LiteLLM credentials and rate limits | `atomr-infer-remote-providers` |
| Compose hybrid local→remote pipelines | `atomr-infer-pipelines` |
| Deploy to a cluster, handle hot-swaps and credential rotation | `atomr-infer-deployment` |
| Diagnose 429 storms / circuit-open / CUDA-context-poisoned | `atomr-infer-troubleshooting` |
| Add a new backend (Bedrock, Cohere, custom kernel pkg) | `atomr-infer-extending` |
| Author the actors themselves (Msg types, supervision, FSM) | `atomr-actor-design` (atomr workspace) |

## Canonical references

- [`README.md`](https://github.com/rustakka/atomr-infer/blob/main/README.md) — value-prop overview + 30-second tour
- [`docs/feature-matrix.md`](https://github.com/rustakka/atomr-infer/blob/main/docs/feature-matrix.md) — every feature, what it pulls in, four canonical shapes
- [`docs/architecture.md`](https://github.com/rustakka/atomr-infer/blob/main/docs/architecture.md) — the 1,459-line RFC
- [`crates/inference/`](https://github.com/rustakka/atomr-infer/blob/main/crates/inference/) — the rollup
- [`examples/remote_only_demo/`](https://github.com/rustakka/atomr-infer/blob/main/examples/remote_only_demo/) — runnable end-to-end demo

## Common mistakes

- **Adding `cudarc` / `candle` to your `Cargo.toml` directly.** Don't.
  Flip the rollup's `cudarc` / `candle` features and the dep graph
  computes itself.
- **Forgetting `--no-default-features` for remote-only builds.** The
  rollup's `default = []` is empty, so this is rarely needed in
  practice — but when in doubt, pass it to assert intent.
- **Hard-coding API keys.** Use `SecretRef::Env { name: "..." }` in
  the per-runtime config; the typed `SecretString` won't `Debug` /
  `Display`, so it can't accidentally land in logs.
