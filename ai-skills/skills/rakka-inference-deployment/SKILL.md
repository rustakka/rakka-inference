---
name: rakka-inference-deployment
description: Use when deploying rakka-inference to a cluster — choosing feature flags, the `remote-only` invariant, `rakka serve --config` project files, hot-swap and credential rotation, container image sizes. Triggers on writing a `inference.toml`, a Dockerfile for an inference service, configuring `kubectl apply` / Helm, or asking "how do I deploy rakka-inference to production".
---

# Deploying rakka-inference

The release ships one binary (`rakka` from `inference-cli`) plus the
library rollup `inference`. Most deployments are: one container, one
project-file TOML, the right feature flags.

## The four canonical deployment shapes

| Shape | When | Features | What ships in the binary |
|---|---|---|---|
| **Pure-remote router** | Front OpenAI / Anthropic / Gemini with rate limiting, fallback, observability. No GPU. | `remote-only` | All four remote provider runtimes + pipeline + circuit breakers. **Zero GPU deps.** |
| **Rust-native LLM box** | Owned hardware running Candle / mistral.rs without Python. | `candle, mistralrs, pipeline` | Candle + mistral.rs runtimes + rakka-accel substrate + pipeline. |
| **Hybrid agent** | Local classify → remote plan; falls back across providers on saturation. | `mistralrs, openai, anthropic, accel-patterns` | Local + remote + the §9 pipeline blueprints (cascade, replica pool, hot-swap). |
| **vLLM cluster** | Production LLM on owned GPUs. | `vllm, tensorrt, openai, pipeline` | Python-bridged vLLM + TensorRT for non-LLM + remote burst. |

## The `remote-only` invariant

```sh
$ cargo tree -p inference --no-default-features --features remote-only \
    | grep -Ec 'cudarc|rakka-accel|candle|pyo3'
0
```

Why this matters: a remote-only deployment doesn't need to drag CUDA
toolchains, candle's ML stack, or a Python interpreter into its
container image. The feature gate enforces it. CI's
`remote-only-invariant` job greps for any leak; PRs that violate it
fail.

## Container image sketch

```dockerfile
# remote-only deployment (~30MB final image vs ~3GB with vLLM)
FROM rust:1.78-slim AS build
WORKDIR /src
COPY . .
RUN cargo build --release -p inference-cli \
    --no-default-features --features remote-only

FROM debian:trixie-slim
COPY --from=build /src/target/release/rakka /usr/local/bin/rakka
COPY inference.toml /etc/inference/inference.toml
ENTRYPOINT ["rakka", "serve", "--config", "/etc/inference/inference.toml"]
```

For a vLLM image, pivot to a CUDA base + Python venv + `--features
default-prod`. The `default-prod` aggregate covers the typical
production preset (`vllm`, `tensorrt`, `ort`, `openai`, `anthropic`,
`pipeline`).

## Project-file (`inference.toml`)

```toml
[cluster]
name = "production"
bind = "0.0.0.0:8080"

# --- Local LLM ---
[[deployment]]
name     = "llama-3.1-70b-instruct"
model    = "meta-llama/Llama-3.1-70B-Instruct"
runtime  = "vllm"
gpus     = 4
replicas = 2

# --- Remote OpenAI ---
[[deployment]]
name     = "gpt-4o-mini"
model    = "gpt-4o-mini"
runtime  = "open_ai"
replicas = 2

[deployment.runtime_config]
endpoint = "https://api.openai.com/v1"
api_key  = { from_env = "OPENAI_API_KEY" }

[deployment.runtime_config.rate_limits]
requests_per_minute = 10_000
tokens_per_minute   = 10_000_000

[deployment.runtime_config.retry]
max_retries          = 3
respect_retry_after  = true

[deployment.runtime_config.circuit_breaker]
failure_threshold   = 10
open_duration_ms    = 30_000

[deployment.serving]
max_concurrent        = 50
on_capacity_exhausted = "queue"     # queue | reject | fallback

[deployment.budget]
max_spend_per_hour_usd = 50.00
on_exceeded            = "reject"   # reject | warn | throttle
```

## Required environment for remote runtimes

| Runtime | Env vars |
|---|---|
| OpenAI | `OPENAI_API_KEY`. Optionally `OPENAI_ORG`, `OPENAI_PROJECT`. |
| Azure OpenAI | `AZURE_OPENAI_KEY` (the same `OpenAiConfig::Azure` variant). |
| Anthropic | `ANTHROPIC_API_KEY`. |
| Gemini AI Studio | `GOOGLE_API_KEY`. |
| Gemini Vertex | `GOOGLE_APPLICATION_CREDENTIALS` (or your `CredentialProvider` impl's preferred source). |
| LiteLLM | `LITELLM_KEY` against the proxy URL. |

Secrets are typed (`SecretString`) and never logged. Reference them in
the TOML via `api_key = { from_env = "..." }`.

## Hot-swap operations

| Operation | What it does |
|---|---|
| `rakka rotate-credentials <name>` | Triggers `RemoteSessionActor::rebuild` on the named deployment. In-flight requests finish on the old credential; new ones use the rotated value. |
| `rakka cost-report` | Per-deployment USD spend from the running `MetricsActor`. |
| `rakka status --config <path>` | Validates the project file without running. |
| Operator API: `cluster.deployment("X").force_open(duration)` | Manually trip the circuit breaker for incident response. |

## Cluster topology

| Node role | What runs there | Required |
|---|---|---|
| **Control nodes** | `DeploymentManagerActor`, per-model `DpCoordinatorActor` cluster singletons, strict `RateLimiterActor` singletons. | At least 1 (HA: 3). |
| **GPU/serving nodes** | Local `WorkerActor`s, `EngineCoreActor`s. | Per local deployment. |
| **Egress nodes** | `RemoteEngineCoreActor`s, `RemoteWorkerActor`s. CPU-only OK. | Per remote provider, ideally with shared `(provider, api_key)` placement so the rate limiter co-locates. |
| **Edge/router nodes** | Optional extra `ApiGatewayActor` replicas. | Optional. |

For a 1–2 node setup, all four roles collapse onto every node. For
5+ nodes, separate them for failure isolation.

## Required CI / release infrastructure

The repo's [`RELEASING.md`](https://github.com/rustakka/rakka-inference/blob/main/RELEASING.md)
documents the version-bump-on-Conventional-Commit + tag-fires-release
pipeline. For your downstream deployment automation:

- Pin `inference = "=0.2.1"` (or your tag) in `Cargo.toml`.
- Use `cargo build --release -p inference-cli --features <preset>` in
  your container build.
- Mount the project-file TOML at `/etc/inference/inference.toml` (or
  whatever path you pass to `--config`).

## Canonical references

- [`README.md`](https://github.com/rustakka/rakka-inference/blob/main/README.md) — top-level overview
- [`docs/feature-matrix.md`](https://github.com/rustakka/rakka-inference/blob/main/docs/feature-matrix.md) — every feature, what it pulls
- [`RELEASING.md`](https://github.com/rustakka/rakka-inference/blob/main/RELEASING.md) — versioning, allowlist, secrets
- [Architecture doc §7](https://github.com/rustakka/rakka-inference/blob/main/docs/rustakka-inference-architecture-v4.md) — cluster operation
- [`crates/inference-cli/README.md`](https://github.com/rustakka/rakka-inference/blob/main/crates/inference-cli/README.md) — `rakka serve` subcommands

## Common mistakes

- **Building the container with `--all-features`.** Pulls candle,
  cudarc, mistral.rs, pyo3 — gigabytes of deps you don't use.
- **Running multiple `vllm` deployments on one node without
  reviewing GIL placement.** Two GIL-pinned deployments don't share an
  interpreter by default; verify your placement actor logs.
- **Setting `on_capacity_exhausted = "fallback"` without configuring
  a fallback chain.** The `RequestActor` will immediately fail
  back upstream. Use `"queue"` unless you've authored a fallback.
- **Forgetting `[deployment.budget]`.** Without one, a misconfigured
  rate limit + a runaway agent loop can rack up real spend on remote
  providers in minutes.
