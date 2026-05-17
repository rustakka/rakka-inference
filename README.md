# atomr-infer

A native Rust **multi-runtime inference layer** built as a supervised
actor topology on top of [atomr](https://github.com/rustakka/atomr).
atomr-infer gives you a single mental model — one `Deployment` value
object, one routing CRDT, one supervision tree — that scales from a
single OpenAI-key script to a heterogeneous cluster blending owned GPU
hardware with managed APIs. The same `actor_ref.tell(msg)` lands a
request on an H100 two racks away or in another company's data center.

```rust
use atomr_infer::prelude::*;

// Same value object describes a vLLM-on-4×H100 replica or a Gemini
// Vertex deployment. The `runtime` field is the only thing that
// changes — and it's auto-inferred from the model name when omitted.
let dep = Deployment {
    name: "gpt-4o-mini".into(),
    model: "gpt-4o-mini".into(),
    runtime: None,
    runtime_config: None,
    gpus: None,
    replicas: 1,
    serving: Serving::default(),
    budget: None,
    idempotent: true,
};
```

## Why multi-runtime inference, in Rust, now

Production AI rarely runs only on owned hardware. Frontier models,
burst capacity, and compliance edge cases all push work onto managed
APIs. Bolting a provider SDK onto a separate retry / rate-limit /
observability stack from your local GPU pool fragments the system —
and the cracks are exactly where 3 a.m. pages come from.

**Heterogeneous workloads are the norm, not the exception.** vLLM on a
DGX node, a Candle CPU model in a sidecar, an OpenAI call for the long
tail of hard prompts, an Anthropic fallback when OpenAI rate-limits —
that's one application, but today it's three SDKs, three retry
policies, three observability stacks. atomr-infer treats every runtime
as just-another-`ModelRunner`. The gateway, request actor, and routing
CRDT don't know — and don't care — whether a request lands on a local
GPU or a remote API.

**Cost, latency, and reliability are coupled.** A pipeline that
classifies cheaply on a local model and escalates to GPT-4o for hard
cases is also the pipeline that needs to fall back to Anthropic when
OpenAI is saturated and shed traffic when the hourly budget hits.
Threading those concerns by hand produces brittle glue. atomr-infer
encodes them as composable actors —
`InferenceCascade`, `RateLimiterActor` (CRDT-backed),
`CircuitBreakerActor`, `Budget { on_exceeded: Reject }` — under one
supervision tree with one trace and one backpressure story.

**Granular efficiency.** Rust gives us deterministic resource use,
zero-cost abstractions, and ownership-as-concurrency-safety. Per-actor
footprint stays small; per-message cost stays low. The remote-network
tier is HTTP/2 + SSE + connection pooling with structured retry; the
local-GPU tier rides on top of [atomr-accel][atomr-accel]'s two-tier
device supervision. A `cargo build --features remote-only` produces a
binary with **zero `cudarc`, zero `atomr-accel`, zero `candle`, zero
`pyo3`** in the dependency graph — the layered crate split makes the
invariant load-bearing, not aspirational.

[atomr-accel]: https://github.com/rustakka/atomr-accel

## What's in the box

| Crate | What it does |
|---|---|
| `atomr-infer` | Umbrella facade re-exporting the public surface, feature-flag-driven |
| `atomr-infer-core` | `Deployment` value object, `ModelRunner` trait, typed `InferenceError`, batch primitives |
| `atomr-infer-runtime` | Gateway, request actor, dp-coordinator, engine-core, two-tier worker, placement, deployment manager, metrics |
| `atomr-infer-remote-core` | Distributed rate limiter (CRDT), circuit breaker, retry/backoff, SSE parser, session lifecycle |
| `atomr-infer-runtime-{openai,anthropic,gemini,litellm}` | Per-provider `ModelRunner` against `api.openai.com`, `api.anthropic.com`, Vertex AI / AI Studio, and the LiteLLM proxy |
| `atomr-infer-runtime-tensorrt` | NVIDIA TensorRT runner over `atomr-accel-tensorrt`'s `TrtRuntime` / `ExecutionContext` (Phase 8); ONNX / INT8 / FP8 / IPluginV3 sub-features forwarded |
| `atomr-infer-runtime-mistralrs` | Mistral.rs LLM runtime via `TextModelBuilder` + token-streaming through `mpsc` |
| `atomr-infer-runtime-{vllm,ort,candle,cudarc}` | Per-backend `ModelRunner` for the remaining local Rust-native and FFI runtimes; feature-gated so absent system libs don't break the workspace |
| `atomr-infer-runtime-{openai-tts,openai-realtime,elevenlabs,gemini-live}` | Remote TTS `SpeechRunner` / `RealtimeRunner` against OpenAI batch + Realtime, ElevenLabs, and Gemini Live |
| `atomr-infer-runtime-{piper,kokoro,xtts,moss}` | Local TTS `SpeechRunner` over ONNX (Piper / Kokoro / XTTS) and native MOSS-TTS |
| `atomr-infer-runtime-{openai-stt,whisper-local,deepgram,assemblyai}` | STT `AudioRunner` for OpenAI Whisper / gpt-4o-transcribe, whisper.cpp local, Deepgram, AssemblyAI |
| `atomr-infer-runtime-audio2face` | NVIDIA Audio2Face-3D `A2FRunner` over gRPC — streams ARKit 52-blendshape frames (Linux x86_64) |
| `atomr-infer-runtime-ws-core` | Shared WSS transport with TLS, reconnect honoring `BackoffPolicy`, ping/pong keepalive, backpressure-aware split tx/rx |
| `atomr-infer-pipeline` | `atomr-streams` integration plus `DynamicBatchingServer` / `InferenceCascade` / `ModelReplicaPool` / `FairShareScheduler` / `ModelHotSwapServer` / `SpeculativeDecoder` blueprints |
| `atomr-infer-testkit` | `MockRunner` + `wiremock`-backed provider mocks (`inject_429`, `inject_5xx`, …) |
| `atomr-infer-cli` | `atomr-infer serve --config <toml>` |
| `atomr-infer-py-bindings` | PyO3 bindings for `Cluster` / `Deployment` |
| `atomr-infer-python-bridge` | `PythonGpuBridge` + python-pinned dispatcher for vLLM-style runners |

Plus a Python facade — `pip install atomr-infer` — that exposes the
same `Cluster.connect(...).deploy(deployment)` shape from Python.

## Quick start (Rust)

The umbrella crate is published on crates.io as **`atomr-infer`**:

```toml
[dependencies]
atomr-infer = { version = "0.4", features = ["openai", "anthropic", "pipeline"] }
```

Or pull in subsystem crates directly — `atomr-infer-core`,
`atomr-infer-runtime`, `atomr-infer-remote-core`, and the four
`atomr-infer-runtime-{openai,anthropic,gemini,litellm}` providers are
all on crates.io.

```rust
use atomr_infer::prelude::*;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let cluster = Cluster::create("inference", Config::empty()).await?;
cluster.deploy(Deployment {
    name: "gpt-4o-mini".into(),
    model: "gpt-4o-mini".into(),
    replicas: 1,
    ..Default::default()
}).await?;
cluster.serve("0.0.0.0:8080").await?;
# Ok(()) }
```

```sh
# OpenAI-compatible gateway over real (or mocked) providers.
cargo run -p atomr-infer-cli --features all-remote -- serve --config demo.toml

# End-to-end demo (happy path / 429 retry / circuit-open) without
# spending a cent — wiremock under the hood.
cargo run --bin remote_only_demo

# Pure-remote binary, zero GPU deps in the graph.
cargo build -p atomr-infer --no-default-features --features remote-only
```

## Zero-config local LLM

If you have a workstation with a CUDA GPU + Python 3.10+, you can
auto-provision a local Gemma 4 deployment with no project-file
edits:

```sh
pip install 'vllm>=0.6.4' timm
hf auth login    # then accept the ToS at https://huggingface.co/google/gemma-4-E4B-it
cargo run -p atomr-infer-cli --features gemma-default -- serve --config demo.toml
```

The feature auto-probes for GPU + Python + vLLM + HF token at boot;
on success it registers a `gemma-local` deployment backed by
`google/gemma-4-E4B-it`. Probe failure logs a one-line tip and
continues. All four Gemma 4 variants (`E2B`, `E2B-it`, `E4B`,
`E4B-it`) are reachable via `ATOMR_INFER_GEMMA_MODEL=...`. Cache
respects `$HF_HOME` so multiple instances on the same workstation
share one on-disk model.

Full reference: [`docs/local-gemma.md`](docs/local-gemma.md).

## Quick start (Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install atomr-infer
```

```python
from atomr_infer import Cluster, Deployment

cluster = Cluster.connect("inproc://app")
cluster.deploy(Deployment(name="gpt-4o-mini", model="gpt-4o-mini", replicas=1))
```

The 0.4 surface is intentionally narrow — `Deployment` value objects
and `Cluster.connect(...).deploy(...)`. Decorators and direct
`ActorRef` escape hatches land as the underlying Rust surface
stabilises.

## Building from source

```bash
# Rust
cargo build --workspace
cargo test --workspace
cargo build -p atomr-infer --no-default-features --features remote-only  # zero-GPU build

# Python bindings (requires maturin + a Python dev toolchain)
maturin develop --release
pytest python/tests -v
```

## Crate-layer picker

The workspace splits into layers so a remote-only egress server pulls
no GPU dependencies whatsoever, while a heterogeneous cluster pulls
exactly the runtimes it serves. Three preset shapes:

| Preset                | What you get                                                 | What you skip               |
|-----------------------|--------------------------------------------------------------|-----------------------------|
| `remote-only`         | OpenAI + Anthropic + Gemini + LiteLLM + pipeline + rate-limiting / circuit-breaker / cost tracking | All GPU code |
| `default-prod`        | vLLM + TensorRT + ORT + OpenAI + Anthropic + pipeline        | Other GPU runtimes; LiteLLM; Gemini |
| `all-runtimes`        | Everything                                                   | —                           |

Detailed feature matrix: [`docs/feature-matrix.md`](docs/feature-matrix.md).

## Layout

```
crates/                       Rust workspace
  atomr-infer-core/           foundation: traits, types, no actor / GPU / HTTP deps
  atomr-infer-runtime/        gateway, request, dp-coordinator, two-tier worker
  atomr-infer-remote-core/    rate limiter (CRDT), circuit breaker, retry, SSE
  atomr-infer-runtime-*/      per-provider / per-backend ModelRunner
  atomr-infer-pipeline/       atomr-streams + batching/cascade/replica blueprints
  atomr-infer-testkit/        MockRunner + wiremock-backed provider mocks
  atomr-infer-cli/            `atomr-infer serve --config <toml>`
  atomr-infer-py-bindings/    PyO3 bindings
  atomr-infer/                rollup
ai-skills/                    Claude / Cursor / Codex / Gemini SKILL.md bundle
docs/                         Architecture (RFC v4), feature matrix, deployment guide
examples/remote_only_demo/    end-to-end happy-path / 429 / circuit-open demo
xtask/                        Cargo xtask (audit, bump, verify, release-checklist)
```

## AI-assisted development

If you're using Claude Code, Cursor, or another AI coding assistant on
a project that depends on `atomr-infer`, install our
**[ai-skills bundle](ai-skills/)** — seven skills covering quickstart,
choosing a runtime, wiring remote providers, composing pipelines,
deployment, typed-error troubleshooting, and extending with a new
backend.

```text
/plugin marketplace add rustakka/atomr-infer
/plugin install atomr-infer-ai-skills@atomr-infer
```

Each `SKILL.md` is a thin router into the canonical docs. Other
harnesses (Cursor, Codex CLI, Gemini CLI, Aider, etc.) have install
instructions in [`ai-skills/README.md`](ai-skills/README.md).

Companion bundles for the broader stack:

- [`atomr` ai-skills](https://github.com/rustakka/atomr/tree/main/ai-skills)
  — actor design, supervision, persistence, clustering, Python bindings.
- [`atomr-accel` ai-skills](https://github.com/rustakka/atomr-accel/tree/main/ai-skills)
  — DeviceActor, kernel selection, two-tier GPU supervision, backend choice.

## Learn more

- [`docs/architecture.md`](docs/architecture.md) — full RFC v4 design
  (~1,400 lines): supervision tree, routing CRDT, distributed rate
  limiter, hybrid pipelines.
- [`docs/local-gemma.md`](docs/local-gemma.md) — zero-config local
  Gemma 4 via the `gemma-default` feature: probe behaviour, variant
  matrix, env var reference.
- [`docs/feature-matrix.md`](docs/feature-matrix.md) — every feature
  flag, what it pulls into the dep graph, when to enable it.
- [`CHANGELOG.md`](CHANGELOG.md) — release history, including the
  upstream-alignment + TensorRT/Mistral.rs notes.
- [`RELEASING.md`](RELEASING.md) — versioning, allowlist, secrets,
  emergency-release runbook.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup, conventional
  commits, audit baselines.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
