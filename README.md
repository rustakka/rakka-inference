# rakka-inference

**Multi-runtime GPU + remote inference as a supervised actor system on
[`rakka`](../rakka).**

One actor topology hosts heterogeneous inference workloads — local GPU
runtimes (vLLM, TensorRT, ONNX Runtime, Candle, cudarc, mistral.rs) and
remote inference providers (OpenAI, Anthropic, Gemini, LiteLLM) — under
the same supervision tree, the same routing CRDT, the same backpressure
story. Cost, latency, and reliability stop being three pipelines and
become one.

> Treat managed APIs and owned hardware as just another runtime. The
> gateway, request actor, and routing CRDT don't know — and don't care
> — whether a request lands on an H100 two racks away or in another
> company's data center.

---

## Why this exists

Production AI rarely runs only on owned hardware. Frontier models, burst
capacity, and compliance edge cases all benefit from offloading to
managed APIs. Stitching managed APIs onto a separate retry / rate-limit /
observability stack from your local GPU pool fragments the system. This
crate folds the two together:

- **Unified routing.** `model="gpt-4o"` lands on a remote deployment;
  `model="llama-3.1-70b"` lands on a local one. One CRDT, one HTTP path.
- **Distributed rate limiting.** Multiple cluster nodes calling the
  same provider with the same API key share a CRDT-backed bucket. No
  surprise 429s from naïve client-side limits firing on each node
  independently.
- **Supervised failure handling.** Circuit breakers, retry with jitter,
  content-filter detection, sticky CUDA-context recovery, credential
  rotation — all under one `OneForOne` strategy with the upstream
  `rakka_accel::error::device_supervisor_strategy()` for the
  GPU-bearing tier.
- **Hybrid pipelines compose.** A request that classifies cheaply on a
  local model and escalates to GPT-4o for hard cases is one supervised
  actor graph spanning local and remote, with one trace, one
  backpressure story, and falls back to Anthropic when OpenAI is
  saturated.
- **Pure-remote builds compile zero GPU deps.** A no-GPU egress server
  is `cargo build -p inference --features remote-only` away — `cudarc`
  and `rakka-accel` aren't even in the dependency graph.

---

## 30-second tour

```sh
# Stand up an OpenAI-compatible gateway over real (or mocked) providers.
cargo run -p inference-cli --features all-remote -- serve --config demo.toml

# Try the end-to-end demo (happy path / 429 retry / circuit-open) without
# spending a cent — uses wiremock under the hood.
cargo run --bin remote_only_demo

# Build a binary with no GPU dependencies whatsoever.
cargo build -p inference --no-default-features --features remote-only
```

```rust
use inference::prelude::*;

let dep = Deployment {
    name: "gpt-4o-mini".into(),
    model: "gpt-4o-mini".into(),
    runtime: None,                   // inferred from model name
    runtime_config: None,            // defaults from rate_limits / retry / circuit_breaker
    gpus: None,
    replicas: 1,
    serving: Serving::default(),     // 32 concurrent, queue on overflow
    budget: None,
    idempotent: true,
};
```

The same `Deployment` value object describes a vLLM-on-4×H100 replica
or a Gemini Vertex deployment. The `runtime` field is the only thing
that changes.

---

## Architecture

The full design lives in
[`docs/rustakka-inference-architecture-v4.md`](docs/rustakka-inference-architecture-v4.md)
(1,459 lines, RFC v4). Short version:

```
                      [HTTP clients]
                            │
                            ▼
                   ApiGatewayActor                   runtime-agnostic
                            │ spawns one per request (inference-runtime)
                            ▼
                    RequestActor
                            │   ask(routing target)
                            ▼
                  DpCoordinatorActor                cluster-singleton
                            │   tell(AddRequest)
                            ▼
            ┌───────────────┴───────────────┐
            ▼                               ▼
   EngineCoreActor (LOCAL)          RemoteEngineCoreActor (REMOTE)
   ┌──────────────────────┐         ┌────────────────────────────┐
   │ scheduler/batcher    │         │ request queue (priority)   │
   │ kv_cache_mgr (LLM)   │         │ rate-limit-aware dispatch  │
   │ ModelExecutorActor   │         │ ┌─────────────────────────┐│
   │   ├─ WorkerActor     │         │ │ WorkerPool              ││
   │   │   └─ ContextActor│         │ │  ├─ RemoteWorkerActor   ││
   │   │       ├─ ModelRunner       │ │  └─ RemoteWorkerActor   ││
   │   │       └─ rakka_accel::*     │ └─────────────────────────┘│
   │   └─ ...                       │ uses:                      │
   └──────────────────────┘         │   RateLimiterActor (CRDT)  │
                                    │   CircuitBreakerActor      │
                                    │   RemoteSessionActor       │
                                    └────────────────────────────┘
```

The local-GPU tier rides on top of [`rakka-accel`](../rakka-accel)'s
substrate: `DeviceActor`, `ContextActor`, `GpuRef<T>`, `GpuDispatcher`,
`PerActorAllocator`, `PlacementActor`, `BlasActor`/`CudnnActor`/etc.
We don't reinvent two-tier supervision; we adopt
`rakka_accel::error::device_supervisor_strategy()` and add the
inference-specific `Box<dyn ModelRunner>` slot on top.

The remote-network tier is HTTP/2 + SSE + connection pooling, with
distributed rate limiting via `rakka_distributed_data::GCounter` and
circuit breaking + retry/backoff inside
[`inference-remote-core`](crates/inference-remote-core/).

---

## Crate layout — pick what you need

The workspace is **18 crates** plus `xtask` and the demo. Each layer is
optional via Cargo features so you only compile what you use. Three
recommended preset shapes:

| Preset                | What you get                                                 | What you skip               |
|-----------------------|--------------------------------------------------------------|-----------------------------|
| `remote-only`         | OpenAI + Anthropic + Gemini + LiteLLM + pipeline + rate-limiting / circuit-breaker / cost tracking | All GPU code (cudarc, rakka-accel, candle, pyo3) |
| `default-prod`        | vLLM + TensorRT + ORT + OpenAI + Anthropic + pipeline        | Other GPU runtimes; LiteLLM; Gemini |
| `all-runtimes`        | Everything                                                   | —                           |

Detailed feature matrix:
[`docs/feature-matrix.md`](docs/feature-matrix.md).

```
inference                                              ← rollup; one dep, feature-flag-driven
   │
   ├── inference-core                                  ← traits, types, no actor / GPU / HTTP deps
   │
   ├── inference-runtime                               ← gateway, request, dp-coordinator,
   │      [feature: local-gpu → rakka-accel]              engine-core, worker (two-tier),
   │                                                    placement, deployment-mgr, metrics
   │
   ├── inference-remote-core                           ← rate limiter (GCounter CRDT),
   │                                                    circuit breaker, retry/backoff,
   │                                                    SSE parser, session lifecycle
   │
   ├── inference-runtime-{openai, anthropic, gemini,   ← per-provider ModelRunner + cost table
   │   litellm}
   │
   ├── inference-runtime-{vllm, tensorrt, ort, candle, ← per-backend ModelRunner; feature-gated
   │   cudarc, mistralrs}                                so absent system libs don't break the
   │                                                    workspace build
   │
   ├── inference-python-bridge                         ← PythonGpuBridge + python-pinned dispatcher
   │      [feature: python → pyo3]                       (will lift to rakka-accel F4 — see TODO)
   │
   ├── inference-pipeline                              ← rakka-streams + re-export of
   │      [feature: cuda-patterns → rakka-accel-patterns] DynamicBatchingServer / InferenceCascade /
   │                                                    ModelReplicaPool / FairShareScheduler /
   │                                                    ModelHotSwapServer / SpeculativeDecoder
   │
   ├── inference-testkit                               ← MockRunner + wiremock-backed provider
   │                                                    mocks (inject_429, inject_5xx, ...)
   │
   ├── inference-cli                                   ← `rakka serve --config <toml>`
   │
   └── inference-py-bindings                           ← PyO3 bindings for Cluster / Deployment
          [feature: python]
```

### How to add only the runtimes you need

```toml
# Just OpenAI + Anthropic, no GPU code, no Python:
inference = { workspace = true, features = ["openai", "anthropic", "pipeline"] }
```

```toml
# Local Candle + remote OpenAI fallback:
inference = { workspace = true, features = ["candle", "openai", "pipeline"] }
# (Pulls rakka-accel + cudarc + candle-* automatically via the `candle` feature.)
```

```toml
# Everything, including the testkit:
inference = { workspace = true, features = ["all-runtimes", "testkit"] }
```

The rollup's job is exactly this: make `Cargo.toml` declare *intent*
and let the feature graph compute *deps*.

---

## Highlights

### Hot-path actor primitives, reused

The `local-gpu` feature wires `inference-runtime`'s `WorkerActor` /
`ContextActor` two-tier supervision directly to
`rakka_accel::error::device_supervisor_strategy()`. The supervisor
recognizes panic-string markers (`ContextPoisoned`, `OutOfMemory`,
`Unrecoverable`) and routes them to `Restart` / `Resume` / `Stop` —
exactly what `rakka-accel` does for its own `DeviceActor` ↔
`ContextActor` pair. No reinvention.

### Distributed rate limiting that actually works in a cluster

The `RateLimiterActor` uses `rakka_distributed_data::counters::GCounter`
to share its token-spent log across nodes. Two cluster members calling
OpenAI with the same API key collectively respect the bucket —
no surprise 429 storms on scale-out.

### Circuit breakers that propagate the right typed error

When the breaker opens, downstream sees
`InferenceError::CircuitOpen { provider, opened_at_unix_ms, retry_at_unix_ms }`.
The `RequestActor` decides whether to fall back to a different
deployment, surface a 429 to the caller, or queue with a timeout —
all without knowing whether the bottleneck was GPU memory or a remote
provider's outage.

### Pipelines for free

The `cuda-patterns` feature on the rollup makes
[`rakka-accel-patterns`](../rakka-accel/crates/rakka-accel-patterns/)
visible as `inference::cuda_patterns`:

```rust
use inference::cuda_patterns::{DynamicBatchingServer, InferenceCascade, ModelReplicaPool};
```

You get dynamic batching, cascade routing (cheap classifier →
escalation), N-replica pools with round-robin / least-loaded, fair-share
WFQ scheduling, hot-swap servers, speculative decoding, and MoE routers
without writing a single new actor. Plug each into a closure that
forwards into `ModelRunner::execute` and you've composed §9 of the
architecture doc.

### Selective compilation guaranteed

The dependency-budget invariant is enforced by the feature graph:

```sh
$ cargo tree -p inference --features remote-only --no-default-features | grep -E 'cudarc|rakka-accel|candle|pyo3'
$  # ← no output: zero GPU deps
```

Because the inference crates are layered (core → runtime → remote-core
→ per-runtime → rollup), a remote-only build skips all of
`inference-runtime-{vllm,tensorrt,ort,candle,cudarc,mistralrs}`,
`inference-python-bridge`, `rakka-accel`, and the entire cudarc dep
tree. **Any** consumer can pick the *exact* runtime mix without
dragging unrelated system libraries into their binary.

---

## Developer experience

### Six layers, surface up to depth

1. **`Deployment` value object.** Most users never go deeper. `runtime`
   is auto-inferred from model name when omitted (`gpt-*` → openai,
   `claude-*` → anthropic, …).
2. **Per-runtime configs.** `OpenAiConfig`, `AnthropicConfig`,
   `GeminiConfig` (Vertex + AI Studio), `LiteLlmConfig`, `CandleConfig`,
   `VllmConfig`, etc. for explicit overrides.
3. **`<config>.toml` project files.** `rakka serve --config foo.toml`
   reads the §11.3 schema and applies every `[[deployment]]`.
4. **Python decorators.** `@inference_actor` for orchestration actors
   that compose deployments without touching a GPU directly. Skeleton
   in `inference-py-bindings`.
5. **Escape hatches.** `cluster.deployment("gpt-4o").rate_limiter()`,
   `.circuit_breaker()`, `.workers()` — direct `ActorRef`s for
   incident response (`force_open`, `rebuild_session`, etc.).
6. **Raw rakka actors.** When you need it, you have the full actor
   system underneath. Unprivileged.

### Footgun-resistant by design

- **Secrets are typed.** `inference_core::SecretString` (re-export of
  `secrecy::SecretString`) — won't `Debug`, won't `Display`, never
  appears in logs.
- **Rate-limit validation at deploy time.** Catches a deployment
  claiming `rpm = 100_000` against a free-tier API key with a typed
  error before the first user request hits.
- **Network egress checked at deploy time.** The placement actor pings
  the provider from each chosen node before flipping the deployment to
  `Serving`.
- **Hot-swappable credentials.** Updating the secret source triggers
  `RemoteSessionActor::rebuild` on the next pulse; in-flight requests
  drain on the old credential, new ones use the rotated value. Zero
  dropped traffic.
- **Cost guardrails.** `Budget { max_spend_per_hour_usd, on_exceeded: Reject }`
  on a `Deployment` makes runaway provider spend physically impossible.

---

## Verification

Every PR runs:

```sh
cargo build --workspace
cargo build -p inference --features remote-only          # zero GPU deps
cargo build -p inference --features cuda,cuda-patterns   # local + patterns
cargo build -p inference --features all-runtimes
cargo test --workspace
cargo run --bin remote_only_demo
```

The demo asserts the §13 Phase-1 + Phase-2c exit criteria end-to-end
against a `wiremock`-driven OpenAI mock: happy-path streaming, 429
retry-after, and circuit-breaker open after consecutive 5xx.

---

## Status

| Layer                           | Status                                                                                       |
|---------------------------------|----------------------------------------------------------------------------------------------|
| Foundation (`inference-core`)   | ✅ stable surface; serde round-trips for every `RuntimeConfig` variant                       |
| Runtime-agnostic actors         | ✅ gateway, request, dp-coordinator, engine-core, worker, placement, manager, metrics        |
| Remote infrastructure           | ✅ rate limiter (CRDT), strict variant (singleton), circuit breaker, retry, SSE, session     |
| OpenAI / Anthropic / Gemini / LiteLLM | ✅ ModelRunner + wire types + error classification + pricing tables                          |
| Local Rust-native runtimes      | 🟡 trait satisfied; forward-pass bodies are stubs pinned to the doc's §13 Phase 2b roadmap   |
| vLLM / TensorRT FFI             | 🟡 stubs that compile against the trait; full bodies on §13 Phase 2a/2b                      |
| Pipeline (rakka-streams + cuda-patterns) | ✅ re-export shim + reference hybrid graph                                                |
| CLI (`rakka serve`)             | ✅ TOML config → ActorSystem → gateway; `cost-report`/`rotate-credentials` are stubs         |
| Python bindings                 | 🟡 PyO3 skeleton (`Cluster`, `Deployment`); decorator surface deferred                       |

---

## AI-assisted development

If you're using Claude Code, Cursor, or another AI coding assistant on
a project that depends on `rakka-inference`, install our
**[ai-skills bundle](ai-skills/)**:

```sh
/plugin install /path/to/rakka-inference/ai-skills
```

Seven skills cover the consumer-facing surface — quickstart, choosing
a runtime, wiring remote providers, composing pipelines, deploying to
production, troubleshooting typed errors, and extending with a new
backend. Each `SKILL.md` is a thin router into the canonical docs
(this README, the per-crate READMEs, the architecture RFC), so it
stays in sync with the code instead of restating API surfaces that
belong in rustdoc.

The companion [rakka ai-skills bundle](https://github.com/rustakka/rakka/tree/main/ai-skills)
ships skills for actor design, supervision, persistence, clustering,
and Python bindings. Install both bundles together when you're
building a service that uses both rakka primitives and rakka-inference
runtimes.

---

## Release management

Releases are fully automated. Land a `feat:` / `fix:` commit on `main`
and the version-bump workflow tags `vX.Y.Z`; the release workflow
fires on the tag, runs `cargo xtask verify`, builds binaries for five
platforms, generates release notes from `git log`, and publishes the
allowlisted crates to crates.io in dependency order with idempotent
retry.

| Task                                        | How                                                            |
|---------------------------------------------|----------------------------------------------------------------|
| Bump + tag based on Conventional Commits    | Auto on push to `main` via `.github/workflows/version-bump.yml`. |
| Force a specific version                    | `Release-As: x.y.z` in commit footer.                          |
| Run the full release pipeline manually      | Actions → Release → Run workflow.                              |
| Dry-run before tagging                      | Actions → Release → Run workflow → `dry_run: true`.            |
| Inspect publishable vs gated crates         | `cargo xtask release-checklist`.                               |
| Audit anti-pattern regressions              | `cargo xtask audit` / `cargo xtask audit --check`.             |
| Run the same checks CI runs                 | `cargo xtask verify`.                                          |

Full operator runbook: **[`RELEASING.md`](RELEASING.md)**.
Contributor guide: **[`CONTRIBUTING.md`](CONTRIBUTING.md)**.

## License

Apache-2.0. See [`LICENSE`](LICENSE) once it lands; the workspace
inherits the rakka project license.
