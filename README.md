# atomr-infer

**One supervised actor topology for every place a model can run.**
Local GPU runtimes (vLLM, TensorRT, ONNX Runtime, Candle, cudarc,
mistral.rs) and managed APIs (OpenAI, Anthropic, Gemini, LiteLLM)
sit under the same routing CRDT, the same supervision tree, the same
backpressure story. A request doesn't know — and doesn't need to — whether
it landed on an H100 two racks away or in another company's data center.

```toml
[dependencies]
inference = { version = "0.2", features = ["openai", "anthropic", "candle", "pipeline"] }
```

```rust
use inference::prelude::*;

// Same value object describes a vLLM-on-4×H100 replica or a Gemini Vertex
// deployment. The `runtime` field is the only thing that changes —
// and it's auto-inferred from the model name when omitted.
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

Built on [`rakka`](../rakka) for actor supervision, clustering, and
CRDTs, and on [`rakka-accel`](../rakka-accel) for two-tier GPU
supervision. Cost, latency, and reliability stop being three pipelines
and become one.

---

## Why

Production AI rarely runs only on owned hardware. Frontier models,
burst capacity, and compliance edge cases all push work onto managed
APIs. Bolting providers onto a separate retry / rate-limit /
observability stack from your local GPU pool fragments the system —
and the cracks are exactly where 3 a.m. pages come from.

| You'd otherwise hand-roll                                   | atomr-infer gives you                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------- |
| One routing layer for local pools, another for the API SDK  | Single routing CRDT — `gpt-4o` and `llama-3.1-70b` resolve through the same path |
| Per-process token buckets that 429 on cluster scale-out     | `RateLimiterActor` over `atomr_distributed_data::GCounter` — one bucket, all nodes |
| Hand-written retry / breaker / backoff per provider         | `CircuitBreakerActor` + jittered retry + content-filter triage, one strategy    |
| Sticky CUDA-context recovery glued to async tasks           | `rakka_accel::error::device_supervisor_strategy()` adopted unchanged            |
| Cascade graphs duct-taped from threadpools and channels     | `InferenceCascade` / `DynamicBatchingServer` / `ModelReplicaPool` actors        |
| Credential rotation that drops in-flight traffic            | `RemoteSessionActor::rebuild` drains old, routes new — zero dropped requests    |
| A no-GPU egress server that still pulls `cudarc` transitively | `--features remote-only` ⇒ `cudarc`, `rakka-accel`, `candle` not in the graph |
| Cost guardrails as Slack alerts after the bill arrives      | `Budget { max_spend_per_hour_usd, on_exceeded: Reject }` enforced at the actor  |

Every concern that's normally a separate library or a separate
incident is folded into one supervised graph with typed messages.

---

## 30-second tour

```sh
# Stand up an OpenAI-compatible gateway over real (or mocked) providers.
cargo run -p inference-cli --features all-remote -- serve --config demo.toml

# End-to-end demo (happy path / 429 retry / circuit-open) without
# spending a cent — wiremock under the hood.
cargo run --bin remote_only_demo

# Pure-remote binary, zero GPU deps in the graph.
cargo build -p inference --no-default-features --features remote-only
```

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
distributed rate limiting via `atomr_distributed_data::GCounter` and
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

## What you don't have to think about

- **Two-tier GPU supervision.** `local-gpu` wires `WorkerActor` /
  `ContextActor` to `rakka_accel::error::device_supervisor_strategy()`.
  Sticky-error CUDA contexts get `Restart`; OOM gets `Resume`;
  unrecoverable failures `Stop`. No panic-string parsing in your code.
- **Distributed rate limits.** `RateLimiterActor` shares its
  token-spent log across cluster nodes through
  `atomr_distributed_data::GCounter`. Two members calling OpenAI on
  the same API key collectively respect the bucket — no surprise 429
  storms on scale-out.
- **Typed circuit-breaker propagation.** When the breaker opens, the
  caller sees
  `InferenceError::CircuitOpen { provider, opened_at_unix_ms, retry_at_unix_ms }`.
  Fall back, surface a 429, or queue — without knowing whether the
  bottleneck was GPU memory or a remote outage.
- **Pipelines from blueprints, not threadpools.** Enable
  `cuda-patterns` and `inference::cuda_patterns::{DynamicBatchingServer,
  InferenceCascade, ModelReplicaPool, FairShareScheduler,
  ModelHotSwapServer, SpeculativeDecoder, MoeRouter}` are one import
  away. Plug a closure into `ModelRunner::execute` and you've composed
  §9 of the architecture doc.
- **Compile-time dependency budgets.**
  `cargo build -p inference --features remote-only` produces a binary
  with zero `cudarc`, zero `rakka-accel`, zero `candle`, zero `pyo3`
  in the graph. Layered crates make the invariant load-bearing, not
  aspirational.
- **Hot credential rotation.** `RemoteSessionActor::rebuild` drains
  in-flight requests on the old credential and routes new ones on the
  rotated value. Zero dropped traffic.

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
a project that depends on `atomr-infer`, install our
**[ai-skills bundle](ai-skills/)** — seven skills covering quickstart,
choosing a runtime, wiring remote providers, composing pipelines,
deployment, typed-error troubleshooting, and extending with a new
backend.

```text
/plugin marketplace add rustakka/atomr-infer
/plugin install atomr-infer-ai-skills@atomr-infer
```

Each `SKILL.md` is a thin router into the canonical docs (this README,
the per-crate READMEs, the architecture RFC) so the skills stay in
sync with the code instead of restating API surfaces that belong in
rustdoc. Other harnesses (Cursor, Codex CLI, Gemini CLI, Aider, etc.)
have install instructions in [`ai-skills/README.md`](ai-skills/README.md).

Companion bundles for the broader stack:

- [`rakka` ai-skills](https://github.com/rustakka/atomr/tree/main/ai-skills)
  — actor design, supervision, persistence, clustering, Python bindings.
- [`rakka-accel` ai-skills](https://github.com/rustakka/atomr-accel/tree/main/ai-skills)
  — DeviceActor, kernel selection, two-tier GPU supervision, backend choice.

Install all three when you're building a service that uses rakka
primitives, rakka-accel GPU acceleration, and atomr-infer runtimes.

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
