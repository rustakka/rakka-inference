# `rustakka-inference`: Multi-Runtime GPU and Remote Inference as a Supervised Actor System

**Status:** Draft / RFC v4
**Scope:** Architectural design for hosting heterogeneous inference workloads — local GPU runtimes (autoregressive LLMs, diffusion, TTS, STT, embeddings, vision) **and remote inference providers** (OpenAI, Anthropic, Google Gemini, LiteLLM proxy) — as actors within the [rakka](https://github.com/rustakka/rakka) runtime. vLLM is the canonical backend for local LLM workloads; Rust-native runtimes (TensorRT, ONNX Runtime, Candle, cudarc) are first-class for non-LLM local workloads; remote provider runtimes are first-class for offloading inference to managed APIs.
**Companion document:** [`rustakka-cuda-architecture.md`](./rustakka-cuda-architecture.md). This doc references its sections as `CUDA §N`.
**Supersedes:**
- v1 (`rustakka-vllm-architecture.md`)
- v2 (`rustakka-inference-architecture-v2.md`)
- v3 (`rustakka-inference-architecture-v3.md`)

**v4 changes:** integrates remote inference providers (OpenAI, Anthropic, Gemini, LiteLLM) as a first-class runtime category. Adds the actor primitives needed to handle remote-specific concerns: distributed rate limiting, exponential-backoff retry, circuit breakers, request queuing under provider capacity constraints, and timeout handling. The `ModelRunner` trait is extended with `transport_kind()` to distinguish local-GPU from remote-network execution. Three new crates are added: `inference-remote-core` (shared HTTP / rate-limit / retry primitives) and per-provider crates (`inference-runtime-openai`, `inference-runtime-anthropic`, `inference-runtime-gemini`, `inference-runtime-litellm`). The user-facing API is unchanged: a remote deployment looks identical to a local one apart from its `runtime` field.

---

## 1. Motivation

The architecture from v3 covered local GPU inference comprehensively, but a production AI system rarely runs only on owned hardware. Frontier models, scale-elastic workloads, and capabilities not yet available locally (e.g., the largest closed models) all benefit from offload to managed APIs. Treating these as a separate, parallel infrastructure — distinct routing, distinct retry logic, distinct observability — fragments the system. Treating them as just another runtime under the same actor decomposition unifies it.

The strategic shape is therefore:

- **Local GPU runtimes** (`vllm`, `tensorrt`, `ort`, `candle`, `cudarc`, `mistralrs`) for hardware-resident inference, as in v3.
- **Remote provider runtimes** (`openai`, `anthropic`, `gemini`, `litellm`) for managed-API offload. Implemented in Rust, GIL-free, with HTTP/2 and connection pooling.
- **Same `Deployment` value object** for both. The `runtime` field selects local vs remote; everything else (routing CRDT, `RequestActor` lifecycle, deployment manager, supervision) is shared.
- **Hybrid pipelines compose naturally.** A request that classifies cheaply on a local model and escalates to GPT-4o for hard cases is one supervised actor graph spanning local and remote, with one trace, one backpressure story.

The strategic value over a separate-infrastructure approach:

- **Unified routing.** A request hitting `/v1/chat/completions` with `model="gpt-4o"` lands on a remote deployment; with `model="llama-3.1-70b"` it lands on a local one. The gateway, request actor, and routing CRDT don't know or care.
- **Unified rate limiting and capacity management.** The same backpressure mechanism (bounded mailboxes, `rustakka-streams`) handles "GPU saturated" and "OpenAI rate limit hit" with the same semantics propagated upstream.
- **Distributed rate limit coordination.** Multiple cluster nodes calling the same provider with the same API key share a CRDT-backed rate limit. No surprise 429s from naive client-side limiting on each node independently.
- **Provider failure handling.** Circuit breakers, retry policies, content-filter refusal handling, and timeout semantics are shared infrastructure across providers, configured per-deployment.
- **Hybrid pipelines.** Cost optimization (cheap local for hot path, remote for fallback or escalation), capability fallback (try local model first, escalate to remote on confidence threshold), regional failover (remote provider when local cluster is saturated). All as supervised actor compositions.
- **Observability.** Per-provider latency, per-API-key spend tracking, per-deployment 429 rate, per-deployment circuit-breaker state — all in the same metrics pipeline as GPU utilization and KV cache hit rates.

The strategic value preserved from v3:

- Throughput scales linearly across native runtimes (local) and across remote workers (no GIL cost).
- Multi-modality clusters are first-class.
- Operational visibility of resource costs (GIL exposure for local Python, API spend for remote).
- Selective compilation: deployments that don't use remote runtimes don't compile the HTTP infrastructure.

---

## 2. Engine Anatomies

The actor decomposition is the same across all runtimes — local and remote. Each runtime brings its own engine-step internals.

### 2.1 vLLM V1 (canonical local LLM backend; `inference-runtime-vllm`)

vLLM V1's process structure already mirrors the actor archetypes from the CUDA doc. Engine core owns Scheduler, KVCacheManager, StructuredOutputManager, ModelExecutor, KV Connector. Each Worker owns ModelRunner, paged KV cache tensors, drafter, rejection sampler, NCCL communicator handles. The engine step (Schedule → Forward pass → Postprocess) runs hundreds of times per second.

### 2.2 TensorRT (`inference-runtime-tensorrt`)

Pre-compiled, opaque, optimized kernel sequences. Rust path is FFI to `libnvinfer.so`. Engine: serialized plan + `ExecutionContext` per concurrent request + caller-provided CUDA stream. No scheduler, no KV cache, batching is stacking inputs along batch dim.

### 2.3 ONNX Runtime (`inference-runtime-ort`)

Rust path via `ort` crate. Engine: Session + IoBinding + async Run on caller-provided CUDA stream. Same shape as TensorRT.

### 2.4 Candle (`inference-runtime-candle`)

Pure Rust transformers on `cudarc`. Model-specific Rust code; no separate runtime layer.

### 2.5 Custom local (`inference-runtime-cudarc`, `inference-runtime-mistralrs`, third-party)

User-supplied `ModelRunner` for novel architectures, research code, model-specific runtimes.

### 2.6 Python-only local (XTTS, Bark, diffusers without Rust path)

Python kept where no Rust binding exists. `PythonGpuBridge` (CUDA §5.9, in `inference-python-bridge`) handles allocator and stream entry. Marked `runtime = "python:<name>"`.

### 2.7 Remote inference providers (`inference-runtime-openai`, `inference-runtime-anthropic`, `inference-runtime-gemini`, `inference-runtime-litellm`)

**New in v4.** Remote providers expose HTTP/JSON APIs over TLS. The "engine" is fundamentally different from local runtimes:

| Component | Local GPU runtime | Remote provider runtime |
|---|---|---|
| Compute substrate | CUDA stream | TCP connection |
| Concurrency unit | GPU + CUDA context | API-key + concurrent-slot |
| Scheduler | Continuous batching, token budget | Bounded request queue, rate-limit-aware worker pool |
| Capacity bottleneck | VRAM, SM utilization | Provider rate limits (RPM, TPM, concurrency) |
| Failure mode | Context poisoning, OOM | 429s, 5xx, timeouts, content-filter refusals, auth errors |
| Recovery | Two-tier `WorkerActor` → `ContextActor` rebuild | Reconnect, refresh credentials, circuit breaker |
| Cost model | Hardware capex / GPU-hours | Per-token billing |

The actor decomposition still works — `WorkerActor`, `EngineCoreActor`, `RequestActor`, `DpCoordinatorActor` all apply — but the underlying mechanisms are network-shaped instead of GPU-shaped:

- **`RemoteEngineCoreActor`** owns a request queue (priority + arrival order), a worker pool, a rate limiter, a circuit breaker, and per-deployment configuration. No KV cache, no continuous batching scheduler.
- **`RemoteWorkerActor`** owns one HTTP/2 connection (or connection pool) to the provider. Pulls requests from the engine's queue, checks the rate limiter, executes the HTTP call (streaming or unary), emits `Tokens` events on the per-request output stream.
- **`RateLimiterActor`** holds the token bucket(s) for a `(provider, api_key, model)` tuple. Cluster-shared via `rustakka-distributed-data` so multiple nodes coordinate.
- **`CircuitBreakerActor`** holds the state machine (closed / open / half-open) per `(provider, endpoint)`. Opens on sustained failures; fast-fails requests during open period.
- **`RemoteSessionActor`** is the remote analog of `ContextActor` — handles credential refresh, connection pool rebuild, recovery from auth failures.

The `ModelRunner` trait still applies. The `RemoteRunner` impl (one per provider crate) wraps the HTTP client and translates between `ExecuteBatch` and the provider's wire format.

#### Provider specifics

| Provider | API surface | Streaming | Auth | Notable |
|---|---|---|---|---|
| **OpenAI** | OpenAI Chat Completions / Responses | SSE | Bearer token | Azure OpenAI variant uses different deployment names; same wire format |
| **Anthropic** | Messages API | SSE | x-api-key header | Different rate limit semantics for streaming vs non-streaming |
| **Gemini** | Vertex AI / AI Studio | SSE | OAuth2 (Vertex) or API key (AI Studio) | Regional endpoints; safety settings configurable per-request |
| **LiteLLM** | OpenAI-compatible proxy | SSE | Configurable | Translates to any provider; has its own caching, fallback chains, retries |

LiteLLM is special: it's a proxy that exposes an OpenAI-compatible API in front of any backend. `inference-runtime-litellm` is implemented as a thin wrapper over `inference-runtime-openai` configured with a different endpoint URL. The distinction is preserved at the runtime level so operators can see "this is going through LiteLLM" vs "this is going direct to OpenAI" in deployment configs.

---

## 3. Runtime Backends

### 3.1 Backend taxonomy

| Backend | Crate | Implementation | GIL? | Transport | Dispatcher |
|---|---|---|---|---|---|
| `vllm` | `inference-runtime-vllm` | Python + Triton + PyTorch | Yes (`python-pinned`) | Local GPU | Python-pinned |
| `tensorrt` | `inference-runtime-tensorrt` | FFI to `libnvinfer.so` | No | Local GPU | Native GPU |
| `ort` | `inference-runtime-ort` | `ort` crate (CUDA EP) | No | Local GPU | Native GPU |
| `candle` | `inference-runtime-candle` | Pure Rust on `cudarc` | No | Local GPU | Native GPU |
| `mistralrs` | `inference-runtime-mistralrs` | Rust LLM runtime | No | Local GPU | Native GPU |
| `cudarc` | `inference-runtime-cudarc` | Direct kernel dispatch | No | Local GPU | Native GPU |
| `custom` | (third-party impl) | User-supplied `ModelRunner` | Depends | Local GPU or Remote | Depends |
| `python:<name>` | `inference-runtime-vllm` (scaffolding) | Python wrapper, no Rust path | Yes (`python-pinned`) | Local GPU | Python-pinned |
| **`openai`** | **`inference-runtime-openai`** | **Rust HTTP/2 client** | **No** | **Remote network** | **Standard tokio multi-threaded** |
| **`anthropic`** | **`inference-runtime-anthropic`** | **Rust HTTP/2 client** | **No** | **Remote network** | **Standard tokio multi-threaded** |
| **`gemini`** | **`inference-runtime-gemini`** | **Rust HTTP/2 client** | **No** | **Remote network** | **Standard tokio multi-threaded** |
| **`litellm`** | **`inference-runtime-litellm`** | **`inference-runtime-openai` re-targeted** | **No** | **Remote network** | **Standard tokio multi-threaded** |

Two new dimensions: `Transport` (Local GPU vs Remote network) and the dispatcher (remote runtimes use standard tokio, not pinned).

### 3.2 The default rule (updated)

For any model, the default runtime is selected by this hierarchy:

1. If the model name is a known remote provider model (`gpt-4o`, `claude-sonnet-4`, `gemini-2.0-pro`, etc.) and credentials are configured: use the matching remote runtime.
2. If the model name matches a local model family with a Rust-native backend: use that backend.
3. If it matches a local LLM family without a Rust backend: use `vllm`.
4. If a Python-only runtime has been registered for the model: use `python:<name>`.

The selection logic lives in `inference-core`'s `Deployment::infer_runtime()`. Operators can always override.

### 3.3 GIL isolation between Python deployments

Unchanged from v3. Two GIL-pinned deployments don't share an interpreter by default. Remote runtimes are GIL-free and don't participate in this constraint.

### 3.4 Runtime-specific lifecycle hooks

Local runtimes follow CUDA §5.11 two-tier supervision with runtime-specific rebuild logic.

Remote runtimes have an analogous two-tier shape but the rebuild semantics differ:

| Runtime | On session rebuild |
|---|---|
| `openai` | Refresh API key from secret store; rebuild HTTP client; re-establish HTTP/2 connection pool |
| `anthropic` | Same as openai, with different headers |
| `gemini` | Refresh OAuth2 token (Vertex) or API key (AI Studio); rebuild client |
| `litellm` | Refresh credentials for proxy; rebuild client; verify proxy reachable |

Triggers for rebuild are different too:
- Local: CUDA `ContextPoisoned` from sticky errors.
- Remote: sustained auth failures (rotated key), connection pool exhaustion, configuration change to endpoint URL.

### 3.5 Rate limiting and provider reliability (new)

Remote runtimes face a class of concerns local runtimes don't: provider-imposed limits and provider-side failures. These are first-class architectural concerns, not retry-loop afterthoughts.

#### Rate limiting

Provider rate limits are enforced server-side by the provider. The client must respect them or face 429 responses. Limits are typically:

- **Requests per minute (RPM)** per API key, per model.
- **Tokens per minute (TPM)** per API key, per model. Often tracked separately for input and output.
- **Concurrent requests** soft cap, often unstated.

Multiple clients (cluster nodes, deployments, applications) using the same API key share these limits. A naive per-process token bucket on each node will collectively exceed the limit and trigger 429s across the fleet.

**Solution:** `RateLimiterActor` per `(provider, api_key, model)` tuple, cluster-distributed via `rustakka-distributed-data` CRDT. The CRDT is a `GCounter`-based token-spent log; each node maintains a local view and periodically syncs. Approximate distributed token bucket: each node holds local capacity proportional to its share of the cluster; over-spend is bounded by sync interval and local capacity.

For deployments where strict adherence is required (premium API keys with hard caps), a stronger primitive is available: `StrictRateLimiterActor` runs as a cluster-singleton; every request waits for permission via `ask`. Higher latency, exact accounting.

#### Backoff on 429

When a `RemoteWorkerActor` receives a 429:

1. Read the `Retry-After` header if present; otherwise use exponential backoff (initial 1s, doubling, capped at 60s).
2. Inform the local rate limiter that the actual rate is lower than configured (auto-tune).
3. Sleep for the backoff period.
4. Retry the same request, incrementing the request's retry counter.
5. After `max_retries` (default 3), fail the request with `RateLimitExceeded` propagated to the `RequestActor`.

#### Circuit breaker

When sustained failures occur (5xx errors, timeouts, repeated 429s after backoff), opening the circuit prevents the deployment from making bad worse. `CircuitBreakerActor` per `(provider, endpoint)`:

- **Closed (normal):** requests pass through.
- **Open:** all requests fail-fast with `CircuitOpen`. Duration configurable (default 30s).
- **Half-open:** after the open duration, allow one probe request. If it succeeds, close the circuit. If it fails, return to open.

Configuration in deployment:

```toml
[deployment.runtime_config.circuit_breaker]
failure_threshold     = 10        # consecutive failures to open
open_duration_ms      = 30_000    # how long to stay open
half_open_max_probes  = 1
```

#### Timeout handling

Two timeouts per request:

- **Request timeout:** time from send to first byte received. Default 30s for non-streaming, 60s for streaming (allowing for queue time on provider side).
- **Read timeout:** for streaming responses, time between consecutive bytes. Default 10s.

Either timeout triggers the same retry path as a 5xx error, subject to the circuit breaker.

#### Content-filter refusals

Providers return structured errors when content is blocked by safety filters. These are **not retryable** — retrying with the same input gets the same refusal. The `RemoteWorkerActor` recognizes the error shapes per provider and propagates them as typed `ContentFiltered { reason: String }` to the `RequestActor` without retry. This avoids wasting quota and makes the failure visible to upstream.

Other non-retryable errors:
- 400 Bad Request (malformed input, context too long)
- 401 Unauthorized (auth failure — triggers session rebuild instead)
- 403 Forbidden (permission denied for the model/feature)

#### Backpressure on capacity exhaustion

When all `RemoteWorkerActor`s are busy and the request queue is full:

1. New `AddRequest` messages to `RemoteEngineCoreActor` get rejected with `Backpressure` typed error.
2. `RequestActor` either retries on a different deployment (fallback chain), surfaces a 429 to the upstream HTTP client, or queues with a configurable timeout.
3. The behavior is configurable per-deployment: `on_capacity_exhausted = "reject" | "queue" | "fallback"`.

This is the same backpressure mechanism as local GPU deployments, with the same upstream semantics. The upstream `RequestActor` doesn't need to know whether the bottleneck is GPU memory or provider quota.

---

## 4. Foundational Mapping

| Concept | Actor concept | Crate | Notes |
|---|---|---|---|
| API server | `ApiGatewayActor` | `inference-runtime` | Replaces FastAPI/Uvicorn or Triton's HTTP server |
| Per-request lifecycle | `RequestActor` | `inference-runtime` | One per active client request |
| DP coordinator | `DpCoordinatorActor` (cluster-singleton) | `inference-runtime` | Cluster-tools singleton with handover |
| Local engine core | `EngineCoreActor` | `inference-runtime` | Per-replica GPU orchestrator |
| **Remote engine core** | **`RemoteEngineCoreActor`** | **`inference-remote-core`** | **Per-replica HTTP orchestrator; queue + worker pool** |
| Scheduler / batching (local) | Module inside `EngineCoreActor` | runtime crate | NOT an actor |
| **Request queue (remote)** | **Module inside `RemoteEngineCoreActor`** | **`inference-remote-core`** | **Bounded priority queue** |
| KVCacheManager (vLLM) | Module inside `EngineCoreActor` | `inference-runtime-vllm` | LLM-specific |
| ModelExecutor | `ModelExecutorActor` | `inference-runtime` | Local-only; remote uses worker pool directly |
| Local worker | `WorkerActor` ≡ CUDA §3.1 `DeviceActor` | `inference-runtime` | Two-tier supervision for context poisoning |
| **Remote worker** | **`RemoteWorkerActor`** | **`inference-remote-core`** | **One per concurrent slot; HTTP client** |
| ModelRunner impl | `ModelRunnerActor` | runtime crate | Local: GPU kernels. Remote: HTTP request |
| Paged KV cache bytes | `PagedKvCacheActor` | `inference-runtime-vllm` | LLM-specific, local-only |
| **Rate limiter** | **`RateLimiterActor`** | **`inference-remote-core`** | **Cluster-distributed via `rustakka-distributed-data`** |
| **Circuit breaker** | **`CircuitBreakerActor`** | **`inference-remote-core`** | **Per `(provider, endpoint)`** |
| **Remote session** | **`RemoteSessionActor`** | **`inference-remote-core`** | **Analog of `ContextActor`; handles credential refresh** |
| NCCL communicator | `CollectiveActor` (CUDA §3.3, §6.1) | `inference-runtime` | Local multi-GPU only |
| ZMQ / shared-memory IPC | rakka transport | `rustakka-remote` (transitive) | One transport everywhere |

The pattern: **local-specific concerns (GPU, KV cache, NCCL) live in `inference-runtime` and per-runtime crates; remote-specific concerns (HTTP, rate limiting, circuit breakers) live in `inference-remote-core` and per-provider crates. Runtime-agnostic concerns (gateway, request actor, deployment manager, placement) are shared.**

---

## 5. Actor Decomposition

### 5.1 Topology — local and remote

```
                      [HTTP clients]
                            │
                            ▼
                   ApiGatewayActor                      runtime-agnostic
                            │ spawns one per request    (inference-runtime)
                            ▼
                    RequestActor
                            │   ask(routing target)
                            ▼
                  DpCoordinatorActor                     cluster-singleton
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
   │   │       ├─ ModelRunner       │ │  ├─ RemoteWorkerActor   ││
   │   │       ├─ PagedKv (LLM)     │ │  ├─ ...                 ││
   │   │       └─ CollectiveActor   │ │  └─ RemoteWorkerActor   ││
   │   └─ ...                       │ └─────────────────────────┘│
   └──────────────────────┘         │ uses:                      │
                                     │   RateLimiterActor         │
                                     │   CircuitBreakerActor      │
                                     │   RemoteSessionActor       │
                                     └────────────────────────────┘

         ↑                                       ↑
   inference-runtime                inference-remote-core
   inference-runtime-vllm           inference-runtime-openai
   inference-runtime-tensorrt       inference-runtime-anthropic
   inference-runtime-ort            inference-runtime-gemini
   inference-runtime-candle         inference-runtime-litellm
   inference-runtime-cudarc
   inference-runtime-mistralrs
```

The `RequestActor` interaction is identical for both branches: it sends `AddRequest` and receives `Tokens` events on its output stream. The branching happens inside the engine core based on the deployment's runtime.

### 5.2 What's an actor and what's a module

Unchanged from v3. Per-step / per-token operations stay as in-process modules. New cases:

| Component | Decision | Why |
|---|---|---|
| `RemoteEngineCoreActor` | Actor | Per-replica orchestrator; supervised; addressable |
| `RemoteWorkerActor` | Actor | One per concurrent slot; standard worker-pool pattern |
| `RateLimiterActor` | Actor (distributed-data backed) | Cross-deployment, cross-node coordination |
| `CircuitBreakerActor` | Actor | Per `(provider, endpoint)`; stateful |
| `RemoteSessionActor` | Actor | Analog of `ContextActor`; credential lifecycle |
| Request queue (remote) | **Module** | In-process priority queue; per-message mailbox hop would add latency |
| HTTP request building | **Module** | Per-request; lives inside `RemoteWorkerActor` |
| Token bucket math | **Module** | Per-permit-check; lives inside `RateLimiterActor` |

### 5.3 Two-tier supervision (extended)

Local:
- `WorkerActor` (stable parent) → `ContextActor` (restartable, owns CUDA context) → child actors.
- Restart trigger: `ContextPoisoned`.

Remote:
- `RemoteEngineCoreActor` (stable parent) → `RemoteSessionActor` (restartable, owns HTTP client + credentials) → `RemoteWorkerActor` children.
- Restart trigger: sustained auth failures, configuration change, manual rebuild RPC.

The two-tier shape is shared infrastructure; the rebuild logic differs per runtime.

### 5.4 The `ModelRunner` trait (extended)

```rust
// in inference-core
#[async_trait]
pub trait ModelRunner: Send + Sync {
    /// Runs an inference. For local runtimes, dispatches kernels to a stream.
    /// For remote runtimes, sends an HTTP request.
    /// Returns immediately; completion observed via the appropriate strategy.
    async fn execute(&mut self, batch: ExecuteBatch) -> Result<RunHandle, InferenceError>;

    /// Local runtimes load weights to GPU; remote runtimes no-op.
    async fn load_weights(
        &mut self,
        ctx: Option<&Arc<CudaContext>>,
        source: WeightSource,
    ) -> Result<(), Error> {
        Ok(())  // default: no-op for runtimes that don't load weights
    }

    /// Local runtimes rebuild after CUDA context poison; remote runtimes
    /// rebuild after auth failure or connection pool exhaustion.
    async fn rebuild_session(
        &mut self,
        cause: SessionRebuildCause,
    ) -> Result<(), Error>;

    /// Runtime metadata for placement, observability, and dispatcher choice.
    fn runtime_kind(&self) -> RuntimeKind;
    fn transport_kind(&self) -> TransportKind;     // LocalGpu | RemoteNetwork
    fn gil_pinned(&self) -> bool;

    /// Rate limit metadata. For local, returns None. For remote, returns the
    /// configured limits so the RateLimiterActor can be initialized.
    fn rate_limits(&self) -> Option<&RateLimits> { None }
}

pub enum TransportKind {
    LocalGpu,
    RemoteNetwork { provider: ProviderKind },
}

pub enum ProviderKind {
    OpenAi,
    Anthropic,
    Gemini,
    LiteLlm,
    Custom(String),
}

pub enum SessionRebuildCause {
    CudaContextPoisoned,
    RemoteAuthFailure,
    RemoteConfigChange,
    Manual,
}

pub struct RateLimits {
    pub requests_per_minute: Option<u64>,
    pub tokens_per_minute: Option<u64>,
    pub concurrent_requests: Option<u32>,
}
```

`transport_kind()` is what `PlacementActor` and the worker-spawning logic read to decide whether to spawn a `WorkerActor` (local) or `RemoteWorkerActor` (remote), and which dispatcher to use.

### 5.5 Mapping IPC layers (unchanged)

vLLM's ZMQ + shared-memory + asyncio collapses to rakka transport. Remote runtimes use HTTP/2 over TLS for provider communication; this is application-protocol traffic, not IPC, and goes through standard `reqwest` or `hyper` clients held by `RemoteWorkerActor`.

### 5.6 Native local runtimes — the default path (unchanged)

Rust-native local runtimes run on standard rakka GPU dispatcher. No GIL, no allocator collision, CUDA-graph capture works directly.

### 5.7 Python local runtimes — `vllm` and the marked exception (unchanged)

`vllm` and `python:*` runtimes go through `inference-python-bridge`'s `PythonGpuBridge` on a `python-pinned` dispatcher. Default placement is dedicated-interpreter-per-deployment.

### 5.8 Remote runtimes — the network path (new)

For remote runtimes (`openai`, `anthropic`, `gemini`, `litellm`, custom), the `RemoteEngineCoreActor` runs on a standard tokio multi-threaded dispatcher with a configurable worker pool. Properties:

- **No GIL.** Pure Rust HTTP/2 client; throughput scales with cores.
- **No GPU.** Deployment can be scheduled on control nodes or any node with network egress; doesn't consume GPU inventory.
- **No CUDA context.** `WorkerActor`'s context-poisoning recovery doesn't apply; `RemoteSessionActor` handles the analogous credential / connection lifecycle.
- **Bounded concurrency.** Worker pool size = `max_concurrent` config, typically matching provider's concurrent-request soft limit.
- **Cluster-distributed rate limiting.** `RateLimiterActor` syncs token-spent counters across nodes via `rustakka-distributed-data` CRDT.
- **Standard backpressure.** Bounded mailbox on `RemoteEngineCoreActor`; bounded request queue inside; `Backpressure` typed error returned when exhausted.
- **Streaming first-class.** `RemoteWorkerActor` consumes provider SSE streams and emits `Tokens` events on the per-request output stream — same shape as local runtimes' continuous-batch output.
- **Connection pooling.** Each `RemoteWorkerActor` holds its share of an HTTP/2 connection pool (typically multiplexed over fewer than N TCP connections via H2 streams).
- **Cost tracking.** `MetricsActor` aggregates per-provider token usage from response headers or response bodies; emits `inference_remote_tokens_in` / `inference_remote_tokens_out` per deployment per model.

```rust
// in inference-runtime-openai
pub struct OpenAiRunner {
    client: reqwest::Client,
    api_key: SecretString,
    endpoint: Url,
    model: String,
    rate_limiter: ActorRef<RateLimiterActor>,
    circuit_breaker: ActorRef<CircuitBreakerActor>,
    config: OpenAiConfig,
}

#[async_trait]
impl ModelRunner for OpenAiRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> Result<RunHandle, InferenceError> {
        // 1. Wait for rate limiter permit.
        let permit = self.rate_limiter.ask(AcquirePermit {
            requests: 1,
            tokens_estimate: batch.estimated_tokens(),
        }).await?;

        // 2. Check circuit breaker.
        self.circuit_breaker.ask(CheckOpen).await?;

        // 3. Build request, send, return future that resolves on first byte.
        let req = self.build_request(&batch)?;
        let response = self.client.post(self.endpoint.clone())
            .bearer_auth(self.api_key.expose_secret())
            .json(&req)
            .send()
            .await
            .map_err(|e| self.classify_error(e))?;

        // 4. Wrap response stream in RunHandle that emits Tokens events.
        Ok(RunHandle::streaming(
            response.bytes_stream(),
            self.deserialize_chunks(),
        ))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> Result<(), Error> {
        self.api_key = SecretString::from(self.config.api_key.read()?);
        self.client = build_client(&self.config)?;
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind { RuntimeKind::OpenAi }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork { provider: ProviderKind::OpenAi }
    }
    fn gil_pinned(&self) -> bool { false }
    fn rate_limits(&self) -> Option<&RateLimits> { Some(&self.config.rate_limits) }
}
```

The classification of errors (`classify_error`) is provider-specific and produces typed errors that flow up to the `RequestActor`:

| Error class | Retryable? | Affects circuit breaker? |
|---|---|---|
| `RateLimited { retry_after }` | Yes (with backoff) | After threshold |
| `ServerError(5xx)` | Yes | Yes |
| `Timeout` | Yes | Yes |
| `ContentFiltered { reason }` | No | No |
| `ContextLengthExceeded` | No | No |
| `BadRequest(400)` | No | No |
| `Unauthorized(401)` | No (triggers session rebuild) | No |
| `Forbidden(403)` | No | No |
| `NetworkError(io)` | Yes (network blip) | Yes |

---

## 6. Request Lifecycle

### 6.1 Local deployment

A walk through `curl -X POST /v1/chat/completions` with `model="llama-3.1-70b"`:

1. `ApiGatewayActor` accepts the HTTP request, spawns a `RequestActor`.
2. `RequestActor` `ask`s `DpCoordinatorActor` for the best `EngineCoreActor` (lowest load score).
3. `RequestActor` `tell`s the chosen `EngineCoreActor` an `AddRequest`.
4. `EngineCoreActor` enqueues; scheduler runs each step; `ModelExecutorActor` dispatches to `WorkerActor`s; `ModelRunner::execute` runs forward pass.
5. `EngineCoreActor` emits `Tokens` events; `RequestActor` accumulates and streams to client.
6. On stop, `RequestActor` cleans up.

Same as v3.

### 6.2 Remote deployment (new)

A walk through `curl -X POST /v1/chat/completions` with `model="gpt-4o"`:

1. `ApiGatewayActor` accepts the HTTP request, spawns a `RequestActor`.
2. `RequestActor` `ask`s `DpCoordinatorActor` for the best `RemoteEngineCoreActor`.
3. `RequestActor` `tell`s the chosen `RemoteEngineCoreActor` an `AddRequest`.
4. `RemoteEngineCoreActor` enqueues request in its priority queue.
5. A free `RemoteWorkerActor` from the pool picks up the next queued request.
6. Worker `ask`s `RateLimiterActor` for a permit (blocks until allowed).
7. Worker `ask`s `CircuitBreakerActor` to verify circuit is closed (fast-fails if open).
8. Worker sends HTTP/2 POST to provider; receives SSE stream.
9. Worker parses chunks; emits `Tokens` events on the per-request output stream.
10. `RequestActor` accumulates tokens, streams to client.
11. On stream end, worker reports usage to `MetricsActor`, returns to pool.
12. On stop, `RequestActor` cleans up.

Failure paths:
- 429 → worker backs off, retries up to `max_retries`, eventually fails to `RequestActor` with `RateLimitExceeded`.
- 5xx / timeout → worker retries with exponential backoff, contributes to circuit breaker failure count.
- Content filter → worker fails to `RequestActor` immediately with `ContentFiltered`, no retry.
- Auth failure → worker triggers `RemoteSessionActor::rebuild`, retries the request once.
- Backpressure (queue full) → `AddRequest` is rejected; `RequestActor` either retries on a fallback deployment or surfaces 429 to the upstream client.

The `RequestActor` doesn't know whether the bottleneck was GPU memory, GIL contention, or provider rate limit. The error types are uniform.

---

## 7. Cluster Operation

### 7.1 Cluster shape (extended)

Three node roles; remote deployments add a fourth implicit role:

- **Control nodes** — host cluster-singletons (`DeploymentManagerActor`, per-model `DpCoordinatorActor`s, `RateLimiterActor`s for strict providers).
- **GPU/serving nodes** — host local `WorkerActor`s and `EngineCoreActor`s.
- **Edge/router nodes** (optional) — extra `ApiGatewayActor` replicas.
- **Egress nodes** (optional, new) — nodes designated for hosting `RemoteEngineCoreActor`s and `RemoteWorkerActor`s. CPU-only; require network egress to the relevant provider; can be the same as control nodes for small deployments.

A small cluster can collapse all four roles onto every node; a large cluster separates them for failure isolation.

### 7.2 Runtime-aware placement (extended)

Three constraint classes now:

1. **Topology constraints** (local only): TP NVLink, DP failure domains, PP link costs.
2. **Runtime constraints** (local Python): two GIL-pinned deployments don't share interpreters.
3. **Network constraints** (remote): remote deployments need egress to the provider; some providers have regional endpoints (Vertex AI Gemini); some deployments require specific outbound IPs (corporate proxy / managed identity).

`PlacementActor` reads `transport_kind()` to choose a node:
- `LocalGpu` → place on a node with available GPUs satisfying topology constraints.
- `RemoteNetwork { provider }` → place on any egress-capable node; prefer nodes already hosting other deployments using the same `(provider, api_key)` to enable shared `RateLimiterActor` co-location for low-latency rate-limit checks.

Example mixed placement:

```
deployment llama-3.1-70b-instruct (runtime=vllm, transport=LocalGpu)
  replica dp=0  on node-h100-a  gpus 0,1,2,3   (TP=4, NVLink island A)
  replica dp=1  on node-h100-b  gpus 0,1,2,3

deployment gpt-4o (runtime=openai, transport=RemoteNetwork)
  replica dp=0  on node-egress-1  workers=50    (HTTP/2 client pool, no GPU needed)
  replica dp=1  on node-egress-2  workers=50    (independent pool, same API key)

deployment claude-sonnet-4 (runtime=anthropic, transport=RemoteNetwork)
  replica dp=0  on node-egress-1  workers=20

deployment whisper-tensorrt (runtime=tensorrt, transport=LocalGpu)
  replica dp=0..3 on GPU 4..7 across nodes-h100-a/b
```

The local LLM, remote OpenAI deployment, remote Anthropic deployment, and local TensorRT whisper coexist. Each has its own placement and resource constraints; the routing CRDT serves all of them uniformly.

### 7.3 Routing (unchanged)

Same routing CRDT, same lookup. A request's runtime is determined by the deployment it routes to, not by the gateway logic.

### 7.4 Multi-tenancy (extended)

For local deployments: per-deployment buffer table (CUDA §5.8), MIG hardware isolation, priority-mapped `StreamAllocator`.

For remote deployments: per-deployment `RateLimiterActor` (shared across nodes for the same `(provider, api_key, model)`), per-deployment `CircuitBreakerActor`, per-deployment cost tracking. Two deployments using different API keys against the same provider don't share rate limit state. Two deployments using the same API key automatically share via the CRDT key.

### 7.5 Lifecycle operations (unchanged)

Same scale-out / scale-in / hot-swap / canary patterns. For remote deployments, "scale" means worker pool size or replica count (independent worker pools, possibly with different API keys for higher aggregate throughput).

### 7.6 Failure handling (extended)

| Tier | Detection | Recovery |
|---|---|---|
| Local context poisoned | `ContextPoisoned` | CUDA §5.11 in-place rebuild |
| Local worker / node loss | Cluster gossip | Replica re-placed |
| Control plane loss | Cluster-tools singleton handover | Recover from `rustakka-persistence` |
| **Remote provider rate limit** | **429 from provider** | **`RateLimiterActor` auto-tunes; worker backoff; retry** |
| **Remote provider outage** | **Sustained 5xx / timeouts** | **`CircuitBreakerActor` opens; fast-fail; periodic probe** |
| **Remote auth failure** | **401 / 403** | **`RemoteSessionActor` rebuilds (refresh creds, rebuild client); retry once** |
| **Remote network partition** | **Connection errors** | **Worker reports to circuit breaker; retry on different connection or fail to fallback deployment** |
| **API key compromised / rotated** | **Operator action** | **Hot-swap deployment with new credentials; old deployment drains** |

### 7.7 Observability (extended)

Local metrics (KV cache hit rate, GPU SM utilization, context rebuild events) plus remote metrics:

- **Per-provider request rate** (succeeded, 429, 5xx, content-filtered, timed-out)
- **Per-deployment p50/p95/p99 latency** (TTFT, ITL, total)
- **Per-deployment circuit breaker state** (closed / open / half-open + transition events)
- **Per-deployment rate limit utilization** (used vs configured RPM / TPM)
- **Per-deployment cost** (input tokens × $/Mtok + output tokens × $/Mtok)
- **Worker pool utilization** (busy / idle workers per `RemoteEngineCoreActor`)
- **Queue depth** (requests waiting in `RemoteEngineCoreActor` queue)

All emitted via the same `MetricsActor` pipeline as local metrics; same Prometheus / OpenTelemetry export; same dashboard.

---

## 8. Multi-Model Deployment Shapes

### 8.1 Shape A — Single big cluster, mixed local and remote

A 16×H100 cluster with two egress-capable nodes:

```
node-h100-a (8 GPUs):
  llama-70b   dp=0   on GPUs 0..3   (vllm, local, NVLink island A)
  qwen-32b    dp=0   on GPUs 4..5   (vllm, local, NVLink island B)
  whisper     dp=0..1 on GPU 6     (tensorrt, local)
  bge-large   dp=0..3 on GPU 7     (ort, local)

node-h100-b (8 GPUs):
  llama-70b   dp=1   on GPUs 0..3
  mistral-7b  dp=0..3 on GPUs 4..7  (mistralrs, local)

node-egress-1 (CPU only):
  gpt-4o      dp=0   workers=50    (openai, remote)
  claude-sonnet-4  dp=0   workers=20   (anthropic, remote)
  gemini-2.0-pro   dp=0   workers=30   (gemini, remote)

node-egress-2 (CPU only):
  gpt-4o      dp=1   workers=50    (openai, remote, second pool)
  litellm-fallback  dp=0   workers=100  (litellm, remote, proxy with fallback chain)
```

Local and remote coexist; routing serves both via the same CRDT. **Use when:** mixed workload with cost-optimization needs (local for high-volume cheap workloads, remote for capability needs or burst capacity).

### 8.2 Shape B — Microservice per model (remote variant)

A pure-remote deployment can run as a microservice cluster too, useful when:
- Different teams own different remote API keys.
- Per-team cost attribution requires per-team Kubernetes namespace boundaries.
- Specific compliance requirements (e.g., a deployment must run only in EU regions for GDPR).

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: gpt-4o-egress, namespace: team-research }
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: engine
        image: rakka-inference:1.0
        command: ["rakka", "serve", "--config", "/etc/gpt-4o.toml"]
        env:
        - { name: RAKKA_RUNTIME, value: "openai" }
        - { name: OPENAI_API_KEY, valueFrom: { secretKeyRef: { name: openai-key, key: token } } }
        # No GPU resources requested
```

### 8.3 Shape C — Federated (unchanged)

Each model is its own cluster; gateway federates.

### 8.4 What's identical across shapes

Same as v3. Remote deployments use the same actor topology regardless of shape.

---

## 9. Pipeline Composition

### 9.1 Hybrid local-remote agent (new in v4)

A cost-optimized agent that classifies cheaply on a local model and escalates to a frontier remote model for hard queries:

```python
@inference_actor
class HybridAgent:
    async def pre_start(self, ctx):
        self.local_router    = await ctx.lookup("llama-3.1-8b-router")     # local, cheap
        self.local_executor  = await ctx.lookup("mistral-7b")              # local, cheap
        self.remote_planner  = await ctx.lookup("gpt-4o")                  # remote, smart
        self.remote_fallback = await ctx.lookup("claude-sonnet-4")         # remote, fallback

    async def handle(self, ctx, query: str):
        # Cheap local classification
        intent = await self.local_router.ask(Classify(query))

        if intent.complexity == "simple":
            # All-local path
            return await self.local_executor.ask(query)

        # Complex query: escalate to remote
        try:
            plan = await self.remote_planner.ask(Plan(query), timeout=30s)
        except RateLimitExceeded:
            # OpenAI saturated, fall back to Anthropic
            plan = await self.remote_fallback.ask(Plan(query))
        except CircuitOpen:
            # OpenAI circuit breaker open, fall back
            plan = await self.remote_fallback.ask(Plan(query))
        except ContentFiltered as e:
            # Don't fall back — content was rejected
            return f"Request was filtered: {e.reason}"

        # Execute plan locally (cheap)
        return await self.local_executor.ask(Execute(plan))
```

Properties:

- **Cost optimization.** Simple queries pay only for local compute; complex queries pay for remote API call but reuse local execution.
- **Availability through diversity.** When OpenAI is rate-limited, the actor falls back to Anthropic. When both are unavailable, the actor can degrade gracefully (return a partial answer from the local model, or surface a typed error to the user).
- **One supervised graph.** Failures at any stage propagate as actor failures; fallback is explicit in the actor logic, not buried in HTTP retry interceptors.
- **One trace.** A flame graph for one query shows: classify (local) → plan (remote, OpenAI) → execute (local). Each stage is annotated with runtime, latency, and cost.
- **One backpressure story.** If local executor is GPU-saturated, the agent's mailbox fills, applying backpressure to upstream. Same for remote rate-limit hits.

### 9.2 Routing by capability and budget (new)

```python
@inference_actor
class TieredRouter:
    """Routes by quality tier and cost budget."""

    async def pre_start(self, ctx):
        self.tiers = {
            "premium":  await ctx.lookup("gpt-4o"),                    # remote, $$$
            "standard": await ctx.lookup("claude-sonnet-4"),           # remote, $$
            "fast":     await ctx.lookup("llama-3.1-70b-instruct"),    # local, $
            "cheap":    await ctx.lookup("mistral-7b"),                # local, ¢
        }

    async def handle(self, ctx, query: str, tier: str = "standard", budget_usd: float = 0.10):
        deployment = self.tiers[tier]
        estimated_cost = await deployment.ask(EstimateCost(query))

        if estimated_cost > budget_usd:
            # Downgrade to a cheaper tier
            deployment = self.tiers["fast"]

        return await deployment.ask(query)
```

The `EstimateCost` message is a typed query that returns predicted cost based on input length, model pricing, and expected output length. Built into `inference-runtime-{openai,anthropic,gemini,litellm}`; defaults to compute-cost-amortized for local deployments.

### 9.3 Other compositions (extended)

- **Voice agent (v3 example, unchanged):** STT (TRT) → LLM (vLLM) → TTS (ORT). All local.
- **Voice agent with remote LLM:** STT (TRT) → LLM (OpenAI) → TTS (ORT). Local audio processing, remote intelligence. Different cost profile.
- **Document understanding with remote summarization:** OCR (TRT) → layout (ORT) → reranker (ORT) → summarizer (Claude via remote). Vision processing local; summarization remote.
- **Agentic loop with mixed tools:** LLM (vLLM, local) → tool calls (custom actors, mostly CPU) → escalation to GPT-4o for hard tool selection → response synthesis (LLM, local).

The mix-and-match is operationally invisible. The agent code uses the same `ctx.lookup` and `ask` regardless of whether the target is on a GPU two racks away or on a server in another company's data center.

---

## 10. Crate Topology

The architecture is realized as a Rust workspace with strict layering. v4 adds remote runtime crates following the same patterns established in v3.

### 10.1 Workspace layout

```
rustakka-inference/                       (workspace root)
│
├── crates/
│   ├── inference-core/                   foundation: traits, types, no actor deps
│   ├── inference-runtime/                actor implementations on rakka
│   ├── inference-python-bridge/          Python interop (CUDA §5.9)
│   │
│   ├── inference-runtime-vllm/           vLLM (local, Python)
│   ├── inference-runtime-tensorrt/       TensorRT (local, Rust FFI)
│   ├── inference-runtime-ort/            ONNX Runtime (local)
│   ├── inference-runtime-candle/         Candle (local, pure Rust)
│   ├── inference-runtime-cudarc/         direct kernel dispatch (local)
│   ├── inference-runtime-mistralrs/      Rust LLM alternative (local)
│   │
│   ├── inference-remote-core/            shared HTTP / rate-limit / retry / circuit-breaker
│   ├── inference-runtime-openai/         OpenAI (+ Azure OpenAI variant)
│   ├── inference-runtime-anthropic/      Anthropic
│   ├── inference-runtime-gemini/         Google Gemini (Vertex + AI Studio)
│   ├── inference-runtime-litellm/        LiteLLM proxy (re-uses openai)
│   │
│   ├── inference-pipeline/               rustakka-streams integration
│   ├── inference-testkit/                mock runtimes (local + remote), deterministic completion
│   ├── inference-cli/                    `rakka` binary
│   ├── inference-py-bindings/            PyO3 bridge for Python users
│   │
│   └── inference/                        rollup; re-exports only, no logic
│
├── examples/
├── benches/
├── xtask/
└── Cargo.toml                            workspace
```

Eighteen crates total in v4 (up from fourteen in v3). The four added: `inference-remote-core`, `inference-runtime-openai`, `inference-runtime-anthropic`, `inference-runtime-gemini`, `inference-runtime-litellm`.

### 10.2 Dependency direction (extended)

```
inference-core         (leaf; no internal deps)
       ▲
       │
inference-runtime      ← inference-python-bridge      ← inference-remote-core
       ▲                       ▲                              ▲
       │                       │                              │
       │   ┌───────────────────┴────────────┐    ┌────────────┴──────────────┐
       │   │                                │    │                            │
inference-runtime-{tensorrt, ort,      inference-runtime-vllm    inference-runtime-{openai, anthropic, gemini}
   candle, cudarc, mistralrs}                                               ▲
                                                                            │
                                                                inference-runtime-litellm
       ▲                                                                    ▲
       │                                                                    │
inference-pipeline     inference-cli      inference-testkit ─────────────────┘
       ▲                       ▲                   ▲
       │                       │                   │
       └─────── inference (rollup) ────────────────┘
                       ▲
                       │
              inference-py-bindings
```

`inference-remote-core` is parallel to `inference-python-bridge`: a layer between `inference-runtime` and the per-runtime crates that need the shared infrastructure. Provider crates depend on `inference-remote-core`; the rollup doesn't see this directly (it only sees the per-runtime crates).

`inference-runtime-litellm` depends on `inference-runtime-openai` because LiteLLM exposes the OpenAI API; this is a thin re-targeting wrapper. Per the v3 rule about layering: this is the only intra-runtime dependency, and it's justified by API compatibility, not code reuse for its own sake.

### 10.3 Crate-by-crate description (additions)

**`inference-remote-core`** — shared remote infrastructure. Contains:

- HTTP client primitives (built on `reqwest` with `hyper` HTTP/2)
- `RemoteEngineCoreActor` (generic over `Box<dyn ModelRunner>` with `transport_kind = RemoteNetwork`)
- `RemoteWorkerActor` (worker pool member)
- `RemoteSessionActor` (credential and connection lifecycle)
- `RateLimiterActor` (cluster-distributed via `rustakka-distributed-data` GCounter; strict variant as cluster-singleton)
- `CircuitBreakerActor` (per `(provider, endpoint)` state machine)
- Retry policy types (exponential backoff, jitter, retry-after header parsing)
- Error classification helpers (HTTP status → `InferenceError` variants)
- SSE stream parsing utilities
- Cost estimation primitives

This is the crate third-party providers (Cohere, Mistral API, AWS Bedrock, custom internal proxies) depend on to add new remote runtimes.

**`inference-runtime-openai`** — OpenAI Chat Completions API. Includes:

- `OpenAiRunner: ModelRunner`
- `OpenAiConfig: RuntimeConfig` (api_key, endpoint, organization, project, rate_limits, retry, circuit_breaker, timeouts)
- Request / response wire types
- SSE chunk deserialization
- Azure OpenAI variant (`OpenAiConfig::Azure { resource, deployment, api_version }`)
- Error classification per OpenAI error codes
- Cost estimation using OpenAI's published pricing

**`inference-runtime-anthropic`** — Anthropic Messages API. Includes:

- `AnthropicRunner: ModelRunner`
- `AnthropicConfig: RuntimeConfig`
- Request / response wire types
- SSE chunk deserialization (Anthropic's event types)
- Vision message handling (multimodal)
- Tool-use serialization (Anthropic's structured tool calling)
- Error classification per Anthropic error codes
- Cost estimation using Anthropic's published pricing

**`inference-runtime-gemini`** — Google Gemini. Includes:

- `GeminiRunner: ModelRunner`
- `GeminiConfig: RuntimeConfig`
- Two variants: AI Studio (API key) and Vertex AI (OAuth2 + project + region)
- Request / response wire types
- SSE chunk deserialization (Gemini's chunk format)
- Safety settings configuration
- Multimodal (text, image, audio, video)
- Function calling
- Cost estimation using Vertex AI pricing

**`inference-runtime-litellm`** — LiteLLM proxy. Includes:

- `LiteLlmRunner: ModelRunner` — wraps `OpenAiRunner` with LiteLLM-specific defaults
- `LiteLlmConfig: RuntimeConfig` — endpoint URL (proxy), api_key, model name
- Awareness of LiteLLM-specific extensions (tags, virtual keys, cache control headers)
- Different default retry policy (LiteLLM has its own retries; lower client-side max_retries)

### 10.4 Dependency budgets (extended)

| Crate | Allowed deps | Forbidden deps |
|---|---|---|
| `inference-core` | `serde`, `thiserror`, `bytes`, `secrecy` | tokio, async-trait, rakka, pyo3, runtime libs, http libs |
| `inference-runtime` | `inference-core`, rakka crates, tokio, async-trait, tracing | pyo3, runtime libs, http libs |
| `inference-python-bridge` | `inference-core`, `inference-runtime`, pyo3 | runtime libs, http libs |
| `inference-remote-core` | `inference-core`, `inference-runtime`, `reqwest`, `hyper`, `eventsource-stream`, `tower` | pyo3, GPU runtime libs |
| `inference-runtime-vllm` | core/runtime/python-bridge, pyo3 | other runtime libs |
| `inference-runtime-{tensorrt,ort,candle,cudarc,mistralrs}` | core/runtime, respective lib | pyo3, other runtime libs |
| `inference-runtime-{openai,anthropic,gemini}` | core/runtime/remote-core | pyo3, GPU runtime libs |
| `inference-runtime-litellm` | core/runtime/remote-core/openai | pyo3, GPU runtime libs |
| `inference-pipeline` | core/runtime, `rustakka-streams` | runtime libs, pyo3 |
| `inference-testkit` | core/runtime, `rustakka-testkit`, `proptest`, `wiremock` (for remote mocks) | runtime libs, pyo3 |
| `inference` (rollup) | all above optionally | nothing forbidden |

The `secrecy` crate is added to `inference-core` because credentials need to be a part of the type system from the bottom up (deployments declare API keys; the type prevents accidental logging).

### 10.5 Feature flag conventions (extended)

```toml
# inference/Cargo.toml
[features]
default = []

# Local runtimes (unchanged from v3)
vllm      = ["dep:inference-runtime-vllm",      "inference-runtime-vllm/vllm"]
tensorrt  = ["dep:inference-runtime-tensorrt",  "inference-runtime-tensorrt/tensorrt"]
ort       = ["dep:inference-runtime-ort",       "inference-runtime-ort/ort"]
candle    = ["dep:inference-runtime-candle",    "inference-runtime-candle/candle"]
cudarc    = ["dep:inference-runtime-cudarc",    "inference-runtime-cudarc/cudarc"]
mistralrs = ["dep:inference-runtime-mistralrs", "inference-runtime-mistralrs/mistralrs"]

# Remote runtimes (new in v4)
openai    = ["dep:inference-runtime-openai"]
anthropic = ["dep:inference-runtime-anthropic"]
gemini    = ["dep:inference-runtime-gemini"]
litellm   = ["dep:inference-runtime-litellm"]

# Pipeline
pipeline  = ["dep:inference-pipeline"]

# Convenience aggregates
all-native      = ["tensorrt", "ort", "candle", "cudarc", "mistralrs"]
all-python      = ["vllm"]
all-local       = ["all-native", "all-python"]
all-remote      = ["openai", "anthropic", "gemini", "litellm"]
all-runtimes    = ["all-local", "all-remote"]
default-prod    = ["vllm", "tensorrt", "ort", "openai", "anthropic", "pipeline"]
remote-only     = ["all-remote", "pipeline"]                    # no GPU needed
```

The `remote-only` aggregate is significant: a deployment that wants to be a pure-remote inference router (federating remote providers into one OpenAI-compatible endpoint with rate limiting, fallback, and observability) can build with `cargo build --features remote-only` and skip every GPU-related dependency. This is a compelling deployment for organizations that don't own GPU hardware but want the actor-based orchestration story.

### 10.6 Versioning policy (unchanged)

`inference-core` and `inference-runtime` version slowly. Per-runtime crates version independently. Rollup is the synchronization point.

For remote runtimes specifically: providers occasionally make breaking changes to their wire formats (deprecating endpoints, adding required fields). Per-runtime crates absorb these as patch / minor versions; the trait surface stays stable.

### 10.7 Workspace `Cargo.toml` (additions)

```toml
[workspace.dependencies]
# ... existing (tokio, serde, etc.) ...

# Remote-runtime infrastructure
reqwest             = { version = "0.12", default-features = false, features = ["rustls-tls", "http2", "json", "stream"] }
hyper               = "1"
eventsource-stream  = "0.2"
tower               = { version = "0.4", features = ["limit", "retry", "timeout"] }
secrecy             = "0.10"

# Remote provider crates
inference-remote-core         = { version = "0.1.0", path = "crates/inference-remote-core" }
inference-runtime-openai      = { version = "0.1.0", path = "crates/inference-runtime-openai" }
inference-runtime-anthropic   = { version = "0.1.0", path = "crates/inference-runtime-anthropic" }
inference-runtime-gemini      = { version = "0.1.0", path = "crates/inference-runtime-gemini" }
inference-runtime-litellm     = { version = "0.1.0", path = "crates/inference-runtime-litellm" }
```

### 10.8 What this enables (extended)

- **Pure-remote deployments don't compile GPU dependencies.** `cargo build --features remote-only` produces a binary suitable for a no-GPU egress server.
- **Pure-local deployments don't compile HTTP infrastructure.** A bare-metal cluster running only vLLM and TensorRT depends on `inference[vllm,tensorrt]` and `reqwest`/`hyper` are not in the dependency graph.
- **Mixed deployments compile what they use.** The default-prod aggregate covers the common case (vLLM + TensorRT + ORT + OpenAI + Anthropic + pipeline) and is what most production builds will use.
- **Third-party remote providers are first-class.** A custom provider (Cohere, Mistral API, AWS Bedrock, internal LLM gateway) is implemented by depending on `inference-remote-core` + `inference-core` and producing a `MyProviderRunner: ModelRunner`. No fork, no vendor.
- **CI matrix is tractable.** Test combinations: `default`, `all-local`, `all-remote`, `default-prod`, `remote-only`. Each combination is a distinct CI job; failures isolate to specific feature combinations.

---

## 11. Developer Experience

### 11.1 Layer 1 — Declarative surface

Same shape as v3; remote deployments use the same `Deployment` value object:

```python
from rustakka_inference import Deployment, Cluster

cluster = Cluster.connect("rakka://prod:7355")

# Local
cluster.deploy(Deployment(
    name="llama-3.1-70b-instruct",
    model="meta-llama/Llama-3.1-70B-Instruct",
    gpus=4,
    replicas=2,
))

# Remote — same Deployment value object, no GPU spec
cluster.deploy(Deployment(
    name="gpt-4o",
    model="gpt-4o",
    runtime="openai",  # auto-inferred from model name when credentials present
    replicas=1,        # for remote, replicas = independent worker pools
))
```

Runtime inference: `gpt-4o`, `gpt-4-turbo`, `o1-*` → `openai`. `claude-*` → `anthropic`. `gemini-*` → `gemini`. Local model paths → local runtime per the v3 rules. Operators can always override.

### 11.2 Layer 2 — Shape and runtime overrides (extended)

```python
# Explicit remote runtime with full configuration
Deployment(
    name="gpt-4o",
    model="gpt-4o",
    runtime="openai",
    runtime_config=OpenAiConfig(
        endpoint="https://api.openai.com/v1",
        api_key=Secret.from_env("OPENAI_API_KEY"),
        organization="org-xxx",
        project="proj-xxx",
        rate_limits=RateLimits(
            requests_per_minute=10_000,
            tokens_per_minute=10_000_000,
        ),
        retry=RetryPolicy(
            max_retries=3,
            initial_backoff_ms=1000,
            max_backoff_ms=60_000,
            respect_retry_after=True,
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=10,
            open_duration_ms=30_000,
        ),
        timeouts=Timeouts(
            request_timeout_ms=60_000,
            read_timeout_ms=10_000,
        ),
    ),
    serving=Serving(
        max_concurrent=50,                # worker pool size
        on_capacity_exhausted="queue",    # queue | reject | fallback
    ),
    replicas=2,
)

# Anthropic with different defaults
Deployment(
    name="claude-sonnet-4",
    model="claude-sonnet-4",
    runtime="anthropic",
    runtime_config=AnthropicConfig(
        api_key=Secret.from_env("ANTHROPIC_API_KEY"),
        rate_limits=RateLimits(
            requests_per_minute=4_000,
            tokens_per_minute=400_000,
        ),
    ),
)

# LiteLLM proxy — single endpoint, multiple backends
Deployment(
    name="multi-provider-fallback",
    model="claude-sonnet-4",       # primary; LiteLLM fallback chain configured proxy-side
    runtime="litellm",
    runtime_config=LiteLlmConfig(
        endpoint="http://litellm.internal:4000",
        api_key=Secret.from_env("LITELLM_KEY"),
        rate_limits=RateLimits(requests_per_minute=20_000),
    ),
)

# Azure OpenAI
Deployment(
    name="gpt-4o-azure",
    model="gpt-4o",
    runtime="openai",
    runtime_config=OpenAiConfig.azure(
        resource="my-azure-resource",
        deployment="gpt-4o-deployment",
        api_version="2024-08-01-preview",
        api_key=Secret.from_env("AZURE_OPENAI_KEY"),
    ),
)
```

### 11.3 Layer 3 — Project file (extended)

```toml
[cluster]
name     = "production-inference"
endpoint = "rakka://controlplane.prod:7355"

# Local LLM
[[deployment]]
name     = "llama-3.1-70b-instruct"
model    = "meta-llama/Llama-3.1-70B-Instruct"
runtime  = "vllm"
gpus     = 4
replicas = 2

# Local TensorRT
[[deployment]]
name     = "whisper-tensorrt"
model    = "openai/whisper-large-v3"
runtime  = "tensorrt"
gpus     = 1
replicas = 4

# Remote OpenAI
[[deployment]]
name     = "gpt-4o"
model    = "gpt-4o"
runtime  = "openai"
replicas = 2
[deployment.runtime_config]
endpoint = "https://api.openai.com/v1"
api_key  = { from_env = "OPENAI_API_KEY" }
[deployment.runtime_config.rate_limits]
requests_per_minute = 10_000
tokens_per_minute   = 10_000_000
[deployment.runtime_config.retry]
max_retries = 3
[deployment.runtime_config.circuit_breaker]
failure_threshold = 10
open_duration_ms  = 30_000
[deployment.serving]
max_concurrent = 50

# Remote Anthropic
[[deployment]]
name     = "claude-sonnet-4"
model    = "claude-sonnet-4"
runtime  = "anthropic"
[deployment.runtime_config]
api_key = { from_env = "ANTHROPIC_API_KEY" }

# Remote Gemini
[[deployment]]
name     = "gemini-2.0-pro"
model    = "gemini-2.0-pro"
runtime  = "gemini"
[deployment.runtime_config]
variant     = "vertex"
project     = "my-gcp-project"
region      = "us-central1"
credentials = { from_file = "/etc/gcp/sa.json" }
```

The `from_env` and `from_file` secret indirection ensures secrets are never inline in the file. The diff-and-apply CLI shows secret references but never values.

### 11.4 Layer 4 — Python actor decorators (extended)

Two decorators now:

- `@gpu_actor(deployment="...")` — for actors that themselves use a GPU (custom samplers, per-request transformations that touch tensors). Pinned to GPU dispatcher, GIL bridge applies.
- `@inference_actor` — for orchestration actors that don't use a GPU directly but compose deployments. Standard tokio dispatcher, no GIL pinning. Useful for the hybrid-agent and tiered-router patterns from §9.

Most multi-runtime / remote-aware actors are `@inference_actor`. The `@gpu_actor` is only needed when the actor's own handler runs CUDA kernels.

### 11.5 Layer 5 — Escape hatches (unchanged)

Placement hints, manual placement, direct actor access. New escape hatches for remote:

```python
deployment = cluster.deployment("gpt-4o")
rate_limiter = deployment.rate_limiter()              # ActorRef[RateLimiterActor]
circuit_breaker = deployment.circuit_breaker()        # ActorRef[CircuitBreakerActor]
workers = deployment.workers()                         # List[ActorRef[RemoteWorkerActor]]

# Inspect rate limit state
state = await rate_limiter.ask(GetState())
print(f"Tokens spent this minute: {state.tokens_used} / {state.tokens_per_minute}")

# Manually trip the circuit breaker for incident response
await circuit_breaker.tell(ForceOpen(duration_ms=300_000))
```

### 11.6 Conventions that prevent footguns (extended)

All v3 conventions plus:

- **Secrets are typed.** `Secret<T>` types from the `secrecy` crate prevent accidental logging. Python bindings expose them as opaque objects with no `__repr__` or `__str__`.
- **Rate limits validated against provider tiers.** `Deployment::validate()` checks declared rate limits against known provider tier limits. A deployment claiming `requests_per_minute = 100_000` against a free OpenAI tier fails validation with a clear error.
- **Network egress checked at deploy time.** Before the deployment goes `Serving`, the placement actor on each chosen node performs a connectivity check (tiny test request to the provider). If unreachable from the placed node, deployment fails with the specific network error instead of mysteriously failing on the first user request.
- **API key rotation hot-swappable.** Updating the secret reference triggers `RemoteSessionActor::rebuild` on the next deploy step; in-flight requests complete with the old credentials, new requests use new credentials.
- **Cost guardrails configurable.** A deployment can set `max_spend_per_hour` and `max_spend_per_day` budgets enforced by the `MetricsActor`; exceeding triggers backpressure (reject new requests) rather than runaway spend.

---

## 12. Cross-cutting Concerns

| Concern | CUDA § / new | Where it surfaces (crate) |
|---|---|---|
| Thread affinity (local) | §5.1 | `inference-runtime` `WorkerActor` dispatcher |
| Asynchrony (local) | §5.2, §5.10 | `inference-core` `ModelRunner::execute` |
| Local supervision | §5.3, §5.11 | `inference-runtime` two-tier; per-runtime crate rebuild logic |
| Backpressure | §5.4 | `inference-runtime` bounded mailboxes; `inference-pipeline` for streams |
| Serialization / location transparency | §5.5, §5.8 | `inference-core` `GpuRef<T>` / `GpuToken<T>` |
| Memory transfer (local) | §5.6 | `inference-runtime` placement; per-runtime allocator hooks |
| Stream allocation (local) | §5.7 | `inference-runtime` `StreamAllocator` |
| `GpuRef<T>` lifecycle | §5.8 | `inference-core` types |
| Python GPU bridge | §5.9 | `inference-python-bridge` |
| Completion strategy (local) | §5.10 | `inference-runtime` `HostFnCompletion` |
| Two-tier supervision | §5.11 | `inference-runtime`; per-runtime rebuild |
| Runtime selection and GIL exposure | new (v2) | `inference-core` types; `inference-runtime` `PlacementActor` |
| Pipeline composition backpressure | new (v2) | `inference-pipeline` |
| Selective compilation | new (v3) | Feature flags forward through `inference` rollup |
| **Distributed rate limiting** | **new (v4, §3.5, §12.1)** | **`inference-remote-core` `RateLimiterActor` via `rustakka-distributed-data`** |
| **Provider failure isolation** | **new (v4, §3.5, §12.2)** | **`inference-remote-core` `CircuitBreakerActor`** |
| **Remote retry semantics** | **new (v4, §3.5, §12.3)** | **`inference-remote-core` retry policy module** |
| **Cost tracking and budgets** | **new (v4, §12.4)** | **`MetricsActor` aggregation; budget enforcement in `RemoteEngineCoreActor`** |
| **Credential lifecycle** | **new (v4, §12.5)** | **`RemoteSessionActor`; `secrecy` types in `inference-core`** |

### 12.1 Distributed rate limiting

Approximate distributed token bucket via `GCounter` CRDT:

- Each `RateLimiterActor` maintains a local `GCounter` for tokens spent in the current window.
- The CRDT propagates spent counters across cluster nodes with sync interval (default 1s).
- A request requires a permit; permit is granted if local view of `total_spent` plus the request's estimated cost is under the per-window budget allocated to this node.
- Node allocation is uniform by default (each node gets `total_budget / node_count`); can be weighted by node capacity hints.
- Over-spend bound = `(node_count - 1) × max_request_cost × sync_interval`. Approximate but tunable.

For deployments requiring strict adherence (pay-per-call, hard caps): `StrictRateLimiterActor` is a cluster-singleton; every request `ask`s for permit. Higher latency (~1–10ms per request depending on cluster size), exact accounting.

Default selection: approximate for high-throughput (>100 RPS), strict for low-throughput (<10 RPS) where the latency overhead is acceptable. Operators override via `rate_limits.strict = true` in deployment config.

### 12.2 Provider failure isolation

`CircuitBreakerActor` per `(provider, endpoint)`:

- **Failure threshold:** N consecutive failures within window M open the circuit.
- **Open duration:** time to keep circuit open before half-open probe (exponential, capped).
- **Half-open probe count:** how many requests to allow during half-open before deciding.
- **Failure types counted:** 5xx, timeouts, network errors. Not counted: 429 (handled by rate limiter), 4xx other than 429 (caller error), content-filter (caller content).

State transitions emit observability events. Operators see "Circuit opened for openai (us-east) at T+120s due to 12 consecutive 503s" in the dashboard.

When the circuit is open, requests fail fast with `CircuitOpen { provider, opened_at, retry_at }`. The error is typed; upstream `RequestActor` can fall back to a different deployment based on the deployment's `on_capacity_exhausted` config.

### 12.3 Remote retry semantics

Retry policy is per-deployment, applied inside `RemoteWorkerActor`:

```rust
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub backoff_multiplier: f64,           // default 2.0
    pub jitter: JitterKind,                // None | Full | Equal
    pub respect_retry_after: bool,         // honor server-sent header
    pub retry_on: HashSet<RetryableError>, // RateLimited, ServerError, Timeout, Network
}
```

The retry counter is per-attempt, not global — a request that's retried within the worker doesn't count against backpressure in the engine queue. If max_retries is exhausted, the worker returns the failure to `RequestActor`, which can then make a higher-level decision (fall back to another deployment, retry on a fresh request, surface to user).

Idempotency: remote runtimes treat retries as idempotent. For non-idempotent operations (rare in inference but possible with stateful APIs), `Deployment::idempotent = false` disables retries entirely.

### 12.4 Cost tracking and budgets

`MetricsActor` aggregates per-deployment cost using:
- Provider response usage fields (most providers return token counts in response)
- Provider published pricing (in `inference-runtime-{openai,anthropic,gemini}` defaults; overridable)

Emitted metrics:
- `inference_cost_usd_total{deployment,model}` (counter)
- `inference_tokens_in_total{deployment,model}` (counter)
- `inference_tokens_out_total{deployment,model}` (counter)

Budget enforcement (optional per deployment):

```toml
[deployment.budget]
max_spend_per_hour = 100.00     # USD
max_spend_per_day  = 1000.00
on_exceeded        = "reject"   # reject | warn | throttle
```

`RemoteEngineCoreActor` tracks rolling-window spend; when threshold approached, applies `on_exceeded` action. `throttle` halves the worker pool active concurrency; `reject` fails new requests with `BudgetExceeded`; `warn` only emits an observability event.

### 12.5 Credential lifecycle

Credentials are `Secret<T>` from the `secrecy` crate, sourced via:
- Environment variables (`from_env = "..."`)
- Files (`from_file = "..."`)
- Vault / HSM via plugin (`from_vault = "..."` with adapter crates)
- Inline (discouraged, only for testing)

`RemoteSessionActor` reads the secret at construction and caches in memory (zeroed on drop). Rotation is achieved by:
1. Operator updates the secret source (rotates env var, updates Vault, etc.).
2. Operator triggers `cluster.deployment("...").rebuild_session()` (or this happens automatically on detected 401).
3. `RemoteSessionActor::rebuild` re-reads the secret and rebuilds the HTTP client.
4. In-flight requests complete with old credentials; new requests use new credentials. No traffic dropped.

Secrets never appear in logs, metrics labels, or tracing spans. Type system enforces: `tracing::info!("{key}")` won't compile if `key: Secret<String>` because the type lacks `Debug` / `Display`.

---

## 13. Implementation Roadmap

### Phase 1 — Crate skeleton and drop-in actor shells around vLLM (unchanged)

Workspace skeleton with all 18 crates in v4 (4 added vs v3). vLLM Phase-1 wrapping as before. Remote runtime crates exist as stubs.

### Phase 2a — Python bridge for vLLM (unchanged)

### Phase 2b — Rust-native local runtime backends (unchanged)

### Phase 2c — Remote runtime backends (new in v4)

In parallel with 2a/2b. Remote runtimes don't depend on Python bridge or local runtime crates beyond `inference-core` and `inference-runtime`.

1. `inference-remote-core` complete:
   - HTTP/2 client primitives on `reqwest` + `hyper`
   - `RemoteEngineCoreActor`, `RemoteWorkerActor`, `RemoteSessionActor`
   - `RateLimiterActor` (approximate, CRDT-backed) + `StrictRateLimiterActor` (singleton)
   - `CircuitBreakerActor` with full state machine
   - Retry policy module with backoff + jitter + retry-after parsing
   - SSE stream parsing utilities
   - Error classification helpers
2. `inference-runtime-openai` complete:
   - `OpenAiRunner: ModelRunner`
   - `OpenAiConfig` with full options including Azure variant
   - Wire types for Chat Completions and Responses APIs
   - Cost estimation
3. `inference-runtime-anthropic` complete:
   - `AnthropicRunner: ModelRunner`
   - Messages API wire types
   - Tool use support
4. `inference-runtime-gemini` complete:
   - `GeminiRunner: ModelRunner` for both Vertex and AI Studio variants
   - Multimodal support
   - Safety settings
5. `inference-runtime-litellm` complete:
   - Thin wrapper over `inference-runtime-openai`
   - LiteLLM-specific defaults
6. Cost tracking and budgets in `MetricsActor`
7. `inference-testkit` extended:
   - `wiremock`-based remote provider mocks
   - Deterministic 429 / 5xx / timeout injection
   - Rate limiter time-skip helpers

**Exit criterion:** A multi-runtime cluster running vLLM (local), TensorRT (local), OpenAI (remote), and Anthropic (remote) deployments simultaneously. Demonstrate:
- 10,000+ RPS to OpenAI without exceeding rate limits, distributed across 4 cluster nodes.
- Circuit breaker opens correctly when provider returns 503 for 30s; closes when provider recovers.
- Hybrid agent example (§9.1) runs end-to-end with one trace per request showing local + remote stages.
- `cargo build --features remote-only` produces a binary that runs on a no-GPU machine and serves remote-only deployments.

### Phase 3 — Full §6.1 control / data plane split (unchanged for local; remote N/A)

### Phase 4 — Multi-deployment cluster operations (extended)

Adds remote-specific cluster operations:
- Per-deployment cost dashboards
- Per-deployment circuit breaker management
- API key rotation via `cluster.deployment().rotate_credentials()`
- Multi-API-key rate limit pools (one deployment with N replicas, each replica using a different API key for higher aggregate throughput)

### Phase 5 — Pipeline composition (extended)

Adds reference hybrid pipelines:
- Cost-optimized agent (local cheap → remote expensive on escalation)
- Tiered router (premium / standard / fast / cheap tiers)
- Capability fallback (try local; fall back to remote on confidence threshold)
- Regional failover (local cluster saturated → remote provider)

### Phase 6 — Developer experience surface (extended)

Adds:
- Cost estimation in `Deployment::validate()`
- Network egress check at deploy time
- `rakka cost-report` CLI subcommand
- `rakka rotate-credentials` CLI subcommand
- Provider compatibility matrix in `defaults.toml`

### Phase 7 — Out-of-scope-but-watch (extended)

- Additional remote providers: AWS Bedrock, Cohere, Mistral API, Together AI — community-contributed `inference-runtime-*` crates.
- Caching layer (Redis-backed semantic cache, LiteLLM-style) as `inference-cache` crate.
- A/B testing primitives (route X% to deployment A, Y% to B, compare quality metrics).
- Spot pricing optimization (AWS Bedrock provisioned vs on-demand pricing arbitrage).
- Cross-region failover for remote (us-east-1 GPT-4o down → route to us-west-2 GPT-4o).

---

## 14. Summary

| Question | Answer |
|---|---|
| Is the wrapping sound? | Yes — local GPU and remote network runtimes fit under the same actor decomposition. |
| Where does it pay off? | Unified routing, unified backpressure, distributed rate limiting, provider failure isolation, cost tracking, hybrid pipelines, supervised orchestration across local and remote. |
| Where doesn't it? | Inside per-token operations. Per-token mailbox hops are the CUDA §3.5 anti-pattern. |
| What's the local runtime story? | Rust-native is the default; Python (`vllm`, `python:*`) is the marked exception. TensorRT, ORT, Candle, cudarc, mistralrs cover non-LLM workloads with no GIL. |
| **What's the remote runtime story?** | **First-class `openai`, `anthropic`, `gemini`, `litellm` runtimes; pure Rust HTTP/2; no GIL; bounded worker pools; distributed rate limiting; circuit breakers; cost tracking.** |
| What's the GIL story? | Per-deployment property, syntactically marked, enforced by placement. Remote runtimes are GIL-free. |
| What's the cluster story? | Three shapes — single big cluster, microservice per model, federated. Local and remote coexist. |
| What's the pipeline story? | Multi-runtime pipelines compose as supervised actor graphs. Hybrid local-remote agents, cost-optimized routing, capability fallback. |
| What's the developer story? | Six layers from declarative `Deployment` down to raw `ActorRef`. Local and remote have identical surfaces. |
| What's the workspace story? | 18 crates, strict layering. `inference-core` no actor deps; `inference-remote-core` parallel to `inference-python-bridge`; per-provider crates plug in. Pure-remote builds compile no GPU deps. |
| What's the strategic story? | Orchestration, local inference, and remote inference share one substrate. Failures are typed and recoverable. Cost is observable. The actor system handles high volumes across mixed local and remote without operators stitching together separate tools. |

---

## Appendix A — Differences from prior versions

### v1 → v2

Multi-runtime; Rust-native default; GIL exposure marked; placement runtime-aware.

### v2 → v3

Workspace structure; 14 crates; strict one-way layering; `inference-python-bridge` separated; `inference-testkit` first-class; bare `inference` rollup.

### v3 → v4

| v3 | v4 |
|---|---|
| Local-only runtimes | Local + remote runtimes; remote is first-class |
| `ModelRunner` trait local-shaped | `ModelRunner` trait extended with `transport_kind()`; default no-op for `load_weights` and `rebuild_session` |
| `WorkerActor` is the only worker | `WorkerActor` (local) and `RemoteWorkerActor` (remote); both supervised |
| `EngineCoreActor` only | `EngineCoreActor` (local) and `RemoteEngineCoreActor` (remote) |
| Cluster operation only considers GPU and Python isolation | Adds rate limiting, circuit breakers, network egress, credential lifecycle |
| 14 crates | 18 crates (added `inference-remote-core`, `inference-runtime-openai`, `inference-runtime-anthropic`, `inference-runtime-gemini`, `inference-runtime-litellm`) |
| Cross-cutting concerns: GIL, selective compilation | Adds: distributed rate limiting, provider failure isolation, remote retry, cost tracking, credential lifecycle |
| Pipeline composition: voice agent (all local) | Adds: hybrid local-remote agent, cost-optimized routing, capability fallback |
| `@gpu_actor` decorator | Adds `@inference_actor` for non-GPU orchestration actors |
| Deployment shapes: A, B, C — all local-oriented | Same shapes; remote and hybrid mixes shown explicitly |
| Phase 2 split into 2a/2b | Phase 2 split into 2a (Python bridge), 2b (native local runtimes), 2c (remote runtimes) — all in parallel |

The actor decomposition itself is unchanged across all four versions — `WorkerActor`, `EngineCoreActor`, `RequestActor`, `DpCoordinatorActor`, two-tier supervision, control / data plane split. v4 generalizes the worker / engine pair from "GPU-bound" to "GPU-bound or remote-bound" and adds the supporting infrastructure for the remote case.
