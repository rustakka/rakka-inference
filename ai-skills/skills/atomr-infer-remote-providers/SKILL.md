---
name: atomr-infer-remote-providers
description: Use when wiring a remote inference provider (OpenAI / Anthropic / Gemini / LiteLLM) in a atomr-infer project — credentials, rate limits, retries, circuit breakers, cost estimation, fallback chains. Triggers on configuring `OpenAiConfig` / `AnthropicConfig` / `GeminiConfig` / `LiteLlmConfig`, handling 429 / `InferenceError::RateLimited` / `CircuitOpen`, or asking "how do I add OpenAI to my deployment".
---

# Wiring remote providers

Every remote provider in atomr-infer uses the same shared
infrastructure from
[`inference-remote-core`](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-remote-core/README.md):
HTTP/2 client pooling, distributed rate limiting via
`atomr_distributed_data::GCounter`, circuit breakers, retry-with-backoff,
SSE streaming, and credential refresh. Per-provider crates are thin
adapters that contribute one `ModelRunner` impl plus wire types.

## What each provider gets right

| Provider | What's unique | What's still shared |
|---|---|---|
| **OpenAI** | Direct + Azure variants in one config; SSE chunk format; refines 400→`ContextLengthExceeded` and `content_filter`→`ContentFiltered`. | Rate limiter, circuit breaker, retry, session, cost tracking. |
| **Anthropic** | `x-api-key` + `anthropic-version` headers; richer SSE event types (`message_start` / `content_block_delta` / `message_delta` / `message_stop`); `tool_use` round-trip. | Same. |
| **Gemini** | AI Studio (API key in query string) **and** Vertex (OAuth2 access token) variants; pluggable `CredentialProvider`. | Same. |
| **LiteLLM** | Newtype around `OpenAiRunner` pointed at the proxy URL with lower default `max_retries` (LiteLLM does its own retries). | Same. |

## Required moving parts

A remote runtime needs three things wired up at deploy time:

1. **A `SessionSnapshot`** holding the `reqwest::Client` and
   `SecretString` credential. Created via
   `RemoteSessionActor::bootstrap(SessionConfig { user_agent, timeouts, credential })`
   and shared with workers as `Arc<ArcSwap<SessionSnapshot>>` so
   credential rotation can hot-swap atomically.
2. **A `RateLimiterActor`** keyed by `(provider, api_key, model)`.
   Approximate (CRDT-backed) for high-throughput; strict
   (cluster-singleton) for premium API keys with hard caps.
3. **A `CircuitBreakerHandle`** keyed by `(provider, endpoint)`.
   Opens after `failure_threshold` 5xx/timeouts, half-opens after
   `open_duration`, closes on a successful probe.

In the full actor system, `RemoteEngineCoreActor` owns these. For a
one-off non-actor consumer, you can drive them directly — see
`examples/remote_only_demo/src/main.rs` for a 200-line runnable
example.

## Credentials: typed and rotatable

```rust
use inference_remote_core::session::{CredentialProvider, StaticApiKey};
use inference_core::SecretString;

// Most providers — static API key from env.
let cred: Arc<dyn CredentialProvider> =
    Arc::new(StaticApiKey(SecretString::from(std::env::var("OPENAI_API_KEY")?)));

// Vertex / Bedrock / anything OAuth2 — implement CredentialProvider yourself:
struct GcloudAdc;
#[async_trait::async_trait]
impl CredentialProvider for GcloudAdc {
    async fn token(&self) -> InferenceResult<SecretString> {
        // call `gcloud auth print-access-token`, refresh on a timer, etc.
        Ok(SecretString::from(/* fresh token */))
    }
}
```

`SecretString` re-exports `secrecy::SecretString`. It will NOT `Debug`,
NOT `Display`, and goes through `expose_secret()` to read. This means
secrets cannot accidentally land in `tracing::info!("{cfg:?}")` —
the type system rejects it.

**Hot-swap rotation:** updating the secret source triggers
`RemoteSessionActor::rebuild` on the next pulse. In-flight requests
finish on the old credential; new ones use the rotated value. Zero
dropped traffic.

## Rate limits

```rust
use inference_remote_core::{RateLimiterActor, AcquirePermit};
use inference_core::deployment::RateLimits;

let mut rl = RateLimiterActor::new(
    "node-a",
    RateLimits {
        requests_per_minute: Some(10_000),
        tokens_per_minute:   Some(10_000_000),
        concurrent_requests: Some(50),
        strict: false,           // approximate (GCounter-backed); flip true for cluster-singleton
    },
);
```

In a cluster, multiple nodes calling the same provider with the same
API key share the bucket via
`atomr_distributed_data::counters::GCounter` deltas. No surprise 429s
from naïve client-side limits firing per-node.

## Circuit breakers

```rust
use inference_remote_core::CircuitBreakerHandle;
use inference_core::runtime::{CircuitBreakerConfig, ProviderKind};
use std::time::Duration;

let breaker = CircuitBreakerHandle::new(
    ProviderKind::OpenAi,
    CircuitBreakerConfig {
        failure_threshold:    10,
        open_duration:        Duration::from_secs(30),
        half_open_max_probes: 1,
    },
);

// Wrap a call; success/failure feed the state machine.
let result = breaker.run(|| async { http_post(/* ... */).await }).await;
```

When the breaker is **Open**, `breaker.check()` returns
`InferenceError::CircuitOpen { provider, opened_at_unix_ms, retry_at_unix_ms }`.
Upstream `RequestActor`s use this to fall back to a different
deployment (see the `atomr-infer-pipelines` skill).

**429s and content-filter refusals deliberately don't count toward
the breaker.** Those are the rate-limiter's and the per-provider
classifier's domain, respectively.

## Error classification

Every remote runner returns the same `InferenceError` enum:

| Error variant | When | Retryable? | Counts toward breaker? |
|---|---|---|---|
| `RateLimited { retry_after }` | 429 | Yes (with backoff, honors `Retry-After`) | Only after threshold |
| `ServerError { status: 5xx }` | 5xx | Yes | Yes |
| `Timeout { elapsed_ms }` | request or read timeout | Yes | Yes |
| `NetworkError(String)` | TCP / TLS / DNS | Yes | Yes |
| `ContentFiltered { reason }` | provider safety blocked the request | **No** | No |
| `ContextLengthExceeded { tokens, max_tokens }` | input too long | No | No |
| `BadRequest { message }` | 400 (caller bug) | No | No |
| `Unauthorized { message }` | 401 (rotated key) | No (triggers session rebuild) | No |
| `Forbidden { message }` | 403 (model/feature denied) | No | No |
| `CircuitOpen { ... }` | breaker open | No | No |
| `BudgetExceeded { deployment }` | spend ceiling tripped | No | No |

Per-provider crates *upgrade* generic errors to typed ones — e.g.
OpenAI's `error.code == "context_length_exceeded"` becomes
`ContextLengthExceeded`, not a generic `BadRequest`.

## Cost tracking

Every provider crate ships a pricing table:

```rust
use inference_runtime_openai::OpenAiPricing;
let p = OpenAiPricing::published().get("gpt-4o-mini").unwrap();
// p.input_per_mtok_usd, p.output_per_mtok_usd
```

`MetricsActor` aggregates real usage from response bodies/headers and
emits `inference_cost_usd_total{deployment,model}` counters. Set
`Deployment.budget = Some(Budget { max_spend_per_hour_usd, on_exceeded })`
to make runaway provider spend physically impossible — `Reject`,
`Warn`, or `Throttle` (halve the worker pool concurrency).

## Canonical references

- [`inference-remote-core` README](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-remote-core/README.md)
- [`inference-runtime-openai` README](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-openai/README.md)
- [`inference-runtime-anthropic` README](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-anthropic/README.md)
- [`inference-runtime-gemini` README](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-gemini/README.md)
- [`examples/remote_only_demo/`](https://github.com/rustakka/atomr-infer/blob/main/examples/remote_only_demo/) — full end-to-end demo
- [Architecture doc §3.5, §12](https://github.com/rustakka/atomr-infer/blob/main/docs/architecture.md) — rate limiting, circuit breaking, retries, credentials

## Common mistakes

- **Putting an API key inline in the TOML file.** Use
  `api_key = { from_env = "OPENAI_API_KEY" }` — `SecretRef::Env`
  is the indirection that keeps secrets out of git.
- **Setting both `RetryPolicy.max_retries` and configuring a
  retrying upstream proxy.** Compound retries amplify outages.
  LiteLLM's `LiteLlmRunner` handles this by lowering its default
  `max_retries` to 1.
- **Counting 429s toward the circuit breaker.** Don't — that's a
  capacity signal handled by the rate limiter. The breaker is for
  *outage* signals (5xx / timeouts / network).
- **Treating `ContentFiltered` as retryable.** Same input → same
  refusal. Surface to the user.
