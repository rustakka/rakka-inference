---
name: atomr-infer-troubleshooting
description: Use when debugging atomr-infer symptoms — `InferenceError::RateLimited` storms, `CircuitOpen`, `ContentFiltered`, `CudaContextPoisoned`, mailbox backpressure, gateway 429s, missing fallbacks, the `remote-only` invariant violation. Triggers on a stack trace mentioning `inference-*`, an unexplained typed error, or asking "why is my deployment returning X".
---

# Troubleshooting atomr-infer

The typed `InferenceError` enum tells you exactly which subsystem
emitted the failure and what to do about it. This skill maps each
typed variant to its recovery path.

## Typed errors → recovery paths

| Variant | What it means | Where it came from | Recovery |
|---|---|---|---|
| `RateLimited { provider, retry_after }` | Provider 429 | Provider HTTP response → `classify_http_status` | Worker retries with `Retry-After` honored. If you see this propagate to the caller, your `RetryPolicy.max_retries` is too low or `Retry-After` is too long for your timeout. |
| `CircuitOpen { provider, opened_at_unix_ms, retry_at_unix_ms }` | Breaker is open after sustained 5xx/timeouts | `CircuitBreakerHandle::check` | Fall back to a different deployment, or wait for `retry_at_unix_ms`. Investigate the provider's status page. |
| `ContentFiltered { reason }` | Provider safety blocked the request | Provider 400 with `error.type = "content_filter"` (OpenAI) or status hint (Gemini) | **Don't retry.** Surface to the user. Same input → same refusal. |
| `ContextLengthExceeded { tokens, max_tokens }` | Input too long | Provider 400 with `error.code = "context_length_exceeded"` | Truncate or summarise input; choose a longer-context model. |
| `BadRequest { message }` | 400 (caller bug) | Provider | Inspect the message — usually a malformed message, missing field, or invalid sampling param. |
| `Unauthorized { message }` | 401 | Provider | API key rotated/expired. Triggers `RemoteSessionActor::rebuild` automatically; if that loops, your secret source is misconfigured. |
| `Forbidden { message }` | 403 | Provider | Model/feature access denied (org policy, region restriction, billing). |
| `Backpressure(String)` | Engine queue full | `RemoteEngineCoreActor` | Either increase `serving.max_concurrent`, change `on_capacity_exhausted` to `"fallback"`, or scale out replicas. |
| `BudgetExceeded { deployment }` | Spend ceiling tripped | `MetricsActor` | Check `Deployment.budget`. The breaker is intentional — investigate why the deployment is more expensive than budgeted. |
| `NetworkError(String)` | TCP / TLS / DNS | `reqwest` | Counts toward circuit breaker. Check upstream connectivity. |
| `ServerError { status, body }` | 5xx | Provider | Counts toward circuit breaker. Wait or fall back. |
| `Timeout { elapsed_ms }` | Hit `request_timeout` or `read_timeout` | `Timeouts` config | Counts toward circuit breaker. Either bump timeouts or fall back. |
| `CudaContextPoisoned(String)` | Sticky CUDA error on a local runtime | `ContextActor` panics with the `CONTEXT_POISONED_TAG` marker | Two-tier supervision restarts the `ContextActor` automatically. If it loops past `max_retries`, the GPU may have failed. |
| `Internal(String)` | Catch-all for runtime-internal bugs | Various | Read the message — usually a deserialisation issue, a feature-disabled stub firing, or an unimplemented path. |

## "I'm seeing 429s I shouldn't be"

Three places to check, in order:

1. **`Deployment.runtime_config.rate_limits`** — the configured RPM /
   TPM. Lower than reality? Bump them.
2. **Distributed sync interval** — multiple cluster nodes calling the
   same provider with the same API key share a CRDT-backed counter,
   but it syncs on an interval (~1s by default). During a burst,
   nodes may collectively over-commit by `(N-1) × max_request_cost`.
   Either bump `RateLimits.strict = true` (cluster singleton, exact
   accounting at the cost of an extra mailbox hop) or accept the
   approximate over-shoot.
3. **Provider auto-tune** — the worker reports observed 429s back to
   the rate limiter, which lowers its local view of available
   capacity. Check the limiter's `snapshot()`.

## "The circuit is open and I don't know why"

```rust
let state = breaker_actor.ask_with(|reply| CircuitBreakerMsg::GetState { reply },
                                    Duration::from_secs(1)).await?;
println!("breaker state: {:?}", state);   // Closed | Open | HalfOpen
```

Inspect logs for the `WARN inference_remote_core::circuit_breaker:
circuit opened provider=... failures=N` line — that's the threshold
crossing. Common upstream causes: provider outage, regional quota,
model retirement.

For incident response, an operator can force the breaker open from
the cluster API:

```rust
breaker_actor.tell(CircuitBreakerMsg::ForceOpen {
    duration: Duration::from_secs(300)
});
```

## "My remote-only build pulled `cudarc`"

```sh
$ cargo tree -p inference --no-default-features --features remote-only \
    | grep -Ec 'cudarc|atomr-accel|candle|pyo3'
1   # ← invariant violated
```

Almost always means: your `Cargo.toml` enabled a feature that
transitively pulls a GPU runtime. Check:

- Did you accidentally enable `candle`, `cudarc`, `accel`, or
  `accel-patterns` in addition to `remote-only`?
- Did a downstream crate enable them via its own `default-features`?
- Did you add a `dep:` line in your service crate that pulls
  `atomr-accel` directly?

Run `cargo xtask verify` locally — the verify gate fails with a
human-readable list of leaked GPU-dep lines.

## "Two-tier supervision keeps restarting"

`WorkerActor` (stable) → `ContextActor` (restartable) restarts on
`ContextPoisoned` panic-string markers. If you see a restart loop:

1. **`max_retries` exhausted** — check the supervisor strategy.
   Default for the `local-gpu` feature is 3 retries / 60s window
   from `atomr_accel_cuda::error::device_supervisor_strategy()`.
   Past that, the device stops.
2. **Sticky GPU fault** — physical GPU error. The machine needs
   manual recovery.
3. **Wrong runner factory** — if `WorkerSlot` factory captures
   mutable state, restart is broken. Capture `Arc<Mutex<...>>` if
   you must share, or stateless data.

## "My PR fails CI's `audit` job"

```
audit regressions vs baseline:
  inference-runtime: unwrap_used 1 -> 3 (+2)
```

You added `unwrap()` / `panic!` / `todo!` / `Box<dyn Any>` / etc. in
non-test code. Either:

1. Remove the new instances and use `?` / typed errors.
2. If they're genuinely justified, regenerate the baseline:
   ```sh
   cargo xtask audit --json docs/reports/audit-baseline.json
   git add docs/reports/audit-baseline.json
   git commit -m "chore(audit): refresh baseline after <reason>"
   ```

## "How do I see the actor system's behavior?"

```sh
RUST_LOG=inference=trace,inference_remote_core=debug cargo run --bin remote_only_demo
```

Useful log targets:

- `inference_remote_core::circuit_breaker` — `WARN` on Open transitions.
- `inference_remote_core::rate_limit` — bucket exhaustion messages.
- `inference_remote_core::worker` — per-attempt retry decisions.
- `inference_runtime::gateway` — per-request lifecycle.
- `inference_runtime::engine_core` — local engine queue depth.
- `inference_runtime::metrics` — per-deployment counters at TRACE.

## Canonical references

- [`inference-core::error`](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-core/src/error.rs) — the canonical `InferenceError` definition
- [Architecture doc §7.6](https://github.com/rustakka/atomr-infer/blob/main/docs/architecture.md) — failure handling matrix
- [`examples/remote_only_demo`](https://github.com/rustakka/atomr-infer/blob/main/examples/remote_only_demo/) — happy path / 429 / circuit open
- [`atomr-troubleshooting`](https://github.com/rustakka/atomr/blob/main/ai-skills/skills/atomr-troubleshooting/SKILL.md) skill — for the underlying actor-system errors

## Common mistakes

- **Treating `Backpressure` as transient.** It's a deliberate signal
  that your deployment is at capacity. Either scale out, change
  `on_capacity_exhausted`, or surface the 429 upstream.
- **Catching `ContentFiltered` and retrying.** Same prompt → same
  refusal. Surface to the user with the `reason` string.
- **Bumping `RetryPolicy.max_retries` to mask 429s.** That's how you
  earn an outage when your provider's rate-limit window resets.
- **Logging `InferenceError::Unauthorized.message`.** If the message
  contains a credential fragment from the provider, redact it. The
  typed `SecretString` only protects the credential at rest, not the
  provider's echo.
