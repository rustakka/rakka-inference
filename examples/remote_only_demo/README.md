# remote_only_demo

> End-to-end smoke test for the remote-only path. Stands up a
> [`wiremock`](https://crates.io/crates/wiremock) server speaking the
> OpenAI Chat Completions wire format and exercises every doc §13
> Phase-1/2c invariant in one process.

## Run it

```sh
cargo run --bin remote_only_demo
```

Expected output (abbreviated):

```
== happy path ==
Hello from the mock!

== 429 retry ==
attempt 0 → rate-limited (retry after Some(1s))
succeeded after retry

== circuit breaker ==
call 0: Err(ServerError { status: 503, ... })  state=Closed
call 1: Err(ServerError { status: 503, ... })  state=Closed
WARN inference_remote_core::circuit_breaker: circuit opened ...
call 2: Err(ServerError { status: 503, ... })  state=Open
call 3: Err(CircuitOpen { provider: OpenAi, ... })  state=Open
circuit final state: Open
```

## Why this exists

Three scenarios in the architecture doc that have to keep working:

1. **Happy path** — SSE streaming returns `TokenChunk`s through to the
   caller.
2. **429 retry** — `RetryEngine` honours `Retry-After`, sleeps, and
   succeeds on the next attempt.
3. **Circuit breaker** — three consecutive 5xx flip the state machine
   to `Open`; the next call short-circuits with
   `InferenceError::CircuitOpen { provider, opened_at_unix_ms,
   retry_at_unix_ms }`.

The demo doubles as a regression test — if any of these stops working,
the binary's output changes and CI catches it. No real API key, no
spend.

## What it shows about the workspace

- `OpenAiRunner` mounted on a real `RemoteSessionActor`-shaped
  `SessionSnapshot` (built in-process for the demo).
- `wiremock` driven via `inference-testkit`'s `MockOpenAi`,
  `mount_chat_happy_path`, `inject_429_once`, `inject_5xx_once`.
- `RetryEngine` decisions and `CircuitBreakerHandle` state queried
  directly — same primitives a `RemoteWorkerActor` uses inside the
  full actor system.

## Companion config

[`demo.toml`](demo.toml) shows the corresponding
`atomr-infer serve` project file shape. Run
`cargo run -p atomr-infer-cli -- serve --config examples/remote_only_demo/demo.toml`
to boot the same providers through the full actor system.
