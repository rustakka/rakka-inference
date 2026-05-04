# inference-testkit

> Mocks and harnesses for testing `atomr-infer`. The crate the
> demo and your tests use to exercise gateway → request → engine
> without spending real money on a remote provider.

## What's in here

| Item                              | Use it for                                                          |
|-----------------------------------|---------------------------------------------------------------------|
| `MockRunner` / `MockScript`       | Drive a deterministic stream of `TokenChunk`s through any actor that holds a `Box<dyn ModelRunner>`. |
| `MockOpenAi`                      | A `wiremock::MockServer` factory speaking the OpenAI Chat Completions wire format. |
| `mount_chat_happy_path(server, content)` | Deterministic single-chunk SSE response.                     |
| `inject_429_once(server)`         | First request gets `429 Retry-After: 1`; subsequent fall through to mounted handlers. |
| `inject_5xx_once(server, n)`      | N consecutive 503s — drives the circuit breaker open in tests.      |

## A test that asserts the real failure recovery

```rust
use inference_testkit::{inject_429_once, mount_chat_happy_path, MockOpenAi};

#[tokio::test]
async fn retries_through_a_429() {
    let mock = MockOpenAi::start().await;
    inject_429_once(&mock.server).await;
    mount_chat_happy_path(&mock.server, "ok").await;

    let runner = build_openai_runner_pointing_at(&mock.url()).await?;
    let result = drive_with_retries(runner, /* batch */).await?;

    assert_eq!(result.text, "ok");
}
```

The `examples/remote_only_demo` binary runs exactly this shape against
all three doc-mandated scenarios (happy path / 429 retry / circuit
open) — see `cargo run --bin remote_only_demo`.

## MockRunner

```rust
use inference_testkit::{MockRunner, MockScript};

let mut runner = MockRunner::new(MockScript::from_text(["hello ", "world"]));
let handle = runner.execute(batch).await?;
let chunks: Vec<_> = handle.into_stream().collect().await;
// "hello " then "world", final chunk carries finish_reason: Stop.
```

Use this to test `RequestActor`, `EngineCoreActor`, gateway streaming
behaviour, accumulator semantics, etc. — anywhere you'd otherwise need
a real GPU or HTTP call.
