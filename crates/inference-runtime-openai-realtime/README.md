# atomr-infer-runtime-openai-realtime

OpenAI Realtime API provider for `atomr-infer`.

Implements [`atomr_infer_core::RealtimeRunner`] against the OpenAI
Realtime WebSocket API (`wss://api.openai.com/v1/realtime`).

## Build profiles

| Profile | Feature flag | What you get |
|---|---|---|
| Default | *(none)* | Stub runner — returns `InferenceError::Internal` on all calls |
| Full | `tts-openai-realtime` | Real WebSocket adapter |

## Quick start

```toml
[dependencies]
atomr-infer-runtime-openai-realtime = { version = "0.8", features = ["tts-openai-realtime"] }
```

```rust,no_run
use atomr_infer_runtime_openai_realtime::config::OpenAiRealtimeConfig;
use atomr_infer_runtime_openai_realtime::runner::OpenAiRealtimeRunner;

let cfg = OpenAiRealtimeConfig::new_with_env_key("OPENAI_API_KEY");
let mut runner = OpenAiRealtimeRunner::new(cfg);
```

## Reference

- [OpenAI Realtime guide](https://platform.openai.com/docs/guides/realtime)
- FR-TTS-001 (atomr-infer program of work, M9-A)
