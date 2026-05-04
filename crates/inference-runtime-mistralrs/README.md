# atomr-infer-runtime-mistralrs

> Thin wrapper over [`mistralrs`](https://github.com/EricLBuehler/mistral.rs).

## Why this exists alongside Candle

`mistralrs` is the most production-ready Rust-native LLM runtime
available today: KV-cache management, paged attention, GGUF + safetensors
loading, quantisation. For Mistral / Llama / Gemma models, it's
typically the right default for a Rust-only deployment.

The `infer_runtime("mistralai/...")` registry returns
`RuntimeKind::MistralRs` so deployments with Mistral-family model names
land here automatically.

## Build profiles

| Build                                                                | Result                            |
|----------------------------------------------------------------------|-----------------------------------|
| `cargo build -p atomr-infer-runtime-mistralrs` (default)               | Stub.                             |
| `cargo build -p atomr-infer-runtime-mistralrs --features mistralrs`    | Pulls the `mistralrs` crate.      |

## Configuration

```rust
use inference_runtime_mistralrs::MistralRsConfig;

let cfg = MistralRsConfig {
    model_id: "mistralai/Mistral-7B-Instruct-v0.3".into(),
    quant: Some("Q4_0".into()),
};
```
