# atomr-infer-runtime-ort

> ONNX Runtime backend via the [`ort`](https://crates.io/crates/ort)
> crate. Targets pre-compiled ONNX graphs.

Two entry points:

- `ModelRunner::execute` — chat-style. Tokenises an `ExecuteBatch`,
  runs an autoregressive sampling loop on ONNX-exported causal LMs
  (HuggingFace Optimum-ONNX layout), streams `TokenChunk`s.
- `OrtRunner::infer` — low-level. Takes raw `f32` / `i64` tensors,
  returns `f32` outputs. For embeddings (BGE / E5), rerankers,
  Whisper encoders, vision classifiers, and anything that's "ONNX
  graph + tensors in / tensors out".

## Use cases

- Whisper variants exported to ONNX (Whisper encoder via `infer()`)
- Embedding models (BGE, E5) via `infer()`
- Rerankers via `infer()`
- Vision classifiers via `infer()`
- ONNX-exported causal LMs (GPT-2, Phi, Qwen, Gemma exports) via
  `execute()`

## Build profiles

| Build                                                                    | Result                                                |
|--------------------------------------------------------------------------|-------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-ort` (default)                        | Stub — `ort` not in dep graph.                        |
| `cargo build -p atomr-infer-runtime-ort --features ort`                   | Real path with `ort` crate (CPU EP).                  |
| `cargo build -p atomr-infer-runtime-ort --features ort,ort-cuda`          | Adds the CUDA EP. Requires a working CUDA toolkit.    |
| `cargo build -p atomr-infer-runtime-ort --features ort,ort-load-dynamic`  | Loads `libonnxruntime` at runtime via `ORT_DYLIB_PATH`. |
| `cargo build -p atomr-infer-runtime-ort --features ort,ort-hf-hub`        | Resolves `tokenizer.json` from HuggingFace if local fails. |

## Configuration

```rust
use atomr_infer_runtime_ort::{ExecutionProvider, OrtConfig, OrtRunner};

let cfg = OrtConfig {
    onnx_path: "/etc/models/bge-large-v1.5.onnx".into(),
    execution_provider: ExecutionProvider::Cuda,
    device_id: 0,
    tokenizer_path: None,        // ⇒ probe `tokenizer.json` next to the ONNX file
    hf_repo: None,               // ⇒ optional fallback if `ort-hf-hub` is on
    intra_threads: None,         // ⇒ ort default
    default_max_new_tokens: 256, // chat-style execute() ceiling
};
let mut runner = OrtRunner::new(cfg);
```

## Low-level `infer()` (embeddings, encoders)

```rust
use std::collections::HashMap;
use atomr_infer_runtime_ort::InferTensor;

let mut inputs = HashMap::new();
inputs.insert("input_ids".into(), InferTensor::I64 {
    shape: vec![1, 8],
    data: vec![101, 7592, 2088, 102, 0, 0, 0, 0],
});
inputs.insert("attention_mask".into(), InferTensor::I64 {
    shape: vec![1, 8],
    data: vec![1, 1, 1, 1, 0, 0, 0, 0],
});
let outputs = runner.infer(inputs).await?;
let (shape, data) = &outputs.f32["last_hidden_state"];
```

## Chat-style `execute()` (ONNX-exported causal LMs)

`execute()` expects the HuggingFace Optimum-ONNX export shape:

- `input_ids: [batch, seq]` (i64)
- `attention_mask: [batch, past + seq]` (i64) — optional but standard
- `position_ids: [batch, seq]` (i64) — optional
- `past_key_values.{i}.{key,value}: [batch, kv_heads, past, head_dim]` (f32)
- output `logits: [batch, seq, vocab]` (f32)
- output `present.{i}.{key,value}` matching past shapes

The probe is tolerant of name variants (`past`/`past_key_values`/`past_key_values.0_key`/etc.) but assumes f32 logits and f32 KV cache. For other dtypes the runner fails with a `BadRequest` echoing the probed shape so the operator can debug.

## CUDA execution provider

When `execution_provider == Cuda` and the `ort-cuda` feature is on,
the runner builds a `CUDAExecutionProvider` with the configured
`device_id` and falls back to CPU for unsupported ops. Sharing a
`cudarc::driver::CudaStream` with `atomr-accel-cuda`'s
`PerActorAllocator` is a follow-up — see `session.rs::providers_for`
for the seam.

## MSRV note

`ort` 2.0 requires Rust 1.85+. The atomr-infer workspace MSRV is
1.78 for `remote-only` builds; operators enabling this runner need a
toolchain that satisfies `ort`'s own MSRV.
