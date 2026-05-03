# inference-runtime-ort

> ONNX Runtime backend via the [`ort`](https://crates.io/crates/ort)
> crate. Targets pre-compiled ONNX graphs.

## Use cases

- Whisper variants exported to ONNX
- Embedding models (BGE, E5)
- Rerankers
- Vision classifiers
- Anything that ships as an ONNX graph + a tokenizer

## Build profiles

| Build                                                       | Result                                                  |
|-------------------------------------------------------------|---------------------------------------------------------|
| `cargo build -p inference-runtime-ort` (default)            | Stub — `ort` not in dep graph.                          |
| `cargo build -p inference-runtime-ort --features ort`       | Real path with `ort` crate (CUDA EP available).         |

## Configuration

```rust
use inference_runtime_ort::{ExecutionProvider, OrtConfig};

let cfg = OrtConfig {
    onnx_path: "/etc/models/bge-large-v1.5.onnx".into(),
    execution_provider: ExecutionProvider::Cuda,
};
```

## CUDA execution provider

When `execution_provider == Cuda`, the runner binds the ORT session to
a `cudarc` stream allocated by
`rakka_accel::cuda::stream::PerActorAllocator`. That keeps GPU contention
predictable when the same node also runs a vLLM or Candle deployment.
