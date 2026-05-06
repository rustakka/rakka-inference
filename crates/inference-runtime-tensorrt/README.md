# atomr-infer-runtime-tensorrt

> NVIDIA TensorRT runtime — pre-compiled `nvinfer` plans driven via
> FFI. Doc §2.2.

## Build profiles

| Build                                                            | Result                                                           |
|------------------------------------------------------------------|------------------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-tensorrt` (default)            | Stub — no `extern "C"` block, no link to `libnvinfer.so`.        |
| `cargo build -p atomr-infer-runtime-tensorrt --features tensorrt`  | Real path: opens the FFI surface, requires `libnvinfer.so` at link time. |

The default-stub-when-feature-off pattern lets CI on machines without
TensorRT installed still run `cargo check --workspace`.

## Configuration

```rust
use inference_runtime_tensorrt::TensorRtConfig;

let cfg = TensorRtConfig {
    plan_path: "/etc/models/whisper-large-v3.plan".into(),
    max_batch_size: 8,
};
```

## What it's for

TensorRT shines on workloads with stable shapes and pre-compiled
optimisation: Whisper, vision pipelines, embedding models, OCR. The
engine is opaque (a serialised `ICudaEngine`) and uses one
`ExecutionContext` per concurrent request; concurrency is bounded by
your max-batch-size + max-concurrent budget.

The `cudarc` driver / context lifecycle is handled by
`atomr_accel_cuda::device::DeviceActor` (when the rollup's `cuda` feature is
on), so the TensorRT runner doesn't manage a CUDA context itself —
it lives inside the two-tier supervision tree just like any other
local runtime.
