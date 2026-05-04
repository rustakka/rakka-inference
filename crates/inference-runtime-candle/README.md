# atomr-infer-runtime-candle

> Pure-Rust transformer inference via the
> [`candle`](https://github.com/huggingface/candle) family of crates.

## Why Candle

- No GIL.
- No system Python install.
- Tight memory footprint.
- Quantised models (Q4_0 GGUF) run in ~CPU-RAM-sized footprints — great
  for edge deployments and CI smoke tests.

## Build profiles

| Build                                                          | Result                                            |
|----------------------------------------------------------------|---------------------------------------------------|
| `cargo build -p atomr-infer-runtime-candle` (default)            | Stub.                                             |
| `cargo build -p atomr-infer-runtime-candle --features candle`    | Pulls `candle-core`, `candle-nn`, `candle-transformers`, and `rakka-accel` for `GpuDispatcher` + `PerActorAllocator`. |

## Configuration

```rust
use inference_runtime_candle::{CandleConfig, CandleDevice, CandleDtype};

let cfg = CandleConfig {
    model_path: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
    device: CandleDevice::Cuda,
    dtype: CandleDtype::Q4_0,
};
```

## How it integrates with rakka-accel

The runner uses upstream substrate, not local re-implementations:

- `atomr_accel::cuda::dispatcher::GpuDispatcher` for thread pinning
- `atomr_accel::cuda::stream::PerActorAllocator` for per-request stream allocation
- `atomr_accel::cuda::device::DeviceActor` two-tier supervision (via the
  rollup's `cuda` feature)

The Candle-specific bit is the model loader and the forward-pass
loop — everything around it is shared infrastructure.
