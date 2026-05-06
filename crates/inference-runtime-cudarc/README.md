# atomr-infer-runtime-cudarc

> Direct CUDA kernel dispatch via `cudarc` and `atomr-accel`'s kernel
> actors. The escape hatch for novel architectures, research code, and
> custom CUDA kernel packages that don't fit any framework.

## When to use this

- You have hand-tuned CUDA kernels you want to expose as a model
  runtime.
- Your model isn't supported by any framework yet (research code).
- You need direct access to cuBLAS / cuDNN / cuFFT actors at the
  atomr-infer-runtime level.

## Build profiles

| Build                                                          | Result                                            |
|----------------------------------------------------------------|---------------------------------------------------|
| `cargo build -p atomr-infer-runtime-cudarc` (default)            | Stub.                                             |
| `cargo build -p atomr-infer-runtime-cudarc --features cudarc`    | Pulls `cudarc` + `atomr-accel`.                    |

## What it gives you

```rust
use inference_runtime_cudarc::{CudarcConfig, CudarcRunner};

let cfg = CudarcConfig {
    device: 0,
    kernel_package: "my_org/llama-3-custom-kernels".into(),
};
```

The runner doesn't manage a CUDA context itself — that lives in
`atomr_accel::cuda::device::DeviceActor`. The runner simply posts typed
kernel messages (e.g. `atomr_accel::cuda::kernel::BlasMsg::Sgemm`) to the
appropriate child actor and lifts replies into `TokenChunk`s.

A canonical hand-roll:

```rust
// Inside the runner's execute() body, gated on `feature = "cudarc"`:
//
// 1. Pin to a thread via atomr_accel::cuda::dispatcher::GpuDispatcher.
// 2. Allocate a stream from atomr_accel::cuda::stream::PerActorAllocator.
// 3. Launch your kernel via cudarc::driver::CudaSlice + cudarc::nvrtc.
// 4. Sync via atomr_accel::cuda::completion::HostFnCompletion (sub-microsecond).
// 5. Stream tokenised output as TokenChunks.
```

The §13 Phase-2b roadmap adds a registry that maps
`CudarcConfig.kernel_package` to a concrete launcher closure, so
operators don't write that boilerplate by hand.
