# atomr-infer-runtime-vllm

> vLLM (Python) runtime — the canonical local LLM backend per
> architecture v4 §2.1.

## Build profiles

| Build                                                                   | Result                                            |
|-------------------------------------------------------------------------|---------------------------------------------------|
| `cargo build -p atomr-infer-runtime-vllm` (default)                       | Stub — runner returns `InferenceError::Internal("vllm feature disabled at build time")`. Useful when `Cargo.toml` references the crate via the rollup but the host has no Python venv. |
| `cargo build -p atomr-infer-runtime-vllm --features vllm`                 | Pulls `atomr-infer-python-bridge/python` (PyO3 + tokio + parking_lot) and wires the runner over vLLM's `EngineCore` via the `PythonGpuBridge`. Phase-2a wiring lands as the surface stabilises upstream. |

## Why a stub by default

`vllm` requires a working Python environment with the `vllm` package
installed at runtime. Linking against PyO3 at build time pulls
substantial deps and a `python3` development install; we don't want
the workspace's default `cargo build --workspace` to fail on a host
without those. The stub keeps the trait satisfied and the rollup
buildable; flipping the feature swaps in the real path.

## Configuration

```rust
use inference_runtime_vllm::VllmConfig;

let cfg = VllmConfig {
    model: "meta-llama/Llama-3.1-70B-Instruct".into(),
    tensor_parallel_size: 4,                  // TP across 4 GPUs
    dtype: "bfloat16".into(),
    gpu_memory_utilization: Some(0.9),
};
```

## GIL story

vLLM is GIL-pinned. `VllmRunner::gil_pinned()` returns `true`, which
the placement actor reads to ensure two GIL-pinned deployments don't
share an interpreter. Each vLLM deployment gets its own
`PythonGpuBridge` instance (one interpreter per deployment by
default).

## Roadmap

Per the architecture doc's §13 Phase 2a, the vLLM runner is wired
through `atomr-infer-python-bridge` once that bridge gets re-exported
from `rakka-accel` (planned F4). The current local `PythonGpuBridge`
implementation is a drop-in placeholder; the lift is mechanical.
