# inference-python-bridge

> Bridge between Python-resident GPU runtimes (vLLM, XTTS, Bark, …)
> and the rakka-inference actor system.

## Build profiles

| Build                                                                  | Result                                                  |
|------------------------------------------------------------------------|---------------------------------------------------------|
| `cargo build -p inference-python-bridge` (default)                     | Empty stub — no PyO3, no Python venv needed.            |
| `cargo build -p inference-python-bridge --features python`             | Real `PythonGpuBridge` + `python_pinned_dispatcher()`.  |

## What it provides (`features = ["python"]`)

- **`PythonGpuBridge`** — entry point for kernel launches that
  originate from Python. Wraps `pyo3::Python::with_gil` plus a
  serialiser so only one Python call runs through a given bridge at a
  time.
- **`python_pinned_dispatcher(name)`** — a `tokio::runtime::Runtime`
  configured one-thread-per-task so each Python interpreter stays
  pinned to a single OS thread (the GIL constrains us to one Python
  execution per interpreter).

## TODO(rakka-accel F4)

The architecture doc places `PythonGpuBridge` in this crate. As of
today, the upstream `rakka-accel` lib.rs lists `PythonGpuBridge` as a
deferred F4 phase. When upstream ships it, this crate switches to a
re-export:

```rust
// after F4 lands:
pub use rakka_accel::python::PythonGpuBridge;
```

The public surface here is intentionally narrow so that lift is
mechanical.
