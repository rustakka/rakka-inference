//! # inference-python-bridge
//!
//! Allocator + stream entry for Python-resident GPU runtimes (vLLM,
//! XTTS, Bark, …). Doc §2.6, §5.7, §5.9.
//!
//! Architecture-doc reference §10.1 places `PythonGpuBridge` in this
//! crate. As of the rakka-accel F1–F5 surfaces, the upstream crate
//! still notes the `PythonGpuBridge` as F4-deferred (see
//! `rakka-accel/src/lib.rs` header). We implement it here in the
//! meantime. When rakka-accel exposes its own `PythonGpuBridge`, this
//! crate switches to a re-export — see `bridge.rs` for the lift
//! marker (`TODO(rakka-accel F4)`).
//!
//! The crate is feature-gated: with `--features python` it pulls in
//! PyO3 and exposes the real `PythonGpuBridge`. Without the feature
//! the crate compiles to a no-op stub so the workspace builds without
//! a Python venv installed.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

#[cfg(feature = "python")]
pub mod bridge;

#[cfg(feature = "python")]
pub use bridge::{python_pinned_dispatcher, PythonGpuBridge};

/// Stub used when the `python` feature is off. Returning a typed error
/// from runtime crates (e.g. `inference-runtime-vllm`) is preferable
/// to a build-time `unimplemented!()` — operators see a clear
/// "vllm feature disabled at build time" error in their logs.
#[cfg(not(feature = "python"))]
pub fn feature_disabled<T>() -> Result<T, atomr_infer_core::error::InferenceError> {
    Err(atomr_infer_core::error::InferenceError::Internal(
        "python feature disabled at build time — rebuild with --features python".into(),
    ))
}
