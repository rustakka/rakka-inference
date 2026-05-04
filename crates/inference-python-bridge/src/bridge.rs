//! `PythonGpuBridge` — Python-resident GPU runtime wiring. Doc §5.9.
//!
//! Provides:
//! - `python_pinned_dispatcher()` — a factory for a `GpuDispatcher`-shaped
//!   thread pool that pins each interpreter to a single OS thread (the
//!   GIL constrains us to one Python execution per interpreter).
//! - `PythonGpuBridge` — entry point for kernel launches that originate
//!   from Python (vLLM's worker process, etc.). Currently a thin shell
//!   around `pyo3::Python::with_gil` plus a dedicated runtime.
//!
//! TODO(atomr-accel F4): atomr-accel's lib.rs lists `PythonGpuBridge` as
//! a deferred phase. When upstream ships it, replace this body with
//! `pub use atomr_accel::python::PythonGpuBridge;` plus a thin
//! `python_pinned_dispatcher` that delegates to
//! `atomr_accel::cuda::dispatcher::GpuDispatcher::python_pinned()`. The
//! public surface this module exposes (`PythonGpuBridge::with_python`,
//! `python_pinned_dispatcher`) is intentionally narrow so that lift
//! is mechanical.

use std::sync::Arc;

use parking_lot::Mutex;
use pyo3::prelude::*;

use atomr_infer_core::error::{InferenceError, InferenceResult};

pub struct PythonGpuBridge {
    /// Per-interpreter thread pool token. One bridge instance maps to
    /// exactly one interpreter for GIL isolation between deployments.
    interpreter_id: u64,
    /// Lock used so only one Python call runs through this bridge at
    /// a time — defensive even with PyO3's GIL handling.
    serializer: Mutex<()>,
}

impl PythonGpuBridge {
    pub fn new(interpreter_id: u64) -> Arc<Self> {
        Arc::new(Self {
            interpreter_id,
            serializer: Mutex::new(()),
        })
    }

    pub fn interpreter_id(&self) -> u64 {
        self.interpreter_id
    }

    /// Run `f` while holding the GIL. Returns the closure's `T` lifted
    /// into `InferenceResult<T>` (Python errors → `InferenceError::Internal`).
    pub fn with_python<F, T>(&self, f: F) -> InferenceResult<T>
    where
        F: FnOnce(Python<'_>) -> PyResult<T> + Send,
        T: Send,
    {
        let _ord = self.serializer.lock();
        Python::with_gil(|py| f(py)).map_err(|e| InferenceError::Internal(format!("python: {e}")))
    }
}

/// Construct a tokio runtime configured one-thread-per-task to keep
/// each Python interpreter pinned to a single OS thread. This matches
/// the doc-described `python-pinned` dispatcher (§5.7).
pub fn python_pinned_dispatcher(name: impl Into<String>) -> InferenceResult<tokio::runtime::Runtime> {
    let name = name.into();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name(name)
        .enable_all()
        .build()
        .map_err(|e| InferenceError::Internal(format!("python-pinned dispatcher: {e}")))
}
