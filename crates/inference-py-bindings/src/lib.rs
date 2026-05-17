//! # inference-py-bindings
//!
//! PyO3 bindings exposing `atomr-infer`'s data types, deployment
//! configuration, error hierarchy, and an async `Cluster.execute`
//! pipeline. Doc §11.1.
//!
//! With default features the crate is an empty rlib so the rest of the
//! workspace builds without a Python toolchain. Build the wheel via
//! `maturin develop --features python-extension`; the cdylib loads as
//! `atomr_infer._native` and is re-exported by the pure-Python
//! `python/atomr_infer/` package.
//!
//! Module layout (mirrors upstream `atomr/pycore`):
//! - `atomr_infer._native.errors`  — exception hierarchy
//! - `atomr_infer._native.core`    — `Deployment`, `ExecuteBatch`,
//!   `Message`, `Tokens`, …
//! - `atomr_infer._native.runtime` — `RuntimeKind`, `RuntimeConfig`,
//!   `ProviderKind`, …
//! - `atomr_infer._native.config`  — `Serving`, `RateLimits`,
//!   `RetryPolicy`, `Budget`, …
//! - `atomr_infer._native.cluster` — `Cluster.deploy/execute`

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

#[cfg(feature = "python")]
mod audio;
#[cfg(feature = "python")]
mod cluster;
#[cfg(feature = "python")]
mod config;
#[cfg(feature = "python")]
mod core;
#[cfg(feature = "python")]
mod errors;
#[cfg(feature = "python")]
mod runtime;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    errors::register(py, m)?;
    core::register(py, m)?;
    runtime::register(py, m)?;
    config::register(py, m)?;
    cluster::register(py, m)?;
    audio::register(py, m)?;
    Ok(())
}
