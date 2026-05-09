//! Python exception hierarchy mirroring `inference_core::error::InferenceError`.
//!
//! Pattern follows upstream `atomr-accel/crates/atomr-accel-py/src/errors.rs`:
//! a base exception inheriting from `PyException`, one subclass per Rust
//! enum variant, and a `map()` helper that pattern-matches a Rust error
//! into the right Python type.

// pyo3 0.22's `create_exception!` macro emits `cfg(feature = "gil-refs")`
// gates that the bindings crate doesn't (and shouldn't) declare. Silence
// the noise — it goes away with the planned pyo3 0.24 bump.
#![allow(unexpected_cfgs)]

use atomr_infer_core::error::InferenceError as RsInferenceError;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    atomr_infer,
    InferenceError,
    PyException,
    "Base atomr-infer error. All concrete variants subclass this."
);
create_exception!(
    atomr_infer,
    RateLimited,
    InferenceError,
    "429 from a remote provider; worker backs off and retries."
);
create_exception!(
    atomr_infer,
    CircuitOpen,
    InferenceError,
    "Circuit breaker open for (provider, endpoint); fail-fast."
);
create_exception!(
    atomr_infer,
    ContentFiltered,
    InferenceError,
    "Provider safety filter rejected the input/output."
);
create_exception!(
    atomr_infer,
    ContextLengthExceeded,
    InferenceError,
    "Input exceeded the model's context window."
);
create_exception!(
    atomr_infer,
    BadRequest,
    InferenceError,
    "400 from the provider — caller-side bug."
);
create_exception!(
    atomr_infer,
    Unauthorized,
    InferenceError,
    "401 — triggers RemoteSessionActor::rebuild."
);
create_exception!(
    atomr_infer,
    Forbidden,
    InferenceError,
    "403 — model/feature access denied."
);
create_exception!(
    atomr_infer,
    Backpressure,
    InferenceError,
    "Mailbox / engine queue full."
);
create_exception!(
    atomr_infer,
    BudgetExceeded,
    InferenceError,
    "Spend ceiling reached for the named deployment."
);
create_exception!(
    atomr_infer,
    NetworkError,
    InferenceError,
    "Network blip below the HTTP layer."
);
create_exception!(
    atomr_infer,
    ServerError,
    InferenceError,
    "5xx from provider; counts toward circuit breaker."
);
create_exception!(
    atomr_infer,
    Timeout,
    InferenceError,
    "Request or read timeout."
);
create_exception!(
    atomr_infer,
    CudaContextPoisoned,
    InferenceError,
    "Local CUDA context poisoned (sticky failure)."
);
create_exception!(
    atomr_infer,
    Internal,
    InferenceError,
    "Catch-all for runtime-internal bugs."
);

/// Map a Rust `InferenceError` onto the matching Python exception type.
pub fn map(e: RsInferenceError) -> PyErr {
    let msg = e.to_string();
    match e {
        RsInferenceError::RateLimited { .. } => PyErr::new::<RateLimited, _>(msg),
        RsInferenceError::CircuitOpen { .. } => PyErr::new::<CircuitOpen, _>(msg),
        RsInferenceError::ContentFiltered { .. } => PyErr::new::<ContentFiltered, _>(msg),
        RsInferenceError::ContextLengthExceeded { .. } => {
            PyErr::new::<ContextLengthExceeded, _>(msg)
        }
        RsInferenceError::BadRequest { .. } => PyErr::new::<BadRequest, _>(msg),
        RsInferenceError::Unauthorized { .. } => PyErr::new::<Unauthorized, _>(msg),
        RsInferenceError::Forbidden { .. } => PyErr::new::<Forbidden, _>(msg),
        RsInferenceError::Backpressure(_) => PyErr::new::<Backpressure, _>(msg),
        RsInferenceError::BudgetExceeded { .. } => PyErr::new::<BudgetExceeded, _>(msg),
        RsInferenceError::NetworkError(_) => PyErr::new::<NetworkError, _>(msg),
        RsInferenceError::ServerError { .. } => PyErr::new::<ServerError, _>(msg),
        RsInferenceError::Timeout { .. } => PyErr::new::<Timeout, _>(msg),
        RsInferenceError::CudaContextPoisoned(_) => PyErr::new::<CudaContextPoisoned, _>(msg),
        RsInferenceError::Internal(_) => PyErr::new::<Internal, _>(msg),
        // `InferenceError` is `#[non_exhaustive]`; bucket future variants
        // under the generic base until each gets its own subclass.
        _ => PyErr::new::<InferenceError, _>(msg),
    }
}

/// Convenience: convert any displayable error into a generic
/// `InferenceError`. Used at FFI boundaries where the Rust side returns
/// a non-`InferenceError` failure (e.g. unknown deployment lookup).
pub fn map_str<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<InferenceError, _>(e.to_string())
}

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new_bound(py, "errors")?;
    sub.add("InferenceError", py.get_type_bound::<InferenceError>())?;
    sub.add("RateLimited", py.get_type_bound::<RateLimited>())?;
    sub.add("CircuitOpen", py.get_type_bound::<CircuitOpen>())?;
    sub.add("ContentFiltered", py.get_type_bound::<ContentFiltered>())?;
    sub.add(
        "ContextLengthExceeded",
        py.get_type_bound::<ContextLengthExceeded>(),
    )?;
    sub.add("BadRequest", py.get_type_bound::<BadRequest>())?;
    sub.add("Unauthorized", py.get_type_bound::<Unauthorized>())?;
    sub.add("Forbidden", py.get_type_bound::<Forbidden>())?;
    sub.add("Backpressure", py.get_type_bound::<Backpressure>())?;
    sub.add("BudgetExceeded", py.get_type_bound::<BudgetExceeded>())?;
    sub.add("NetworkError", py.get_type_bound::<NetworkError>())?;
    sub.add("ServerError", py.get_type_bound::<ServerError>())?;
    sub.add("Timeout", py.get_type_bound::<Timeout>())?;
    sub.add(
        "CudaContextPoisoned",
        py.get_type_bound::<CudaContextPoisoned>(),
    )?;
    sub.add("Internal", py.get_type_bound::<Internal>())?;
    m.add_submodule(&sub)?;
    Ok(())
}
