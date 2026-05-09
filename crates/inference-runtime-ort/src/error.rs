//! Error mapping from `ort::Error` to `InferenceError`.

use atomr_infer_core::error::InferenceError;

pub(crate) fn map_ort_err<T>(err: ort::Error<T>) -> InferenceError {
    let msg = err.to_string();
    let lower = msg.to_ascii_lowercase();

    // Heuristic: invalid-argument / shape errors look like caller bugs
    // (wrong tensor shape, missing input). Surface as BadRequest so
    // `RequestActor` can map to a 4xx; everything else is Internal.
    if lower.contains("invalid")
        || lower.contains("shape")
        || lower.contains("missing")
        || lower.contains("expected")
    {
        InferenceError::BadRequest {
            message: format!("ort: {msg}"),
        }
    } else {
        InferenceError::Internal(format!("ort: {msg}"))
    }
}

pub(crate) fn internal<E: std::fmt::Display>(prefix: &str, e: E) -> InferenceError {
    InferenceError::Internal(format!("ort: {prefix}: {e}"))
}
