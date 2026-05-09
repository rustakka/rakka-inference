//! Low-level inference API for non-LLM uses (embeddings, rerankers,
//! Whisper, vision classifiers). Callers stage their own tensors
//! instead of going through the chat-style `execute()` adapter.
//!
//! Scope: f32 + i64 inputs, f32 outputs. This covers the bulk of
//! ONNX-deployed encoder models. Models with mixed-dtype outputs
//! (e.g. detection heads with both class scores and box coords as
//! different dtypes) can use the f32-only path here for the float
//! outputs and ignore the rest, or drop down to a follow-up API once
//! we add it.

use std::collections::HashMap;
use std::sync::Arc;

use atomr_infer_core::error::InferenceResult;
use ort::value::Tensor;

use crate::error::{internal, map_ort_err};
use crate::session::OrtState;

/// One input tensor in a typed form. The crate handles `f32` and
/// `i64` because together they cover the overwhelming majority of
/// ONNX-exported encoders. Add variants here as concrete demand
/// emerges.
#[derive(Debug, Clone)]
pub enum InferTensor {
    F32 { shape: Vec<i64>, data: Vec<f32> },
    I64 { shape: Vec<i64>, data: Vec<i64> },
}

#[derive(Debug, Clone, Default)]
pub struct InferOutputs {
    pub f32: HashMap<String, (Vec<i64>, Vec<f32>)>,
}

pub(crate) async fn run_infer(
    state: Arc<OrtState>,
    inputs: HashMap<String, InferTensor>,
) -> InferenceResult<InferOutputs> {
    tokio::task::spawn_blocking(move || run_infer_blocking(state, inputs))
        .await
        .map_err(|e| internal("spawn_blocking join", e))?
}

fn run_infer_blocking(
    state: Arc<OrtState>,
    inputs: HashMap<String, InferTensor>,
) -> InferenceResult<InferOutputs> {
    use ort::session::SessionInputValue;
    use std::borrow::Cow;

    let mut entries: Vec<(Cow<'static, str>, SessionInputValue<'_>)> = Vec::with_capacity(inputs.len());
    for (name, t) in inputs {
        match t {
            InferTensor::F32 { shape, data } => {
                let tensor = Tensor::from_array((shape, data)).map_err(map_ort_err)?;
                entries.push((Cow::Owned(name), tensor.into()));
            }
            InferTensor::I64 { shape, data } => {
                let tensor = Tensor::from_array((shape, data)).map_err(map_ort_err)?;
                entries.push((Cow::Owned(name), tensor.into()));
            }
        }
    }

    let mut session = state.session.lock();
    // Snapshot output names before run() so we can extract by name
    // without re-borrowing the session through the live SessionOutputs.
    let output_names: Vec<String> =
        session.outputs().iter().map(|o| o.name().to_owned()).collect();
    let outputs = session.run(entries).map_err(map_ort_err)?;

    let mut out = InferOutputs::default();
    for name in output_names {
        let Some(value) = outputs.get(name.as_str()) else {
            continue;
        };
        if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            out.f32.insert(name, (shape.to_vec(), data.to_vec()));
        }
    }
    Ok(out)
}
