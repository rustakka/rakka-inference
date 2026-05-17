//! Session lifecycle: build the `ort::Session` with execution providers,
//! probe its topology, and bundle it with the (optional) tokenizer.

use std::sync::Arc;

use atomr_infer_core::error::{InferenceError, InferenceResult};
use ort::session::Session;

use crate::config::{ExecutionProvider, OrtConfig};
use crate::error::{internal, map_ort_err};
use crate::topology::Topology;

/// Bundle of immutable + lazily-mutable state shared by every
/// generation invoked on this runner.
pub(crate) struct OrtState {
    /// Session is `&mut self`-only for `run`, so we wrap in a sync
    /// mutex. One generation per runner at a time matches the actor
    /// model (one `WorkerActor` owns one runner).
    pub(crate) session: parking_lot::Mutex<Session>,
    pub(crate) topology: Topology,
    pub(crate) tokenizer: Option<Arc<tokenizers::Tokenizer>>,
}

pub(crate) fn build_state(cfg: &OrtConfig) -> InferenceResult<Arc<OrtState>> {
    let mut builder = Session::builder().map_err(map_ort_err)?;

    if let Some(threads) = cfg.intra_threads {
        builder = builder.with_intra_threads(threads).map_err(map_ort_err)?;
    }

    let eps = providers_for(cfg)?;
    if !eps.is_empty() {
        builder = builder.with_execution_providers(eps).map_err(map_ort_err)?;
    }

    let session = builder
        .commit_from_file(&cfg.onnx_path)
        .map_err(|e| internal(&format!("commit_from_file({})", cfg.onnx_path.display()), e))?;

    let topology = Topology::probe(&session);
    let tokenizer = crate::tokenizer::resolve_tokenizer(cfg)?;

    Ok(Arc::new(OrtState {
        session: parking_lot::Mutex::new(session),
        topology,
        tokenizer: tokenizer.map(Arc::new),
    }))
}

#[allow(unused_variables)] // device_id consumed only when ort-cuda is on
fn providers_for(
    cfg: &OrtConfig,
) -> InferenceResult<Vec<ort::execution_providers::ExecutionProviderDispatch>> {
    use ort::execution_providers::CPUExecutionProvider;

    let mut out: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();

    match cfg.execution_provider {
        ExecutionProvider::Cpu => {
            out.push(CPUExecutionProvider::default().build());
        }
        ExecutionProvider::Cuda => {
            #[cfg(feature = "ort-cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                // Future hook: when atomr-accel-cuda's PerActorAllocator
                // hands us a `cudarc::driver::CudaStream`, swap in
                // `.with_compute_stream(stream.cu_stream() as *mut ())`
                // (unsafe) so ORT shares the timeline. Today we let ORT
                // pick its own stream — fine for single-runner deploys.
                out.push(
                    CUDAExecutionProvider::default()
                        .with_device_id(cfg.device_id as i32)
                        .build(),
                );
                // Always keep CPU as a fallback for ops the CUDA EP
                // can't handle.
                out.push(CPUExecutionProvider::default().build());
            }
            #[cfg(not(feature = "ort-cuda"))]
            {
                return Err(InferenceError::BadRequest {
                    message: "ort: execution_provider=cuda requires the `ort-cuda` cargo feature \
                         (rebuild with --features ort,ort-cuda)"
                        .into(),
                });
            }
        }
        ExecutionProvider::TensorRt => {
            return Err(InferenceError::BadRequest {
                message: "ort: execution_provider=tensor_rt is not wired in this build — \
                          use the `tensorrt` runner crate instead, or open an issue if you \
                          need ORT's TensorRT EP specifically"
                    .into(),
            });
        }
        ExecutionProvider::DirectMl => {
            return Err(InferenceError::BadRequest {
                message: "ort: execution_provider=direct_ml is not wired in this build".into(),
            });
        }
    }

    Ok(out)
}
