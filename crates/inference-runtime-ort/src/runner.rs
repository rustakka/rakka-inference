//! `OrtRunner` — the public-facing type. Implements `ModelRunner` and
//! exposes the low-level `infer()` escape hatch for non-LLM models.

use async_trait::async_trait;
use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

use crate::config::OrtConfig;

#[cfg(feature = "ort")]
use std::sync::Arc;
#[cfg(feature = "ort")]
use crate::session::{build_state, OrtState};
#[cfg(feature = "ort")]
use crate::infer::{run_infer, InferOutputs, InferTensor};

pub struct OrtRunner {
    cfg: OrtConfig,
    #[cfg(feature = "ort")]
    state: tokio::sync::OnceCell<Arc<OrtState>>,
}

impl OrtRunner {
    pub fn new(cfg: OrtConfig) -> Self {
        Self {
            cfg,
            #[cfg(feature = "ort")]
            state: tokio::sync::OnceCell::new(),
        }
    }

    pub fn config(&self) -> &OrtConfig {
        &self.cfg
    }

    /// Low-level inference entry point. Bypasses tokenizer / sampling /
    /// streaming — feeds raw tensors to the ONNX session and returns
    /// f32 outputs. The chat-style [`ModelRunner::execute`] is built
    /// on top of this for ONNX-exported causal LMs; for embeddings,
    /// rerankers, Whisper encoders, and vision models, call this
    /// method directly.
    #[cfg(feature = "ort")]
    pub async fn infer(
        &mut self,
        inputs: std::collections::HashMap<String, InferTensor>,
    ) -> InferenceResult<InferOutputs> {
        let state = self.ensure_state().await?;
        run_infer(state, inputs).await
    }

    #[cfg(feature = "ort")]
    async fn ensure_state(&self) -> InferenceResult<Arc<OrtState>> {
        self.state
            .get_or_try_init(|| async {
                let cfg = self.cfg.clone();
                tokio::task::spawn_blocking(move || build_state(&cfg))
                    .await
                    .map_err(|e| {
                        InferenceError::Internal(format!("ort: spawn_blocking join: {e}"))
                    })?
            })
            .await
            .cloned()
    }
}

#[async_trait]
impl ModelRunner for OrtRunner {
    #[cfg_attr(
        feature = "ort",
        tracing::instrument(skip(self, _batch), fields(onnx = %self.cfg.onnx_path.display()))
    )]
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "ort"))]
        {
            let _ = _batch;
            Err(InferenceError::Internal(
                "ort feature disabled at build time — rebuild with --features ort".into(),
            ))
        }
        #[cfg(feature = "ort")]
        {
            let state = self.ensure_state().await?;
            let stream = crate::generate::run_generation(state, self.cfg.clone(), _batch).await?;
            Ok(RunHandle::streaming(stream))
        }
    }

    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()> {
        #[cfg(feature = "ort")]
        {
            if matches!(
                cause,
                SessionRebuildCause::CudaContextPoisoned | SessionRebuildCause::Manual
            ) {
                self.state = tokio::sync::OnceCell::new();
            }
        }
        let _ = cause;
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Ort
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
}
