//! # inference-runtime-ort
//!
//! ONNX Runtime backend via the `ort` crate. Doc §10.3. Targets
//! pre-compiled ONNX graphs (Whisper, embedding models, BGE rerankers).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrtConfig {
    pub onnx_path: std::path::PathBuf,
    #[serde(default)]
    pub execution_provider: ExecutionProvider,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    Cuda,
    TensorRt,
    DirectMl,
}

pub struct OrtRunner {
    #[allow(dead_code)]
    config: OrtConfig,
}

impl OrtRunner {
    pub fn new(config: OrtConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ModelRunner for OrtRunner {
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "ort"))]
        {
            return Err(InferenceError::Internal(
                "ort feature disabled at build time — rebuild with --features ort".into(),
            ));
        }
        #[cfg(feature = "ort")]
        {
            return Err(InferenceError::Internal(
                "ort runner: session.run wiring pending (see doc §13 Phase 2b)".into(),
            ));
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Ort
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
}
