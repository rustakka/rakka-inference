//! # inference-runtime-vllm
//!
//! vLLM (Python) runtime. Doc §2.1, §10.3. The canonical local-LLM
//! backend per architecture v4.
//!
//! Default-features-off the crate compiles to a stub. With
//! `--features vllm` it pulls in `inference-python-bridge` (`python`
//! feature) and wraps vLLM's `EngineCore` over PyO3.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use inference_core::batch::ExecuteBatch;
use inference_core::error::{InferenceError, InferenceResult};
use inference_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use inference_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    pub model: String,
    #[serde(default = "default_tp")]
    pub tensor_parallel_size: u32,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default)]
    pub gpu_memory_utilization: Option<f32>,
}

fn default_tp() -> u32 {
    1
}
fn default_dtype() -> String {
    "auto".to_string()
}

pub struct VllmRunner {
    #[allow(dead_code)]
    config: VllmConfig,
}

impl VllmRunner {
    pub fn new(config: VllmConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ModelRunner for VllmRunner {
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "vllm"))]
        {
            return Err(InferenceError::Internal(
                "vllm feature disabled at build time — rebuild with --features vllm".into(),
            ));
        }
        #[cfg(feature = "vllm")]
        {
            return Err(InferenceError::Internal(
                "vllm runner: PythonGpuBridge wiring pending (see doc §13 Phase 2a)".into(),
            ));
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Vllm
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
    fn gil_pinned(&self) -> bool {
        true
    }
}
