//! # inference-runtime-tensorrt
//!
//! NVIDIA TensorRT runtime — opaque pre-compiled `nvinfer` plans.
//! Doc §2.2, §10.3.
//!
//! Default-features-off the crate compiles to a stub (no `extern "C"`
//! block). Operators with `libnvinfer.so` enable `--features tensorrt`
//! to pull in the FFI surface.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use inference_core::batch::ExecuteBatch;
use inference_core::error::{InferenceError, InferenceResult};
use inference_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use inference_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRtConfig {
    pub plan_path: std::path::PathBuf,
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: u32,
}

fn default_max_batch_size() -> u32 {
    1
}

pub struct TensorRtRunner {
    #[allow(dead_code)]
    config: TensorRtConfig,
}

impl TensorRtRunner {
    pub fn new(config: TensorRtConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ModelRunner for TensorRtRunner {
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "tensorrt"))]
        {
            return Err(InferenceError::Internal(
                "tensorrt feature disabled at build time — rebuild with --features tensorrt".into(),
            ));
        }
        #[cfg(feature = "tensorrt")]
        {
            return Err(InferenceError::Internal(
                "tensorrt runner: ExecutionContext wiring pending (see doc §13 Phase 2b)".into(),
            ));
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::TensorRt
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
}
