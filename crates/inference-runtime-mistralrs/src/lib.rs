//! # inference-runtime-mistralrs
//!
//! Thin wrapper around the `mistralrs` Rust LLM runtime. Doc §10.3.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use inference_core::batch::ExecuteBatch;
use inference_core::error::{InferenceError, InferenceResult};
use inference_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use inference_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsConfig {
    pub model_id: String,
    #[serde(default)]
    pub quant: Option<String>,
}

pub struct MistralRsRunner {
    #[allow(dead_code)]
    config: MistralRsConfig,
}

impl MistralRsRunner {
    pub fn new(config: MistralRsConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ModelRunner for MistralRsRunner {
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "mistralrs"))]
        {
            return Err(InferenceError::Internal(
                "mistralrs feature disabled at build time — rebuild with --features mistralrs".into(),
            ));
        }
        #[cfg(feature = "mistralrs")]
        {
            return Err(InferenceError::Internal(
                "mistralrs runner: integration pending (see doc §13 Phase 2b)".into(),
            ));
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::MistralRs
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
}
