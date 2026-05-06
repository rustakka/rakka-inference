//! # inference-runtime-candle
//!
//! Pure-Rust transformer inference via the `candle` family of crates.
//! Doc §10.3.
//!
//! Default-features-off the crate compiles to a typed-error stub so
//! the workspace builds without `candle-core`, `candle-nn`,
//! `candle-transformers` (which pull large dependency trees). Operators
//! enable `--features candle` to wire in the real model runtime.
//!
//! When the feature is on, the runner uses
//! `atomr_accel_cuda::dispatcher::GpuDispatcher` for thread pinning and
//! `atomr_accel_cuda::stream::PerActorAllocator` for per-request stream
//! allocation — both are upstream substrate, not redefined here.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::deployment::RateLimits;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleConfig {
    /// HuggingFace repo id, or local path.
    pub model_path: String,
    #[serde(default)]
    pub device: CandleDevice,
    #[serde(default)]
    pub dtype: CandleDtype,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CandleDevice {
    #[default]
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CandleDtype {
    F32,
    #[default]
    F16,
    Bf16,
    Q4_0,
}

pub struct CandleRunner {
    #[allow(dead_code)]
    config: CandleConfig,
}

impl CandleRunner {
    pub fn new(config: CandleConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ModelRunner for CandleRunner {
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "candle"))]
        {
            return Err(InferenceError::Internal(
                "candle feature disabled at build time — rebuild with --features candle".into(),
            ));
        }
        #[cfg(feature = "candle")]
        {
            // Real wiring lands in Phase 2b of the doc roadmap. Shape:
            //   1. Pin to a thread via `atomr_accel_cuda::dispatcher::GpuDispatcher`.
            //   2. Allocate a stream from `atomr_accel_cuda::stream::PerActorAllocator`.
            //   3. Run forward pass with `candle_transformers::models::*`.
            //   4. Stream de-tokenized text out as `TokenChunk`s.
            // The ModelRunner trait is satisfied; the body is a smoke
            // value until the model-specific code lands.
            let _ = (); // ensure atomr_accel is referenced once the body lands
            return Err(InferenceError::Internal(
                "candle runner: forward pass pending — wire via \
                 atomr_accel_cuda::dispatcher::GpuDispatcher (doc §13 Phase 2b)"
                    .into(),
            ));
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Candle
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
    fn rate_limits(&self) -> Option<&RateLimits> {
        None
    }
}
