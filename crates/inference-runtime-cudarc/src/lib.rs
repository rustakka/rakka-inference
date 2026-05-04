//! # inference-runtime-cudarc
//!
//! Direct CUDA kernel dispatch via `cudarc` + `rakka-accel` primitives.
//! Doc §10.3.
//!
//! With `--features cudarc` the runner becomes a thin wrapper around
//! the rakka-accel kernel-actor mailbox: `ExecuteBatch` → user-supplied
//! kernel-launch closure → tokens. `rakka_accel::cuda::dispatcher::GpuDispatcher`
//! handles thread pinning and `rakka_accel::cuda::stream::PerActorAllocator`
//! handles per-request stream allocation, so this crate does not
//! re-implement either. Default-features-off the crate compiles to a
//! typed-error stub.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudarcConfig {
    /// CUDA device ordinal.
    pub device: u32,
    /// Logical name of the kernel package; resolved by the operator.
    pub kernel_package: String,
}

pub struct CudarcRunner {
    #[allow(dead_code)]
    config: CudarcConfig,
}

impl CudarcRunner {
    pub fn new(config: CudarcConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl ModelRunner for CudarcRunner {
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "cudarc"))]
        {
            return Err(InferenceError::Internal(
                "cudarc feature disabled at build time — rebuild with --features cudarc".into(),
            ));
        }
        #[cfg(feature = "cudarc")]
        {
            // Real wiring: rakka_accel::cuda::device::DeviceActor owns the
            // `Arc<CudaContext>`; rakka_accel::cuda::kernel::BlasActor and
            // friends sit underneath it as KernelChildren. The runner
            // selected at deploy time is a closure that posts a
            // typed kernel message (e.g. BlasMsg::Sgemm) to the
            // appropriate child and lifts the reply into a
            // TokenChunk. The closure-wiring lives in caller code; the
            // §13 Phase 2b follow-up adds a registry that maps
            // `CudarcConfig.kernel_package` to a concrete launcher.
            //
            // See:
            //   rakka_accel::cuda::dispatcher::GpuDispatcher
            //   rakka_accel::cuda::stream::PerActorAllocator
            //   rakka_accel::cuda::kernel::BlasActor
            return Err(InferenceError::Internal(
                "cudarc runner: kernel registry pending — wire via \
                 rakka_accel::cuda::kernel::BlasActor (doc §13 Phase 2b)"
                    .into(),
            ));
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Cudarc
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
}
