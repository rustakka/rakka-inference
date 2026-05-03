//! `ModelRunner` — the trait every runtime backend implements.
//!
//! This is the seam that makes the actor decomposition work for both
//! local-GPU and remote-network runtimes. Doc §5.4. The trait is
//! deliberately small; backend-specific scheduling lives inside the
//! runner's `execute` body.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::batch::ExecuteBatch;
use crate::deployment::RateLimits;
use crate::error::{InferenceError, InferenceResult};
use crate::runtime::{RuntimeKind, TransportKind};
use crate::tokens::TokenChunk;

/// The result of `ModelRunner::execute`. Local runtimes typically
/// return `Streaming` even for unary calls (one final chunk); remote
/// runtimes return `Streaming` for SSE responses and a single-chunk
/// stream otherwise. Callers always treat it as a stream.
pub struct RunHandle {
    inner: BoxStream<'static, InferenceResult<TokenChunk>>,
}

impl RunHandle {
    pub fn streaming(inner: BoxStream<'static, InferenceResult<TokenChunk>>) -> Self {
        Self { inner }
    }

    pub fn into_stream(self) -> BoxStream<'static, InferenceResult<TokenChunk>> {
        self.inner
    }
}

impl std::fmt::Debug for RunHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunHandle").finish_non_exhaustive()
    }
}

/// Where to load weights from. Local runtimes implement; remote
/// runtimes no-op.
#[derive(Debug, Clone)]
pub enum WeightSource {
    HuggingFace {
        repo: String,
        revision: Option<String>,
    },
    LocalPath {
        path: std::path::PathBuf,
    },
    /// The runtime knows how to fetch its own weights (vLLM, mistralrs).
    RuntimeManaged,
}

/// Why a session rebuild was requested. Drives the runtime-specific
/// rebuild behaviour described in §3.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRebuildCause {
    CudaContextPoisoned,
    RemoteAuthFailure,
    RemoteConfigChange,
    Manual,
}

/// Opaque CUDA-context handle. Real local runtimes downcast to
/// `Arc<rakka_accel::cuda::device::DeviceState>` (which itself wraps the
/// `cudarc::driver::CudaContext`); tests and remote runtimes pass
/// `None`. Kept type-erased so `inference-core` doesn't depend on
/// `rakka-accel`/`cudarc` — preserves the §10.4 dependency budget so
/// `inference --features remote-only` builds compile no GPU deps at
/// all. Local-runtime crates downcast at the seam.
pub type CudaContextHandle = Arc<dyn std::any::Any + Send + Sync>;

#[async_trait]
pub trait ModelRunner: Send + Sync {
    /// Run an inference. For local runtimes, dispatches kernels; for
    /// remote runtimes, sends an HTTP request. Returns immediately;
    /// completion is observed via the returned `RunHandle` stream.
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle>;

    /// Local runtimes load weights to GPU; remote runtimes default to
    /// a no-op.
    async fn load_weights(
        &mut self,
        _ctx: Option<&CudaContextHandle>,
        _source: WeightSource,
    ) -> InferenceResult<()> {
        Ok(())
    }

    /// Local runtimes rebuild after CUDA context poison; remote
    /// runtimes rebuild after auth failure or config change.
    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()>;

    fn runtime_kind(&self) -> RuntimeKind;
    fn transport_kind(&self) -> TransportKind;
    fn gil_pinned(&self) -> bool {
        matches!(self.runtime_kind(), RuntimeKind::Vllm | RuntimeKind::Python(_))
    }

    /// Rate-limit metadata. Returns `None` for local runtimes; remote
    /// runtimes return their configured limits so the
    /// `RateLimiterActor` can be initialized at deploy time.
    fn rate_limits(&self) -> Option<&RateLimits> {
        None
    }

    /// Best-effort cost estimate for the given batch (USD). Used by
    /// `TieredRouter`-style actors and budget enforcement. Local
    /// runtimes default to 0 (compute cost is amortized).
    fn estimate_cost_usd(&self, _batch: &ExecuteBatch) -> f64 {
        0.0
    }
}

/// Helper: convert a generic error string to an `InferenceError`. Useful
/// inside `RunHandle` stream futures that need to lift unrelated errors.
pub fn lift_internal<E: std::fmt::Display>(err: E) -> InferenceError {
    InferenceError::Internal(err.to_string())
}
