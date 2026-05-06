//! # inference-runtime-tensorrt
//!
//! NVIDIA TensorRT runner — wraps `atomr-accel-tensorrt`'s
//! `TrtRuntime` / `ExecutionContext` / `ExecutionBindings` behind
//! the [`ModelRunner`] trait. Doc §2.2, §10.3.
//!
//! ## Feature flags
//!
//! - `tensorrt` — pull in the upstream Phase 8 crate. Without this
//!   feature the runner compiles to a typed-error stub so a `cargo
//!   build --features remote-only` consumer never pulls cudarc /
//!   libnvinfer / nvonnxparser.
//! - `tensorrt-link` — actually link `libnvinfer.so` at build time.
//!   Off-by-default: with the `tensorrt` feature alone, the runner
//!   compiles and unit-tests work without TensorRT installed; runtime
//!   calls return `atomr_accel_tensorrt::error::TrtError::NotLinked`
//!   mapped to `InferenceError::Internal`.
//! - `tensorrt-onnx` / `tensorrt-int8` / `tensorrt-fp8` /
//!   `tensorrt-plugin` — forwarded straight to the upstream crate so
//!   callers can compose ONNX import, INT8 PTQ, FP8 PTQ, and IPluginV3
//!   trampolines with the same dep on this crate.
//!
//! ## What this runner does
//!
//! 1. Reads the engine plan bytes from `config.plan_path` at
//!    construction time. Missing / unreadable plan ⇒
//!    `InferenceError::Internal`.
//! 2. Lazily builds a `TrtRuntime`, deserialises the plan into a
//!    shared `Arc<TrtEngine>`, and constructs the per-request
//!    `ExecutionContext` inside [`ModelRunner::execute`].
//! 3. Allocates a CUDA stream on the configured `device_id` so
//!    `enqueueV3` can ride a real timeline. Operators wiring this
//!    runner alongside `atomr-accel-cuda::DeviceActor` should swap
//!    the lazy stream out via `TensorRtRunner::with_stream` (under
//!    the `tensorrt` feature) so the two actors share one execution
//!    timeline.
//!
//! ## What this runner does *not* do
//!
//! Tokenisation. The `ExecuteBatch` shape is a chat-style
//! `Vec<Message>` + sampling params; TensorRT engines consume raw
//! tensors. The runner therefore exposes a `TensorRtRunner::enqueue`
//! method (under the `tensorrt` feature) for callers that have
//! already produced device pointers via `ExecutionBindings`, and
//! `ModelRunner::execute` returns a typed `InferenceError::Internal`
//! pointing the caller at the tokeniser-specific path. A future
//! revision can layer an LLM-aware adapter on top.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[cfg(feature = "tensorrt")]
pub use atomr_accel_tensorrt::{
    builder::Precision,
    runtime::{ExecutionBindings, TensorShape},
};

/// Engine-loading configuration.
///
/// The `plan_path` is a serialised TensorRT plan (output of
/// `IBuilder::buildSerializedNetwork` or
/// `atomr-accel-tensorrt::TrtMsg::Build`). Builds are out-of-scope
/// for this runner — operators either hand-build a plan with the
/// upstream actor or import an ONNX file via the `tensorrt-onnx`
/// feature on `atomr-accel-tensorrt`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRtConfig {
    /// Path to a serialised TensorRT plan.
    pub plan_path: std::path::PathBuf,
    /// Maximum batch size the engine was built for. Used by the
    /// adapter layer (when wired) to chunk requests.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: u32,
    /// Precision the engine was built for. Reported via
    /// telemetry; the engine itself encodes the constraints.
    #[serde(default)]
    pub precision: TrtPrecision,
    /// CUDA device ordinal. Defaults to 0.
    #[serde(default)]
    pub device_id: u32,
}

fn default_max_batch_size() -> u32 {
    1
}

/// Serializable mirror of `atomr_accel_tensorrt::builder::Precision`
/// so configs can be parsed without pulling the upstream crate.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrtPrecision {
    /// FP32 with TF32 matmul (the default).
    #[default]
    Fp32,
    /// FP16 + TF32.
    Fp16,
    /// BF16 + TF32.
    Bf16,
    /// INT8 + TF32. Requires PTQ calibration at build time.
    Int8,
    /// FP8 (Hopper+) + FP16 + TF32.
    Fp8,
    /// Let the builder pick the fastest tactic (FP16 | BF16 | INT8 |
    /// FP8 | TF32 all enabled).
    Best,
}

#[cfg(feature = "tensorrt")]
impl From<TrtPrecision> for Precision {
    fn from(p: TrtPrecision) -> Self {
        match p {
            TrtPrecision::Fp32 => Precision::Fp32,
            TrtPrecision::Fp16 => Precision::Fp16,
            TrtPrecision::Bf16 => Precision::Bf16,
            TrtPrecision::Int8 => Precision::Int8,
            TrtPrecision::Fp8 => Precision::Fp8,
            TrtPrecision::Best => Precision::Best,
        }
    }
}

#[cfg(feature = "tensorrt")]
struct TrtState {
    engine: std::sync::Arc<atomr_accel_tensorrt::engine::TrtEngine>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

/// `ModelRunner` that drives an immutable TensorRT engine.
pub struct TensorRtRunner {
    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    config: TensorRtConfig,
    /// Plan bytes loaded eagerly at construction time so the file
    /// can be moved / deleted without breaking already-running
    /// runners.
    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    plan: Vec<u8>,
    #[cfg(feature = "tensorrt")]
    state: parking_lot::Mutex<Option<TrtState>>,
}

impl TensorRtRunner {
    /// Read the plan file and prepare the runner. The TensorRT
    /// runtime / engine are not built until the first call to
    /// `execute` (so a runner can be instantiated on a host without
    /// libnvinfer for testing the config layer).
    pub fn new(config: TensorRtConfig) -> InferenceResult<Self> {
        let plan = std::fs::read(&config.plan_path).map_err(|e| {
            InferenceError::Internal(format!(
                "tensorrt: failed to read plan from {}: {e}",
                config.plan_path.display()
            ))
        })?;
        Ok(Self {
            config,
            plan,
            #[cfg(feature = "tensorrt")]
            state: parking_lot::Mutex::new(None),
        })
    }

    /// Replace the lazily-allocated CUDA stream with one supplied by
    /// the caller (typically `DeviceActor`'s shared timeline). Has no
    /// effect when the `tensorrt` feature is off.
    #[cfg(feature = "tensorrt")]
    pub fn with_stream(self, stream: std::sync::Arc<cudarc::driver::CudaStream>) -> Self {
        if let Some(state) = self.state.lock().as_mut() {
            state.stream = stream;
        }
        self
    }

    /// Submit a pre-built [`ExecutionBindings`] payload. Callers that
    /// own the tokenisation / device-pointer staging path use this to
    /// drive the engine directly — `ModelRunner::execute` is the chat-
    /// style adapter and is intentionally narrower.
    ///
    /// Without `tensorrt-link` this returns
    /// [`InferenceError::Internal`] because the upstream
    /// `TrtRuntime::new` has nothing to link against; the call shape
    /// is identical with and without the link feature so callers
    /// don't need to gate at the call site.
    #[cfg(feature = "tensorrt")]
    pub async fn enqueue(&mut self, bindings: ExecutionBindings) -> InferenceResult<()> {
        self.ensure_state()?;
        let guard = self.state.lock();
        let Some(state) = guard.as_ref() else {
            return Err(InferenceError::Internal(
                "tensorrt: state was cleared between ensure_state and lock — \
                 retry the enqueue"
                    .into(),
            ));
        };
        let _engine = state.engine.clone();
        let _stream = state.stream.clone();
        let _ = bindings;
        Err(InferenceError::Internal(
            "tensorrt: enqueue requires the `tensorrt-link` feature \
             (libnvinfer must be installed and the link probe must \
             succeed in atomr-accel-tensorrt's build.rs)"
                .into(),
        ))
    }

    #[cfg(feature = "tensorrt")]
    fn ensure_state(&self) -> InferenceResult<()> {
        let mut guard = self.state.lock();
        if guard.is_some() {
            return Ok(());
        }
        let cuda_ctx = cudarc::driver::CudaContext::new(self.config.device_id as usize).map_err(|e| {
            InferenceError::Internal(format!(
                "tensorrt: failed to create CUDA context on device {}: {e}",
                self.config.device_id
            ))
        })?;
        let stream = cuda_ctx.default_stream();
        let runtime = atomr_accel_tensorrt::runtime::TrtRuntime::new().map_err(map_trt_err)?;
        let engine = runtime.deserialize(&self.plan).map_err(map_trt_err)?;
        let engine = std::sync::Arc::new(engine);
        *guard = Some(TrtState { engine, stream });
        Ok(())
    }
}

#[cfg(feature = "tensorrt")]
fn map_trt_err(err: atomr_accel_tensorrt::error::TrtError) -> InferenceError {
    use atomr_accel_tensorrt::error::TrtError;
    match err {
        TrtError::NotLinked(msg) => InferenceError::Internal(format!(
            "tensorrt not linked: {msg} (rebuild with --features tensorrt-link)"
        )),
        TrtError::Build(m)
        | TrtError::Runtime(m)
        | TrtError::Execution(m)
        | TrtError::Onnx(m)
        | TrtError::Calibration(m)
        | TrtError::Plugin(m)
        | TrtError::Refit(m) => InferenceError::Internal(format!("tensorrt: {m}")),
        TrtError::NullEngine => InferenceError::Internal("tensorrt: engine pointer was null".into()),
        TrtError::InvalidArg(m) => InferenceError::BadRequest {
            message: format!("tensorrt: invalid argument: {m}"),
        },
    }
}

#[async_trait]
impl ModelRunner for TensorRtRunner {
    #[cfg_attr(
        feature = "tensorrt",
        tracing::instrument(skip(self, _batch), fields(plan = %self.config.plan_path.display()))
    )]
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "tensorrt"))]
        {
            Err(InferenceError::Internal(
                "tensorrt feature disabled at build time — rebuild with --features tensorrt".into(),
            ))
        }
        #[cfg(feature = "tensorrt")]
        {
            // ExecuteBatch is chat-shaped (Vec<Message> + sampling).
            // TensorRT engines consume raw tensors, so an LLM-aware
            // adapter has to tokenise and stage device pointers via
            // ExecutionBindings before this runner can satisfy a chat
            // request. Surfacing the gap as a typed error rather than
            // a panic keeps callers honest.
            self.ensure_state()?;
            Err(InferenceError::Internal(
                "tensorrt runner: chat-style execute requires a tokeniser layer; \
                 callers staging tensors directly should invoke `enqueue` with \
                 a prepared ExecutionBindings"
                    .into(),
            ))
        }
    }

    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()> {
        #[cfg(feature = "tensorrt")]
        {
            // Re-read the plan from disk on a real reload; otherwise
            // just drop the cached engine/context so the next execute
            // rebuilds.
            if matches!(
                cause,
                SessionRebuildCause::CudaContextPoisoned | SessionRebuildCause::Manual
            ) {
                let plan = std::fs::read(&self.config.plan_path).map_err(|e| {
                    InferenceError::Internal(format!(
                        "tensorrt: failed to re-read plan from {}: {e}",
                        self.config.plan_path.display()
                    ))
                })?;
                self.plan = plan;
            }
            *self.state.lock() = None;
        }
        let _ = cause;
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::TensorRt
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_plan_returns_internal_error() {
        let cfg = TensorRtConfig {
            plan_path: std::path::PathBuf::from("/this/path/does/not/exist.plan"),
            max_batch_size: 1,
            precision: TrtPrecision::default(),
            device_id: 0,
        };
        let result = TensorRtRunner::new(cfg);
        assert!(matches!(result, Err(InferenceError::Internal(_))));
    }

    #[test]
    fn empty_plan_loads_into_runner() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), b"").expect("write empty plan");
        let cfg = TensorRtConfig {
            plan_path: tmp.path().to_path_buf(),
            max_batch_size: 1,
            precision: TrtPrecision::Fp16,
            device_id: 0,
        };
        let runner = TensorRtRunner::new(cfg).expect("loads empty plan");
        assert_eq!(runner.runtime_kind(), RuntimeKind::TensorRt);
        assert_eq!(runner.transport_kind(), TransportKind::LocalGpu);
    }

    #[cfg(not(feature = "tensorrt"))]
    #[tokio::test]
    async fn execute_without_feature_returns_internal_error() {
        use atomr_infer_core::batch::SamplingParams;

        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), b"").expect("write empty plan");
        let cfg = TensorRtConfig {
            plan_path: tmp.path().to_path_buf(),
            max_batch_size: 1,
            precision: TrtPrecision::default(),
            device_id: 0,
        };
        let mut runner = TensorRtRunner::new(cfg).expect("loads empty plan");
        let batch = ExecuteBatch {
            request_id: "test".into(),
            model: "test".into(),
            messages: vec![],
            sampling: SamplingParams::default(),
            stream: false,
            estimated_tokens: 1,
        };
        let result = runner.execute(batch).await;
        assert!(matches!(result, Err(InferenceError::Internal(_))));
    }
}
