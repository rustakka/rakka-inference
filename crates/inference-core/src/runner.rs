//! `ModelRunner` — the trait every runtime backend implements.
//!
//! This is the seam that makes the actor decomposition work for both
//! local-GPU and remote-network runtimes. Doc §5.4. The trait is
//! deliberately small; backend-specific scheduling lives inside the
//! runner's `execute` body.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::audio::{AudioBatch, BlendshapeChunk, RealtimeBatch, SpeechBatch, SpeechChunk, TranscriptChunk};
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
#[non_exhaustive]
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
#[non_exhaustive]
pub enum SessionRebuildCause {
    CudaContextPoisoned,
    RemoteAuthFailure,
    RemoteConfigChange,
    Manual,
}

/// Opaque CUDA-context handle. Real local runtimes downcast to
/// `Arc<atomr_accel_cuda::device::DeviceState>` (which itself wraps the
/// `cudarc::driver::CudaContext`); tests and remote runtimes pass
/// `None`. Kept type-erased so `inference-core` doesn't depend on
/// `atomr-accel`/`cudarc` — preserves the §10.4 dependency budget so
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

// ─────────────────────────────────────────────────────────────────────────────
// Audio sibling handles
// ─────────────────────────────────────────────────────────────────────────────
//
// We keep [`RunHandle`] monomorphic over [`TokenChunk`] (see the comment
// at the top of this file). Sibling handles per audio modality let each
// modality's chunk type stay statically known to consumers, avoiding a
// `match` per chunk on the hot path.

/// Result of [`AudioRunner::execute_audio`] — streamed transcript chunks.
pub struct AudioRunHandle {
    inner: BoxStream<'static, InferenceResult<TranscriptChunk>>,
}

impl AudioRunHandle {
    pub fn streaming(inner: BoxStream<'static, InferenceResult<TranscriptChunk>>) -> Self {
        Self { inner }
    }
    pub fn into_stream(self) -> BoxStream<'static, InferenceResult<TranscriptChunk>> {
        self.inner
    }
}

impl std::fmt::Debug for AudioRunHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioRunHandle").finish_non_exhaustive()
    }
}

/// Result of [`SpeechRunner::speak`] — streamed synthesized speech chunks.
pub struct SpeechRunHandle {
    inner: BoxStream<'static, InferenceResult<SpeechChunk>>,
}

impl SpeechRunHandle {
    pub fn streaming(inner: BoxStream<'static, InferenceResult<SpeechChunk>>) -> Self {
        Self { inner }
    }
    pub fn into_stream(self) -> BoxStream<'static, InferenceResult<SpeechChunk>> {
        self.inner
    }
}

impl std::fmt::Debug for SpeechRunHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechRunHandle").finish_non_exhaustive()
    }
}

/// Result of [`A2FRunner::execute_audio2face`] — streamed blendshape frames.
pub struct A2FRunHandle {
    inner: BoxStream<'static, InferenceResult<BlendshapeChunk>>,
}

impl A2FRunHandle {
    pub fn streaming(inner: BoxStream<'static, InferenceResult<BlendshapeChunk>>) -> Self {
        Self { inner }
    }
    pub fn into_stream(self) -> BoxStream<'static, InferenceResult<BlendshapeChunk>> {
        self.inner
    }
}

impl std::fmt::Debug for A2FRunHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2FRunHandle").finish_non_exhaustive()
    }
}

/// Handle to a live bidirectional session opened by
/// [`RealtimeRunner::open_session`]. The session's channels are owned
/// by the caller and the runner adapter; this handle exposes the
/// session's lifecycle for telemetry and cancellation.
pub struct RealtimeSession {
    request_id: String,
    cancel: futures::future::AbortHandle,
}

impl RealtimeSession {
    pub fn new(request_id: String, cancel: futures::future::AbortHandle) -> Self {
        Self { request_id, cancel }
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Cancel the underlying session adapter task. The outbound channel
    /// will close shortly after.
    pub fn cancel(&self) {
        self.cancel.abort();
    }
}

impl std::fmt::Debug for RealtimeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeSession")
            .field("request_id", &self.request_id)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio sibling traits
// ─────────────────────────────────────────────────────────────────────────────

/// Speech-to-text runner. Implemented by `inference-runtime-openai-stt`,
/// `inference-runtime-whisper-local`, `inference-runtime-deepgram`,
/// `inference-runtime-assemblyai`.
///
/// Source: `FR-STT-001`.
#[async_trait]
pub trait AudioRunner: Send + Sync {
    /// Submit one transcription request. Returns immediately;
    /// transcript chunks arrive on the returned [`AudioRunHandle`].
    ///
    /// # Errors
    ///
    /// - [`InferenceError::UnsupportedAudioFormat`] when the input's
    ///   [`crate::audio::AudioParams::format`] is not handled by this
    ///   runtime.
    /// - [`InferenceError::BadRequest`] for malformed options.
    /// - [`InferenceError::Unauthorized`] / [`InferenceError::RateLimited`]
    ///   / [`InferenceError::ServerError`] for remote-provider faults.
    async fn execute_audio(&mut self, batch: AudioBatch) -> InferenceResult<AudioRunHandle>;

    /// Rebuild any remote session or local pipeline.
    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()>;

    fn runtime_kind(&self) -> RuntimeKind;
    fn transport_kind(&self) -> TransportKind;

    /// Rate-limit metadata. Returns `None` for local runtimes.
    fn rate_limits(&self) -> Option<&RateLimits> {
        None
    }
}

/// Text-to-speech runner. Implemented by `inference-runtime-openai-tts`,
/// `inference-runtime-elevenlabs`, `inference-runtime-piper`,
/// `inference-runtime-kokoro`, `inference-runtime-xtts`,
/// `inference-runtime-moss`.
///
/// Source: `FR-TTS-001`.
#[async_trait]
pub trait SpeechRunner: Send + Sync {
    /// Submit one synthesis request. Returns immediately; speech
    /// chunks arrive on the returned [`SpeechRunHandle`].
    ///
    /// # Errors
    ///
    /// - [`InferenceError::BadRequest`] for malformed options or
    ///   an unknown voice.
    /// - [`InferenceError::Unauthorized`] / [`InferenceError::RateLimited`]
    ///   / [`InferenceError::ServerError`] for remote-provider faults.
    async fn speak(&mut self, batch: SpeechBatch) -> InferenceResult<SpeechRunHandle>;

    /// Rebuild any remote session or local pipeline.
    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()>;

    fn runtime_kind(&self) -> RuntimeKind;
    fn transport_kind(&self) -> TransportKind;

    fn rate_limits(&self) -> Option<&RateLimits> {
        None
    }
}

/// Bidirectional realtime speech runner. Implemented by
/// `inference-runtime-openai-realtime` and `inference-runtime-gemini-live`.
///
/// Source: `FR-TTS-001` (realtime section).
#[async_trait]
pub trait RealtimeRunner: Send + Sync {
    /// Open one bidirectional session. The runner spawns its own
    /// session adapter task and returns a [`RealtimeSession`] handle
    /// for cancellation and telemetry. The session's `outbound`
    /// channel closes when the adapter shuts down.
    ///
    /// # Errors
    ///
    /// - [`InferenceError::Unauthorized`] when the provider rejects
    ///   the session's credentials.
    /// - [`InferenceError::NetworkError`] when the underlying transport
    ///   fails to connect.
    /// - [`InferenceError::RealtimeClosed`] when the session is closed
    ///   before any frames are exchanged (rare).
    async fn open_session(&mut self, batch: RealtimeBatch) -> InferenceResult<RealtimeSession>;

    /// Rebuild any persistent transport state (e.g. WebSocket pool).
    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()>;

    fn runtime_kind(&self) -> RuntimeKind;
    fn transport_kind(&self) -> TransportKind;

    fn rate_limits(&self) -> Option<&RateLimits> {
        None
    }
}

/// Audio2Face runner. Implemented by `inference-runtime-audio2face`.
///
/// Shares [`AudioBatch`] with [`AudioRunner`] because both ingest
/// audio. Returns a [`BlendshapeChunk`] stream rather than a
/// [`TranscriptChunk`] stream — hence a separate trait.
///
/// Source: `FR-A2F-001`.
#[async_trait]
pub trait A2FRunner: Send + Sync {
    /// Submit one audio-to-blendshape request. Returns immediately;
    /// blendshape frames arrive on the returned [`A2FRunHandle`].
    ///
    /// # Errors
    ///
    /// - [`InferenceError::UnsupportedAudioFormat`] when the input's
    ///   [`crate::audio::AudioParams::format`] is not handled by this
    ///   runtime (A2F-3D expects 16 kHz mono PCM).
    /// - [`InferenceError::BadRequest`] for malformed options.
    /// - [`InferenceError::NetworkError`] / [`InferenceError::ServerError`]
    ///   for gRPC transport faults.
    async fn execute_audio2face(&mut self, batch: AudioBatch) -> InferenceResult<A2FRunHandle>;

    /// Rebuild the gRPC session if the persistent stream drops.
    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()>;

    fn runtime_kind(&self) -> RuntimeKind;
    fn transport_kind(&self) -> TransportKind;

    fn rate_limits(&self) -> Option<&RateLimits> {
        None
    }
}
