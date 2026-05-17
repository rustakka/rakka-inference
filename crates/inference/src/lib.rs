//! # atomr-infer
//!
//! Multi-runtime GPU + remote inference as a supervised actor system
//! on top of [atomr](https://github.com/rustakka/atomr) and the
//! backend-agnostic [atomr-accel](https://github.com/rustakka/atomr-accel)
//! compute substrate. See `docs/architecture.md`
//! for the design.
//!
//! This crate is a **rollup**: it re-exports the public surface of the
//! workspace's per-runtime crates behind feature flags so downstream
//! consumers depend on a single crate (`inference`) and pick the
//! backends they need at compile time.
//!
//! ## Pure-remote builds
//!
//! ```sh
//! cargo build -p inference --features remote-only
//! ```
//!
//! produces a binary that compiles no GPU dependencies at all — useful
//! for pure-remote routers (a deployment that fronts OpenAI /
//! Anthropic / Gemini / LiteLLM with rate limiting, fallback and
//! observability but owns no hardware).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub use atomr_infer_core as core;
pub use atomr_infer_runtime as runtime;

#[cfg(feature = "candle")]
pub use atomr_infer_runtime_candle as runtime_candle;
#[cfg(feature = "cudarc")]
pub use atomr_infer_runtime_cudarc as runtime_cudarc;
#[cfg(feature = "mistralrs")]
pub use atomr_infer_runtime_mistralrs as runtime_mistralrs;
#[cfg(feature = "ort")]
pub use atomr_infer_runtime_ort as runtime_ort;
#[cfg(feature = "tensorrt")]
pub use atomr_infer_runtime_tensorrt as runtime_tensorrt;
#[cfg(feature = "vllm")]
pub use atomr_infer_runtime_vllm as runtime_vllm;

#[cfg(feature = "anthropic")]
pub use atomr_infer_runtime_anthropic as runtime_anthropic;
#[cfg(feature = "gemini")]
pub use atomr_infer_runtime_gemini as runtime_gemini;
#[cfg(feature = "litellm")]
pub use atomr_infer_runtime_litellm as runtime_litellm;
#[cfg(feature = "openai")]
pub use atomr_infer_runtime_openai as runtime_openai;

// Audio modality runtimes — gated on per-provider features.
#[cfg(feature = "stt-assemblyai")]
pub use atomr_infer_runtime_assemblyai as runtime_stt_assemblyai;
#[cfg(feature = "audio2face")]
pub use atomr_infer_runtime_audio2face as runtime_audio2face;
#[cfg(feature = "stt-deepgram")]
pub use atomr_infer_runtime_deepgram as runtime_stt_deepgram;
#[cfg(feature = "tts-elevenlabs")]
pub use atomr_infer_runtime_elevenlabs as runtime_tts_elevenlabs;
#[cfg(feature = "tts-gemini-live")]
pub use atomr_infer_runtime_gemini_live as runtime_tts_gemini_live;
#[cfg(feature = "tts-kokoro")]
pub use atomr_infer_runtime_kokoro as runtime_tts_kokoro;
#[cfg(feature = "tts-moss")]
pub use atomr_infer_runtime_moss as runtime_tts_moss;
#[cfg(feature = "tts-openai-realtime")]
pub use atomr_infer_runtime_openai_realtime as runtime_tts_openai_realtime;
#[cfg(feature = "stt-openai")]
pub use atomr_infer_runtime_openai_stt as runtime_stt_openai;
#[cfg(feature = "tts-openai")]
pub use atomr_infer_runtime_openai_tts as runtime_tts_openai;
#[cfg(feature = "tts-piper")]
pub use atomr_infer_runtime_piper as runtime_tts_piper;
#[cfg(feature = "stt-whisper")]
pub use atomr_infer_runtime_whisper_local as runtime_stt_whisper;
#[cfg(feature = "tts-xtts")]
pub use atomr_infer_runtime_xtts as runtime_tts_xtts;

#[cfg(feature = "pipeline")]
pub use atomr_infer_pipeline as pipeline;

#[cfg(feature = "testkit")]
pub use atomr_infer_testkit as testkit;

/// Re-export of the upstream `atomr-accel` trait surface so callers
/// can reach `AccelBackend`, `AccelRef<T>`, `AccelError`,
/// `CompletionStrategy`, `KernelOp`, etc. without taking a separate
/// dependency.
#[cfg(feature = "accel")]
pub use atomr_accel as accel;

/// Re-export of the NVIDIA CUDA backend (`atomr-accel-cuda`, split
/// out of the umbrella in atomr-accel 0.3) so callers can reach
/// `DeviceActor`, `ContextActor`, `GpuRef`, `GpuDispatcher`,
/// `PerActorAllocator`, `PlacementActor`, and the kernel actors at
/// `atomr_infer::accel_cuda::*`. Doc §4 ("Foundational Mapping" —
/// `WorkerActor` ≡ `DeviceActor`).
#[cfg(feature = "accel")]
pub use atomr_accel_cuda as accel_cuda;

/// Re-export of `atomr-accel-patterns` so callers can compose §9
/// pipelines (`DynamicBatchingServer`, `InferenceCascade`,
/// `ModelReplicaPool`, `FairShareScheduler`, `ModelHotSwapServer`,
/// `SpeculativeDecoder`, `MoeRouter`) without a second dep.
#[cfg(feature = "accel-patterns")]
pub use atomr_accel_patterns as accel_patterns;

/// Zero-config defaults — auto-provisioning helpers for common
/// dev-experience setups.
///
/// Currently ships [`defaults::gemma`] for local Gemma 4 via the
/// vLLM runner. Off by default; opt in with the `gemma-default`
/// feature.
#[cfg(feature = "gemma-default")]
pub mod defaults {
    /// Gemma 4 auto-provisioner. See
    /// `atomr_infer_runtime_vllm::defaults` for the full surface;
    /// re-exported here so rollup consumers don't need a second dep.
    pub use atomr_infer_runtime_vllm::defaults as gemma;
}

/// Re-export the most commonly used types so callers can `use
/// atomr_infer::prelude::*;` and have everything they need to declare
/// `Deployment`s and write actors.
pub mod prelude {
    pub use atomr_infer_core::{
        Deployment, ExecuteBatch, InferenceError, InferenceResult, ModelRunner, ProviderKind, RateLimits,
        RetryPolicy, RuntimeConfig, RuntimeKind, SecretString, Serving, Timeouts, TokenChunk, Tokens,
        TransportKind,
    };

    // Audio modality types — always available since they live in
    // `inference-core` (not gated on a feature). Provider runners are
    // still gated; these types let callers describe audio work without
    // depending on a specific provider crate.
    pub use atomr_infer_core::audio::{
        A2FOptions, AlignmentDelta, AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams,
        AudioPayload, BlendshapeChunk, RealtimeBatch, RealtimeIn, RealtimeOut, SpeechBatch, SpeechChunk,
        SynthOptions, TranscribeOptions, TranscriptChunk, TranscriptRole, Viseme, VoiceRef, WordTiming,
    };
    pub use atomr_infer_core::runner::{
        A2FRunHandle, A2FRunner, AudioRunHandle, AudioRunner, RealtimeRunner, RealtimeSession,
        SpeechRunHandle, SpeechRunner,
    };
}
