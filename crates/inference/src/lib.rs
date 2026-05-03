//! # rakka-inference
//!
//! Multi-runtime GPU + remote inference as a supervised actor system
//! on top of [rakka](https://github.com/local/rakka). See
//! `docs/rustakka-inference-architecture-v4.md` for the design.
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

pub use inference_core as core;
pub use inference_runtime as runtime;

#[cfg(feature = "vllm")]
pub use inference_runtime_vllm as runtime_vllm;
#[cfg(feature = "tensorrt")]
pub use inference_runtime_tensorrt as runtime_tensorrt;
#[cfg(feature = "ort")]
pub use inference_runtime_ort as runtime_ort;
#[cfg(feature = "candle")]
pub use inference_runtime_candle as runtime_candle;
#[cfg(feature = "cudarc")]
pub use inference_runtime_cudarc as runtime_cudarc;
#[cfg(feature = "mistralrs")]
pub use inference_runtime_mistralrs as runtime_mistralrs;

#[cfg(feature = "openai")]
pub use inference_runtime_openai as runtime_openai;
#[cfg(feature = "anthropic")]
pub use inference_runtime_anthropic as runtime_anthropic;
#[cfg(feature = "gemini")]
pub use inference_runtime_gemini as runtime_gemini;
#[cfg(feature = "litellm")]
pub use inference_runtime_litellm as runtime_litellm;

#[cfg(feature = "pipeline")]
pub use inference_pipeline as pipeline;

#[cfg(feature = "testkit")]
pub use inference_testkit as testkit;

/// Re-export of the upstream `rakka-cuda` substrate so callers can
/// reach `DeviceActor`, `ContextActor`, `GpuRef`, `GpuDispatcher`,
/// `PerActorAllocator`, `PlacementActor`, the kernel actors, etc.,
/// without taking a separate dependency. Doc §4 ("Foundational
/// Mapping" — `WorkerActor` ≡ `DeviceActor`).
#[cfg(feature = "cuda")]
pub use rakka_cuda as cuda;

/// Re-export of `rakka-cuda-patterns` so callers can compose §9
/// pipelines (`DynamicBatchingServer`, `InferenceCascade`,
/// `ModelReplicaPool`, `FairShareScheduler`, `ModelHotSwapServer`,
/// `SpeculativeDecoder`, `MoeRouter`) without a second dep.
#[cfg(feature = "cuda-patterns")]
pub use rakka_cuda_patterns as cuda_patterns;

/// Re-export the most commonly used types so callers can `use
/// inference::prelude::*;` and have everything they need to declare
/// `Deployment`s and write actors.
pub mod prelude {
    pub use inference_core::{
        Deployment, ExecuteBatch, InferenceError, InferenceResult, ModelRunner, ProviderKind,
        RateLimits, RetryPolicy, RuntimeConfig, RuntimeKind, SecretString, Serving, Timeouts,
        TokenChunk, Tokens, TransportKind,
    };
}
