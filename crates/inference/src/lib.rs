//! # atomr-infer
//!
//! Multi-runtime GPU + remote inference as a supervised actor system
//! on top of [rakka](https://github.com/rustakka/atomr) and the
//! backend-agnostic [rakka-accel](https://github.com/rustakka/atomr-accel)
//! compute substrate. See `docs/rustakka-inference-architecture-v4.md`
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

// candle / cudarc re-exports are intentionally absent in 0.3 — both
// runners depend on atomr-accel which is mid-rakka→atomr rename and
// not yet usable as a sibling workspace dep.
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

#[cfg(feature = "pipeline")]
pub use atomr_infer_pipeline as pipeline;

#[cfg(feature = "testkit")]
pub use atomr_infer_testkit as testkit;

/// Re-export of the upstream `rakka-accel` substrate so callers can
/// reach `AccelBackend`, `AccelRef<T>`, `AccelError`, and (with the
/// `cuda` backend re-exported at `rakka_accel::cuda`) `DeviceActor`,
/// `ContextActor`, `GpuRef`, `GpuDispatcher`, `PerActorAllocator`,
/// `PlacementActor`, the kernel actors, etc., without taking a
/// separate dependency. Doc §4 ("Foundational Mapping" —
/// `WorkerActor` ≡ `DeviceActor`).
#[cfg(any())] // atomr-accel-gated; disabled until atomr-accel renames
pub use rakka_accel as accel;

/// Re-export of `rakka-accel-patterns` so callers can compose §9
/// pipelines (`DynamicBatchingServer`, `InferenceCascade`,
/// `ModelReplicaPool`, `FairShareScheduler`, `ModelHotSwapServer`,
/// `SpeculativeDecoder`, `MoeRouter`) without a second dep.
#[cfg(any())] // atomr-accel-gated; disabled until atomr-accel renames
pub use rakka_accel_patterns as accel_patterns;

// Back-compat aliases for the v0.1 names. Will be removed in v0.4.
#[cfg(any())] // atomr-accel-gated; disabled until atomr-accel renames
#[doc(hidden)]
pub use rakka_accel as cuda;
#[cfg(any())] // atomr-accel-gated; disabled until atomr-accel renames
#[doc(hidden)]
pub use rakka_accel_patterns as cuda_patterns;

/// Re-export the most commonly used types so callers can `use
/// atomr_infer::prelude::*;` and have everything they need to declare
/// `Deployment`s and write actors.
pub mod prelude {
    pub use atomr_infer_core::{
        Deployment, ExecuteBatch, InferenceError, InferenceResult, ModelRunner, ProviderKind, RateLimits,
        RetryPolicy, RuntimeConfig, RuntimeKind, SecretString, Serving, Timeouts, TokenChunk, Tokens,
        TransportKind,
    };
}
