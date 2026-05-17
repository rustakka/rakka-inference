//! # atomr-infer-runtime-audio2face
//!
//! NVIDIA Omniverse Audio2Face-3D runtime for `atomr-infer`.
//!
//! Implements [`atomr_infer_core::A2FRunner`] ingesting an
//! [`atomr_infer_core::AudioBatch`] (16 kHz mono PCM16LE is the common
//! case) and emitting a stream of [`atomr_infer_core::BlendshapeChunk`]s
//! with 52 ARKit-canonical blendshape weights each.
//!
//! ## Build profiles
//!
//! | Feature flag    | Behaviour |
//! |-----------------|-----------|
//! | *(off)*         | All APIs compile; `execute_audio2face` returns `InferenceError::Internal("feature disabled")` |
//! | `audio2face`    | Real path active (see arch gate below) |
//!
//! ## Architecture gate
//!
//! The `audio2face` feature requires **Linux x86_64**. On other platforms
//! (macOS, Windows, aarch64) the real path returns
//! `InferenceError::Internal("audio2face requires Linux x86_64")` at
//! runtime. This design lets the crate appear in the dependency graph on
//! all platforms — downstream code compiles everywhere — while the
//! actually-unsupported path fails gracefully at request time.
//!
//! ## ARKit canonical ordering
//!
//! `BlendshapeChunk::weights` carries 52 weights in the order mandated by
//! Apple's `ARBlendShapeLocation`. See [`arkit::ARKIT_BLENDSHAPE_NAMES`]
//! for the ordered list and
//! <https://developer.apple.com/documentation/arkit/arblendshapelocation>
//! for the upstream reference.
//!
//! The A2F→ARKit index permutation is currently a **passthrough**;
//! see [`arkit::a2f_to_arkit`] and the `TODO(a2f-normalization)` marker
//! therein.
//!
//! ## Transport status (FR-A2F-001 M11)
//!
//! The gRPC transport path (`tonic::Channel` → `PushAudioStream`) is
//! scaffolded with `TODO(grpc-transport)` markers in `runner.rs`. For
//! this milestone the runner uses an **in-memory blendshape generator**
//! that produces deterministic 30 fps output driven by the input audio's
//! frame count. This validates the full trait-implementation and
//! ARKit-ordering story without requiring a live NVIDIA server.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod arkit;
pub mod config;
pub mod cost;
pub mod error;
pub mod wire;

mod runner;

pub use config::Audio2FaceConfig;
pub use error::Audio2FaceError;
pub use runner::Audio2FaceRunner;
