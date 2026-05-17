//! # inference-runtime-ort
//!
//! ONNX Runtime backend via the [`ort`](https://crates.io/crates/ort)
//! crate. Doc §2.3, §10.3.
//!
//! ## Two entry points
//!
//! - `ModelRunner::execute` — chat-style, takes an `ExecuteBatch`
//!   of messages, runs a tokenizer + autoregressive sampling loop on
//!   ONNX-exported causal LMs (HuggingFace Optimum-ONNX layout).
//!   Streams `TokenChunk`s like every other runtime.
//! - `OrtRunner::infer` — low-level, takes raw tensors, returns f32
//!   outputs. For embeddings (BGE / E5), rerankers, Whisper encoders,
//!   vision classifiers, and anything else that's "ONNX graph + maybe
//!   a tokenizer + tensors in/out".
//!
//! ## Build profiles
//!
//! | Build                                                                  | Result                                                |
//! |------------------------------------------------------------------------|-------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-ort`                                | Stub — `ort` not in dep graph.                        |
//! | `cargo build -p atomr-infer-runtime-ort --features ort`                 | Real path with `ort` crate (CPU EP).                  |
//! | `cargo build -p atomr-infer-runtime-ort --features ort,ort-cuda`        | Adds the CUDA EP — needs a working CUDA toolkit.      |
//! | `cargo build -p atomr-infer-runtime-ort --features ort,ort-load-dynamic`| Loads `libonnxruntime` at runtime via `ORT_DYLIB_PATH`. |
//! | `cargo build -p atomr-infer-runtime-ort --features ort,ort-hf-hub`      | Resolves `tokenizer.json` from HuggingFace if local fails. |
//!
//! ## MSRV note
//!
//! The `ort` 2.0 release line requires Rust 1.85+. The atomr-infer
//! workspace MSRV is 1.78 for `remote-only` builds; operators
//! enabling this runner need a toolchain that satisfies `ort`'s own
//! MSRV.
//!
//! ## CUDA execution provider
//!
//! When `execution_provider == Cuda` and the `ort-cuda` feature is
//! on, the runner constructs a `CUDAExecutionProvider` with the
//! configured device id and falls back to CPU for unsupported ops.
//! Sharing a `cudarc::driver::CudaStream` with `atomr-accel-cuda`'s
//! `PerActorAllocator` is a follow-up — see the comment in
//! `session.rs::providers_for` for the seam.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;

#[cfg(feature = "ort")]
mod error;
#[cfg(feature = "ort")]
mod generate;
#[cfg(feature = "ort")]
mod infer;
mod runner;
#[cfg(feature = "ort")]
mod sampling;
#[cfg(feature = "ort")]
mod session;
#[cfg(feature = "ort")]
mod tokenizer;
#[cfg(feature = "ort")]
mod topology;

pub use config::{ExecutionProvider, OrtConfig};
pub use runner::OrtRunner;

#[cfg(feature = "ort")]
pub use infer::{InferOutputs, InferTensor};

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::ModelRunner;
    use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let runner = OrtRunner::new(OrtConfig {
            onnx_path: "/tmp/does-not-exist.onnx".into(),
            execution_provider: ExecutionProvider::default(),
            device_id: 0,
            tokenizer_path: None,
            hf_repo: None,
            intra_threads: None,
            default_max_new_tokens: 256,
        });
        assert_eq!(runner.runtime_kind(), RuntimeKind::Ort);
        assert_eq!(runner.transport_kind(), TransportKind::LocalGpu);
    }

    #[test]
    fn config_round_trips_through_serde() {
        let cfg = OrtConfig {
            onnx_path: "/etc/models/bge.onnx".into(),
            execution_provider: ExecutionProvider::Cuda,
            device_id: 1,
            tokenizer_path: Some("/etc/models/bge.tok.json".into()),
            hf_repo: Some("BAAI/bge-large-en-v1.5".into()),
            intra_threads: Some(4),
            default_max_new_tokens: 128,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: OrtConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.onnx_path, cfg.onnx_path);
        assert_eq!(back.execution_provider, cfg.execution_provider);
        assert_eq!(back.device_id, cfg.device_id);
        assert_eq!(back.tokenizer_path, cfg.tokenizer_path);
        assert_eq!(back.hf_repo, cfg.hf_repo);
        assert_eq!(back.intra_threads, cfg.intra_threads);
        assert_eq!(back.default_max_new_tokens, cfg.default_max_new_tokens);
    }

    #[cfg(not(feature = "ort"))]
    #[tokio::test]
    async fn execute_without_feature_returns_internal_error() {
        use atomr_infer_core::batch::SamplingParams;
        use atomr_infer_core::error::InferenceError;

        let mut runner = OrtRunner::new(OrtConfig {
            onnx_path: "/tmp/does-not-exist.onnx".into(),
            execution_provider: ExecutionProvider::default(),
            device_id: 0,
            tokenizer_path: None,
            hf_repo: None,
            intra_threads: None,
            default_max_new_tokens: 256,
        });
        let batch = atomr_infer_core::batch::ExecuteBatch {
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
