//! # inference-runtime-vllm
//!
//! vLLM (Python) runtime — canonical local-LLM backend. Doc §2.1, §10.3.
//!
//! ## Feature flags
//!
//! - `vllm` — pull in PyO3 + the `AsyncLLMEngine` bridge. Without
//!   this feature the runner compiles to a typed-error stub so a
//!   `cargo build --features remote-only` consumer never pulls
//!   pyo3 / vllm / cudarc.
//! - `gemma-default` — adds the env probe + HuggingFace cache
//!   resolver + optional `hf-hub` pre-download path so an operator
//!   can auto-provision a Gemma 4 deployment when the host has a
//!   workable GPU + Python + vLLM + HF token. See
//!   `inference::defaults::gemma` for the rollup-side adapter.
//!
//! ## Lifecycle
//!
//! `VllmRunner::new` is cheap and synchronous — it stores the config.
//! The Python `AsyncLLMEngine` is built lazily on the first
//! [`ModelRunner::execute`] call, so a runner can be instantiated
//! on hosts without a GPU (handy for config-layer tests).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::InferenceResult;
#[cfg(any(not(feature = "vllm"), all(test, not(feature = "vllm"))))]
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[cfg(feature = "vllm")]
mod engine;

#[cfg(feature = "gemma-default")]
pub mod defaults;
#[cfg(feature = "gemma-default")]
pub mod hf_cache;
#[cfg(feature = "gemma-default")]
pub mod probe;

/// vLLM engine configuration. Pass-through for the Python builder
/// arguments (`AsyncEngineArgs`); the perf knobs at the bottom map
/// 1:1 to vLLM's own settings of the same name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VllmConfig {
    /// HuggingFace repo id or local path the engine loads.
    pub model: String,
    #[serde(default = "default_tp")]
    pub tensor_parallel_size: u32,
    /// Numeric dtype: `"auto"`, `"float16"`, `"bfloat16"`, `"float32"`.
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// Fraction of GPU memory the engine pre-allocates. Defaults to
    /// vLLM's `0.9`; the Gemma auto-provisioner overrides to `0.5`
    /// to leave room for dev tools.
    #[serde(default)]
    pub gpu_memory_utilization: Option<f32>,
    /// Maximum sequence length. `None` ⇒ vLLM picks from the model
    /// config.
    #[serde(default)]
    pub max_model_len: Option<u32>,
    /// Optional HuggingFace cache directory. When set, the engine is
    /// constructed with `HF_HOME` pointing here so multi-instance
    /// deployments share a single on-disk cache.
    #[serde(default)]
    pub hf_cache_dir: Option<std::path::PathBuf>,
    /// Disable CUDA graphs (vLLM `enforce_eager`). Default `None`
    /// ⇒ vLLM picks (graphs on for most models). Enabling this is
    /// the easiest perf experiment — CUDA graphs typically give
    /// 1.5–2× throughput on small models.
    #[serde(default)]
    pub enforce_eager: Option<bool>,
    /// Cache common prompt prefixes across requests. Useful for chat
    /// with shared system prompts. Default `None` ⇒ vLLM default
    /// (off in v0.6, on in v0.7+).
    #[serde(default)]
    pub enable_prefix_caching: Option<bool>,
    /// Chunked prefill: split long prompts so prefill interleaves
    /// with decode. Improves TTFT under concurrent load.
    #[serde(default)]
    pub enable_chunked_prefill: Option<bool>,
    /// Maximum concurrent sequences the scheduler runs. Higher ⇒
    /// better steady-state throughput at cost of per-request latency.
    /// Default `None` ⇒ vLLM picks (256 in v0.6).
    #[serde(default)]
    pub max_num_seqs: Option<u32>,
    /// PagedAttention block size in tokens. Default `None` ⇒ vLLM
    /// picks (16). Larger blocks ⇒ better throughput, smaller ⇒
    /// finer-grained memory packing.
    #[serde(default)]
    pub block_size: Option<u32>,
    /// Quantization scheme: `"awq"`, `"gptq"`, `"squeezellm"`,
    /// `"fp8"`, etc. `None` ⇒ unquantized (whatever the checkpoint
    /// natively is).
    #[serde(default)]
    pub quantization: Option<String>,
    /// Per-prompt multimodal-input cap, e.g. `{"image": 0,
    /// "audio": 0}`. For multimodal models like Gemma 4, setting
    /// these to 0 tells vLLM the workload is text-only and lets it
    /// skip the worst-case vision/audio buffer allocation during
    /// KV-cache profiling — often the difference between fitting
    /// in 16 GB and OOMing. Note: vLLM 0.20's Gemma 4 text-only
    /// path is buggy (per-layer-embeddings share mm plumbing); use
    /// `cpu_offload_gb` instead on small GPUs.
    #[serde(default)]
    pub limit_mm_per_prompt: Option<std::collections::BTreeMap<String, u32>>,
    /// Offload N GB of model weights to CPU RAM (vLLM
    /// `cpu_offload_gb`). On a 16 GB GPU running Gemma 4 E4B,
    /// `Some(4)` is enough to fit the multimodal profile pass that
    /// otherwise OOMs at ~15.5 GB. Trade-off: per-token decode
    /// slows ~30–50 % because each forward pass copies offloaded
    /// weights GPU↔CPU.
    #[serde(default)]
    pub cpu_offload_gb: Option<u32>,
}

fn default_tp() -> u32 {
    1
}
fn default_dtype() -> String {
    "auto".to_string()
}

/// vLLM runner. Constructs in O(1); the engine boots lazily on the
/// first call to [`ModelRunner::execute`].
pub struct VllmRunner {
    #[cfg_attr(not(feature = "vllm"), allow(dead_code))]
    config: VllmConfig,
    #[cfg(feature = "vllm")]
    engine: tokio::sync::OnceCell<std::sync::Arc<engine::VllmEngine>>,
}

impl VllmRunner {
    pub fn new(config: VllmConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "vllm")]
            engine: tokio::sync::OnceCell::new(),
        }
    }

    #[cfg(feature = "vllm")]
    async fn ensure_engine(&self) -> InferenceResult<std::sync::Arc<engine::VllmEngine>> {
        self.engine
            .get_or_try_init(|| async {
                engine::VllmEngine::launch(&self.config)
                    .await
                    .map(std::sync::Arc::new)
            })
            .await
            .cloned()
    }
}

#[async_trait]
impl ModelRunner for VllmRunner {
    #[cfg_attr(
        feature = "vllm",
        tracing::instrument(skip(self, batch), fields(request_id = %batch.request_id, model = %batch.model))
    )]
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "vllm"))]
        {
            let _ = batch;
            Err(InferenceError::Internal(
                "vllm feature disabled at build time — rebuild with --features vllm".into(),
            ))
        }
        #[cfg(feature = "vllm")]
        {
            let engine = self.ensure_engine().await?;
            engine.generate(batch).await
        }
    }

    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()> {
        #[cfg(feature = "vllm")]
        {
            // CudaContextPoisoned / Manual ⇒ tear down the cached
            // engine handle. The next `execute` reconstructs it; vLLM
            // V1 doesn't always release VRAM cleanly, so a hard
            // rebuild may need a process restart in practice.
            if matches!(
                cause,
                SessionRebuildCause::CudaContextPoisoned | SessionRebuildCause::Manual
            ) {
                self.engine = tokio::sync::OnceCell::new();
            }
        }
        let _ = cause;
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Vllm
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
    fn gil_pinned(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(model: &str) -> VllmConfig {
        VllmConfig {
            model: model.into(),
            tensor_parallel_size: 1,
            dtype: "auto".into(),
            gpu_memory_utilization: Some(0.5),
            max_model_len: Some(8192),
            hf_cache_dir: None,
            enforce_eager: None,
            enable_prefix_caching: None,
            enable_chunked_prefill: None,
            max_num_seqs: None,
            block_size: None,
            quantization: None,
            limit_mm_per_prompt: None,
            cpu_offload_gb: None,
        }
    }

    #[test]
    fn config_round_trips_through_serde() {
        let cfg = test_config("google/gemma-4-E4B-it");
        let s = serde_json::to_string(&cfg).expect("serialize");
        let back: VllmConfig = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back.model, cfg.model);
        assert_eq!(back.gpu_memory_utilization, cfg.gpu_memory_utilization);
    }

    #[test]
    fn runner_reports_runtime_kind() {
        let r = VllmRunner::new(test_config("test"));
        assert_eq!(r.runtime_kind(), RuntimeKind::Vllm);
        assert_eq!(r.transport_kind(), TransportKind::LocalGpu);
        assert!(r.gil_pinned());
    }

    #[cfg(not(feature = "vllm"))]
    #[tokio::test]
    async fn execute_without_feature_returns_internal_error() {
        use atomr_infer_core::batch::SamplingParams;

        let mut r = VllmRunner::new(test_config("test"));
        let batch = ExecuteBatch {
            request_id: "t".into(),
            model: "t".into(),
            messages: vec![],
            sampling: SamplingParams::default(),
            stream: false,
            estimated_tokens: 1,
        };
        let result = r.execute(batch).await;
        assert!(matches!(result, Err(InferenceError::Internal(_))));
    }
}
