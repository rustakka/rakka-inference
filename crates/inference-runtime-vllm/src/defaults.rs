//! Zero-config Gemma 4 auto-provisioning.
//!
//! Layered on top of the env probe + HF cache resolver: when the
//! `gemma-default` feature is enabled, an operator can call
//! [`provision_if_ready`] after spinning up the actor system and the
//! [`DeploymentManagerActor`] to register a `gemma-local` deployment
//! when the host has a workable GPU + Python + vLLM + HF token.
//!
//! Probe-failure path is intentionally lightweight: this returns a
//! [`ProvisionOutcome::Skipped`] with a human-readable reason and
//! hint so the operator's main loop can `info!` the message and
//! continue without the deployment.
//!
//! ## Environment variables
//!
//! | Variable                          | Effect                                                 |
//! |-----------------------------------|--------------------------------------------------------|
//! | `ATOMR_INFER_GEMMA_AUTO`          | `0` / `false` / `skip-quietly` ⇒ disable               |
//! | `ATOMR_INFER_GEMMA_MODEL`         | Override model id (allow-listed against [`SUPPORTED_VARIANTS`]) |
//! | `ATOMR_INFER_GEMMA_DEPLOYMENT`    | Override deployment name (default `gemma-local`)        |
//! | `ATOMR_INFER_GEMMA_GPU_UTIL`      | Float, defaults `0.5`                                  |
//! | `ATOMR_INFER_GEMMA_MAX_LEN`       | Optional max model context length                      |
//! | `HF_HOME` / `HF_HUB_CACHE`        | Standard HF caches (respected by the engine + probe)   |
//! | `HF_TOKEN`                        | HF auth                                                |

#![cfg(feature = "gemma-default")]

use std::path::PathBuf;

use atomr_core::actor::ActorRef;
use atomr_infer_core::deployment::Deployment;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runtime::RuntimeConfig;
use atomr_infer_runtime::{DeploymentManagerActor, DeploymentManagerMsg};

use crate::probe::{probe, ProbeResult};
use crate::VllmConfig;

/// Default `Deployment::name` for the auto-provisioned Gemma instance.
pub const DEFAULT_DEPLOYMENT_NAME: &str = "gemma-local";

/// Default HuggingFace repo id. The auto-provisioner pairs E4B-it
/// with `cpu_offload_gb=4` so it fits in 16 GB at the cost of a
/// ~30–50 % decode-throughput hit (forward pass copies offloaded
/// weights GPU↔CPU each step). Operators with ≥24 GB GPUs should
/// drop `cpu_offload_gb` for full GPU-side throughput; operators
/// constrained to even smaller GPUs override via
/// `ATOMR_INFER_GEMMA_MODEL=google/gemma-4-E2B-it`.
pub const DEFAULT_MODEL_ID: &str = "google/gemma-4-E4B-it";

/// Allow-list of supported Gemma 4 variants. Validated against
/// `model_id` so a typo fails fast with a clear error rather than a
/// multi-GB HF download into a 404.
pub const SUPPORTED_VARIANTS: &[&str] = &[
    "google/gemma-4-E2B",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B",
    "google/gemma-4-E4B-it",
];

/// Recommended VRAM floor in GB per variant (fp16). The probe uses
/// this to decide whether to surface an "insufficient VRAM, try the
/// smaller variant" hint.
pub fn min_vram_gb(model_id: &str) -> Option<f32> {
    match model_id {
        "google/gemma-4-E2B" | "google/gemma-4-E2B-it" => Some(2.5),
        "google/gemma-4-E4B" | "google/gemma-4-E4B-it" => Some(4.5),
        _ => None,
    }
}

/// Approximate on-disk size in GB after HF download (model weights +
/// tokenizer + config). Used to surface a disk-space hint.
pub fn min_disk_gb(model_id: &str) -> Option<f32> {
    match model_id {
        "google/gemma-4-E2B" | "google/gemma-4-E2B-it" => Some(4.0),
        "google/gemma-4-E4B" | "google/gemma-4-E4B-it" => Some(7.0),
        _ => None,
    }
}

/// For an E4B variant, point at the matching E2B variant; for E2B,
/// there's no smaller supported option.
pub fn fallback_variant(model_id: &str) -> Option<&'static str> {
    match model_id {
        "google/gemma-4-E4B-it" => Some("google/gemma-4-E2B-it"),
        "google/gemma-4-E4B" => Some("google/gemma-4-E2B"),
        _ => None,
    }
}

/// Validate `model_id` against [`SUPPORTED_VARIANTS`].
pub fn validate_variant(model_id: &str) -> InferenceResult<()> {
    if SUPPORTED_VARIANTS.contains(&model_id) {
        Ok(())
    } else {
        Err(InferenceError::BadRequest {
            message: format!(
                "unsupported Gemma variant `{model_id}` — supported: {}",
                SUPPORTED_VARIANTS.join(", ")
            ),
        })
    }
}

/// Operator-facing configuration. Built either from env vars
/// ([`Self::from_env`]) or directly by the operator.
#[derive(Debug, Clone)]
pub struct GemmaDefaults {
    pub model_id: String,
    pub deployment_name: String,
    pub cache_dir: Option<PathBuf>,
    pub gpu_memory_utilization: f32,
    pub max_model_len: Option<u32>,
    pub auto_provision: bool,
}

impl Default for GemmaDefaults {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.into(),
            deployment_name: DEFAULT_DEPLOYMENT_NAME.into(),
            cache_dir: None,
            gpu_memory_utilization: 0.5,
            max_model_len: None,
            auto_provision: true,
        }
    }
}

impl GemmaDefaults {
    /// Read the standard `ATOMR_INFER_GEMMA_*` env vars and apply on
    /// top of [`Default::default`].
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Some(v) = env_string("ATOMR_INFER_GEMMA_AUTO") {
            cfg.auto_provision = !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off" | "skip" | "skip-quietly"
            );
        }
        if let Some(v) = env_string("ATOMR_INFER_GEMMA_MODEL") {
            cfg.model_id = v;
        }
        if let Some(v) = env_string("ATOMR_INFER_GEMMA_DEPLOYMENT") {
            cfg.deployment_name = v;
        }
        if let Some(v) = env_string("ATOMR_INFER_GEMMA_GPU_UTIL") {
            if let Ok(f) = v.parse::<f32>() {
                cfg.gpu_memory_utilization = f;
            }
        }
        if let Some(v) = env_string("ATOMR_INFER_GEMMA_MAX_LEN") {
            if let Ok(u) = v.parse::<u32>() {
                cfg.max_model_len = Some(u);
            }
        }
        cfg
    }
}

/// Outcome of [`provision_if_ready`].
#[derive(Debug)]
pub enum ProvisionOutcome {
    /// The deployment was successfully registered with the manager.
    Ready { deployment_name: String },
    /// Probe found a missing prereq the user can fix. The
    /// auto-provisioner did not register a deployment; nothing else
    /// changed.
    Skipped { reason: String, hint: String },
}

/// Probe the environment and, if all checks pass, register a
/// [`Deployment`] for the configured Gemma variant with the
/// [`DeploymentManagerActor`].
///
/// Errors from this function are limited to "the probe itself
/// crashed" or "the deployment manager rejected the apply" —
/// missing-prereq cases come back as
/// [`ProvisionOutcome::Skipped`] with a human-readable message.
pub async fn provision_if_ready(
    manager: &ActorRef<DeploymentManagerMsg>,
    cfg: &GemmaDefaults,
) -> InferenceResult<ProvisionOutcome> {
    validate_variant(&cfg.model_id)?;

    let min_vram = min_vram_gb(&cfg.model_id).unwrap_or(4.5);
    let min_disk = min_disk_gb(&cfg.model_id).unwrap_or(7.0);
    let fallback = fallback_variant(&cfg.model_id);

    match probe(&cfg.model_id, min_vram, min_disk, fallback) {
        ProbeResult::Skipped { reason, hint } => return Ok(ProvisionOutcome::Skipped { reason, hint }),
        ProbeResult::Error(e) => return Err(e),
        ProbeResult::Ready {
            vram_free_gb,
            hf_cache,
        } => {
            tracing::info!(
                model = %cfg.model_id,
                deployment = %cfg.deployment_name,
                vram_free_gb,
                hf_cache = %hf_cache.hub_cache.display(),
                "probe ok — provisioning Gemma deployment"
            );
        }
    }

    // Build the VllmConfig that the runner will consume. We
    // deliberately leave `limit_mm_per_prompt` unset because vLLM
    // 0.20's Gemma 4 text-only fast-path is broken (the
    // per-layer-embeddings input shares plumbing with the
    // multimodal encoder and ends up `None`). Letting the full
    // multimodal pipeline run is fine for E2B-it on 16 GB cards;
    // E4B-it operators want a ≥24 GB GPU.
    let vllm_cfg = VllmConfig {
        model: cfg.model_id.clone(),
        tensor_parallel_size: 1,
        dtype: "auto".into(),
        gpu_memory_utilization: Some(cfg.gpu_memory_utilization),
        max_model_len: cfg.max_model_len,
        hf_cache_dir: cfg.cache_dir.clone(),
        // Eager mode by default: CUDA-graph capture across many
        // batch sizes adds ~5 GB on Gemma 4, which forces a
        // smaller card into OOM. The bench harness toggles graphs
        // on for the perf comparison.
        enforce_eager: Some(true),
        enable_prefix_caching: None,
        enable_chunked_prefill: None,
        max_num_seqs: Some(16),
        block_size: None,
        quantization: None,
        limit_mm_per_prompt: None,
        // Pair with E4B-it's 10 GiB weights to keep peak GPU usage
        // under 12 GiB on 16 GB cards. Operators with ≥24 GB GPUs
        // should override this to None for full GPU-side throughput.
        cpu_offload_gb: Some(4),
    };

    let runtime_config = serde_json::to_value(&vllm_cfg)
        .map(RuntimeConfig::Vllm)
        .map_err(|e| InferenceError::Internal(format!("gemma defaults: serialise VllmConfig: {e}")))?;

    let deployment = Deployment {
        name: cfg.deployment_name.clone(),
        model: cfg.model_id.clone(),
        runtime: Some(atomr_infer_core::runtime::RuntimeKind::Vllm),
        runtime_config: Some(runtime_config),
        gpus: Some(1),
        replicas: 1,
        serving: Default::default(),
        budget: None,
        idempotent: true,
    };

    let (tx, rx) = tokio::sync::oneshot::channel();
    manager.tell(DeploymentManagerMsg::Apply {
        deployment,
        reply: tx,
    });
    match rx.await {
        Ok(Ok(())) => Ok(ProvisionOutcome::Ready {
            deployment_name: cfg.deployment_name.clone(),
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(InferenceError::Internal(
            "gemma defaults: deployment manager dropped reply channel".into(),
        )),
    }
}

fn env_string(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// `DeploymentManagerActor` is referenced for documentation purposes
// — the public function takes an `ActorRef<DeploymentManagerMsg>`,
// which is the message-handling shape callers care about.
#[allow(dead_code)]
type _ManagerType = DeploymentManagerActor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_pick_e4b_it() {
        let d = GemmaDefaults::default();
        assert_eq!(d.model_id, "google/gemma-4-E4B-it");
        assert_eq!(d.deployment_name, "gemma-local");
        assert_eq!(d.gpu_memory_utilization, 0.5);
        assert!(d.auto_provision);
    }

    #[test]
    fn validate_variant_accepts_all_four() {
        for v in SUPPORTED_VARIANTS {
            assert!(validate_variant(v).is_ok(), "{v} should be supported");
        }
    }

    #[test]
    fn validate_variant_rejects_unknown() {
        assert!(matches!(
            validate_variant("google/some-other-model"),
            Err(InferenceError::BadRequest { .. })
        ));
    }

    #[test]
    fn fallback_e4b_to_e2b() {
        assert_eq!(
            fallback_variant("google/gemma-4-E4B-it"),
            Some("google/gemma-4-E2B-it")
        );
        assert_eq!(fallback_variant("google/gemma-4-E4B"), Some("google/gemma-4-E2B"));
        // E2B has no smaller supported variant.
        assert_eq!(fallback_variant("google/gemma-4-E2B-it"), None);
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn from_env_respects_skip_quietly() {
        let _g = env_lock();
        std::env::set_var("ATOMR_INFER_GEMMA_AUTO", "skip-quietly");
        let d = GemmaDefaults::from_env();
        assert!(!d.auto_provision);
        std::env::remove_var("ATOMR_INFER_GEMMA_AUTO");
    }

    #[test]
    fn from_env_overrides_model_id() {
        let _g = env_lock();
        std::env::set_var("ATOMR_INFER_GEMMA_MODEL", "google/gemma-4-E2B-it");
        let d = GemmaDefaults::from_env();
        assert_eq!(d.model_id, "google/gemma-4-E2B-it");
        std::env::remove_var("ATOMR_INFER_GEMMA_MODEL");
    }

    #[test]
    fn vram_floors_match_table() {
        assert_eq!(min_vram_gb("google/gemma-4-E2B-it"), Some(2.5));
        assert_eq!(min_vram_gb("google/gemma-4-E4B-it"), Some(4.5));
        assert_eq!(min_vram_gb("unknown"), None);
    }
}
