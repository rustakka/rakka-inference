//! Runtime / transport / provider taxonomy and per-runtime configuration.
//!
//! Doc references: §3.1 (backend taxonomy), §5.4 (`TransportKind` /
//! `ProviderKind` enums), §10.5 (feature flags).

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Identifies the runtime *backend* that hosts a model.
///
/// Maps 1:1 to the per-runtime crates listed in §10.1. `Custom(String)`
/// is the escape hatch third-party runtimes use until they're added to
/// the canonical enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeKind {
    Vllm,
    TensorRt,
    Ort,
    Candle,
    Cudarc,
    MistralRs,
    /// Locally-hosted Python runtime without a Rust binding (e.g. XTTS,
    /// Bark, diffusers). Doc §2.6.
    Python(String),
    OpenAi,
    Anthropic,
    Gemini,
    LiteLlm,
    Custom(String),
}

impl RuntimeKind {
    pub fn is_remote(&self) -> bool {
        matches!(
            self,
            RuntimeKind::OpenAi | RuntimeKind::Anthropic | RuntimeKind::Gemini | RuntimeKind::LiteLlm
        )
    }

    pub fn is_local(&self) -> bool {
        !self.is_remote()
    }
}

/// Where the runtime executes — local GPU vs remote network. Read by
/// `PlacementActor` and the worker-spawning logic to decide what kind of
/// `WorkerActor` to spin up. Doc §5.4.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TransportKind {
    LocalGpu,
    RemoteNetwork { provider: ProviderKind },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenAi,
    Anthropic,
    Gemini,
    LiteLlm,
    Custom(String),
}

impl From<&RuntimeKind> for TransportKind {
    fn from(kind: &RuntimeKind) -> Self {
        match kind {
            RuntimeKind::OpenAi => Self::RemoteNetwork {
                provider: ProviderKind::OpenAi,
            },
            RuntimeKind::Anthropic => Self::RemoteNetwork {
                provider: ProviderKind::Anthropic,
            },
            RuntimeKind::Gemini => Self::RemoteNetwork {
                provider: ProviderKind::Gemini,
            },
            RuntimeKind::LiteLlm => Self::RemoteNetwork {
                provider: ProviderKind::LiteLlm,
            },
            _ => Self::LocalGpu,
        }
    }
}

/// Per-deployment runtime configuration. The `runtime` discriminator
/// drives both the backend selection and the shape of the inner config
/// blob. Per-runtime crates each contribute one variant or expose their
/// own `RuntimeConfig`-shaped struct that can be wrapped in `Custom`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "runtime", rename_all = "snake_case")]
pub enum RuntimeConfig {
    /// vLLM (local Python). Body intentionally opaque here — the real
    /// shape lives in `inference-runtime-vllm` and is parsed lazily.
    Vllm(serde_json::Value),
    TensorRt(serde_json::Value),
    Ort(serde_json::Value),
    Candle(serde_json::Value),
    Cudarc(serde_json::Value),
    MistralRs(serde_json::Value),
    /// Remote OpenAI / Azure OpenAI. Concrete shape in
    /// `inference-runtime-openai::OpenAiConfig`.
    OpenAi(serde_json::Value),
    Anthropic(serde_json::Value),
    Gemini(serde_json::Value),
    LiteLlm(serde_json::Value),
    /// Custom backend (third-party runtime crate).
    Custom {
        kind: String,
        config: serde_json::Value,
    },
}

impl RuntimeConfig {
    pub fn runtime_kind(&self) -> RuntimeKind {
        match self {
            RuntimeConfig::Vllm(_) => RuntimeKind::Vllm,
            RuntimeConfig::TensorRt(_) => RuntimeKind::TensorRt,
            RuntimeConfig::Ort(_) => RuntimeKind::Ort,
            RuntimeConfig::Candle(_) => RuntimeKind::Candle,
            RuntimeConfig::Cudarc(_) => RuntimeKind::Cudarc,
            RuntimeConfig::MistralRs(_) => RuntimeKind::MistralRs,
            RuntimeConfig::OpenAi(_) => RuntimeKind::OpenAi,
            RuntimeConfig::Anthropic(_) => RuntimeKind::Anthropic,
            RuntimeConfig::Gemini(_) => RuntimeKind::Gemini,
            RuntimeConfig::LiteLlm(_) => RuntimeKind::LiteLlm,
            RuntimeConfig::Custom { kind, .. } => RuntimeKind::Custom(kind.clone()),
        }
    }

    pub fn transport_kind(&self) -> TransportKind {
        TransportKind::from(&self.runtime_kind())
    }
}

/// Circuit-breaker config (doc §3.5, §12.2). One per `(provider,
/// endpoint)`; opens after sustained failures, half-opens after the
/// configured duration to permit a probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    #[serde(with = "humantime_serde_ms")]
    pub open_duration: Duration,
    pub half_open_max_probes: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 10,
            open_duration: Duration::from_secs(30),
            half_open_max_probes: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JitterKind {
    None,
    Equal,
    Full,
}

/// `Duration` (de)serialization in milliseconds — chosen so the doc's
/// TOML examples (`open_duration_ms = 30_000`) round-trip naturally.
pub(crate) mod humantime_serde_ms {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(d: &Duration, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (d.as_millis() as u64).serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let ms = u64::deserialize(d)?;
        Ok(Duration::from_millis(ms))
    }
}
