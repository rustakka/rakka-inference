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
#[non_exhaustive]
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
    /// Speech-to-text deployments. Concrete provider is determined by
    /// the deployment's [`RuntimeConfig::SpeechToText`] body (OpenAI,
    /// whisper.cpp, Deepgram, AssemblyAI). Source: `FR-STT-001`.
    SpeechToText,
    /// Text-to-speech deployments. Concrete provider is determined by
    /// the deployment's [`RuntimeConfig::TextToSpeech`] body (OpenAI,
    /// ElevenLabs, Piper, Kokoro, XTTS, MOSS). Source: `FR-TTS-001`.
    TextToSpeech,
    /// Bidirectional realtime speech deployments (OpenAI Realtime,
    /// Gemini Live). Source: `FR-TTS-001` realtime section.
    RealtimeSpeech,
    /// NVIDIA Audio2Face-3D deployments — audio → ARKit blendshapes.
    /// Source: `FR-A2F-001`.
    Audio2Face,
    Custom(String),
}

impl RuntimeKind {
    /// True iff this runtime *always* talks to a remote provider.
    ///
    /// The audio-modality variants ([`RuntimeKind::SpeechToText`],
    /// [`RuntimeKind::TextToSpeech`], [`RuntimeKind::RealtimeSpeech`])
    /// are intentionally **not** classified here — they're polymorphic
    /// over multiple providers (local whisper.cpp vs remote Deepgram,
    /// local Piper vs remote ElevenLabs), so transport classification
    /// happens at the runtime config layer via
    /// [`crate::runner::ModelRunner::transport_kind`] /
    /// [`crate::runner::AudioRunner::transport_kind`] /
    /// [`crate::runner::SpeechRunner::transport_kind`] /
    /// [`crate::runner::RealtimeRunner::transport_kind`] /
    /// [`crate::runner::A2FRunner::transport_kind`], which can consult
    /// the config blob.
    pub fn is_remote(&self) -> bool {
        matches!(
            self,
            RuntimeKind::OpenAi | RuntimeKind::Anthropic | RuntimeKind::Gemini | RuntimeKind::LiteLlm
        )
    }

    pub fn is_local(&self) -> bool {
        !self.is_remote()
    }

    /// True for the audio-modality variants regardless of underlying
    /// provider — STT, TTS, realtime speech, A2F. Useful for routing
    /// requests to the correct sibling engine actor.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            RuntimeKind::SpeechToText
                | RuntimeKind::TextToSpeech
                | RuntimeKind::RealtimeSpeech
                | RuntimeKind::Audio2Face
        )
    }
}

/// Where the runtime executes — local GPU vs remote network. Read by
/// `PlacementActor` and the worker-spawning logic to decide what kind of
/// `WorkerActor` to spin up. Doc §5.4.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TransportKind {
    /// Local execution on the host's GPU(s).
    LocalGpu,
    /// Local CPU-only execution — whisper.cpp, Piper voices, etc.
    /// Placement does not allocate a GPU ordinal for these.
    LocalCpu,
    /// Outbound network call to a provider.
    RemoteNetwork { provider: ProviderKind },
    /// Default classification when the runtime cannot be inferred from
    /// [`RuntimeKind`] alone (audio modalities — concrete transport is
    /// determined by inspecting [`RuntimeConfig`]).
    UnknownTransport,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProviderKind {
    OpenAi,
    Anthropic,
    Gemini,
    LiteLlm,
    /// Deepgram STT (WSS). Source: `FR-STT-001`.
    Deepgram,
    /// AssemblyAI STT (WSS). Source: `FR-STT-001`.
    AssemblyAi,
    /// ElevenLabs TTS (HTTPS + WSS). Source: `FR-TTS-001`.
    ElevenLabs,
    /// NVIDIA Audio2Face-3D (gRPC). Source: `FR-A2F-001`.
    NvidiaA2F,
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
            // Local-GPU backends.
            RuntimeKind::Vllm
            | RuntimeKind::TensorRt
            | RuntimeKind::Ort
            | RuntimeKind::Candle
            | RuntimeKind::Cudarc
            | RuntimeKind::MistralRs => Self::LocalGpu,
            // Locally-hosted Python runtime; assume LocalGpu unless the
            // specific kind name implies otherwise — adapters override
            // via the runner's `transport_kind` method.
            RuntimeKind::Python(_) => Self::LocalGpu,
            // Audio modality variants are polymorphic over providers;
            // their transport is determined by the deployment's
            // `RuntimeConfig` body. Adapters override via the runner's
            // `transport_kind` method.
            RuntimeKind::SpeechToText
            | RuntimeKind::TextToSpeech
            | RuntimeKind::RealtimeSpeech
            | RuntimeKind::Audio2Face => Self::UnknownTransport,
            RuntimeKind::Custom(_) => Self::UnknownTransport,
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
    /// Speech-to-text — body discriminates by provider field
    /// (`{"provider": "openai" | "whisper" | "deepgram" | "assemblyai", ...}`).
    /// Concrete shapes live in the per-provider crates.
    SpeechToText(serde_json::Value),
    /// Text-to-speech — body discriminates by provider field
    /// (`{"provider": "openai" | "elevenlabs" | "piper" | "kokoro" | "xtts" | "moss", ...}`).
    TextToSpeech(serde_json::Value),
    /// Bidirectional realtime speech — body discriminates by provider
    /// field (`{"provider": "openai_realtime" | "gemini_live", ...}`).
    RealtimeSpeech(serde_json::Value),
    /// NVIDIA Audio2Face-3D gRPC config.
    Audio2Face(serde_json::Value),
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
            RuntimeConfig::SpeechToText(_) => RuntimeKind::SpeechToText,
            RuntimeConfig::TextToSpeech(_) => RuntimeKind::TextToSpeech,
            RuntimeConfig::RealtimeSpeech(_) => RuntimeKind::RealtimeSpeech,
            RuntimeConfig::Audio2Face(_) => RuntimeKind::Audio2Face,
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
#[non_exhaustive]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_kinds_classified_as_audio() {
        assert!(RuntimeKind::SpeechToText.is_audio());
        assert!(RuntimeKind::TextToSpeech.is_audio());
        assert!(RuntimeKind::RealtimeSpeech.is_audio());
        assert!(RuntimeKind::Audio2Face.is_audio());
        assert!(!RuntimeKind::OpenAi.is_audio());
        assert!(!RuntimeKind::Vllm.is_audio());
    }

    #[test]
    fn audio_kinds_not_classified_as_remote_or_local_polymorphic() {
        // Audio modalities are provider-polymorphic — they're neither
        // "always remote" nor "always local".
        assert!(!RuntimeKind::SpeechToText.is_remote());
        assert!(!RuntimeKind::TextToSpeech.is_remote());
        assert!(!RuntimeKind::RealtimeSpeech.is_remote());
        assert!(!RuntimeKind::Audio2Face.is_remote());
    }

    #[test]
    fn transport_kind_explicit_arms() {
        // Existing arms unchanged.
        assert_eq!(
            TransportKind::from(&RuntimeKind::OpenAi),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::OpenAi,
            }
        );
        assert_eq!(TransportKind::from(&RuntimeKind::Vllm), TransportKind::LocalGpu);
        // Audio modalities yield UnknownTransport — runner overrides.
        assert_eq!(
            TransportKind::from(&RuntimeKind::SpeechToText),
            TransportKind::UnknownTransport
        );
        assert_eq!(
            TransportKind::from(&RuntimeKind::TextToSpeech),
            TransportKind::UnknownTransport
        );
        assert_eq!(
            TransportKind::from(&RuntimeKind::RealtimeSpeech),
            TransportKind::UnknownTransport
        );
        assert_eq!(
            TransportKind::from(&RuntimeKind::Audio2Face),
            TransportKind::UnknownTransport
        );
        assert_eq!(
            TransportKind::from(&RuntimeKind::Custom("xyz".into())),
            TransportKind::UnknownTransport
        );
    }

    #[test]
    fn provider_kind_new_arms_serde() {
        for p in [
            ProviderKind::Deepgram,
            ProviderKind::AssemblyAi,
            ProviderKind::ElevenLabs,
            ProviderKind::NvidiaA2F,
        ] {
            let json = serde_json::to_string(&p).unwrap();
            let back: ProviderKind = serde_json::from_str(&json).unwrap();
            assert_eq!(p, back);
        }
    }

    #[test]
    fn runtime_config_runtime_kind_round_trip() {
        let configs = [
            (
                RuntimeConfig::SpeechToText(serde_json::json!({"provider":"openai"})),
                RuntimeKind::SpeechToText,
            ),
            (
                RuntimeConfig::TextToSpeech(serde_json::json!({"provider":"piper"})),
                RuntimeKind::TextToSpeech,
            ),
            (
                RuntimeConfig::RealtimeSpeech(serde_json::json!({"provider":"openai_realtime"})),
                RuntimeKind::RealtimeSpeech,
            ),
            (
                RuntimeConfig::Audio2Face(serde_json::json!({"endpoint":"localhost:50051"})),
                RuntimeKind::Audio2Face,
            ),
        ];
        for (cfg, expected_kind) in &configs {
            assert_eq!(cfg.runtime_kind(), *expected_kind);
        }
    }

    #[test]
    fn runtime_kind_serde_round_trip_audio() {
        for kind in [
            RuntimeKind::SpeechToText,
            RuntimeKind::TextToSpeech,
            RuntimeKind::RealtimeSpeech,
            RuntimeKind::Audio2Face,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: RuntimeKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, back);
        }
    }

    #[test]
    fn transport_kind_local_cpu_serde() {
        let t = TransportKind::LocalCpu;
        let json = serde_json::to_string(&t).unwrap();
        let back: TransportKind = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
    }
}
