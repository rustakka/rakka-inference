//! `OpenAiTtsConfig` — parameters for the OpenAI TTS runtime.
//!
//! Mirrors the auth surface of `atomr_infer_runtime_openai::OpenAiConfig`
//! so an operator can hand the same credential store to both Chat
//! Completions and TTS deployments. The endpoint URL is held verbatim
//! and combined with the static path `audio/speech` at request time.

#[cfg(feature = "tts-openai")]
use atomr_infer_core::deployment::{RateLimits, RetryPolicy, Timeouts};
#[cfg(feature = "tts-openai")]
use atomr_infer_core::runtime::CircuitBreakerConfig;
#[cfg(feature = "tts-openai")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "tts-openai")]
use url::Url;

#[cfg(feature = "tts-openai")]
use atomr_infer_runtime_openai::config::SecretRef;

/// Live config for [`crate::OpenAiTtsRunner`]. Mirrors the auth +
/// retry surface of the sibling chat-completions runtime so a single
/// `OpenAI` deployment block can host both.
#[cfg(feature = "tts-openai")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiTtsConfig {
    /// `https://api.openai.com/v1/` or an OpenAI-compatible base URL.
    pub endpoint: Url,
    /// Bearer token reference. Resolved at deploy time by
    /// `inference-cli` and handed to the runner as a
    /// `secrecy::SecretString`.
    pub api_key: SecretRef,
    /// Optional `OpenAI-Organization` header.
    #[serde(default)]
    pub organization: Option<String>,
    /// Optional `OpenAI-Project` header.
    #[serde(default)]
    pub project: Option<String>,
    /// PCM streaming chunk size in bytes — the response body is split
    /// at this boundary before being emitted to the caller. Default
    /// 8192 (≈170 ms at 24 kHz mono PCM16).
    #[serde(default = "default_chunk_bytes")]
    pub chunk_bytes: usize,
    #[serde(default)]
    pub rate_limits: RateLimits,
    #[serde(default)]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub timeouts: Timeouts,
}

#[cfg(feature = "tts-openai")]
fn default_chunk_bytes() -> usize {
    8_192
}

#[cfg(feature = "tts-openai")]
impl OpenAiTtsConfig {
    /// Default config pointing at `https://api.openai.com/v1/`.
    pub fn defaults_for_openai(api_key: SecretRef) -> Self {
        Self {
            endpoint: Url::parse("https://api.openai.com/v1/").expect("static url"),
            api_key,
            organization: None,
            project: None,
            chunk_bytes: default_chunk_bytes(),
            rate_limits: RateLimits::default(),
            retry: RetryPolicy::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            timeouts: Timeouts::default(),
        }
    }

    /// Override the base endpoint (useful for `MockOpenAi` and Azure-
    /// compatible bridges).
    pub fn with_endpoint(mut self, endpoint: Url) -> Self {
        self.endpoint = endpoint;
        self
    }

    /// Build the concrete URL for the speech endpoint.
    pub fn speech_url(&self) -> Result<Url, url::ParseError> {
        self.endpoint.join("audio/speech")
    }
}

// Stub when feature off — keeps `pub use` in lib.rs honest even on
// remote-only builds.
#[cfg(not(feature = "tts-openai"))]
#[derive(Debug, Clone, Default)]
pub struct OpenAiTtsConfig;

#[cfg(test)]
mod tests {
    #[cfg(feature = "tts-openai")]
    use super::*;

    #[cfg(feature = "tts-openai")]
    #[test]
    fn defaults_point_at_openai_v1() {
        let cfg = OpenAiTtsConfig::defaults_for_openai(SecretRef::Env {
            name: "OPENAI_API_KEY".into(),
        });
        let url = cfg.speech_url().unwrap();
        assert_eq!(url.as_str(), "https://api.openai.com/v1/audio/speech");
    }

    #[cfg(feature = "tts-openai")]
    #[test]
    fn with_endpoint_overrides_base() {
        let cfg = OpenAiTtsConfig::defaults_for_openai(SecretRef::Env {
            name: "OPENAI_API_KEY".into(),
        })
        .with_endpoint(Url::parse("http://127.0.0.1:1234/v1/").unwrap());
        assert_eq!(
            cfg.speech_url().unwrap().as_str(),
            "http://127.0.0.1:1234/v1/audio/speech"
        );
    }

    #[cfg(feature = "tts-openai")]
    #[test]
    fn serde_round_trip_minimal() {
        let cfg = OpenAiTtsConfig::defaults_for_openai(SecretRef::Env {
            name: "OPENAI_API_KEY".into(),
        });
        let json = serde_json::to_string(&cfg).unwrap();
        let back: OpenAiTtsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.endpoint, cfg.endpoint);
        assert_eq!(back.chunk_bytes, cfg.chunk_bytes);
    }
}
