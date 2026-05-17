//! `OpenAiSttConfig` — parameters for the OpenAI STT runtime.

#[cfg(feature = "stt-openai")]
use atomr_infer_core::deployment::{RateLimits, RetryPolicy, Timeouts};
#[cfg(feature = "stt-openai")]
use atomr_infer_core::runtime::CircuitBreakerConfig;
#[cfg(feature = "stt-openai")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "stt-openai")]
use url::Url;

#[cfg(feature = "stt-openai")]
use atomr_infer_runtime_openai::config::SecretRef;

/// Live config for [`crate::OpenAiSttRunner`]. Mirrors the auth +
/// retry surface of the sibling chat-completions runtime so a single
/// `OpenAI` deployment block can host both modalities.
#[cfg(feature = "stt-openai")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiSttConfig {
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
    #[serde(default)]
    pub rate_limits: RateLimits,
    #[serde(default)]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub timeouts: Timeouts,
}

#[cfg(feature = "stt-openai")]
impl OpenAiSttConfig {
    /// Default config pointing at `https://api.openai.com/v1/`.
    pub fn defaults_for_openai(api_key: SecretRef) -> Self {
        Self {
            endpoint: Url::parse("https://api.openai.com/v1/").expect("static url"),
            api_key,
            organization: None,
            project: None,
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

    /// Build the concrete URL for the transcriptions endpoint.
    pub fn transcriptions_url(&self) -> Result<Url, url::ParseError> {
        self.endpoint.join("audio/transcriptions")
    }
}

// Stub when feature off — keeps `pub use` in lib.rs honest even on
// remote-only builds.
#[cfg(not(feature = "stt-openai"))]
#[derive(Debug, Clone, Default)]
pub struct OpenAiSttConfig;

#[cfg(test)]
mod tests {
    #[cfg(feature = "stt-openai")]
    use super::*;

    #[cfg(feature = "stt-openai")]
    #[test]
    fn defaults_point_at_openai_v1() {
        let cfg = OpenAiSttConfig::defaults_for_openai(SecretRef::Env {
            name: "OPENAI_API_KEY".into(),
        });
        let url = cfg.transcriptions_url().unwrap();
        assert_eq!(url.as_str(), "https://api.openai.com/v1/audio/transcriptions");
    }

    #[cfg(feature = "stt-openai")]
    #[test]
    fn with_endpoint_overrides_base() {
        let cfg = OpenAiSttConfig::defaults_for_openai(SecretRef::Env {
            name: "OPENAI_API_KEY".into(),
        })
        .with_endpoint(Url::parse("http://127.0.0.1:1234/v1/").unwrap());
        assert_eq!(
            cfg.transcriptions_url().unwrap().as_str(),
            "http://127.0.0.1:1234/v1/audio/transcriptions"
        );
    }

    #[cfg(feature = "stt-openai")]
    #[test]
    fn serde_round_trip_minimal() {
        let cfg = OpenAiSttConfig::defaults_for_openai(SecretRef::Env {
            name: "OPENAI_API_KEY".into(),
        });
        let json = serde_json::to_string(&cfg).unwrap();
        let back: OpenAiSttConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.endpoint, cfg.endpoint);
    }
}
