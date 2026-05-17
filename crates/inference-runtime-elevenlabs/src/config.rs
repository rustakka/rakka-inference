//! `ElevenLabsTtsConfig` — parameters for the ElevenLabs TTS runtime.
//!
//! Holds the bearer credential reference, the HTTPS + WSS base URLs,
//! and the streaming knobs (chunk boundary, connect / read timeouts).
//! Mirrors the shape of the sibling OpenAI TTS config so an operator
//! can keep their `deployment.toml`s symmetrical.

#[cfg(feature = "tts-elevenlabs")]
use atomr_infer_core::deployment::{RateLimits, RetryPolicy, Timeouts};
#[cfg(feature = "tts-elevenlabs")]
use atomr_infer_core::runtime::CircuitBreakerConfig;
#[cfg(feature = "tts-elevenlabs")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "tts-elevenlabs")]
use std::time::Duration;
#[cfg(feature = "tts-elevenlabs")]
use url::Url;

/// Where the ElevenLabs API key comes from. Mirrors
/// `atomr_infer_runtime_openai::config::SecretRef` so a shared
/// secret-store implementation can resolve both.
#[cfg(feature = "tts-elevenlabs")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ElevenLabsSecret {
    /// Read the API key from the named environment variable at deploy
    /// time. Deployment-side resolution; the runner only ever sees the
    /// already-resolved `SecretString`.
    Env { name: String },
    /// Read the API key from a file (deploy-time resolution).
    File { path: String },
}

#[cfg(feature = "tts-elevenlabs")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElevenLabsTtsConfig {
    /// `https://api.elevenlabs.io/v1/` — the HTTPS base.
    pub endpoint: Url,
    /// `wss://api.elevenlabs.io/v1/` — the WSS base. Defaults to the
    /// HTTPS base with the scheme swapped to `wss`.
    pub ws_endpoint: Url,
    /// Bearer credential reference.
    pub api_key: ElevenLabsSecret,
    /// Default voice id used when `SpeechBatch::voice` is empty. Not
    /// required; the runner returns `BadRequest` if both are missing.
    #[serde(default)]
    pub default_voice_id: Option<String>,
    /// HTTPS streaming chunk boundary — the response body is split
    /// here before being emitted to the caller. Default 8192 (≈170 ms
    /// at 24 kHz mono PCM16).
    #[serde(default = "default_chunk_bytes")]
    pub chunk_bytes: usize,
    /// WS connect timeout. Default 5 s.
    #[serde(default = "default_ws_connect_timeout")]
    #[serde(with = "duration_ms")]
    pub ws_connect_timeout: Duration,
    #[serde(default)]
    pub rate_limits: RateLimits,
    #[serde(default)]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub timeouts: Timeouts,
}

#[cfg(feature = "tts-elevenlabs")]
fn default_chunk_bytes() -> usize {
    8_192
}

#[cfg(feature = "tts-elevenlabs")]
fn default_ws_connect_timeout() -> Duration {
    Duration::from_secs(5)
}

#[cfg(feature = "tts-elevenlabs")]
mod duration_ms {
    use serde::{de::Error, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_millis() as u64)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let ms = u64::deserialize(d).map_err(D::Error::custom)?;
        Ok(Duration::from_millis(ms))
    }
}

#[cfg(feature = "tts-elevenlabs")]
impl ElevenLabsTtsConfig {
    /// Default config pointing at the public ElevenLabs API.
    pub fn defaults_for_elevenlabs(api_key: ElevenLabsSecret) -> Self {
        let endpoint = Url::parse("https://api.elevenlabs.io/v1/").expect("static url");
        let ws_endpoint = Url::parse("wss://api.elevenlabs.io/v1/").expect("static url");
        Self {
            endpoint,
            ws_endpoint,
            api_key,
            default_voice_id: None,
            chunk_bytes: default_chunk_bytes(),
            ws_connect_timeout: default_ws_connect_timeout(),
            rate_limits: RateLimits::default(),
            retry: RetryPolicy::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            timeouts: Timeouts::default(),
        }
    }

    /// Override the HTTPS base. Useful for wiremock-driven tests and
    /// for routing through a corporate proxy.
    pub fn with_endpoint(mut self, endpoint: Url) -> Self {
        self.endpoint = endpoint;
        self
    }

    /// Override the WSS base. The defaults derive from the HTTPS base,
    /// but tests using `ws://` against an in-process mock server need
    /// to override this directly.
    pub fn with_ws_endpoint(mut self, ws_endpoint: Url) -> Self {
        self.ws_endpoint = ws_endpoint;
        self
    }

    /// Build the concrete HTTPS URL for the one-shot TTS endpoint.
    pub fn speech_url(&self, voice_id: &str) -> Result<Url, url::ParseError> {
        self.endpoint.join(&format!("text-to-speech/{}", voice_id))
    }

    /// Build the concrete WSS URL for the streaming endpoint.
    pub fn speech_stream_url(&self, voice_id: &str) -> Result<Url, url::ParseError> {
        self.ws_endpoint
            .join(&format!("text-to-speech/{}/stream-input", voice_id))
    }

    /// Build the concrete URL for voice-cloning multipart upload.
    pub fn add_voice_url(&self) -> Result<Url, url::ParseError> {
        self.endpoint.join("voices/add")
    }
}

// Stub when feature off — keeps `pub use` in lib.rs honest.
#[cfg(not(feature = "tts-elevenlabs"))]
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsTtsConfig;

#[cfg(not(feature = "tts-elevenlabs"))]
#[derive(Debug, Clone, Default)]
pub struct ElevenLabsSecret;

#[cfg(all(test, feature = "tts-elevenlabs"))]
mod tests {
    use super::*;

    #[test]
    fn defaults_point_at_public_elevenlabs() {
        let cfg = ElevenLabsTtsConfig::defaults_for_elevenlabs(ElevenLabsSecret::Env {
            name: "ELEVEN_API_KEY".into(),
        });
        assert_eq!(cfg.endpoint.as_str(), "https://api.elevenlabs.io/v1/");
        assert_eq!(cfg.ws_endpoint.as_str(), "wss://api.elevenlabs.io/v1/");
        assert_eq!(
            cfg.speech_url("21m00Tcm4TlvDq8ikWAM").unwrap().as_str(),
            "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM",
        );
        assert_eq!(
            cfg.speech_stream_url("vid").unwrap().as_str(),
            "wss://api.elevenlabs.io/v1/text-to-speech/vid/stream-input",
        );
        assert_eq!(
            cfg.add_voice_url().unwrap().as_str(),
            "https://api.elevenlabs.io/v1/voices/add",
        );
    }

    #[test]
    fn with_endpoint_overrides() {
        let cfg = ElevenLabsTtsConfig::defaults_for_elevenlabs(ElevenLabsSecret::Env { name: "K".into() })
            .with_endpoint(Url::parse("http://127.0.0.1:1234/v1/").unwrap())
            .with_ws_endpoint(Url::parse("ws://127.0.0.1:1235/v1/").unwrap());
        assert_eq!(
            cfg.speech_url("v").unwrap().as_str(),
            "http://127.0.0.1:1234/v1/text-to-speech/v",
        );
        assert_eq!(
            cfg.speech_stream_url("v").unwrap().as_str(),
            "ws://127.0.0.1:1235/v1/text-to-speech/v/stream-input",
        );
    }

    #[test]
    fn serde_round_trip_minimal() {
        let cfg = ElevenLabsTtsConfig::defaults_for_elevenlabs(ElevenLabsSecret::Env { name: "K".into() });
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ElevenLabsTtsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.endpoint, cfg.endpoint);
        assert_eq!(back.ws_endpoint, cfg.ws_endpoint);
        assert_eq!(back.chunk_bytes, cfg.chunk_bytes);
    }
}
