//! `DeepgramSttConfig` — parameters for the Deepgram STT runtime.
//!
//! Holds the bearer credential reference, the WSS base URL, and the
//! per-session knobs (interim_results, diarize, punctuate, smart_format).
//! Mirrors the shape of the sibling provider configs so an operator
//! can keep their `deployment.toml`s symmetrical.

#[cfg(feature = "stt-deepgram")]
use atomr_infer_core::deployment::{RateLimits, RetryPolicy, Timeouts};
#[cfg(feature = "stt-deepgram")]
use atomr_infer_core::runtime::CircuitBreakerConfig;
#[cfg(feature = "stt-deepgram")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "stt-deepgram")]
use std::time::Duration;
#[cfg(feature = "stt-deepgram")]
use url::Url;

/// Where the Deepgram API key comes from. Mirrors `ElevenLabsSecret`
/// and `OpenAiSecretRef` so a shared secret-store implementation can
/// resolve all three.
#[cfg(feature = "stt-deepgram")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum DeepgramSecret {
    /// Read the API key from the named environment variable at deploy
    /// time. Deployment-side resolution; the runner only ever sees the
    /// already-resolved `SecretString`.
    Env { name: String },
    /// Read the API key from a file (deploy-time resolution).
    File { path: String },
}

/// Per-deployment Deepgram parameters.
///
/// The runner builds the WSS URL by joining `ws_endpoint` with
/// `listen` plus query parameters derived from
/// [`atomr_infer_core::audio::TranscribeOptions`] and the audio
/// `params` carried on the inbound frame stream.
#[cfg(feature = "stt-deepgram")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepgramSttConfig {
    /// `wss://api.deepgram.com/v1/` — the WSS base.
    pub ws_endpoint: Url,
    /// Bearer credential reference.
    pub api_key: DeepgramSecret,
    /// WS connect timeout. Default 5 s.
    #[serde(default = "default_ws_connect_timeout")]
    #[serde(with = "duration_ms")]
    pub ws_connect_timeout: Duration,
    /// Optional `smart_format=true` knob — enables Deepgram's
    /// punctuation + capitalisation post-processor. Off by default
    /// because it adds latency.
    #[serde(default)]
    pub smart_format: bool,
    #[serde(default)]
    pub rate_limits: RateLimits,
    #[serde(default)]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub timeouts: Timeouts,
}

#[cfg(feature = "stt-deepgram")]
fn default_ws_connect_timeout() -> Duration {
    Duration::from_secs(5)
}

#[cfg(feature = "stt-deepgram")]
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

#[cfg(feature = "stt-deepgram")]
impl DeepgramSttConfig {
    /// Default config pointing at the public Deepgram API.
    pub fn defaults_for_deepgram(api_key: DeepgramSecret) -> Self {
        let ws_endpoint = Url::parse("wss://api.deepgram.com/v1/").expect("static url");
        Self {
            ws_endpoint,
            api_key,
            ws_connect_timeout: default_ws_connect_timeout(),
            smart_format: false,
            rate_limits: RateLimits::default(),
            retry: RetryPolicy::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            timeouts: Timeouts::default(),
        }
    }

    /// Override the WSS base. Tests using `ws://` against an in-process
    /// mock server need to override this directly.
    pub fn with_ws_endpoint(mut self, ws_endpoint: Url) -> Self {
        self.ws_endpoint = ws_endpoint;
        self
    }

    /// Build the concrete WSS URL for the streaming-listen endpoint.
    pub fn listen_url(&self) -> Result<Url, url::ParseError> {
        self.ws_endpoint.join("listen")
    }
}

// Stub when feature off — keeps `pub use` in lib.rs honest.
#[cfg(not(feature = "stt-deepgram"))]
#[derive(Debug, Clone, Default)]
pub struct DeepgramSttConfig;

#[cfg(not(feature = "stt-deepgram"))]
#[derive(Debug, Clone, Default)]
pub struct DeepgramSecret;

#[cfg(all(test, feature = "stt-deepgram"))]
mod tests {
    use super::*;

    #[test]
    fn defaults_point_at_public_deepgram() {
        let cfg = DeepgramSttConfig::defaults_for_deepgram(DeepgramSecret::Env {
            name: "DEEPGRAM_API_KEY".into(),
        });
        assert_eq!(cfg.ws_endpoint.as_str(), "wss://api.deepgram.com/v1/");
        assert_eq!(
            cfg.listen_url().unwrap().as_str(),
            "wss://api.deepgram.com/v1/listen",
        );
    }

    #[test]
    fn with_ws_endpoint_overrides() {
        let cfg = DeepgramSttConfig::defaults_for_deepgram(DeepgramSecret::Env { name: "K".into() })
            .with_ws_endpoint(Url::parse("ws://127.0.0.1:1235/v1/").unwrap());
        assert_eq!(
            cfg.listen_url().unwrap().as_str(),
            "ws://127.0.0.1:1235/v1/listen",
        );
    }

    #[test]
    fn serde_round_trip_minimal() {
        let cfg = DeepgramSttConfig::defaults_for_deepgram(DeepgramSecret::Env { name: "K".into() });
        let json = serde_json::to_string(&cfg).unwrap();
        let back: DeepgramSttConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ws_endpoint, cfg.ws_endpoint);
        assert_eq!(back.smart_format, cfg.smart_format);
    }
}
