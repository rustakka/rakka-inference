//! `GeminiLiveConfig` — parameters for the Gemini Live realtime runtime.
//!
//! Holds the API key reference, the WSS base URL, and per-session
//! knobs. Auth differs from OpenAI Realtime: instead of an
//! `Authorization: Bearer` header, Gemini Live embeds the API key as a
//! `?key=` query parameter on the WebSocket upgrade URL.

#[cfg(feature = "tts-gemini-live")]
use atomr_infer_core::deployment::{RateLimits, RetryPolicy, Timeouts};
#[cfg(feature = "tts-gemini-live")]
use atomr_infer_core::runtime::CircuitBreakerConfig;
#[cfg(feature = "tts-gemini-live")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "tts-gemini-live")]
use std::time::Duration;
#[cfg(feature = "tts-gemini-live")]
use url::Url;

/// Where the Gemini API key comes from. Mirrors `DeepgramSecret`
/// and `OpenAiSecretRef` so a shared secret-store implementation can
/// resolve all three.
#[cfg(feature = "tts-gemini-live")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum GeminiLiveApiKey {
    /// Read the API key from the named environment variable at deploy
    /// time.
    Env { name: String },
    /// Read the API key from a file (deploy-time resolution).
    File { path: String },
}

/// Per-deployment Gemini Live parameters.
///
/// The runner constructs the full WebSocket URL by appending the Gemini
/// Live BidiGenerateContent path and `?key=<api_key>` to `ws_endpoint`.
///
/// # Example URL
///
/// ```text
/// wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=<key>
/// ```
#[cfg(feature = "tts-gemini-live")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiLiveConfig {
    /// WSS base URL: `wss://generativelanguage.googleapis.com/`.
    pub ws_endpoint: Url,
    /// API key reference.
    pub api_key: GeminiLiveApiKey,
    /// WS connect timeout. Default 10 s (Gemini Live setup is slower than
    /// Deepgram because it involves a gRPC-over-WS upgrade).
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

#[cfg(feature = "tts-gemini-live")]
fn default_ws_connect_timeout() -> Duration {
    Duration::from_secs(10)
}

#[cfg(feature = "tts-gemini-live")]
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

#[cfg(feature = "tts-gemini-live")]
impl GeminiLiveConfig {
    /// Default config pointing at the public Gemini API.
    pub fn defaults_for_gemini_live(api_key: GeminiLiveApiKey) -> Self {
        let ws_endpoint = Url::parse("wss://generativelanguage.googleapis.com/").expect("static url");
        Self {
            ws_endpoint,
            api_key,
            ws_connect_timeout: default_ws_connect_timeout(),
            rate_limits: RateLimits::default(),
            retry: RetryPolicy::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            timeouts: Timeouts::default(),
        }
    }

    /// Override the WSS base URL. Tests using `ws://` against an
    /// in-process mock server need to override this directly.
    pub fn with_ws_endpoint(mut self, ws_endpoint: Url) -> Self {
        self.ws_endpoint = ws_endpoint;
        self
    }

    /// Build the full Gemini Live WebSocket URL including the path and
    /// the `?key=` query string.
    pub fn live_url(&self, api_key: &str) -> Result<Url, url::ParseError> {
        let path = "ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
        let mut url = self.ws_endpoint.join(path)?;
        url.query_pairs_mut().append_pair("key", api_key);
        Ok(url)
    }
}

// Stubs when feature is off — keeps `pub use` in lib.rs honest.
#[cfg(not(feature = "tts-gemini-live"))]
#[derive(Debug, Clone, Default)]
pub struct GeminiLiveConfig;

#[cfg(not(feature = "tts-gemini-live"))]
#[derive(Debug, Clone, Default)]
pub struct GeminiLiveApiKey;

#[cfg(all(test, feature = "tts-gemini-live"))]
mod tests {
    use super::*;

    #[test]
    fn defaults_point_at_public_gemini() {
        let cfg = GeminiLiveConfig::defaults_for_gemini_live(GeminiLiveApiKey::Env {
            name: "GEMINI_API_KEY".into(),
        });
        assert_eq!(
            cfg.ws_endpoint.as_str(),
            "wss://generativelanguage.googleapis.com/"
        );
    }

    #[test]
    fn live_url_includes_path_and_key() {
        let cfg = GeminiLiveConfig::defaults_for_gemini_live(GeminiLiveApiKey::Env { name: "K".into() });
        let url = cfg.live_url("my-key").unwrap();
        assert!(url.as_str().contains("BidiGenerateContent"), "url: {}", url);
        assert!(url.as_str().contains("key=my-key"), "url: {}", url);
    }

    #[test]
    fn with_ws_endpoint_overrides() {
        let cfg = GeminiLiveConfig::defaults_for_gemini_live(GeminiLiveApiKey::Env { name: "K".into() })
            .with_ws_endpoint(Url::parse("ws://127.0.0.1:1234/").unwrap());
        let url = cfg.live_url("fake-key").unwrap();
        assert!(url.as_str().starts_with("ws://127.0.0.1:1234/"), "{url}");
        assert!(url.as_str().contains("key=fake-key"), "{url}");
    }

    #[test]
    fn serde_round_trip_minimal() {
        let cfg = GeminiLiveConfig::defaults_for_gemini_live(GeminiLiveApiKey::Env { name: "K".into() });
        let json = serde_json::to_string(&cfg).unwrap();
        let back: GeminiLiveConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ws_endpoint, cfg.ws_endpoint);
        assert_eq!(
            back.ws_connect_timeout.as_millis(),
            cfg.ws_connect_timeout.as_millis()
        );
    }
}
