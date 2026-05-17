//! `AssemblyAiSttConfig` — parameters for the AssemblyAI STT runtime.
//!
//! Holds the bearer credential reference, the WSS base URL, and the
//! per-session knobs (format_turns). Mirrors the shape of the sibling
//! provider configs so an operator can keep their `deployment.toml`s
//! symmetrical.

#[cfg(feature = "stt-assemblyai")]
use atomr_infer_core::deployment::{RateLimits, RetryPolicy, Timeouts};
#[cfg(feature = "stt-assemblyai")]
use atomr_infer_core::runtime::CircuitBreakerConfig;
#[cfg(feature = "stt-assemblyai")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "stt-assemblyai")]
use std::time::Duration;
#[cfg(feature = "stt-assemblyai")]
use url::Url;

/// Where the AssemblyAI API key comes from. Mirrors `DeepgramSecret`
/// and `ElevenLabsSecret` so a shared secret-store implementation can
/// resolve all three.
#[cfg(feature = "stt-assemblyai")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum AssemblyAiSecret {
    /// Read the API key from the named environment variable at deploy
    /// time. Deployment-side resolution; the runner only ever sees the
    /// already-resolved `SecretString`.
    Env { name: String },
    /// Read the API key from a file (deploy-time resolution).
    File { path: String },
}

/// Per-deployment AssemblyAI parameters.
///
/// The runner builds the WSS URL by joining `ws_endpoint` with `v3/ws`
/// plus query parameters derived from
/// [`atomr_infer_core::audio::TranscribeOptions`] and the audio
/// `params` carried on the inbound frame stream.
#[cfg(feature = "stt-assemblyai")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyAiSttConfig {
    /// `wss://streaming.assemblyai.com/` — the WSS base. The runner
    /// joins `v3/ws` onto this to form the streaming-listen endpoint.
    pub ws_endpoint: Url,
    /// Bearer credential reference.
    pub api_key: AssemblyAiSecret,
    /// WS connect timeout. Default 5 s.
    #[serde(default = "default_ws_connect_timeout")]
    #[serde(with = "duration_ms")]
    pub ws_connect_timeout: Duration,
    /// `format_turns=true` — request the provider's Punctuated &
    /// Formatted text on `Turn` envelopes. Off by default because it
    /// adds latency.
    #[serde(default)]
    pub format_turns: bool,
    #[serde(default)]
    pub rate_limits: RateLimits,
    #[serde(default)]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub timeouts: Timeouts,
}

#[cfg(feature = "stt-assemblyai")]
fn default_ws_connect_timeout() -> Duration {
    Duration::from_secs(5)
}

#[cfg(feature = "stt-assemblyai")]
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

#[cfg(feature = "stt-assemblyai")]
impl AssemblyAiSttConfig {
    /// Default config pointing at the public AssemblyAI streaming API.
    pub fn defaults_for_assemblyai(api_key: AssemblyAiSecret) -> Self {
        let ws_endpoint = Url::parse("wss://streaming.assemblyai.com/").expect("static url");
        Self {
            ws_endpoint,
            api_key,
            ws_connect_timeout: default_ws_connect_timeout(),
            format_turns: false,
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

    /// Build the concrete WSS URL for the streaming endpoint
    /// (`<base>/v3/ws`).
    pub fn listen_url(&self) -> Result<Url, url::ParseError> {
        self.ws_endpoint.join("v3/ws")
    }
}

// Stub when feature off — keeps `pub use` in lib.rs honest.
#[cfg(not(feature = "stt-assemblyai"))]
#[derive(Debug, Clone, Default)]
pub struct AssemblyAiSttConfig;

#[cfg(not(feature = "stt-assemblyai"))]
#[derive(Debug, Clone, Default)]
pub struct AssemblyAiSecret;

#[cfg(all(test, feature = "stt-assemblyai"))]
mod tests {
    use super::*;

    #[test]
    fn defaults_point_at_public_assemblyai() {
        let cfg = AssemblyAiSttConfig::defaults_for_assemblyai(AssemblyAiSecret::Env {
            name: "ASSEMBLYAI_API_KEY".into(),
        });
        assert_eq!(cfg.ws_endpoint.as_str(), "wss://streaming.assemblyai.com/");
        assert_eq!(
            cfg.listen_url().unwrap().as_str(),
            "wss://streaming.assemblyai.com/v3/ws",
        );
    }

    #[test]
    fn with_ws_endpoint_overrides() {
        let cfg = AssemblyAiSttConfig::defaults_for_assemblyai(AssemblyAiSecret::Env { name: "K".into() })
            .with_ws_endpoint(Url::parse("ws://127.0.0.1:1235/").unwrap());
        assert_eq!(cfg.listen_url().unwrap().as_str(), "ws://127.0.0.1:1235/v3/ws",);
    }

    #[test]
    fn serde_round_trip_minimal() {
        let cfg = AssemblyAiSttConfig::defaults_for_assemblyai(AssemblyAiSecret::Env { name: "K".into() });
        let json = serde_json::to_string(&cfg).unwrap();
        let back: AssemblyAiSttConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ws_endpoint, cfg.ws_endpoint);
        assert_eq!(back.format_turns, cfg.format_turns);
    }
}
