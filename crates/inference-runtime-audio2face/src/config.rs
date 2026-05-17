//! `Audio2FaceConfig` — connection and tuning parameters for the
//! NVIDIA Omniverse Audio2Face-3D gRPC runtime.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Configuration for the Audio2Face-3D gRPC runner.
///
/// # Examples
///
/// ```
/// use atomr_infer_runtime_audio2face::config::Audio2FaceConfig;
///
/// let cfg = Audio2FaceConfig::defaults_for_a2f()
///     .with_endpoint("localhost:50051".into());
/// assert_eq!(cfg.endpoint, "localhost:50051");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Audio2FaceConfig {
    /// gRPC endpoint in `host:port` form (no scheme).
    ///
    /// The runner prefixes `http://` for plaintext or `https://` when
    /// `tls` is enabled.
    pub endpoint: String,

    /// Default emotion preset forwarded to the A2F engine when the
    /// [`AudioBatch`][atomr_infer_core::AudioBatch] does not specify one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_emotion: Option<String>,

    /// Maximum time to wait for the initial gRPC channel to become ready.
    #[serde(with = "duration_ms", default = "default_connect_timeout")]
    pub connect_timeout: Duration,

    /// Per-request deadline. Applies to the streaming RPC as a whole.
    #[serde(with = "duration_ms", default = "default_request_timeout")]
    pub request_timeout: Duration,

    /// Maximum number of consecutive gRPC errors before the runner
    /// surfaces [`Audio2FaceError::Unsupported`][crate::error::Audio2FaceError::Unsupported].
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Enable TLS for the gRPC channel.
    #[serde(default)]
    pub tls: bool,
}

fn default_connect_timeout() -> Duration {
    Duration::from_secs(5)
}
fn default_request_timeout() -> Duration {
    Duration::from_secs(60)
}
fn default_max_retries() -> u32 {
    3
}

impl Default for Audio2FaceConfig {
    fn default() -> Self {
        Self::defaults_for_a2f()
    }
}

impl Audio2FaceConfig {
    /// Sensible defaults for a locally-hosted Audio2Face-3D server.
    pub fn defaults_for_a2f() -> Self {
        Self {
            endpoint: "localhost:50051".into(),
            default_emotion: None,
            connect_timeout: default_connect_timeout(),
            request_timeout: default_request_timeout(),
            max_retries: default_max_retries(),
            tls: false,
        }
    }

    /// Override the gRPC endpoint.
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
        self
    }

    /// Override the default emotion preset.
    pub fn with_emotion(mut self, emotion: impl Into<String>) -> Self {
        self.default_emotion = Some(emotion.into());
        self
    }
}

mod duration_ms {
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
    fn defaults_for_a2f() {
        let cfg = Audio2FaceConfig::defaults_for_a2f();
        assert_eq!(cfg.endpoint, "localhost:50051");
        assert!(cfg.default_emotion.is_none());
        assert_eq!(cfg.connect_timeout, Duration::from_secs(5));
        assert_eq!(cfg.request_timeout, Duration::from_secs(60));
        assert_eq!(cfg.max_retries, 3);
        assert!(!cfg.tls);
    }

    #[test]
    fn builder_with_endpoint() {
        let cfg = Audio2FaceConfig::defaults_for_a2f().with_endpoint("10.0.0.1:50051".into());
        assert_eq!(cfg.endpoint, "10.0.0.1:50051");
    }

    #[test]
    fn builder_with_emotion() {
        let cfg = Audio2FaceConfig::defaults_for_a2f().with_emotion("happy");
        assert_eq!(cfg.default_emotion.as_deref(), Some("happy"));
    }

    #[test]
    fn serde_round_trip() {
        let cfg = Audio2FaceConfig::defaults_for_a2f().with_endpoint("remote:9090".into());
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: Audio2FaceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.endpoint, cfg.endpoint);
        assert_eq!(decoded.connect_timeout, cfg.connect_timeout);
    }
}
