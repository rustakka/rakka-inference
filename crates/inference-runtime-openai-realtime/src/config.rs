//! `OpenAiRealtimeConfig` — connection parameters for the OpenAI Realtime API.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Configuration for the OpenAI Realtime runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiRealtimeConfig {
    /// Bearer token source.
    pub api_key: ApiKeySource,

    /// Override the default endpoint
    /// (`wss://api.openai.com/v1/realtime`).
    #[serde(default = "default_endpoint")]
    pub endpoint: String,

    /// WebSocket handshake timeout.
    #[serde(default = "default_handshake_timeout_ms", rename = "handshake_timeout_ms")]
    pub handshake_timeout_ms: u64,
}

fn default_endpoint() -> String {
    "wss://api.openai.com/v1/realtime".to_string()
}

fn default_handshake_timeout_ms() -> u64 {
    10_000
}

impl OpenAiRealtimeConfig {
    /// Construct config that reads the API key from an environment variable
    /// at runtime.
    pub fn new_with_env_key(env_var: impl Into<String>) -> Self {
        Self {
            api_key: ApiKeySource::Env { name: env_var.into() },
            endpoint: default_endpoint(),
            handshake_timeout_ms: default_handshake_timeout_ms(),
        }
    }

    /// Resolve the API key string from whichever source was configured.
    ///
    /// # Errors
    ///
    /// Returns an error string if the env var is missing or the inline value
    /// is empty.
    pub fn resolve_api_key(&self) -> Result<String, String> {
        match &self.api_key {
            ApiKeySource::Env { name } => {
                std::env::var(name).map_err(|_| format!("env var `{name}` not set"))
            }
            ApiKeySource::Inline { value } => {
                if value.is_empty() {
                    Err("inline api_key is empty".into())
                } else {
                    Ok(value.clone())
                }
            }
        }
    }

    /// Handshake timeout as a `Duration`.
    pub fn handshake_timeout(&self) -> Duration {
        Duration::from_millis(self.handshake_timeout_ms)
    }

    /// The full WSS URL for the given model.
    ///
    /// Appends `?model=<model>` to the configured endpoint.  If the
    /// endpoint already has a trailing slash, the slash is preserved.
    /// If the endpoint has no path component, `/` is prepended so the
    /// resulting HTTP request line is valid (e.g. `GET /?model=… HTTP/1.1`).
    pub fn ws_url(&self, model: &str) -> String {
        // Ensure there is at least a `/` path so the query string is
        // attached to a valid request-target.
        let base = if self.endpoint.contains('/')
            && self
                .endpoint
                .split_once("://")
                .map(|(_, rest)| rest.contains('/'))
                .unwrap_or(false)
        {
            // Has an explicit path after the authority
            self.endpoint.clone()
        } else {
            // No path → append `/` so the URL becomes `scheme://host/?model=…`
            format!("{}/", self.endpoint)
        };
        format!("{base}?model={model}")
    }
}

/// API-key source — mirrors the `SecretRef` pattern in the OpenAI runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum ApiKeySource {
    /// Read from an environment variable at runtime.
    Env { name: String },
    /// Inline literal (discouraged — emits a `warn!` on first use).
    Inline { value: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_url_includes_model() {
        let cfg = OpenAiRealtimeConfig::new_with_env_key("OPENAI_API_KEY");
        let url = cfg.ws_url("gpt-4o-realtime-preview");
        assert_eq!(
            url,
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        );
    }

    #[test]
    fn inline_key_resolve() {
        let cfg = OpenAiRealtimeConfig {
            api_key: ApiKeySource::Inline {
                value: "sk-test".into(),
            },
            endpoint: default_endpoint(),
            handshake_timeout_ms: 5_000,
        };
        assert_eq!(cfg.resolve_api_key().unwrap(), "sk-test");
    }

    #[test]
    fn empty_inline_key_errors() {
        let cfg = OpenAiRealtimeConfig {
            api_key: ApiKeySource::Inline { value: String::new() },
            endpoint: default_endpoint(),
            handshake_timeout_ms: 5_000,
        };
        assert!(cfg.resolve_api_key().is_err());
    }

    #[test]
    fn default_serde_round_trip() {
        let cfg = OpenAiRealtimeConfig::new_with_env_key("OPENAI_API_KEY");
        let s = serde_json::to_string(&cfg).unwrap();
        let back: OpenAiRealtimeConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back.endpoint, cfg.endpoint);
    }

    #[test]
    fn handshake_timeout_converts() {
        let cfg = OpenAiRealtimeConfig::new_with_env_key("K");
        assert_eq!(cfg.handshake_timeout(), Duration::from_millis(10_000));
    }
}
