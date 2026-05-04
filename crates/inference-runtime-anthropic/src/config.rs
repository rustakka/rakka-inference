use serde::{Deserialize, Serialize};
use url::Url;

use atomr_infer_core::deployment::{RateLimits, RetryPolicy, Timeouts};
use atomr_infer_core::runtime::CircuitBreakerConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// Defaults to `https://api.anthropic.com/v1/`.
    #[serde(default = "default_endpoint")]
    pub endpoint: Url,
    pub api_key: SecretRef,
    /// API version pinned via `anthropic-version` header. Defaults to
    /// `2023-06-01`, the current stable.
    #[serde(default = "default_version")]
    pub anthropic_version: String,
    #[serde(default)]
    pub rate_limits: RateLimits,
    #[serde(default)]
    pub retry: RetryPolicy,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    #[serde(default)]
    pub timeouts: Timeouts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum SecretRef {
    Env { name: String },
    File { path: std::path::PathBuf },
    Inline { value: String },
}

fn default_endpoint() -> Url {
    Url::parse("https://api.anthropic.com/v1/").expect("static url")
}
fn default_version() -> String {
    "2023-06-01".to_string()
}

impl AnthropicConfig {
    pub fn defaults(api_key: SecretRef) -> Self {
        Self {
            endpoint: default_endpoint(),
            api_key,
            anthropic_version: default_version(),
            rate_limits: RateLimits::default(),
            retry: RetryPolicy::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            timeouts: Timeouts::default(),
        }
    }

    pub fn messages_url(&self) -> Result<Url, url::ParseError> {
        self.endpoint.join("messages")
    }
}
