//! `OpenAiConfig` — parameters for the OpenAI / Azure OpenAI runtime.

use serde::{Deserialize, Serialize};
use url::Url;

use inference_core::deployment::{RateLimits, RetryPolicy, Timeouts};
use inference_core::runtime::CircuitBreakerConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiConfig {
    #[serde(flatten)]
    pub variant: OpenAiVariant,
    /// Bearer token. Stored opaquely on the wire; surfaced to the
    /// runner via `inference-remote-core::session::CredentialProvider`.
    pub api_key: SecretRef,
    #[serde(default)]
    pub organization: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OpenAiVariant {
    /// `https://api.openai.com/v1` (or any OpenAI-compatible endpoint).
    Direct { endpoint: Url },
    /// Azure OpenAI: builds the endpoint URL from resource +
    /// deployment + api_version per the Azure docs.
    Azure {
        resource: String,
        deployment: String,
        api_version: String,
    },
}

impl OpenAiVariant {
    pub fn chat_completions_url(&self) -> Result<Url, url::ParseError> {
        match self {
            OpenAiVariant::Direct { endpoint } => endpoint.join("chat/completions"),
            OpenAiVariant::Azure { resource, deployment, api_version } => Url::parse(&format!(
                "https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            )),
        }
    }
}

/// Indirection that prevents inline secrets in serialised config. The
/// real value is resolved by `inference-cli` at deploy time and handed
/// to the runner as a `secrecy::SecretString`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum SecretRef {
    Env {
        name: String,
    },
    File {
        path: std::path::PathBuf,
    },
    /// Inline literal — discouraged; emits a warning on load.
    Inline {
        value: String,
    },
}

impl OpenAiConfig {
    pub fn defaults_for_openai(api_key: SecretRef) -> Self {
        Self {
            variant: OpenAiVariant::Direct {
                endpoint: Url::parse("https://api.openai.com/v1/").expect("static url"),
            },
            api_key,
            organization: None,
            project: None,
            rate_limits: RateLimits::default(),
            retry: RetryPolicy::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            timeouts: Timeouts::default(),
        }
    }

    pub fn with_endpoint(mut self, endpoint: Url) -> Self {
        if let OpenAiVariant::Direct { endpoint: ref mut e } = self.variant {
            *e = endpoint;
        }
        self
    }
}
