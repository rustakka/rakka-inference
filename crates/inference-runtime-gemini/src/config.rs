use serde::{Deserialize, Serialize};
use url::Url;

use atomr_infer_core::deployment::{RateLimits, RetryPolicy, Timeouts};
use atomr_infer_core::runtime::CircuitBreakerConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    #[serde(flatten)]
    pub variant: GeminiVariant,
    /// Auth credential. AI Studio uses an API key (`StaticApiKey`);
    /// Vertex uses an OAuth2 access token via the operator-supplied
    /// credential provider.
    pub credential: SecretRef,
    #[serde(default)]
    pub safety: Vec<SafetySetting>,
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
#[serde(tag = "variant", rename_all = "snake_case")]
pub enum GeminiVariant {
    AiStudio {
        #[serde(default = "default_aistudio_endpoint")]
        endpoint: Url,
    },
    Vertex {
        project: String,
        region: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: String,
    pub threshold: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "from", rename_all = "snake_case")]
pub enum SecretRef {
    Env {
        name: String,
    },
    File {
        path: std::path::PathBuf,
    },
    Inline {
        value: String,
    },
    /// Vertex uses application default credentials; resolved by the
    /// operator-supplied `CredentialProvider`.
    Adc,
}

fn default_aistudio_endpoint() -> Url {
    Url::parse("https://generativelanguage.googleapis.com/v1beta/").expect("static url")
}

impl GeminiConfig {
    pub fn generate_content_url(&self, model: &str, stream: bool) -> Result<Url, url::ParseError> {
        let suffix = if stream {
            ":streamGenerateContent?alt=sse"
        } else {
            ":generateContent"
        };
        match &self.variant {
            GeminiVariant::AiStudio { endpoint } => {
                endpoint.join(&format!("models/{model}{suffix}"))
            }
            GeminiVariant::Vertex { project, region } => Url::parse(&format!(
                "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}{suffix}"
            )),
        }
    }
}
