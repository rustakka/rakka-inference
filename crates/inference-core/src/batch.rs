//! Request batch — what the runtime executes.
//!
//! `ExecuteBatch` is intentionally small: it carries one logical
//! request's worth of input. Local runtimes that batch internally (vLLM,
//! TensorRT) batch *across* `ExecuteBatch` instances inside their own
//! engine module — see doc §5.2 ("scheduler and batching are modules,
//! not actors").

use serde::{Deserialize, Serialize};

/// One conversation message. OpenAI-compatible shape so the gateway can
/// pass it through with minimal translation; provider-specific
/// runtimes (Anthropic, Gemini) translate at the edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentPart {
    Text {
        text: String,
    },
    /// Base64-encoded image input. Provider runtimes translate to their
    /// preferred wire format.
    ImageBase64 {
        mime: String,
        data: String,
    },
    /// URL-referenced image (provider-supported only).
    ImageUrl {
        url: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
}

/// One unit of work handed to a `ModelRunner`. `request_id` is the
/// `RequestActor`'s identifier so completions can be correlated back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteBatch {
    pub request_id: String,
    pub model: String,
    pub messages: Vec<Message>,
    pub sampling: SamplingParams,
    /// True if the caller wants token-by-token streaming (`Tokens`
    /// chunks). False if a single final `Tokens` is acceptable.
    pub stream: bool,
    /// Best-effort estimate of input + max_output tokens, used by
    /// `RateLimiterActor` to acquire a TPM permit before the request
    /// hits the wire.
    pub estimated_tokens: u32,
}

impl ExecuteBatch {
    pub fn estimated_tokens(&self) -> u32 {
        self.estimated_tokens.max(1)
    }
}
