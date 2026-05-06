//! OpenAI Chat Completions wire types — request envelope + SSE chunk
//! shape. Kept minimal: only what's needed to round-trip the prompts /
//! deltas / usage info that the actor system needs.

use serde::{Deserialize, Serialize};

use atomr_infer_core::batch::{ExecuteBatch, Message, MessageContent, Role};

#[derive(Debug, Serialize)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    pub stop: Vec<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl ChatRequest<'_> {
    pub fn from_batch<'b>(b: &'b ExecuteBatch) -> ChatRequest<'b> {
        ChatRequest {
            model: &b.model,
            messages: b.messages.iter().map(serialize_message).collect(),
            temperature: b.sampling.temperature,
            top_p: b.sampling.top_p,
            max_tokens: b.sampling.max_tokens,
            stop: b.sampling.stop.clone(),
            stream: b.stream,
            seed: b.sampling.seed,
        }
    }
}

fn serialize_message(m: &Message) -> ChatMessage {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        // `Role` is `#[non_exhaustive]`; default unknown roles to
        // "user" so the request still goes through.
        _ => "user",
    }
    .to_string();
    let content = match &m.content {
        MessageContent::Text(t) => serde_json::Value::String(t.clone()),
        MessageContent::Parts(parts) => {
            serde_json::to_value(parts).unwrap_or(serde_json::Value::String(String::new()))
        }
        _ => serde_json::Value::String(String::new()),
    };
    ChatMessage { role, content }
}

// ---- streaming chunk ------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub choices: Vec<ChoiceDelta>,
    #[serde(default)]
    pub usage: Option<UsageDelta>,
}

#[derive(Debug, Deserialize)]
pub struct ChoiceDelta {
    #[serde(default)]
    pub delta: ContentDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ContentDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct UsageDelta {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Deserialize, Default)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
}

// ---- non-streaming response ----------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<UsageDelta>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    #[serde(default)]
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

impl Default for ChatMessage {
    fn default() -> Self {
        Self {
            role: "assistant".into(),
            content: serde_json::Value::Null,
        }
    }
}
