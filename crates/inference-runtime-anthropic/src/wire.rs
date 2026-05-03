//! Anthropic Messages API wire types.

use serde::{Deserialize, Serialize};

use inference_core::batch::{ContentPart, ExecuteBatch, MessageContent, Role};

#[derive(Debug, Serialize)]
pub struct MessagesRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<MessagesMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct MessagesMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl MessagesRequest<'_> {
    pub fn from_batch<'b>(b: &'b ExecuteBatch) -> MessagesRequest<'b> {
        let mut system: Option<String> = None;
        let mut messages: Vec<MessagesMessage> = Vec::new();
        for m in &b.messages {
            if matches!(m.role, Role::System) {
                if let MessageContent::Text(s) = &m.content {
                    if let Some(prev) = system.as_mut() {
                        prev.push('\n');
                        prev.push_str(s);
                    } else {
                        system = Some(s.clone());
                    }
                }
                continue;
            }
            let role = match m.role {
                Role::User | Role::Tool => "user",
                Role::Assistant => "assistant",
                Role::System => unreachable!(),
            }
            .to_string();
            let content = match &m.content {
                MessageContent::Text(t) => serde_json::Value::String(t.clone()),
                MessageContent::Parts(parts) => {
                    serde_json::Value::Array(parts.iter().map(serialize_part).collect())
                }
            };
            messages.push(MessagesMessage { role, content });
        }
        MessagesRequest {
            model: &b.model,
            messages,
            system,
            max_tokens: b.sampling.max_tokens.or(Some(1024)),
            temperature: b.sampling.temperature,
            top_p: b.sampling.top_p,
            stop_sequences: b.sampling.stop.clone(),
            stream: b.stream,
        }
    }
}

fn serialize_part(p: &ContentPart) -> serde_json::Value {
    match p {
        ContentPart::Text { text } => serde_json::json!({"type": "text", "text": text}),
        ContentPart::ImageBase64 { mime, data } => serde_json::json!({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": data}
        }),
        ContentPart::ImageUrl { url } => serde_json::json!({
            "type": "image",
            "source": {"type": "url", "url": url}
        }),
    }
}

// ---- streaming events -----------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseEvent {
    MessageStart {
        message: MessageStart,
    },
    ContentBlockStart {
        index: u32,
    },
    ContentBlockDelta {
        index: u32,
        delta: BlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: Option<UsageDelta>,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicErrorBody,
    },
}

#[derive(Debug, Deserialize)]
pub struct MessageStart {
    pub id: String,
    pub usage: Option<UsageDelta>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone, Copy)]
pub struct UsageDelta {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

// ---- non-streaming response ----------------------------------------------

#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(default)]
    pub content: Vec<ResponseContent>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<UsageDelta>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}
