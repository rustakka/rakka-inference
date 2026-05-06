//! Output side: the streaming token chunks runners emit and the
//! `RequestActor` accumulates.

use serde::{Deserialize, Serialize};

/// One streamed chunk. Local runtimes emit one per generated token (or
/// per micro-batch); remote runtimes emit one per provider SSE event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenChunk {
    pub request_id: String,
    pub text_delta: String,
    /// Tool-call delta (provider-specific JSON). Carried opaquely
    /// through the runtime; only the gateway interprets it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_delta: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Error,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Model-reported reasoning tokens (o1-style). Optional.
    #[serde(default)]
    pub reasoning_tokens: u32,
    /// Cached prefix tokens, if the provider reports them (Anthropic
    /// prompt-caching, OpenAI cached input).
    #[serde(default)]
    pub cached_tokens: u32,
}

impl TokenUsage {
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.cached_tokens += other.cached_tokens;
    }
}

/// Aggregate of one request's full output. Built by the `RequestActor`
/// from the chunk stream; emitted to the upstream client as either an
/// SSE response (streaming) or a single JSON body (unary).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tokens {
    pub request_id: String,
    pub text: String,
    pub usage: TokenUsage,
    pub finish_reason: Option<FinishReason>,
}

impl Tokens {
    pub fn append(&mut self, chunk: &TokenChunk) {
        self.text.push_str(&chunk.text_delta);
        if let Some(u) = chunk.usage {
            self.usage.add(u);
        }
        if let Some(r) = chunk.finish_reason {
            self.finish_reason = Some(r);
        }
    }
}
