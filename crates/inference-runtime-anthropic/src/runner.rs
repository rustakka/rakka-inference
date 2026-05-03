use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::header;
use secrecy::ExposeSecret;
use url::Url;

use inference_core::batch::ExecuteBatch;
use inference_core::cost::from_rates;
use inference_core::deployment::RateLimits;
use inference_core::error::{InferenceError, InferenceResult};
use inference_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use inference_core::runtime::{ProviderKind, RuntimeKind, TransportKind};
use inference_core::tokens::{FinishReason, TokenChunk, TokenUsage};

use crate::config::AnthropicConfig;
use crate::cost::AnthropicPricing;
use crate::error::classify_anthropic_error;
use crate::wire::{BlockDelta, MessagesRequest, MessagesResponse, SseEvent};

use inference_remote_core::session::SessionSnapshot;
use inference_remote_core::sse::{decode_sse_stream, SseChunk};

pub struct AnthropicRunner {
    config: AnthropicConfig,
    session: Arc<ArcSwap<SessionSnapshot>>,
    messages_url: Url,
}

impl AnthropicRunner {
    pub fn new(
        config: AnthropicConfig,
        session: Arc<ArcSwap<SessionSnapshot>>,
    ) -> InferenceResult<Self> {
        let messages_url = config
            .messages_url()
            .map_err(|e| InferenceError::Internal(format!("anthropic url: {e}")))?;
        Ok(Self { config, session, messages_url })
    }

    fn auth_headers(&self) -> InferenceResult<header::HeaderMap> {
        let mut h = header::HeaderMap::new();
        let snap = self.session.load();
        let token = snap.credential.expose_secret().to_string();
        h.insert(
            header::HeaderName::from_static("x-api-key"),
            header::HeaderValue::from_str(&token)
                .map_err(|e| InferenceError::Internal(format!("invalid api key: {e}")))?,
        );
        h.insert(
            header::HeaderName::from_static("anthropic-version"),
            header::HeaderValue::from_str(&self.config.anthropic_version)
                .map_err(|e| InferenceError::Internal(format!("invalid version: {e}")))?,
        );
        Ok(h)
    }
}

fn map_stop_reason(s: &str) -> FinishReason {
    match s {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
}

fn lift_event(request_id: &str, sc: SseChunk) -> Option<InferenceResult<TokenChunk>> {
    let event_kind = sc.event.unwrap_or_default();
    if event_kind == "ping" || sc.data.is_empty() {
        return None;
    }
    match serde_json::from_str::<SseEvent>(&sc.data) {
        Err(e) => Some(Err(InferenceError::Internal(format!("anthropic event decode: {e}")))),
        Ok(SseEvent::ContentBlockDelta { delta: BlockDelta::TextDelta { text }, .. }) => {
            Some(Ok(TokenChunk {
                request_id: request_id.into(),
                text_delta: text,
                tool_call_delta: None,
                usage: None,
                finish_reason: None,
            }))
        }
        Ok(SseEvent::ContentBlockDelta {
            delta: BlockDelta::InputJsonDelta { partial_json },
            ..
        }) => Some(Ok(TokenChunk {
            request_id: request_id.into(),
            text_delta: String::new(),
            tool_call_delta: Some(serde_json::Value::String(partial_json)),
            usage: None,
            finish_reason: None,
        })),
        Ok(SseEvent::MessageDelta { delta, usage }) => Some(Ok(TokenChunk {
            request_id: request_id.into(),
            text_delta: String::new(),
            tool_call_delta: None,
            usage: usage.map(|u| TokenUsage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                cached_tokens: u.cache_read_input_tokens,
                ..Default::default()
            }),
            finish_reason: delta.stop_reason.as_deref().map(map_stop_reason),
        })),
        Ok(SseEvent::MessageStart { message, .. }) => {
            let _ = message;
            None
        }
        Ok(SseEvent::MessageStop) => Some(Ok(TokenChunk {
            request_id: request_id.into(),
            text_delta: String::new(),
            tool_call_delta: None,
            usage: None,
            finish_reason: Some(FinishReason::Stop),
        })),
        Ok(SseEvent::Error { error }) => {
            Some(Err(InferenceError::Internal(format!("anthropic stream error: {}: {}", error.kind, error.message))))
        }
        Ok(_) => None,
    }
}

#[async_trait]
impl ModelRunner for AnthropicRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        let snap = self.session.load_full();
        let body = MessagesRequest::from_batch(&batch);
        let resp = snap
            .client
            .post(self.messages_url.clone())
            .headers(self.auth_headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let body = resp.text().await.ok();
            return Err(classify_anthropic_error(status, retry_after.as_deref(), body));
        }

        let request_id = batch.request_id.clone();
        if batch.stream {
            let stream = decode_sse_stream(resp.bytes_stream());
            let id = request_id.clone();
            let lifted = stream.filter_map(move |item| {
                let id = id.clone();
                async move {
                    match item {
                        Ok(c) => lift_event(&id, c),
                        Err(e) => Some(Err(e)),
                    }
                }
            });
            Ok(RunHandle::streaming(lifted.boxed()))
        } else {
            let parsed: MessagesResponse = resp
                .json()
                .await
                .map_err(|e| InferenceError::Internal(format!("anthropic decode: {e}")))?;
            let mut text = String::new();
            for c in &parsed.content {
                if let crate::wire::ResponseContent::Text { text: t } = c {
                    text.push_str(t);
                }
            }
            let usage = parsed.usage.map(|u| TokenUsage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                cached_tokens: u.cache_read_input_tokens,
                ..Default::default()
            });
            let finish = parsed.stop_reason.as_deref().map(map_stop_reason).or(Some(FinishReason::Stop));
            let chunk = TokenChunk {
                request_id,
                text_delta: text,
                tool_call_delta: None,
                usage,
                finish_reason: finish,
            };
            let s: BoxStream<'static, InferenceResult<TokenChunk>> =
                stream::iter(vec![Ok(chunk)]).boxed();
            Ok(RunHandle::streaming(s))
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Anthropic
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork { provider: ProviderKind::Anthropic }
    }
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }
    fn estimate_cost_usd(&self, batch: &ExecuteBatch) -> f64 {
        AnthropicPricing::published()
            .get(&batch.model)
            .map(|p| from_rates(p.input_per_mtok_usd, p.output_per_mtok_usd, batch).usd)
            .unwrap_or(0.0)
    }
}
