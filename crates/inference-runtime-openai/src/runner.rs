//! `OpenAiRunner` — `ModelRunner` impl for OpenAI Chat Completions /
//! Azure OpenAI.

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

use crate::config::OpenAiConfig;
use crate::cost::OpenAiPricing;
use crate::error::classify_openai_error;
use crate::wire::{ChatChunk, ChatRequest, ChatResponse};

use inference_remote_core::session::SessionSnapshot;
use inference_remote_core::sse::{decode_sse_stream, SseChunk};

pub struct OpenAiRunner {
    config: OpenAiConfig,
    /// Hot-swappable session snapshot — rebuilt on auth-failure or
    /// operator request via `inference_remote_core::RemoteSessionActor`.
    session: Arc<ArcSwap<SessionSnapshot>>,
    /// Concrete URL for chat-completions, computed once at construction.
    chat_url: Url,
}

impl OpenAiRunner {
    pub fn new(config: OpenAiConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> InferenceResult<Self> {
        let chat_url = config
            .variant
            .chat_completions_url()
            .map_err(|e| InferenceError::Internal(format!("openai endpoint url: {e}")))?;
        Ok(Self { config, session, chat_url })
    }

    fn auth_headers(&self) -> InferenceResult<header::HeaderMap> {
        let mut h = header::HeaderMap::new();
        let snap = self.session.load();
        let token = snap.credential.expose_secret().to_string();
        let value = header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|e| InferenceError::Internal(format!("invalid bearer token: {e}")))?;
        h.insert(header::AUTHORIZATION, value);
        if let Some(org) = &self.config.organization {
            h.insert(
                header::HeaderName::from_static("openai-organization"),
                header::HeaderValue::from_str(org)
                    .map_err(|e| InferenceError::Internal(format!("invalid org header: {e}")))?,
            );
        }
        if let Some(proj) = &self.config.project {
            h.insert(
                header::HeaderName::from_static("openai-project"),
                header::HeaderValue::from_str(proj)
                    .map_err(|e| InferenceError::Internal(format!("invalid project header: {e}")))?,
            );
        }
        Ok(h)
    }

}

/// Lift one OpenAI SSE chunk into a `TokenChunk`. Free function so the
/// per-stream closure that calls this doesn't capture `&OpenAiRunner`.
fn lift_chunk(request_id: &str, sc: SseChunk) -> Option<InferenceResult<TokenChunk>> {
    if sc.data == "[DONE]" {
        return None;
    }
    match serde_json::from_str::<ChatChunk>(&sc.data) {
        Err(e) => Some(Err(InferenceError::Internal(format!("openai chunk decode: {e}")))),
        Ok(parsed) => {
            let mut text_delta = String::new();
            let mut finish = None;
            for ch in &parsed.choices {
                if let Some(c) = &ch.delta.content {
                    text_delta.push_str(c);
                }
                finish = ch.finish_reason.as_deref().and_then(map_finish_reason);
            }
            let usage = parsed.usage.as_ref().map(|u| TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cached_tokens: u
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0),
                reasoning_tokens: u
                    .completion_tokens_details
                    .as_ref()
                    .map(|d| d.reasoning_tokens)
                    .unwrap_or(0),
            });
            Some(Ok(TokenChunk {
                request_id: request_id.to_string(),
                text_delta,
                tool_call_delta: parsed.choices.into_iter().find_map(|c| c.delta.tool_calls),
                usage,
                finish_reason: finish,
            }))
        }
    }
}

fn map_finish_reason(s: &str) -> Option<FinishReason> {
    match s {
        "stop" | "end_turn" => Some(FinishReason::Stop),
        "length" => Some(FinishReason::Length),
        "tool_calls" | "function_call" => Some(FinishReason::ToolCalls),
        "content_filter" => Some(FinishReason::ContentFilter),
        _ => Some(FinishReason::Stop),
    }
}

#[async_trait]
impl ModelRunner for OpenAiRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        let snap = self.session.load_full();
        let body = ChatRequest::from_batch(&batch);
        let req = snap
            .client
            .post(self.chat_url.clone())
            .headers(self.auth_headers()?)
            .json(&body);

        let resp = req
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
            return Err(classify_openai_error(status, retry_after.as_deref(), body));
        }

        let request_id = batch.request_id.clone();
        if batch.stream {
            let stream = decode_sse_stream(resp.bytes_stream());
            let request_id_for_stream = request_id.clone();
            let lifted = stream.filter_map(move |item| {
                let id = request_id_for_stream.clone();
                async move {
                    match item {
                        Ok(chunk) => lift_chunk(&id, chunk),
                        Err(e) => Some(Err(e)),
                    }
                }
            });
            Ok(RunHandle::streaming(lifted.boxed()))
        } else {
            let parsed: ChatResponse = resp
                .json()
                .await
                .map_err(|e| InferenceError::Internal(format!("openai response decode: {e}")))?;
            let mut text = String::new();
            let mut finish = None;
            for ch in &parsed.choices {
                if let Some(s) = ch.message.content.as_str() {
                    text.push_str(s);
                }
                finish = ch.finish_reason.as_deref().and_then(map_finish_reason);
            }
            let usage = parsed.usage.map(|u| TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                ..Default::default()
            });
            let chunk = TokenChunk {
                request_id,
                text_delta: text,
                tool_call_delta: None,
                usage,
                finish_reason: finish.or(Some(FinishReason::Stop)),
            };
            let s: BoxStream<'static, InferenceResult<TokenChunk>> = stream::iter(vec![Ok(chunk)]).boxed();
            Ok(RunHandle::streaming(s))
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        // The session is owned by `RemoteSessionActor`; this hook is a
        // no-op on the runner side. The actor swaps `self.session`'s
        // `ArcSwap` and the next request picks up the fresh snapshot.
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::OpenAi
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork { provider: ProviderKind::OpenAi }
    }
    fn gil_pinned(&self) -> bool {
        false
    }
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }

    fn estimate_cost_usd(&self, batch: &ExecuteBatch) -> f64 {
        OpenAiPricing::published()
            .get(&batch.model)
            .map(|p| from_rates(p.input_per_mtok_usd, p.output_per_mtok_usd, batch).usd)
            .unwrap_or(0.0)
    }
}

