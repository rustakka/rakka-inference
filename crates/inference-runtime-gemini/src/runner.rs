use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::header;
use secrecy::ExposeSecret;

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::cost::from_rates;
use atomr_infer_core::deployment::RateLimits;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};
use atomr_infer_core::tokens::{FinishReason, TokenChunk, TokenUsage};

use crate::config::{GeminiConfig, GeminiVariant};
use crate::cost::GeminiPricing;
use crate::error::classify_gemini_error;
use crate::wire::{GenerateContentRequest, GenerateContentResponse};

use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_remote_core::sse::{decode_sse_stream, SseChunk};

pub struct GeminiRunner {
    config: GeminiConfig,
    session: Arc<ArcSwap<SessionSnapshot>>,
}

impl GeminiRunner {
    pub fn new(config: GeminiConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> Self {
        Self { config, session }
    }

    fn auth_apply(
        &self,
        builder: reqwest::RequestBuilder,
        url: &mut url::Url,
    ) -> InferenceResult<reqwest::RequestBuilder> {
        let snap = self.session.load();
        let token = snap.credential.expose_secret().to_string();
        match self.config.variant {
            GeminiVariant::AiStudio { .. } => {
                // AI Studio takes the API key as a query param.
                url.query_pairs_mut().append_pair("key", &token);
                Ok(builder)
            }
            GeminiVariant::Vertex { .. } => Ok(builder.header(
                header::AUTHORIZATION,
                header::HeaderValue::from_str(&format!("Bearer {token}"))
                    .map_err(|e| InferenceError::Internal(format!("invalid bearer token: {e}")))?,
            )),
        }
    }
}

fn map_finish(s: &str) -> FinishReason {
    match s {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

fn lift_chunk(request_id: &str, sc: SseChunk) -> Option<InferenceResult<TokenChunk>> {
    if sc.data.is_empty() {
        return None;
    }
    match serde_json::from_str::<GenerateContentResponse>(&sc.data) {
        Err(e) => Some(Err(InferenceError::Internal(format!("gemini chunk decode: {e}")))),
        Ok(parsed) => {
            let mut text_delta = String::new();
            let mut finish = None;
            for c in &parsed.candidates {
                if let Some(content) = &c.content {
                    for p in &content.parts {
                        if let Some(t) = &p.text {
                            text_delta.push_str(t);
                        }
                    }
                }
                if let Some(s) = &c.finish_reason {
                    finish = Some(map_finish(s));
                }
            }
            let usage = parsed.usage_metadata.map(|u| TokenUsage {
                input_tokens: u.prompt_token_count,
                output_tokens: u.candidates_token_count,
                cached_tokens: u.cached_content_token_count,
                ..Default::default()
            });
            Some(Ok(TokenChunk {
                request_id: request_id.into(),
                text_delta,
                tool_call_delta: None,
                usage,
                finish_reason: finish,
            }))
        }
    }
}

#[async_trait]
impl ModelRunner for GeminiRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        let snap = self.session.load_full();
        let mut url = self
            .config
            .generate_content_url(&batch.model, batch.stream)
            .map_err(|e| InferenceError::Internal(format!("gemini url: {e}")))?;
        let body = GenerateContentRequest::from_batch(&batch, self.config.safety.clone());
        let request = snap.client.post(url.clone()).json(&body);
        let request = self.auth_apply(request, &mut url)?;
        // Re-issue with the (possibly query-augmented) URL.
        let snap2 = self.session.load_full();
        let _ = snap2; // keep the snapshot alive; we already used it above
        let resp = request
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
            return Err(classify_gemini_error(status, retry_after.as_deref(), body));
        }

        let request_id = batch.request_id.clone();
        if batch.stream {
            let stream = decode_sse_stream(resp.bytes_stream());
            let id = request_id.clone();
            let lifted = stream.filter_map(move |item| {
                let id = id.clone();
                async move {
                    match item {
                        Ok(c) => lift_chunk(&id, c),
                        Err(e) => Some(Err(e)),
                    }
                }
            });
            Ok(RunHandle::streaming(lifted.boxed()))
        } else {
            let parsed: GenerateContentResponse = resp
                .json()
                .await
                .map_err(|e| InferenceError::Internal(format!("gemini decode: {e}")))?;
            let mut text = String::new();
            let mut finish = None;
            for c in &parsed.candidates {
                if let Some(content) = &c.content {
                    for p in &content.parts {
                        if let Some(t) = &p.text {
                            text.push_str(t);
                        }
                    }
                }
                if let Some(s) = &c.finish_reason {
                    finish = Some(map_finish(s));
                }
            }
            let usage = parsed.usage_metadata.map(|u| TokenUsage {
                input_tokens: u.prompt_token_count,
                output_tokens: u.candidates_token_count,
                cached_tokens: u.cached_content_token_count,
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
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Gemini
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork {
            provider: ProviderKind::Gemini,
        }
    }
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }
    fn estimate_cost_usd(&self, batch: &ExecuteBatch) -> f64 {
        GeminiPricing::published()
            .get(&batch.model)
            .map(|p| from_rates(p.input_per_mtok_usd, p.output_per_mtok_usd, batch).usd)
            .unwrap_or(0.0)
    }
}
