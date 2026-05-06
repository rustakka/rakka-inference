//! # inference-runtime-mistralrs
//!
//! `mistralrs` runner for atomr-infer. Wraps `mistralrs::Model` +
//! `mistralrs::TextModelBuilder` behind the `ModelRunner` trait so
//! Mistral.rs participates in the same `Deployment` actor topology as
//! the OpenAI / Anthropic / vLLM / TensorRT runners. Doc §10.3.
//!
//! The model is loaded lazily on the first call to
//! `ModelRunner::execute` (mistralrs's builder downloads from
//! HuggingFace, which can take minutes for 7B+ models — eager loading
//! would block the runner's constructor for too long).
//!
//! Default-features-off the crate compiles to a typed-error stub;
//! `cargo build --features remote-only` therefore pulls no candle /
//! cuda dependencies via this crate.
//!
//! ## MSRV note
//!
//! mistralrs 0.8 declares MSRV 1.88. The atomr-infer workspace MSRV
//! is 1.78 for remote-only builds; operators enabling this runner
//! need a toolchain that satisfies mistralrs's own MSRV.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsConfig {
    /// HuggingFace repo id, e.g. `"mistralai/Mistral-7B-Instruct-v0.3"`.
    pub model_id: String,
    /// Optional in-situ quantisation. Passed verbatim to
    /// `mistralrs::parse_isq_value`; e.g. `"Q4K"` / `"Q8_0"`.
    #[serde(default)]
    pub quant: Option<String>,
    /// Optional HuggingFace revision (branch / tag / commit).
    #[serde(default)]
    pub hf_revision: Option<String>,
    /// Force CPU execution (skip CUDA / Metal device probing).
    #[serde(default)]
    pub force_cpu: bool,
    /// Maximum concurrent sequences the engine schedules. Defaults
    /// to the mistralrs builder default (32).
    #[serde(default)]
    pub max_num_seqs: Option<usize>,
}

pub struct MistralRsRunner {
    #[cfg_attr(not(feature = "mistralrs"), allow(dead_code))]
    config: MistralRsConfig,
    #[cfg(feature = "mistralrs")]
    model: tokio::sync::OnceCell<std::sync::Arc<mistralrs::Model>>,
}

impl MistralRsRunner {
    pub fn new(config: MistralRsConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "mistralrs")]
            model: tokio::sync::OnceCell::new(),
        }
    }

    #[cfg(feature = "mistralrs")]
    async fn ensure_model(&self) -> InferenceResult<std::sync::Arc<mistralrs::Model>> {
        self.model
            .get_or_try_init(|| async {
                let mut builder = mistralrs::TextModelBuilder::new(&self.config.model_id);
                if self.config.force_cpu {
                    builder = builder.with_force_cpu();
                }
                if let Some(rev) = &self.config.hf_revision {
                    builder = builder.with_hf_revision(rev.clone());
                }
                if let Some(max_seqs) = self.config.max_num_seqs {
                    builder = builder.with_max_num_seqs(max_seqs);
                }
                if let Some(q) = &self.config.quant {
                    let isq = mistralrs::parse_isq_value(q, None)
                        .map_err(|e| InferenceError::Internal(format!("mistralrs: bad quant '{q}': {e}")))?;
                    builder = builder.with_isq(isq);
                }
                let model = builder.build().await.map_err(|e| {
                    InferenceError::Internal(format!(
                        "mistralrs: failed to build model '{}': {e}",
                        self.config.model_id
                    ))
                })?;
                Ok(std::sync::Arc::new(model))
            })
            .await
            .cloned()
    }
}

#[cfg(feature = "mistralrs")]
fn map_role(role: atomr_infer_core::batch::Role) -> mistralrs::TextMessageRole {
    use atomr_infer_core::batch::Role;
    match role {
        Role::System => mistralrs::TextMessageRole::System,
        Role::User => mistralrs::TextMessageRole::User,
        Role::Assistant => mistralrs::TextMessageRole::Assistant,
        Role::Tool => mistralrs::TextMessageRole::Tool,
        // `Role` is `#[non_exhaustive]`; default unknown roles to
        // `User` so the request still reaches the model.
        _ => mistralrs::TextMessageRole::User,
    }
}

#[cfg(feature = "mistralrs")]
fn message_text(message: &atomr_infer_core::batch::Message) -> String {
    use atomr_infer_core::batch::{ContentPart, MessageContent};
    match &message.content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        // Forward-compat: drop unknown variants.
        _ => String::new(),
    }
}

#[async_trait]
impl ModelRunner for MistralRsRunner {
    #[cfg_attr(
        feature = "mistralrs",
        tracing::instrument(skip(self, _batch), fields(model = %self.config.model_id))
    )]
    async fn execute(&mut self, _batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        #[cfg(not(feature = "mistralrs"))]
        {
            Err(InferenceError::Internal(
                "mistralrs feature disabled at build time — rebuild with --features mistralrs".into(),
            ))
        }
        #[cfg(feature = "mistralrs")]
        {
            use atomr_infer_core::tokens::{FinishReason, TokenChunk};
            use futures::StreamExt;

            let model = self.ensure_model().await?;
            let request_id = _batch.request_id.clone();

            let mut messages = mistralrs::TextMessages::new();
            for m in &_batch.messages {
                messages = messages.add_message(map_role(m.role), message_text(m));
            }

            let (tx, rx) = tokio::sync::mpsc::channel::<InferenceResult<TokenChunk>>(64);
            let req_id_for_task = request_id.clone();
            tokio::spawn(async move {
                let mut stream = match model.stream_chat_request(messages).await {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx
                            .send(Err(InferenceError::Internal(format!(
                                "mistralrs: stream_chat_request failed: {e}"
                            ))))
                            .await;
                        return;
                    }
                };
                while let Some(resp) = stream.next().await {
                    match resp {
                        mistralrs::Response::Chunk(chunk) => {
                            let choice = chunk.choices.first();
                            let text_delta = choice.and_then(|c| c.delta.content.clone()).unwrap_or_default();
                            let finish_reason = choice
                                .and_then(|c| c.finish_reason.as_deref())
                                .map(map_finish_reason);
                            let chunk_out = TokenChunk {
                                request_id: req_id_for_task.clone(),
                                text_delta,
                                tool_call_delta: None,
                                usage: None,
                                finish_reason,
                            };
                            if tx.send(Ok(chunk_out)).await.is_err() {
                                break;
                            }
                        }
                        mistralrs::Response::Done(full) => {
                            let usage = atomr_infer_core::tokens::TokenUsage {
                                input_tokens: full.usage.prompt_tokens as u32,
                                output_tokens: full.usage.completion_tokens as u32,
                                ..Default::default()
                            };
                            let finish_reason = full
                                .choices
                                .first()
                                .map(|c| map_finish_reason(c.finish_reason.as_str()));
                            let chunk_out = TokenChunk {
                                request_id: req_id_for_task.clone(),
                                text_delta: String::new(),
                                tool_call_delta: None,
                                usage: Some(usage),
                                finish_reason: finish_reason.or(Some(FinishReason::Stop)),
                            };
                            let _ = tx.send(Ok(chunk_out)).await;
                            break;
                        }
                        mistralrs::Response::ModelError(msg, _partial) => {
                            let _ = tx
                                .send(Err(InferenceError::Internal(format!(
                                    "mistralrs model error: {msg}"
                                ))))
                                .await;
                            break;
                        }
                        mistralrs::Response::InternalError(e) | mistralrs::Response::ValidationError(e) => {
                            let _ = tx
                                .send(Err(InferenceError::Internal(format!("mistralrs error: {e}"))))
                                .await;
                            break;
                        }
                        // Other variants (Completion*, ImageGeneration, Speech, Raw,
                        // Embeddings) are not produced by stream_chat_request on a
                        // text model; if the engine ever surfaces one, drop the
                        // stream rather than silently corrupting the token
                        // sequence.
                        _ => {
                            let _ = tx
                                .send(Err(InferenceError::Internal(
                                    "mistralrs: unexpected response variant".into(),
                                )))
                                .await;
                            break;
                        }
                    }
                }
            });

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx).boxed();
            Ok(RunHandle::streaming(stream))
        }
    }

    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()> {
        #[cfg(feature = "mistralrs")]
        {
            // CUDA poisoning or a manual rebuild forces a fresh model
            // load (and re-download if HF cache is gone). Auth /
            // config-change causes are remote-only and ignored here.
            if matches!(
                cause,
                SessionRebuildCause::CudaContextPoisoned | SessionRebuildCause::Manual
            ) {
                self.model = tokio::sync::OnceCell::new();
            }
        }
        let _ = cause;
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::MistralRs
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
}

#[cfg(feature = "mistralrs")]
fn map_finish_reason(s: &str) -> atomr_infer_core::tokens::FinishReason {
    use atomr_infer_core::tokens::FinishReason;
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips_through_serde() {
        let cfg = MistralRsConfig {
            model_id: "mistralai/Mistral-7B-Instruct-v0.3".into(),
            quant: Some("Q4K".into()),
            hf_revision: None,
            force_cpu: false,
            max_num_seqs: Some(16),
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: MistralRsConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model_id, cfg.model_id);
        assert_eq!(back.quant, cfg.quant);
        assert_eq!(back.max_num_seqs, cfg.max_num_seqs);
    }

    #[test]
    fn runner_reports_runtime_kind() {
        let runner = MistralRsRunner::new(MistralRsConfig {
            model_id: "test".into(),
            quant: None,
            hf_revision: None,
            force_cpu: false,
            max_num_seqs: None,
        });
        assert_eq!(runner.runtime_kind(), RuntimeKind::MistralRs);
        assert_eq!(runner.transport_kind(), TransportKind::LocalGpu);
    }

    #[cfg(not(feature = "mistralrs"))]
    #[tokio::test]
    async fn execute_without_feature_returns_internal_error() {
        use atomr_infer_core::batch::SamplingParams;

        let mut runner = MistralRsRunner::new(MistralRsConfig {
            model_id: "test".into(),
            quant: None,
            hf_revision: None,
            force_cpu: false,
            max_num_seqs: None,
        });
        let batch = ExecuteBatch {
            request_id: "test".into(),
            model: "test".into(),
            messages: vec![],
            sampling: SamplingParams::default(),
            stream: false,
            estimated_tokens: 1,
        };
        let result = runner.execute(batch).await;
        assert!(matches!(result, Err(InferenceError::Internal(_))));
    }
}
