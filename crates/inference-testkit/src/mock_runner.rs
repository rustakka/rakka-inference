//! `MockRunner` — deterministic ModelRunner for tests.

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{ModelRunner, RunHandle, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};
use atomr_infer_core::tokens::{FinishReason, TokenChunk, TokenUsage};

#[derive(Debug, Clone, Default)]
pub struct MockScript {
    /// Chunks emitted in order. The mock appends a `Stop` finish on the
    /// last item if none of the user-supplied chunks already carry one.
    pub chunks: Vec<String>,
    /// Inter-chunk delay; useful to simulate streaming latency in
    /// gateway/streams tests.
    pub inter_chunk_delay: Duration,
    /// Force `execute` to return `err` instead of a `RunHandle`.
    pub fail_with: Option<InferenceError>,
}

impl MockScript {
    pub fn from_text<I: IntoIterator<Item: Into<String>>>(chunks: I) -> Self {
        Self {
            chunks: chunks.into_iter().map(Into::into).collect(),
            ..Default::default()
        }
    }
}

pub struct MockRunner {
    script: MockScript,
}

impl MockRunner {
    pub fn new(script: MockScript) -> Self {
        Self { script }
    }
}

#[async_trait]
impl ModelRunner for MockRunner {
    async fn execute(&mut self, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        if let Some(err) = self.script.fail_with.clone() {
            return Err(err);
        }
        let chunks = self.script.chunks.clone();
        let delay = self.script.inter_chunk_delay;
        let request_id = batch.request_id.clone();
        let total = chunks.len();
        let stream: BoxStream<'static, InferenceResult<TokenChunk>> =
            stream::iter(chunks.into_iter().enumerate().map(move |(i, c)| {
                let last = i == total.saturating_sub(1);
                Ok::<_, InferenceError>(TokenChunk {
                    request_id: request_id.clone(),
                    text_delta: c,
                    tool_call_delta: None,
                    usage: last.then(|| TokenUsage {
                        input_tokens: 1,
                        output_tokens: total as u32,
                        ..Default::default()
                    }),
                    finish_reason: last.then_some(FinishReason::Stop),
                })
            }))
            .then(move |item| async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                item
            })
            .boxed();
        Ok(RunHandle::streaming(stream))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Custom("mock".into())
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalGpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_streams_text_in_order() {
        let mut r = MockRunner::new(MockScript::from_text(["hello ", "world"]));
        let batch = ExecuteBatch {
            request_id: "r1".into(),
            model: "mock".into(),
            messages: vec![],
            sampling: Default::default(),
            stream: true,
            estimated_tokens: 1,
        };
        let handle = r.execute(batch).await.unwrap();
        let chunks: Vec<_> = handle.into_stream().collect().await;
        let texts: Vec<_> = chunks
            .iter()
            .filter_map(|c| c.as_ref().ok().map(|c| c.text_delta.clone()))
            .collect();
        assert_eq!(texts, vec!["hello ", "world"]);
        let last = chunks.last().unwrap().as_ref().unwrap();
        assert_eq!(last.finish_reason, Some(FinishReason::Stop));
    }
}
