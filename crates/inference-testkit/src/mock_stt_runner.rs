//! `MockSttRunner` — deterministic [`AudioRunner`] for tests.

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

use atomr_infer_core::audio::{AudioBatch, TranscriptChunk};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{AudioRunHandle, AudioRunner, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Default)]
pub struct MockSttScript {
    /// Final transcripts emitted in order. The mock marks the last
    /// one as `is_final = true`.
    pub transcripts: Vec<String>,
    /// Optional interim (partial) transcripts emitted before the
    /// final list. Each is marked `is_final = false`.
    pub partials: Vec<String>,
    /// Inter-chunk delay; simulates streaming latency.
    pub inter_chunk_delay: Duration,
    /// Force `execute_audio` to return `err` instead of a handle.
    pub fail_with: Option<InferenceError>,
}

impl MockSttScript {
    pub fn from_text<I: IntoIterator<Item: Into<String>>>(transcripts: I) -> Self {
        Self {
            transcripts: transcripts.into_iter().map(Into::into).collect(),
            ..Default::default()
        }
    }
}

pub struct MockSttRunner {
    script: MockSttScript,
}

impl MockSttRunner {
    pub fn new(script: MockSttScript) -> Self {
        Self { script }
    }
}

#[async_trait]
impl AudioRunner for MockSttRunner {
    async fn execute_audio(&mut self, batch: AudioBatch) -> InferenceResult<AudioRunHandle> {
        if let Some(err) = self.script.fail_with.clone() {
            return Err(err);
        }
        let partials = self.script.partials.clone();
        let finals = self.script.transcripts.clone();
        let delay = self.script.inter_chunk_delay;
        let request_id = batch.request_id.clone();
        let mut items: Vec<TranscriptChunk> = Vec::with_capacity(partials.len() + finals.len());
        for p in partials {
            items.push(TranscriptChunk {
                request_id: request_id.clone(),
                is_final: false,
                text: p,
                ts_start_ms: 0,
                ts_end_ms: 0,
                speaker_id: None,
                words: vec![],
                usage: None,
            });
        }
        let total_finals = finals.len();
        for (i, f) in finals.into_iter().enumerate() {
            items.push(TranscriptChunk {
                request_id: request_id.clone(),
                is_final: i == total_finals.saturating_sub(1),
                text: f,
                ts_start_ms: 0,
                ts_end_ms: 0,
                speaker_id: None,
                words: vec![],
                usage: None,
            });
        }
        let stream: BoxStream<'static, InferenceResult<TranscriptChunk>> =
            stream::iter(items.into_iter().map(Ok::<_, InferenceError>))
                .then(move |item| async move {
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    item
                })
                .boxed();
        Ok(AudioRunHandle::streaming(stream))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::SpeechToText
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::UnknownTransport
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::audio::{
        AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscribeOptions,
    };
    use bytes::Bytes;

    #[tokio::test]
    async fn mock_streams_transcripts_in_order() {
        let mut r = MockSttRunner::new(MockSttScript::from_text(["hello", "world"]));
        let batch = AudioBatch {
            request_id: "r1".into(),
            model: "whisper-1".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from_static(&[]),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: true,
            options: AudioOptions::Transcribe(TranscribeOptions::default()),
            estimated_units: 1,
        };
        let h = r.execute_audio(batch).await.unwrap();
        let chunks: Vec<_> = h.into_stream().collect().await;
        assert_eq!(chunks.len(), 2);
        let last = chunks.last().unwrap().as_ref().unwrap();
        assert!(last.is_final);
        assert_eq!(last.text, "world");
    }
}
