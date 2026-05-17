//! `MockTtsRunner` — deterministic [`SpeechRunner`] for tests.

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt};

use atomr_infer_core::audio::{AudioFormat, AudioParams, SpeechBatch, SpeechChunk};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{SessionRebuildCause, SpeechRunHandle, SpeechRunner};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone)]
pub struct MockTtsScript {
    /// Audio chunks emitted in order. Each is a raw PCM buffer.
    pub audio_chunks: Vec<Bytes>,
    /// Sample rate applied to every emitted chunk.
    pub sample_rate: u32,
    /// Encoding applied to every emitted chunk.
    pub format: AudioFormat,
    /// Inter-chunk delay; simulates streaming latency.
    pub inter_chunk_delay: Duration,
    /// Force `speak` to return `err` instead of a handle.
    pub fail_with: Option<InferenceError>,
}

impl Default for MockTtsScript {
    fn default() -> Self {
        Self {
            audio_chunks: Vec::new(),
            sample_rate: 24_000,
            format: AudioFormat::Pcm16Le,
            inter_chunk_delay: Duration::ZERO,
            fail_with: None,
        }
    }
}

impl MockTtsScript {
    pub fn from_audio<I: IntoIterator<Item = Bytes>>(chunks: I) -> Self {
        Self {
            audio_chunks: chunks.into_iter().collect(),
            ..Default::default()
        }
    }
}

pub struct MockTtsRunner {
    script: MockTtsScript,
}

impl MockTtsRunner {
    pub fn new(script: MockTtsScript) -> Self {
        Self { script }
    }
}

#[async_trait]
impl SpeechRunner for MockTtsRunner {
    async fn speak(&mut self, batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        if let Some(err) = self.script.fail_with.clone() {
            return Err(err);
        }
        let chunks = self.script.audio_chunks.clone();
        let delay = self.script.inter_chunk_delay;
        let request_id = batch.request_id.clone();
        let params = AudioParams::new(self.script.sample_rate, 1, self.script.format);
        let total = chunks.len();
        let stream: BoxStream<'static, InferenceResult<SpeechChunk>> =
            stream::iter(chunks.into_iter().enumerate().map(move |(i, c)| {
                let last = i == total.saturating_sub(1);
                Ok::<_, InferenceError>(SpeechChunk {
                    request_id: request_id.clone(),
                    is_final: last,
                    audio_pcm_chunk: c,
                    params,
                    alignment: None,
                    usage: None,
                })
            }))
            .then(move |item| async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                item
            })
            .boxed();
        Ok(SpeechRunHandle::streaming(stream))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::TextToSpeech
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::UnknownTransport
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::audio::{SynthOptions, VoiceRef};

    #[tokio::test]
    async fn mock_streams_audio_in_order() {
        let mut r = MockTtsRunner::new(MockTtsScript::from_audio([
            Bytes::from_static(b"abc"),
            Bytes::from_static(b"def"),
        ]));
        let batch = SpeechBatch {
            request_id: "r1".into(),
            model: "mock".into(),
            text: "hi".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 2,
        };
        let h = r.speak(batch).await.unwrap();
        let chunks: Vec<_> = h.into_stream().collect().await;
        assert_eq!(chunks.len(), 2);
        let first = chunks[0].as_ref().unwrap();
        let last = chunks[1].as_ref().unwrap();
        assert_eq!(first.audio_pcm_chunk.as_ref(), b"abc");
        assert!(last.is_final);
    }
}
