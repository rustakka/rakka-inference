//! `MockA2FRunner` — deterministic [`A2FRunner`] for tests.

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

use atomr_infer_core::audio::{AudioBatch, BlendshapeChunk};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{A2FRunHandle, A2FRunner, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone)]
pub struct MockA2FScript {
    /// Blendshape frames emitted in order. Each is an ARKit-canonical
    /// 52-float weight vector.
    pub frames: Vec<[f32; 52]>,
    /// Frame rate, used to populate `timestamp_ms` on each chunk.
    pub fps: u32,
    /// Inter-chunk delay; simulates streaming latency.
    pub inter_chunk_delay: Duration,
    /// Force `execute_audio2face` to return `err` instead of a handle.
    pub fail_with: Option<InferenceError>,
}

impl Default for MockA2FScript {
    fn default() -> Self {
        Self {
            frames: Vec::new(),
            fps: 30,
            inter_chunk_delay: Duration::ZERO,
            fail_with: None,
        }
    }
}

impl MockA2FScript {
    pub fn from_frames<I: IntoIterator<Item = [f32; 52]>>(frames: I) -> Self {
        Self {
            frames: frames.into_iter().collect(),
            ..Default::default()
        }
    }
}

pub struct MockA2FRunner {
    script: MockA2FScript,
}

impl MockA2FRunner {
    pub fn new(script: MockA2FScript) -> Self {
        Self { script }
    }
}

#[async_trait]
impl A2FRunner for MockA2FRunner {
    async fn execute_audio2face(&mut self, batch: AudioBatch) -> InferenceResult<A2FRunHandle> {
        if let Some(err) = self.script.fail_with.clone() {
            return Err(err);
        }
        let frames = self.script.frames.clone();
        let fps = self.script.fps.max(1);
        let frame_ms = 1000 / fps;
        let delay = self.script.inter_chunk_delay;
        let request_id = batch.request_id.clone();
        let total = frames.len();
        let stream: BoxStream<'static, InferenceResult<BlendshapeChunk>> =
            stream::iter(frames.into_iter().enumerate().map(move |(i, weights)| {
                let last = i == total.saturating_sub(1);
                Ok::<_, InferenceError>(BlendshapeChunk {
                    request_id: request_id.clone(),
                    is_final: last,
                    timestamp_ms: (i as u32) * frame_ms,
                    weights,
                })
            }))
            .then(move |item| async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                item
            })
            .boxed();
        Ok(A2FRunHandle::streaming(stream))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Audio2Face
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::UnknownTransport
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::audio::{
        A2FOptions, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload,
    };
    use bytes::Bytes;

    #[tokio::test]
    async fn mock_streams_blendshape_frames() {
        let mut r = MockA2FRunner::new(MockA2FScript::from_frames([
            [0.1_f32; 52],
            [0.2_f32; 52],
            [0.3_f32; 52],
        ]));
        let batch = AudioBatch {
            request_id: "r1".into(),
            model: "a2f-3d".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from_static(&[]),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: true,
            options: AudioOptions::Audio2Face(A2FOptions::default()),
            estimated_units: 30,
        };
        let h = r.execute_audio2face(batch).await.unwrap();
        let chunks: Vec<_> = h.into_stream().collect().await;
        assert_eq!(chunks.len(), 3);
        let last = chunks.last().unwrap().as_ref().unwrap();
        assert!(last.is_final);
    }
}
