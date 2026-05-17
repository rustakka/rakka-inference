//! `PiperRunner` — public entry point. Implements
//! [`atomr_infer_core::runner::SpeechRunner`] against a local Piper
//! voice pack.

use async_trait::async_trait;
use atomr_infer_core::audio::{AudioFormat, AudioParams, SpeechBatch, SpeechChunk};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{SessionRebuildCause, SpeechRunHandle, SpeechRunner};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

use crate::config::PiperConfig;

#[cfg(feature = "piper")]
use crate::session::{build_state, run_synthesis, PiperState};
#[cfg(feature = "piper")]
use std::sync::Arc;

/// Local Piper TTS runner. See the crate-level docs for build
/// profiles and the M4 phonemization scope note.
pub struct PiperRunner {
    cfg: PiperConfig,
    #[cfg(feature = "piper")]
    state: tokio::sync::OnceCell<Arc<PiperState>>,
}

impl PiperRunner {
    pub fn new(cfg: PiperConfig) -> Self {
        Self {
            cfg,
            #[cfg(feature = "piper")]
            state: tokio::sync::OnceCell::new(),
        }
    }

    pub fn config(&self) -> &PiperConfig {
        &self.cfg
    }

    #[cfg(feature = "piper")]
    async fn ensure_state(&self) -> InferenceResult<Arc<PiperState>> {
        self.state
            .get_or_try_init(|| async {
                let cfg = self.cfg.clone();
                tokio::task::spawn_blocking(move || build_state(&cfg))
                    .await
                    .map_err(|e| InferenceError::Internal(format!("piper: spawn_blocking join: {e}")))?
                    .map_err(InferenceError::from)
            })
            .await
            .cloned()
    }
}

#[async_trait]
impl SpeechRunner for PiperRunner {
    async fn speak(&mut self, _batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        #[cfg(not(feature = "piper"))]
        {
            let _ = _batch;
            Err(InferenceError::Internal(
                "piper feature disabled at build time — rebuild with --features piper".into(),
            ))
        }
        #[cfg(feature = "piper")]
        {
            let state = self.ensure_state().await?;
            let cfg = self.cfg.clone();
            let request_id = _batch.request_id.clone();
            let text = _batch.text.clone();
            let chunk_samples = cfg.chunk_samples.max(1);
            let sample_rate = state.sample_rate;

            let pcm_f32 = tokio::task::spawn_blocking(move || run_synthesis(&state, &text, &cfg))
                .await
                .map_err(|e| InferenceError::Internal(format!("piper: spawn_blocking join: {e}")))?
                .map_err(InferenceError::from)?;

            let chunks = pcm_f32_to_chunks(pcm_f32, request_id, sample_rate, chunk_samples);
            let stream = futures::stream::iter(chunks.into_iter().map(Ok));
            Ok(SpeechRunHandle::streaming(Box::pin(stream)))
        }
    }

    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()> {
        #[cfg(feature = "piper")]
        {
            if matches!(
                cause,
                SessionRebuildCause::CudaContextPoisoned | SessionRebuildCause::Manual
            ) {
                self.state = tokio::sync::OnceCell::new();
            }
        }
        let _ = cause;
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::TextToSpeech
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalCpu
    }
}

#[cfg(feature = "piper")]
fn pcm_f32_to_chunks(
    pcm: Vec<f32>,
    request_id: String,
    sample_rate: u32,
    chunk_samples: usize,
) -> Vec<SpeechChunk> {
    use bytes::Bytes;

    if pcm.is_empty() {
        return vec![SpeechChunk {
            request_id,
            is_final: true,
            audio_pcm_chunk: Bytes::new(),
            params: AudioParams::new(sample_rate, 1, AudioFormat::Pcm16Le),
            alignment: None,
            usage: None,
        }];
    }

    let total = pcm.len();
    let num_chunks = total.div_ceil(chunk_samples);
    let mut out = Vec::with_capacity(num_chunks);
    for (i, window) in pcm.chunks(chunk_samples).enumerate() {
        let is_final = i + 1 == num_chunks;
        let mut bytes: Vec<u8> = Vec::with_capacity(window.len() * 2);
        for &s in window {
            // f32 in roughly [-1.0, 1.0] → i16. Saturate at the edges.
            let clamped = s.clamp(-1.0, 1.0);
            let i16_val = (clamped * i16::MAX as f32) as i16;
            bytes.extend_from_slice(&i16_val.to_le_bytes());
        }
        out.push(SpeechChunk {
            request_id: request_id.clone(),
            is_final,
            audio_pcm_chunk: Bytes::from(bytes),
            params: AudioParams::new(sample_rate, 1, AudioFormat::Pcm16Le),
            alignment: None,
            usage: None,
        });
    }
    out
}

// Suppress unused-import warning when `piper` is off; the imports are
// only used inside the feature-gated chunker.
#[cfg(not(feature = "piper"))]
#[allow(dead_code)]
fn _silence_unused_audio_imports() -> (AudioParams, AudioFormat, SpeechChunk) {
    unimplemented!()
}
