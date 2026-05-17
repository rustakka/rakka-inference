//! `XttsRunner` — public entry point. Implements
//! [`atomr_infer_core::runner::SpeechRunner`] against a local XTTS-v2
//! ONNX model.

// `InferenceError` is referenced in the `#[cfg(not(feature))]` stub path.
#[allow(unused_imports)]
use atomr_infer_core::error::InferenceError;

use async_trait::async_trait;
use atomr_infer_core::audio::{AudioFormat, AudioParams, SpeechBatch, SpeechChunk, VoiceRef};
use atomr_infer_core::error::InferenceResult;
use atomr_infer_core::runner::{SessionRebuildCause, SpeechRunHandle, SpeechRunner};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

use crate::config::XttsConfig;
#[cfg(feature = "tts-xtts")]
use crate::error::XttsError;
#[cfg(feature = "tts-xtts")]
use crate::wire::XTTS_SAMPLE_RATE_HZ;

/// Local XTTS-v2 TTS runner with voice-cloning support.
///
/// When compiled without `tts-xtts`, every call to [`speak`](Self::speak)
/// returns `InferenceError::Internal("tts-xtts feature disabled at build time")`.
///
/// Voice-cloning embedding computation is **stubbed** in M10: the runner
/// validates that the reference audio is materializable, then logs a warning
/// and proceeds with a zero-valued speaker embedding. A follow-up milestone
/// will wire the real speaker-encoder ONNX model.
pub struct XttsRunner {
    cfg: XttsConfig,
}

impl XttsRunner {
    /// Construct a new runner.
    pub fn new(cfg: XttsConfig) -> Self {
        Self { cfg }
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &XttsConfig {
        &self.cfg
    }

    /// Validate batch options before dispatching.
    #[cfg(feature = "tts-xtts")]
    fn validate_batch(&self, batch: &SpeechBatch) -> InferenceResult<()> {
        // Reject non-PCM output format requests.
        if let Some(fmt) = batch.options.format {
            if !fmt.is_pcm() {
                return Err(XttsError::UnsupportedFormat {
                    message: format!("XTTS only emits PCM; requested {fmt:?} is not supported"),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Handle the voice reference. In M10, `ClonedFrom` is stubbed: the
    /// payload is validated as materializable, then a warning is emitted
    /// and a zero embedding is used. Named/Id voices resolve to the
    /// default speaker.
    #[cfg(feature = "tts-xtts")]
    fn handle_voice_ref(&self, voice: &VoiceRef) -> InferenceResult<()> {
        match voice {
            VoiceRef::Named(_) | VoiceRef::Id(_) => {
                // Resolve to default speaker — no extra validation needed.
            }
            VoiceRef::ClonedFrom(payload) => {
                // Validate the payload is well-formed (params must be valid).
                if !payload.params().is_valid() {
                    return Err(XttsError::ReferenceAudioError {
                        reason: "reference audio params out of valid range".into(),
                    }
                    .into());
                }
                // M10 stub: embedding computation not yet implemented.
                tracing::warn!(
                    "xtts voice cloning embedding not yet implemented; \
                     using zero speaker embedding as placeholder"
                );
            }
            // VoiceRef is #[non_exhaustive] — treat unknown variants as Named.
            _ => {}
        }
        Ok(())
    }
}

#[async_trait]
impl SpeechRunner for XttsRunner {
    async fn speak(&mut self, _batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        #[cfg(not(feature = "tts-xtts"))]
        {
            let _ = _batch;
            Err(InferenceError::Internal(
                "tts-xtts feature disabled at build time — rebuild with --features tts-xtts".into(),
            ))
        }
        #[cfg(feature = "tts-xtts")]
        {
            self.validate_batch(&_batch)?;
            self.handle_voice_ref(&_batch.voice)?;

            let model_path = self.cfg.model_path.join("model.onnx");
            if !model_path.exists() {
                return Err(XttsError::ModelNotFound { path: model_path }.into());
            }

            // Real ORT synthesis would happen here. Return a single empty
            // terminal chunk as the feature-on / no-model stub.
            let request_id = _batch.request_id.clone();
            let chunk = SpeechChunk {
                request_id,
                is_final: true,
                audio_pcm_chunk: bytes::Bytes::new(),
                params: AudioParams::new(XTTS_SAMPLE_RATE_HZ, 1, AudioFormat::Pcm16Le),
                alignment: None,
                usage: None,
            };
            let stream = futures::stream::iter(vec![Ok(chunk)]);
            Ok(SpeechRunHandle::streaming(Box::pin(stream)))
        }
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::TextToSpeech
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalCpu
    }
}

// Suppress unused-import warnings when `tts-xtts` is off.
#[cfg(not(feature = "tts-xtts"))]
#[allow(dead_code)]
fn _silence_unused_audio_imports() -> (AudioParams, AudioFormat, SpeechChunk, VoiceRef) {
    unimplemented!()
}
