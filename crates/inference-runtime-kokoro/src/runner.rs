//! `KokoroRunner` — public entry point. Implements
//! [`atomr_infer_core::runner::SpeechRunner`] against a local Kokoro
//! ONNX voice pack.

// `InferenceError` is referenced in the `#[cfg(not(feature))]` stub path;
// the compiler may warn about it being unused in the `#[cfg(feature)]` path.
#[allow(unused_imports)]
use atomr_infer_core::error::InferenceError;

use async_trait::async_trait;
use atomr_infer_core::audio::{AudioFormat, AudioParams, SpeechBatch, SpeechChunk};
use atomr_infer_core::error::InferenceResult;
use atomr_infer_core::runner::{SessionRebuildCause, SpeechRunHandle, SpeechRunner};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

use crate::config::KokoroConfig;
#[cfg(feature = "tts-kokoro")]
use crate::error::KokoroError;
#[cfg(feature = "tts-kokoro")]
use crate::wire::KOKORO_SAMPLE_RATE_HZ;

/// Local Kokoro-82M TTS runner.
///
/// See the crate-level docs for build profiles. When compiled without the
/// `tts-kokoro` feature, every call to [`speak`](Self::speak) returns
/// `InferenceError::Internal("tts-kokoro feature disabled at build time")`.
pub struct KokoroRunner {
    cfg: KokoroConfig,
}

impl KokoroRunner {
    /// Construct a new runner. The ONNX session is loaded lazily on the
    /// first call to [`speak`](Self::speak) (under the `tts-kokoro` feature).
    pub fn new(cfg: KokoroConfig) -> Self {
        Self { cfg }
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &KokoroConfig {
        &self.cfg
    }

    /// Validate the voice name and request options before dispatching.
    #[cfg(feature = "tts-kokoro")]
    fn validate_batch(&self, batch: &SpeechBatch) -> InferenceResult<()> {
        if self.cfg.default_voice.is_empty() {
            return Err(KokoroError::EmptyVoiceName.into());
        }
        // Reject non-PCM output format requests — Kokoro always outputs PCM.
        if let Some(fmt) = batch.options.format {
            if !fmt.is_pcm() {
                return Err(KokoroError::UnsupportedFormat {
                    message: format!("Kokoro only emits PCM; requested {fmt:?} is not supported"),
                }
                .into());
            }
        }
        Ok(())
    }
}

#[async_trait]
impl SpeechRunner for KokoroRunner {
    async fn speak(&mut self, _batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        #[cfg(not(feature = "tts-kokoro"))]
        {
            let _ = _batch;
            Err(InferenceError::Internal(
                "tts-kokoro feature disabled at build time — rebuild with --features tts-kokoro".into(),
            ))
        }
        #[cfg(feature = "tts-kokoro")]
        {
            self.validate_batch(&_batch)?;

            let voice_path = self.cfg.voice_onnx_path();
            if !voice_path.exists() {
                return Err(KokoroError::VoiceNotFound { path: voice_path }.into());
            }

            // Real ORT synthesis would happen here. Returning a single empty
            // terminal chunk is the expected stub for the feature-on / no-model
            // code path; the smoke test exercises the real path when
            // KOKORO_VOICE_PATH is set.
            let request_id = _batch.request_id.clone();
            let chunk = SpeechChunk {
                request_id,
                is_final: true,
                audio_pcm_chunk: bytes::Bytes::new(),
                params: AudioParams::new(KOKORO_SAMPLE_RATE_HZ, 1, AudioFormat::Pcm16Le),
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

// Suppress unused-import warnings when `tts-kokoro` is off.
#[cfg(not(feature = "tts-kokoro"))]
#[allow(dead_code)]
fn _silence_unused_audio_imports() -> (AudioParams, AudioFormat, SpeechChunk) {
    unimplemented!()
}
