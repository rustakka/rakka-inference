//! `MossRunner` — public entry point. Implements
//! [`atomr_infer_core::runner::SpeechRunner`] against a local MOSS-TTSD
//! model. **Linux-only**: on non-Linux targets the feature-on path compiles
//! but `speak` returns `InferenceError::Internal("tts-moss requires Linux")`.

// `InferenceError` is referenced in the `#[cfg(not(feature))]` stub path.
#[allow(unused_imports)]
use atomr_infer_core::error::InferenceError;

use async_trait::async_trait;
use atomr_infer_core::audio::{AudioFormat, AudioParams, SpeechBatch, SpeechChunk};
use atomr_infer_core::error::InferenceResult;
use atomr_infer_core::runner::{SessionRebuildCause, SpeechRunHandle, SpeechRunner};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

use crate::config::MossConfig;
#[cfg(feature = "tts-moss")]
use crate::error::MossError;
#[cfg(all(feature = "tts-moss", target_os = "linux"))]
use crate::wire::MOSS_SAMPLE_RATE_HZ;

/// Local MOSS-TTSD TTS runner (Linux-only).
///
/// When compiled without `tts-moss`, `speak` returns
/// `InferenceError::Internal("tts-moss feature disabled at build time")`.
///
/// When `tts-moss` is on but the host is not Linux, `speak` returns
/// `InferenceError::Internal("tts-moss requires Linux")`.
pub struct MossRunner {
    cfg: MossConfig,
}

impl MossRunner {
    /// Construct a new runner.
    pub fn new(cfg: MossConfig) -> Self {
        Self { cfg }
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &MossConfig {
        &self.cfg
    }
}

#[async_trait]
impl SpeechRunner for MossRunner {
    async fn speak(&mut self, _batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        #[cfg(not(feature = "tts-moss"))]
        {
            let _ = _batch;
            return Err(InferenceError::Internal(
                "tts-moss feature disabled at build time — rebuild with --features tts-moss".into(),
            ));
        }

        #[cfg(feature = "tts-moss")]
        {
            // Platform gate: MOSS-TTSD is Linux-only.
            #[cfg(not(target_os = "linux"))]
            {
                let _ = _batch;
                return Err(MossError::RequiresLinux.into());
            }

            #[cfg(target_os = "linux")]
            {
                // Reject non-PCM format requests.
                if let Some(fmt) = _batch.options.format {
                    if !fmt.is_pcm() {
                        return Err(MossError::UnsupportedFormat {
                            message: format!("MOSS only emits PCM; requested {fmt:?} is not supported"),
                        }
                        .into());
                    }
                }

                let model_dir = &self.cfg.model_dir;
                if !model_dir.exists() {
                    return Err(MossError::ModelNotFound {
                        path: model_dir.clone(),
                    }
                    .into());
                }

                // Real inference would happen here. Return a single empty terminal
                // chunk as the feature-on / model-present stub.
                let request_id = _batch.request_id.clone();
                let chunk = SpeechChunk {
                    request_id,
                    is_final: true,
                    audio_pcm_chunk: bytes::Bytes::new(),
                    params: AudioParams::new(MOSS_SAMPLE_RATE_HZ, 1, AudioFormat::Pcm16Le),
                    alignment: None,
                    usage: None,
                };
                let stream = futures::stream::iter(vec![Ok(chunk)]);
                Ok(SpeechRunHandle::streaming(Box::pin(stream)))
            }
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

// Suppress unused-import warnings when `tts-moss` is off.
#[cfg(not(feature = "tts-moss"))]
#[allow(dead_code)]
fn _silence_unused_audio_imports() -> (AudioParams, AudioFormat, SpeechChunk) {
    unimplemented!()
}
