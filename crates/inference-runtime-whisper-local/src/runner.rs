//! `WhisperRunner` — public entry point. Implements
//! [`atomr_infer_core::runner::AudioRunner`] against a local
//! whisper.cpp model file.

use async_trait::async_trait;
use atomr_infer_core::audio::{AudioBatch, AudioInput, AudioOptions};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{AudioRunHandle, AudioRunner, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

use crate::config::WhisperConfig;
use crate::error::WhisperError;

#[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::session::{build_session, transcribe, WhisperSession};
#[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
use std::sync::Arc;

/// Local whisper.cpp STT runner. See the crate-level docs for the
/// build profile / arch support matrix.
pub struct WhisperRunner {
    cfg: WhisperConfig,
    #[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
    session: tokio::sync::OnceCell<Arc<WhisperSession>>,
}

impl WhisperRunner {
    pub fn new(cfg: WhisperConfig) -> Self {
        Self {
            cfg,
            #[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
            session: tokio::sync::OnceCell::new(),
        }
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.cfg
    }

    #[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
    async fn ensure_session(&self) -> InferenceResult<Arc<WhisperSession>> {
        self.session
            .get_or_try_init(|| async {
                let cfg = self.cfg.clone();
                tokio::task::spawn_blocking(move || build_session(&cfg))
                    .await
                    .map_err(|e| InferenceError::Internal(format!("whisper: spawn_blocking join: {e}")))?
                    .map_err(InferenceError::from)
            })
            .await
            .cloned()
    }
}

#[async_trait]
impl AudioRunner for WhisperRunner {
    async fn execute_audio(&mut self, batch: AudioBatch) -> InferenceResult<AudioRunHandle> {
        // 1. Validate options — only Transcribe is in scope.
        if !matches!(batch.options, AudioOptions::Transcribe(_)) {
            return Err(InferenceError::BadRequest {
                message:
                    "whisper-local: AudioOptions::Audio2Face passed to an STT runner — wire the request to an A2F deployment instead"
                        .into(),
            });
        }
        // 2. Streaming input isn't supported in M5 (would require VAD
        //    chunking upstream of whisper.cpp).
        if matches!(batch.input, AudioInput::Stream { .. }) {
            return Err(WhisperError::StreamingNotSupported.into());
        }

        run_unsupported_arch_or_feature()?;

        #[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // 3. Decode payload to f32 mono 16 kHz.
            let payload = match batch.input {
                AudioInput::Static(payload) => payload,
                AudioInput::Stream { .. } => unreachable!("guarded above"),
            };
            let cfg = self.cfg.clone();
            let request_id = batch.request_id.clone();

            let session = self.ensure_session().await?;
            let cfg_for_blocking = cfg.clone();
            let request_id_for_blocking = request_id.clone();
            let chunks = tokio::task::spawn_blocking(move || -> Result<_, WhisperError> {
                let pcm = crate::audio_decode::payload_to_f32_pcm(&payload, cfg_for_blocking.sample_rate_hz)?;
                transcribe(&session, &pcm, &cfg_for_blocking, &request_id_for_blocking)
            })
            .await
            .map_err(|e| InferenceError::Internal(format!("whisper: spawn_blocking join: {e}")))?
            .map_err(InferenceError::from)?;

            let stream = futures::stream::iter(chunks.into_iter().map(Ok));
            return Ok(AudioRunHandle::streaming(Box::pin(stream)));
        }

        #[allow(unreachable_code)]
        Err(InferenceError::Internal(
            "whisper-local: reached unreachable post-guard path".into(),
        ))
    }

    async fn rebuild_session(&mut self, cause: SessionRebuildCause) -> InferenceResult<()> {
        #[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            if matches!(
                cause,
                SessionRebuildCause::CudaContextPoisoned | SessionRebuildCause::Manual
            ) {
                self.session = tokio::sync::OnceCell::new();
            }
        }
        let _ = cause;
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::SpeechToText
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::LocalCpu
    }
}

/// Return an error if the build can never reach the real whisper.cpp
/// path (feature off, or unsupported host arch). When the build is
/// fully enabled this is a no-op.
fn run_unsupported_arch_or_feature() -> InferenceResult<()> {
    #[cfg(not(feature = "stt-whisper"))]
    {
        Err(WhisperError::FeatureDisabled.into())
    }
    #[cfg(all(
        feature = "stt-whisper",
        not(any(target_arch = "x86_64", target_arch = "aarch64"))
    ))]
    {
        Err(WhisperError::UnsupportedArch {
            arch: std::env::consts::ARCH,
        }
        .into())
    }
    #[cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Ok(())
    }
}
