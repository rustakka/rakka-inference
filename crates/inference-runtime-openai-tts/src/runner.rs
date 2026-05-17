//! `OpenAiTtsRunner` — [`SpeechRunner`] implementation against
//! `POST /v1/audio/speech`.

#[cfg(feature = "tts-openai")]
use std::sync::Arc;

use async_trait::async_trait;

use atomr_infer_core::audio::SpeechBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{SessionRebuildCause, SpeechRunHandle, SpeechRunner};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

#[cfg(feature = "tts-openai")]
use arc_swap::ArcSwap;
#[cfg(feature = "tts-openai")]
use atomr_infer_core::audio::{AudioFormat, AudioParams, SpeechChunk};
#[cfg(feature = "tts-openai")]
use atomr_infer_core::deployment::RateLimits;
#[cfg(feature = "tts-openai")]
use atomr_infer_remote_core::session::SessionSnapshot;
#[cfg(feature = "tts-openai")]
use atomr_infer_runtime_openai::error::classify_openai_error;
#[cfg(feature = "tts-openai")]
use bytes::Bytes;
#[cfg(feature = "tts-openai")]
use futures::stream::{self, BoxStream, StreamExt};
#[cfg(feature = "tts-openai")]
use reqwest::header;
#[cfg(feature = "tts-openai")]
use secrecy::ExposeSecret;
#[cfg(feature = "tts-openai")]
use url::Url;

#[cfg(feature = "tts-openai")]
use crate::config::OpenAiTtsConfig;
#[cfg(feature = "tts-openai")]
use crate::wire::{response_format_str, SpeechRequest};

/// SpeechRunner implementation for OpenAI's `/v1/audio/speech` endpoint.
///
/// Audio bytes arrive in the HTTP response body; the runner re-chunks
/// them at `OpenAiTtsConfig::chunk_bytes` boundaries before emitting
/// [`SpeechChunk`]s to the caller. The terminal chunk carries
/// `is_final = true`.
///
/// [`SpeechChunk`]: atomr_infer_core::audio::SpeechChunk
pub struct OpenAiTtsRunner {
    #[cfg(feature = "tts-openai")]
    config: OpenAiTtsConfig,
    #[cfg(feature = "tts-openai")]
    session: Arc<ArcSwap<SessionSnapshot>>,
    #[cfg(feature = "tts-openai")]
    speech_url: Url,
    // Without the feature the struct still needs to exist (for stub
    // tests and dependency-graph clarity) but holds nothing.
    #[cfg(not(feature = "tts-openai"))]
    _stub: (),
}

#[cfg(feature = "tts-openai")]
impl OpenAiTtsRunner {
    /// Construct a runner. `session` is the shared
    /// `inference-remote-core::SessionSnapshot` swap-cell — when the
    /// session actor rebuilds the credential, this runner's next
    /// request picks up the fresh bearer token automatically.
    pub fn new(config: OpenAiTtsConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> InferenceResult<Self> {
        let speech_url = config
            .speech_url()
            .map_err(|e| InferenceError::Internal(format!("openai-tts endpoint url: {e}")))?;
        Ok(Self {
            config,
            session,
            speech_url,
        })
    }

    fn auth_headers(&self) -> InferenceResult<header::HeaderMap> {
        let mut h = header::HeaderMap::new();
        let snap = self.session.load();
        let token = snap.credential.expose_secret().to_string();
        let value = header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|e| InferenceError::Internal(format!("invalid bearer token: {e}")))?;
        h.insert(header::AUTHORIZATION, value);
        if let Some(org) = &self.config.organization {
            h.insert(
                header::HeaderName::from_static("openai-organization"),
                header::HeaderValue::from_str(org)
                    .map_err(|e| InferenceError::Internal(format!("invalid org header: {e}")))?,
            );
        }
        if let Some(proj) = &self.config.project {
            h.insert(
                header::HeaderName::from_static("openai-project"),
                header::HeaderValue::from_str(proj)
                    .map_err(|e| InferenceError::Internal(format!("invalid project header: {e}")))?,
            );
        }
        Ok(h)
    }
}

#[cfg(not(feature = "tts-openai"))]
impl OpenAiTtsRunner {
    /// Stub constructor — accepts no arguments so callers can still
    /// link without pulling the feature in.
    pub fn new_stub() -> Self {
        Self { _stub: () }
    }
}

#[async_trait]
impl SpeechRunner for OpenAiTtsRunner {
    #[cfg(feature = "tts-openai")]
    async fn speak(&mut self, batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        let response_format = match batch.options.format {
            Some(fmt) => response_format_str(fmt).ok_or_else(|| InferenceError::UnsupportedAudioFormat {
                message: format!("openai-tts: response_format for {fmt:?} not supported"),
            })?,
            None => "pcm",
        };

        let body = SpeechRequest::from_batch(&batch, response_format);

        let snap = self.session.load_full();
        let resp = snap
            .client
            .post(self.speech_url.clone())
            .headers(self.auth_headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());
            let body_text = resp.text().await.ok();
            return Err(classify_openai_error(status, retry_after.as_deref(), body_text));
        }

        // Materialise the audio body (OpenAI's TTS endpoint sends the
        // full payload in one response; chunked-transfer is per-byte,
        // not per-semantic-frame). We re-chunk it ourselves at
        // `chunk_bytes` boundaries to get streaming-friendly
        // backpressure on the consumer side.
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| InferenceError::NetworkError(format!("read tts body: {e}")))?;

        let chunk_bytes = self.config.chunk_bytes.max(1);
        let total = bytes.len();
        let params = output_params(batch.options.format);

        let mut chunks: Vec<InferenceResult<SpeechChunk>> = Vec::new();
        if total == 0 {
            chunks.push(Ok(SpeechChunk {
                request_id: batch.request_id.clone(),
                is_final: true,
                audio_pcm_chunk: Bytes::new(),
                params,
                alignment: None,
                usage: None,
            }));
        } else {
            let mut offset = 0usize;
            while offset < total {
                let end = (offset + chunk_bytes).min(total);
                let slice = bytes.slice(offset..end);
                let is_final = end == total;
                chunks.push(Ok(SpeechChunk {
                    request_id: batch.request_id.clone(),
                    is_final,
                    audio_pcm_chunk: slice,
                    params,
                    alignment: None,
                    usage: None,
                }));
                offset = end;
            }
        }

        let s: BoxStream<'static, InferenceResult<SpeechChunk>> = stream::iter(chunks).boxed();
        Ok(SpeechRunHandle::streaming(s))
    }

    #[cfg(not(feature = "tts-openai"))]
    async fn speak(&mut self, _batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        Err(InferenceError::Internal(
            "tts-openai feature disabled at build time".into(),
        ))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        // The session snapshot is owned by `RemoteSessionActor`; this
        // hook is a no-op on the runner side.
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::TextToSpeech
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork {
            provider: ProviderKind::OpenAi,
        }
    }

    #[cfg(feature = "tts-openai")]
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }
}

/// Map the caller's requested format (if any) to the params the
/// runner attaches to each emitted [`SpeechChunk`]. OpenAI's PCM
/// format is 24 kHz mono signed 16-bit LE; the container-style
/// formats (mp3/wav/opus/flac) keep [`AudioFormat`] tagged but the
/// inner bytes carry their own headers.
#[cfg(feature = "tts-openai")]
fn output_params(requested: Option<AudioFormat>) -> AudioParams {
    let format = requested.unwrap_or(AudioFormat::Pcm16Le);
    let sample_rate_hz = match format {
        AudioFormat::Pcm16Le | AudioFormat::Pcm24Le | AudioFormat::PcmF32Le => 24_000,
        // Containerised formats carry their own header; the
        // `sample_rate_hz` field becomes advisory. 24 kHz matches the
        // public default for `tts-1`.
        _ => 24_000,
    };
    AudioParams::new(sample_rate_hz, 1, format)
}
