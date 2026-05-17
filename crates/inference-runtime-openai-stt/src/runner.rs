//! `OpenAiSttRunner` — [`AudioRunner`] implementation against
//! `POST /v1/audio/transcriptions`.

#[cfg(feature = "stt-openai")]
use std::sync::Arc;

use async_trait::async_trait;

use atomr_infer_core::audio::AudioBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{AudioRunHandle, AudioRunner, SessionRebuildCause};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

#[cfg(feature = "stt-openai")]
use arc_swap::ArcSwap;
#[cfg(feature = "stt-openai")]
use atomr_infer_core::audio::{
    AudioInput, AudioOptions, AudioPayload, TranscribeOptions, TranscriptChunk, WordTiming,
};
#[cfg(feature = "stt-openai")]
use atomr_infer_core::deployment::RateLimits;
#[cfg(feature = "stt-openai")]
use atomr_infer_remote_core::session::SessionSnapshot;
#[cfg(feature = "stt-openai")]
use atomr_infer_runtime_openai::error::classify_openai_error;
#[cfg(feature = "stt-openai")]
use bytes::Bytes;
#[cfg(feature = "stt-openai")]
use futures::stream::{self, BoxStream, StreamExt};
#[cfg(feature = "stt-openai")]
use reqwest::{header, multipart};
#[cfg(feature = "stt-openai")]
use secrecy::ExposeSecret;
#[cfg(feature = "stt-openai")]
use url::Url;

#[cfg(feature = "stt-openai")]
use crate::config::OpenAiSttConfig;
#[cfg(feature = "stt-openai")]
use crate::error::OpenAiSttError;
#[cfg(feature = "stt-openai")]
use crate::wire::{PlainResponse, VerboseResponse};

/// AudioRunner implementation for OpenAI's
/// `/v1/audio/transcriptions` endpoint.
///
/// Builds a `multipart/form-data` request from
/// [`AudioBatch::input`], asks for `json` (single chunk) or
/// `verbose_json` (per-segment chunks) depending on whether the
/// caller wanted timestamps or interim results, and re-emits the
/// parsed response as a sequence of [`atomr_infer_core::audio::TranscriptChunk`]s.
pub struct OpenAiSttRunner {
    #[cfg(feature = "stt-openai")]
    config: OpenAiSttConfig,
    #[cfg(feature = "stt-openai")]
    session: Arc<ArcSwap<SessionSnapshot>>,
    #[cfg(feature = "stt-openai")]
    transcriptions_url: Url,
    #[cfg(not(feature = "stt-openai"))]
    _stub: (),
}

#[cfg(feature = "stt-openai")]
impl OpenAiSttRunner {
    /// Construct a runner. `session` is the shared snapshot owned by
    /// `inference-remote-core::RemoteSessionActor` — credential
    /// rotations are picked up on the next request automatically.
    pub fn new(config: OpenAiSttConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> InferenceResult<Self> {
        let transcriptions_url = config
            .transcriptions_url()
            .map_err(|e| InferenceError::Internal(format!("openai-stt endpoint url: {e}")))?;
        Ok(Self {
            config,
            session,
            transcriptions_url,
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

#[cfg(not(feature = "stt-openai"))]
impl OpenAiSttRunner {
    /// Stub constructor — accepts no arguments so callers can still
    /// link without pulling the feature in.
    pub fn new_stub() -> Self {
        Self { _stub: () }
    }
}

#[async_trait]
impl AudioRunner for OpenAiSttRunner {
    #[cfg(feature = "stt-openai")]
    async fn execute_audio(&mut self, batch: AudioBatch) -> InferenceResult<AudioRunHandle> {
        // Reject A2F options early — this runner is STT-only.
        let opts = match &batch.options {
            AudioOptions::Transcribe(t) => t.clone(),
            AudioOptions::Audio2Face(_) => {
                return Err(OpenAiSttError::BadRequest {
                    message: "openai-stt: AudioOptions::Audio2Face is not supported".into(),
                }
                .into());
            }
            _ => {
                return Err(OpenAiSttError::BadRequest {
                    message: "openai-stt: unknown AudioOptions variant".into(),
                }
                .into());
            }
        };

        // Streaming input is not yet implemented — would need to
        // either accumulate frames before upload or move to the
        // realtime endpoint. M6 scope is upload-based.
        let payload = match batch.input {
            AudioInput::Static(p) => p,
            AudioInput::Stream { .. } => {
                return Err(OpenAiSttError::Unsupported {
                    method: "execute_audio: streaming inputs require the realtime endpoint",
                }
                .into());
            }
        };

        let (audio_bytes, filename) = materialize_payload(payload).await?;

        let want_verbose = opts.word_timestamps || opts.interim_results;
        let response_format = if want_verbose { "verbose_json" } else { "json" };

        let mut form = multipart::Form::new()
            .text("model", batch.model.clone())
            .text("response_format", response_format)
            .part(
                "file",
                multipart::Part::bytes(audio_bytes.to_vec())
                    .file_name(filename)
                    .mime_str("application/octet-stream")
                    .map_err(|e| InferenceError::Internal(format!("openai-stt: mime: {e}")))?,
            );
        if let Some(lang) = &opts.language {
            form = form.text("language", lang.clone());
        }
        if let Some(prompt) = &opts.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if let Some(temp) = opts.temperature {
            form = form.text("temperature", temp.to_string());
        }
        if opts.word_timestamps {
            // verbose_json supports `timestamp_granularities[]=word|segment`.
            form = form.text("timestamp_granularities[]", "word");
            form = form.text("timestamp_granularities[]", "segment");
        }

        let snap = self.session.load_full();
        let resp = snap
            .client
            .post(self.transcriptions_url.clone())
            .headers(self.auth_headers()?)
            .multipart(form)
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

        let request_id = batch.request_id.clone();
        let chunks: Vec<InferenceResult<TranscriptChunk>> = if want_verbose {
            let parsed: VerboseResponse = resp
                .json()
                .await
                .map_err(|e| InferenceError::Internal(format!("openai-stt verbose decode: {e}")))?;
            verbose_to_chunks(request_id, parsed, &opts)
        } else {
            let parsed: PlainResponse = resp
                .json()
                .await
                .map_err(|e| InferenceError::Internal(format!("openai-stt plain decode: {e}")))?;
            vec![Ok(TranscriptChunk {
                request_id,
                is_final: true,
                text: parsed.text,
                ts_start_ms: 0,
                ts_end_ms: 0,
                speaker_id: None,
                words: Vec::new(),
                usage: None,
            })]
        };

        let s: BoxStream<'static, InferenceResult<TranscriptChunk>> = stream::iter(chunks).boxed();
        Ok(AudioRunHandle::streaming(s))
    }

    #[cfg(not(feature = "stt-openai"))]
    async fn execute_audio(&mut self, _batch: AudioBatch) -> InferenceResult<AudioRunHandle> {
        Err(InferenceError::Internal(
            "stt-openai feature disabled at build time".into(),
        ))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::SpeechToText
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork {
            provider: ProviderKind::OpenAi,
        }
    }

    #[cfg(feature = "stt-openai")]
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }
}

#[cfg(feature = "stt-openai")]
async fn materialize_payload(payload: AudioPayload) -> InferenceResult<(Bytes, String)> {
    match payload {
        AudioPayload::Bytes { data, params } => {
            let name = default_filename(params.format);
            Ok((data, name))
        }
        AudioPayload::Path { path, params } => {
            let data = tokio::fs::read(&path)
                .await
                .map_err(|e| InferenceError::BadRequest {
                    message: format!("openai-stt: read {}: {e}", path.display()),
                })?;
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| default_filename(params.format));
            Ok((Bytes::from(data), name))
        }
        AudioPayload::Url { .. } => Err(OpenAiSttError::Unsupported {
            method: "execute_audio: URL audio payloads not implemented (M6)",
        }
        .into()),
        _ => Err(OpenAiSttError::BadRequest {
            message: "openai-stt: unknown AudioPayload variant".into(),
        }
        .into()),
    }
}

#[cfg(feature = "stt-openai")]
fn default_filename(format: atomr_infer_core::audio::AudioFormat) -> String {
    use atomr_infer_core::audio::AudioFormat;
    match format {
        AudioFormat::Pcm16Le | AudioFormat::Pcm24Le | AudioFormat::PcmF32Le => "audio.pcm".into(),
        AudioFormat::Wav => "audio.wav".into(),
        AudioFormat::OggOpus => "audio.ogg".into(),
        AudioFormat::Mp3 => "audio.mp3".into(),
        AudioFormat::Flac => "audio.flac".into(),
        _ => "audio.bin".into(),
    }
}

#[cfg(feature = "stt-openai")]
fn verbose_to_chunks(
    request_id: String,
    parsed: VerboseResponse,
    _opts: &TranscribeOptions,
) -> Vec<InferenceResult<TranscriptChunk>> {
    // Build a flat WordTiming list (already in ms) and a sorted index
    // so we can attribute words to segments by time-range overlap.
    let words_ms: Vec<WordTiming> = parsed
        .words
        .iter()
        .map(|w| WordTiming {
            text: w.word.clone(),
            ts_start_ms: seconds_to_ms(w.start),
            ts_end_ms: seconds_to_ms(w.end),
            confidence: None,
        })
        .collect();

    if parsed.segments.is_empty() {
        // verbose_json without segments — emit one final chunk for the
        // whole text. Word list (if present) rides on the single chunk.
        return vec![Ok(TranscriptChunk {
            request_id,
            is_final: true,
            text: parsed.text,
            ts_start_ms: words_ms.iter().map(|w| w.ts_start_ms).min().unwrap_or(0),
            ts_end_ms: words_ms.iter().map(|w| w.ts_end_ms).max().unwrap_or(0),
            speaker_id: None,
            words: words_ms,
            usage: None,
        })];
    }

    let last_idx = parsed.segments.len() - 1;
    parsed
        .segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            let ts_start_ms = seconds_to_ms(seg.start);
            let ts_end_ms = seconds_to_ms(seg.end);
            let words: Vec<WordTiming> = words_ms
                .iter()
                .filter(|w| w.ts_start_ms >= ts_start_ms && w.ts_end_ms <= ts_end_ms)
                .cloned()
                .collect();
            Ok(TranscriptChunk {
                request_id: request_id.clone(),
                is_final: i == last_idx,
                text: seg.text.clone(),
                ts_start_ms,
                ts_end_ms,
                speaker_id: None,
                words,
                usage: None,
            })
        })
        .collect()
}

#[cfg(feature = "stt-openai")]
fn seconds_to_ms(s: f32) -> u32 {
    (s * 1_000.0).max(0.0) as u32
}

#[cfg(all(test, feature = "stt-openai"))]
mod tests {
    use super::*;

    #[test]
    fn default_filename_covers_common_formats() {
        use atomr_infer_core::audio::AudioFormat;
        assert_eq!(default_filename(AudioFormat::Wav), "audio.wav");
        assert_eq!(default_filename(AudioFormat::Pcm16Le), "audio.pcm");
        assert_eq!(default_filename(AudioFormat::Mp3), "audio.mp3");
        assert_eq!(default_filename(AudioFormat::Flac), "audio.flac");
        assert_eq!(default_filename(AudioFormat::OggOpus), "audio.ogg");
    }

    #[test]
    fn verbose_to_chunks_splits_per_segment_and_marks_final() {
        let parsed = VerboseResponse {
            text: "hello world".into(),
            segments: vec![
                crate::wire::VerboseSegment {
                    start: 0.0,
                    end: 0.5,
                    text: "hello".into(),
                    avg_logprob: None,
                },
                crate::wire::VerboseSegment {
                    start: 0.5,
                    end: 1.0,
                    text: " world".into(),
                    avg_logprob: None,
                },
            ],
            words: vec![
                crate::wire::VerboseWord {
                    word: "hello".into(),
                    start: 0.0,
                    end: 0.5,
                },
                crate::wire::VerboseWord {
                    word: "world".into(),
                    start: 0.5,
                    end: 1.0,
                },
            ],
        };
        let chunks = verbose_to_chunks("r".into(), parsed, &TranscribeOptions::default());
        assert_eq!(chunks.len(), 2);
        let c0 = chunks[0].as_ref().unwrap();
        let c1 = chunks[1].as_ref().unwrap();
        assert!(!c0.is_final);
        assert!(c1.is_final);
        assert_eq!(c0.text, "hello");
        assert_eq!(c0.ts_start_ms, 0);
        assert_eq!(c0.ts_end_ms, 500);
        assert_eq!(c0.words.len(), 1);
        assert_eq!(c0.words[0].text, "hello");
        assert_eq!(c1.words[0].text, "world");
    }

    #[test]
    fn verbose_to_chunks_without_segments_emits_one_final() {
        let parsed = VerboseResponse {
            text: "hi".into(),
            segments: vec![],
            words: vec![crate::wire::VerboseWord {
                word: "hi".into(),
                start: 0.0,
                end: 0.2,
            }],
        };
        let chunks = verbose_to_chunks("r".into(), parsed, &TranscribeOptions::default());
        assert_eq!(chunks.len(), 1);
        let c = chunks[0].as_ref().unwrap();
        assert!(c.is_final);
        assert_eq!(c.text, "hi");
        assert_eq!(c.words.len(), 1);
        assert_eq!(c.ts_end_ms, 200);
    }
}
