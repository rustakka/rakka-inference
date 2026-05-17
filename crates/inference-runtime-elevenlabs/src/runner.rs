//! `ElevenLabsTtsRunner` — [`SpeechRunner`] implementation against
//! ElevenLabs' HTTPS one-shot endpoint
//! (`POST /v1/text-to-speech/{voice_id}`) and WebSocket streaming
//! endpoint (`/v1/text-to-speech/{voice_id}/stream-input`).
//!
//! Routing rule: when [`SpeechBatch::stream`] is `false`, take the
//! HTTPS path; when `true`, take the WS path. The HTTPS path returns
//! after the full body has been received and re-chunked at
//! `chunk_bytes`. The WS path emits one [`SpeechChunk`] per inbound
//! JSON frame, attaching an [`AlignmentDelta`] (with per-character
//! [`WordTiming`]s) when the provider includes one and the caller
//! asked for it via [`SpeechBatch::emit_alignment`].
//!
//! [`SpeechRunner`]: atomr_infer_core::runner::SpeechRunner
//! [`SpeechBatch::stream`]: atomr_infer_core::audio::SpeechBatch::stream
//! [`SpeechBatch::emit_alignment`]: atomr_infer_core::audio::SpeechBatch::emit_alignment
//! [`SpeechChunk`]: atomr_infer_core::audio::SpeechChunk
//! [`AlignmentDelta`]: atomr_infer_core::audio::AlignmentDelta
//! [`WordTiming`]: atomr_infer_core::audio::WordTiming

#[cfg(feature = "tts-elevenlabs")]
use std::sync::Arc;

use async_trait::async_trait;

use atomr_infer_core::audio::SpeechBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{SessionRebuildCause, SpeechRunHandle, SpeechRunner};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

#[cfg(feature = "tts-elevenlabs")]
use arc_swap::ArcSwap;
#[cfg(feature = "tts-elevenlabs")]
use atomr_infer_core::audio::{
    AlignmentDelta, AudioFormat, AudioParams, AudioPayload, SpeechChunk, VoiceRef, WordTiming,
};
#[cfg(feature = "tts-elevenlabs")]
use atomr_infer_core::deployment::RateLimits;
#[cfg(feature = "tts-elevenlabs")]
use atomr_infer_remote_core::session::SessionSnapshot;
#[cfg(feature = "tts-elevenlabs")]
use base64::Engine;
#[cfg(feature = "tts-elevenlabs")]
use bytes::Bytes;
#[cfg(feature = "tts-elevenlabs")]
use futures::stream::{self, BoxStream, StreamExt};
#[cfg(feature = "tts-elevenlabs")]
use reqwest::header;
#[cfg(feature = "tts-elevenlabs")]
use secrecy::ExposeSecret;
#[cfg(feature = "tts-elevenlabs")]
use tokio::sync::mpsc;
#[cfg(feature = "tts-elevenlabs")]
use tokio_stream::wrappers::ReceiverStream;

#[cfg(feature = "tts-elevenlabs")]
use crate::config::ElevenLabsTtsConfig;
#[cfg(feature = "tts-elevenlabs")]
use crate::error::ElevenLabsError;
#[cfg(feature = "tts-elevenlabs")]
use crate::wire::{
    SpeechRequest, VoiceSettings, WsGenerationConfig, WsInboundFrame, WsInitMessage, WsTextMessage,
};

#[cfg(feature = "tts-elevenlabs")]
use atomr_infer_runtime_ws_core::{Frame as WsFrame, WsClient};

/// `SpeechRunner` implementation against ElevenLabs' HTTPS + WSS TTS
/// surface.
///
/// One instance is reusable across batches; per-batch state lives in
/// the returned [`SpeechRunHandle`].
pub struct ElevenLabsTtsRunner {
    #[cfg(feature = "tts-elevenlabs")]
    config: ElevenLabsTtsConfig,
    #[cfg(feature = "tts-elevenlabs")]
    session: Arc<ArcSwap<SessionSnapshot>>,
    #[cfg(not(feature = "tts-elevenlabs"))]
    _stub: (),
}

#[cfg(feature = "tts-elevenlabs")]
impl ElevenLabsTtsRunner {
    /// Construct a runner. `session` is the shared
    /// `inference-remote-core::SessionSnapshot` swap-cell — when the
    /// session actor rebuilds the credential, this runner's next
    /// request picks up the fresh `xi-api-key` automatically.
    pub fn new(config: ElevenLabsTtsConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> InferenceResult<Self> {
        Ok(Self { config, session })
    }

    /// Borrow the runtime configuration.
    pub fn config(&self) -> &ElevenLabsTtsConfig {
        &self.config
    }

    fn auth_header_value(&self) -> InferenceResult<header::HeaderValue> {
        let snap = self.session.load();
        let token = snap.credential.expose_secret().to_string();
        header::HeaderValue::from_str(&token)
            .map_err(|e| InferenceError::Internal(format!("invalid xi-api-key: {e}")))
    }

    fn resolve_voice_id<'a>(&'a self, voice: &'a VoiceRef) -> InferenceResult<&'a str> {
        match voice {
            VoiceRef::Id(s) | VoiceRef::Named(s) => Ok(s.as_str()),
            VoiceRef::ClonedFrom(_) => {
                // Cloning is a separate POST /v1/voices/add round-trip
                // (see `clone_voice`). The runner does not implicitly
                // upload — callers must clone first, then submit a
                // `VoiceRef::Id(returned_id)` to `speak`.
                if let Some(id) = &self.config.default_voice_id {
                    Ok(id.as_str())
                } else {
                    Err(ElevenLabsError::BadRequest {
                        message: "ClonedFrom requires prior /v1/voices/add; \
                                  pass VoiceRef::Id(new_id) to speak, or set \
                                  ElevenLabsTtsConfig::default_voice_id"
                            .into(),
                    }
                    .into())
                }
            }
            _ => Err(ElevenLabsError::BadRequest {
                message: "unsupported VoiceRef variant".into(),
            }
            .into()),
        }
    }

    fn voice_settings_from_batch(batch: &SpeechBatch) -> Option<VoiceSettings> {
        // Map our generic SynthOptions onto ElevenLabs' subset. We only
        // emit a `voice_settings` object when at least one field is
        // populated — otherwise we let ElevenLabs use the voice's
        // server-side defaults.
        let style = batch.options.pitch_semitones.map(|s| {
            // Clamp to 0..=1: -12 → 0, +12 → 1, 0 → 0.5.
            let normalized = (s + 12.0) / 24.0;
            normalized.clamp(0.0, 1.0)
        });
        if style.is_some() {
            Some(VoiceSettings {
                stability: None,
                similarity_boost: None,
                style,
                use_speaker_boost: None,
            })
        } else {
            None
        }
    }

    /// Clone a voice from a reference audio sample. Returns the voice
    /// id the caller should then pass via [`VoiceRef::Id`] on
    /// subsequent [`SpeechRunner::speak`] calls.
    ///
    /// Uses `POST /v1/voices/add` (multipart form upload). The
    /// reference audio must already be materialised — for `Path` /
    /// `Url` payloads, callers should resolve to bytes before calling.
    pub async fn clone_voice(
        &self,
        name: &str,
        sample: AudioPayload,
        description: Option<&str>,
    ) -> InferenceResult<String> {
        let url = self
            .config
            .add_voice_url()
            .map_err(|e| InferenceError::Internal(format!("voices/add url: {e}")))?;

        let bytes = match sample {
            AudioPayload::Bytes { data, .. } => data,
            AudioPayload::Path { .. } | AudioPayload::Url { .. } => {
                return Err(ElevenLabsError::BadRequest {
                    message: "clone_voice requires AudioPayload::Bytes; resolve \
                              Path/Url to bytes first"
                        .into(),
                }
                .into());
            }
            _ => {
                return Err(ElevenLabsError::BadRequest {
                    message: "unsupported AudioPayload variant for clone_voice".into(),
                }
                .into())
            }
        };

        let mut form = reqwest::multipart::Form::new()
            .text("name", name.to_string())
            .part(
                "files",
                reqwest::multipart::Part::bytes(bytes.to_vec()).file_name("sample.wav"),
            );
        if let Some(desc) = description {
            form = form.text("description", desc.to_string());
        }

        let snap = self.session.load_full();
        let resp = snap
            .client
            .post(url)
            .header("xi-api-key", self.auth_header_value()?)
            .multipart(form)
            .send()
            .await
            .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body_text = resp.text().await.unwrap_or_default();
            return Err(classify_elevenlabs_error(status, None, Some(body_text)));
        }

        #[derive(serde::Deserialize)]
        struct AddVoiceResponse {
            voice_id: String,
        }
        let body: AddVoiceResponse = resp
            .json()
            .await
            .map_err(|e| InferenceError::NetworkError(format!("decode voices/add: {e}")))?;
        Ok(body.voice_id)
    }

    async fn speak_https(&mut self, batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        let voice_id = self.resolve_voice_id(&batch.voice)?.to_string();
        let url = self
            .config
            .speech_url(&voice_id)
            .map_err(|e| InferenceError::Internal(format!("elevenlabs speech url: {e}")))?;

        let format_query = elevenlabs_output_format(batch.options.format)?;
        let mut url = url;
        url.query_pairs_mut().append_pair("output_format", format_query);

        let voice_settings = Self::voice_settings_from_batch(&batch);
        let body = SpeechRequest {
            model_id: batch.model.as_str(),
            text: batch.text.as_str(),
            voice_settings,
        };

        let snap = self.session.load_full();
        let resp = snap
            .client
            .post(url)
            .header("xi-api-key", self.auth_header_value()?)
            .header(header::ACCEPT, "audio/*")
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
            return Err(classify_elevenlabs_error(
                status,
                retry_after.as_deref(),
                body_text,
            ));
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| InferenceError::NetworkError(format!("read elevenlabs body: {e}")))?;

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

    async fn speak_ws(&mut self, batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        let voice_id = self.resolve_voice_id(&batch.voice)?.to_string();
        let url = self
            .config
            .speech_stream_url(&voice_id)
            .map_err(|e| InferenceError::Internal(format!("elevenlabs ws url: {e}")))?;

        let format_query = elevenlabs_output_format(batch.options.format)?;
        let mut url = url;
        url.query_pairs_mut().append_pair("output_format", format_query);

        let (mut tx, mut rx) = WsClient::connect(url.as_str(), self.config.ws_connect_timeout)
            .await
            .map_err(|e| InferenceError::NetworkError(format!("ws connect: {e}")))?;

        let api_key = {
            let snap = self.session.load();
            snap.credential.expose_secret().to_string()
        };

        let voice_settings = Self::voice_settings_from_batch(&batch);
        let init = WsInitMessage {
            text: if batch.text.is_empty() {
                " "
            } else {
                batch.text.as_str()
            },
            model_id: batch.model.as_str(),
            xi_api_key: api_key.as_str(),
            voice_settings,
            generation_config: None::<WsGenerationConfig>,
        };
        let init_json = serde_json::to_string(&init)
            .map_err(|e| InferenceError::Internal(format!("encode ws init: {e}")))?;
        tx.send(WsFrame::Text(init_json))
            .await
            .map_err(|e| InferenceError::NetworkError(format!("ws init send: {e}")))?;

        // Immediately follow with the EOS-style flush frame (empty
        // text) so the server starts emitting audio for the full
        // single-shot input. Callers driving multi-chunk streaming
        // would instead push subsequent `WsTextMessage` frames before
        // flushing — that lives in the realtime crate, not here.
        let flush = WsTextMessage {
            text: "",
            try_trigger_generation: Some(true),
        };
        let flush_json = serde_json::to_string(&flush)
            .map_err(|e| InferenceError::Internal(format!("encode ws flush: {e}")))?;
        tx.send(WsFrame::Text(flush_json))
            .await
            .map_err(|e| InferenceError::NetworkError(format!("ws flush send: {e}")))?;

        let request_id = batch.request_id.clone();
        let params = output_params(batch.options.format);
        let emit_alignment = batch.emit_alignment;

        let (out_tx, out_rx) = mpsc::channel::<InferenceResult<SpeechChunk>>(16);

        tokio::spawn(async move {
            let mut sent_final = false;
            loop {
                match rx.next().await {
                    Ok(Some(WsFrame::Text(text))) => {
                        let parsed: Result<WsInboundFrame, _> = serde_json::from_str(&text);
                        match parsed {
                            Ok(frame) => {
                                let audio_bytes = match &frame.audio {
                                    Some(b64) if !b64.is_empty() => {
                                        match base64::engine::general_purpose::STANDARD.decode(b64) {
                                            Ok(v) => Bytes::from(v),
                                            Err(e) => {
                                                let _ = out_tx
                                                    .send(Err(InferenceError::Internal(format!(
                                                        "ws audio b64: {e}"
                                                    ))))
                                                    .await;
                                                break;
                                            }
                                        }
                                    }
                                    _ => Bytes::new(),
                                };

                                let alignment = if emit_alignment {
                                    frame.alignment.as_ref().map(ws_alignment_to_delta)
                                } else {
                                    None
                                };

                                let is_final = frame.is_final.unwrap_or(false);

                                // Skip silent pings: no audio, no
                                // alignment, not the final frame.
                                if audio_bytes.is_empty() && alignment.is_none() && !is_final {
                                    continue;
                                }

                                if out_tx
                                    .send(Ok(SpeechChunk {
                                        request_id: request_id.clone(),
                                        is_final,
                                        audio_pcm_chunk: audio_bytes,
                                        params,
                                        alignment,
                                        usage: None,
                                    }))
                                    .await
                                    .is_err()
                                {
                                    break;
                                }

                                if is_final {
                                    sent_final = true;
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = out_tx
                                    .send(Err(InferenceError::Internal(format!("ws frame decode: {e}"))))
                                    .await;
                                break;
                            }
                        }
                    }
                    Ok(Some(WsFrame::Binary(_) | WsFrame::Ping(_) | WsFrame::Pong(_))) => continue,
                    Ok(Some(WsFrame::Close { code, reason })) => {
                        if code != 1000 {
                            let _ = out_tx
                                .send(Err(InferenceError::NetworkError(format!(
                                    "ws closed code={code} reason={reason}"
                                ))))
                                .await;
                        }
                        break;
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = out_tx
                            .send(Err(InferenceError::NetworkError(format!("ws recv: {e}"))))
                            .await;
                        break;
                    }
                }
            }

            if !sent_final {
                let _ = out_tx
                    .send(Ok(SpeechChunk {
                        request_id: request_id.clone(),
                        is_final: true,
                        audio_pcm_chunk: Bytes::new(),
                        params,
                        alignment: None,
                        usage: None,
                    }))
                    .await;
            }

            let _ = tx.close(1000, "done").await;
        });

        let s: BoxStream<'static, InferenceResult<SpeechChunk>> = ReceiverStream::new(out_rx).boxed();
        Ok(SpeechRunHandle::streaming(s))
    }
}

#[cfg(not(feature = "tts-elevenlabs"))]
impl ElevenLabsTtsRunner {
    /// Stub constructor — accepts no arguments so callers can still
    /// link without pulling the feature in.
    pub fn new_stub() -> Self {
        Self { _stub: () }
    }
}

#[async_trait]
impl SpeechRunner for ElevenLabsTtsRunner {
    #[cfg(feature = "tts-elevenlabs")]
    async fn speak(&mut self, batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        if batch.stream {
            self.speak_ws(batch).await
        } else {
            self.speak_https(batch).await
        }
    }

    #[cfg(not(feature = "tts-elevenlabs"))]
    async fn speak(&mut self, _batch: SpeechBatch) -> InferenceResult<SpeechRunHandle> {
        Err(InferenceError::Internal(
            "tts-elevenlabs feature disabled at build time".into(),
        ))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::TextToSpeech
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork {
            provider: ProviderKind::ElevenLabs,
        }
    }

    #[cfg(feature = "tts-elevenlabs")]
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }
}

/// Map ElevenLabs HTTP status codes onto [`InferenceError`].
///
/// The shape mirrors `atomr_infer_runtime_openai::error::classify_openai_error`
/// — `429` is provider-tagged rate-limit, `5xx` is upstream, `4xx`
/// other than `429` is bad-request.
#[cfg(feature = "tts-elevenlabs")]
fn classify_elevenlabs_error(status: u16, retry_after: Option<&str>, body: Option<String>) -> InferenceError {
    use std::time::Duration;
    let body = body.unwrap_or_default();
    match status {
        429 => {
            let retry = retry_after
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs);
            InferenceError::RateLimited {
                provider: ProviderKind::ElevenLabs,
                retry_after: retry,
            }
        }
        401 => InferenceError::Unauthorized {
            message: if body.is_empty() {
                format!("elevenlabs auth failed (status {status})")
            } else {
                body
            },
        },
        403 => InferenceError::Forbidden {
            message: if body.is_empty() {
                format!("elevenlabs forbidden (status {status})")
            } else {
                body
            },
        },
        400..=499 => InferenceError::BadRequest {
            message: if body.is_empty() {
                format!("elevenlabs rejected request (status {status})")
            } else {
                body
            },
        },
        500..=599 => InferenceError::ServerError {
            status,
            body: if body.is_empty() { None } else { Some(body) },
        },
        _ => InferenceError::Internal(format!("elevenlabs returned unexpected status {status}: {body}")),
    }
}

/// Translate a caller-requested [`AudioFormat`] into ElevenLabs'
/// `output_format` query string. The ElevenLabs API uses
/// dash-separated names like `pcm_24000`, `mp3_44100_128`,
/// `opus_48000_64`. Returns `Err` if the requested format has no
/// ElevenLabs equivalent.
#[cfg(feature = "tts-elevenlabs")]
fn elevenlabs_output_format(format: Option<AudioFormat>) -> InferenceResult<&'static str> {
    Ok(match format.unwrap_or(AudioFormat::Mp3) {
        AudioFormat::Pcm16Le => "pcm_24000",
        AudioFormat::Pcm24Le => "pcm_24000",
        AudioFormat::PcmF32Le => {
            return Err(ElevenLabsError::UnsupportedFormat {
                message: "ElevenLabs does not emit PcmF32Le; choose Pcm16Le, Mp3, or OggOpus".into(),
            }
            .into());
        }
        AudioFormat::Mp3 => "mp3_44100_128",
        AudioFormat::OggOpus => "opus_48000_64",
        AudioFormat::Flac | AudioFormat::Wav => {
            return Err(ElevenLabsError::UnsupportedFormat {
                message: "ElevenLabs does not support Flac/Wav output; choose Mp3, Pcm16Le, or OggOpus"
                    .into(),
            }
            .into());
        }
        _ => {
            return Err(ElevenLabsError::UnsupportedFormat {
                message: "unsupported AudioFormat variant".into(),
            }
            .into())
        }
    })
}

/// Map the caller's requested format to [`AudioParams`] attached to
/// emitted [`SpeechChunk`]s. ElevenLabs' PCM is 24 kHz mono signed
/// 16-bit LE; container formats keep [`AudioFormat`] tagged but the
/// inner bytes carry their own headers.
#[cfg(feature = "tts-elevenlabs")]
fn output_params(requested: Option<AudioFormat>) -> AudioParams {
    let format = requested.unwrap_or(AudioFormat::Mp3);
    let sample_rate_hz = match format {
        AudioFormat::Pcm16Le | AudioFormat::Pcm24Le => 24_000,
        AudioFormat::OggOpus => 48_000,
        _ => 44_100,
    };
    AudioParams::new(sample_rate_hz, 1, format)
}

/// Translate the wire-level `WsAlignment` (char-indexed) into our
/// shared [`AlignmentDelta`] (word-indexed). ElevenLabs emits one
/// entry per character; we expose them as single-character
/// [`WordTiming`] entries so downstream consumers get the same
/// surface as STT word timings.
#[cfg(feature = "tts-elevenlabs")]
fn ws_alignment_to_delta(a: &crate::wire::WsAlignment) -> AlignmentDelta {
    let n = a
        .chars
        .len()
        .min(a.char_start_times_ms.len())
        .min(a.char_durations_ms.len());
    let mut words = Vec::with_capacity(n);
    for i in 0..n {
        let start = a.char_start_times_ms[i];
        let end = start.saturating_add(a.char_durations_ms[i]);
        words.push(WordTiming {
            text: a.chars[i].clone(),
            ts_start_ms: start,
            ts_end_ms: end,
            confidence: None,
        });
    }
    AlignmentDelta {
        words,
        visemes: Vec::new(),
    }
}

#[cfg(all(test, feature = "tts-elevenlabs"))]
mod tests {
    use super::*;
    use crate::wire::WsAlignment;

    #[test]
    fn classify_429_carries_provider_and_retry_after() {
        let e = classify_elevenlabs_error(429, Some("2"), Some("rate limited".into()));
        match e {
            InferenceError::RateLimited {
                provider,
                retry_after,
            } => {
                assert_eq!(provider, ProviderKind::ElevenLabs);
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(2)));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[test]
    fn classify_401_is_unauthorized() {
        let e = classify_elevenlabs_error(401, None, Some("nope".into()));
        match e {
            InferenceError::Unauthorized { message } => assert!(message.contains("nope")),
            other => panic!("expected Unauthorized, got {other:?}"),
        }
    }

    #[test]
    fn classify_500_is_server_error() {
        let e = classify_elevenlabs_error(503, None, None);
        match e {
            InferenceError::ServerError { status, body } => {
                assert_eq!(status, 503);
                assert!(body.is_none());
            }
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[test]
    fn output_format_maps_known_variants() {
        assert_eq!(
            elevenlabs_output_format(Some(AudioFormat::Mp3)).unwrap(),
            "mp3_44100_128"
        );
        assert_eq!(
            elevenlabs_output_format(Some(AudioFormat::Pcm16Le)).unwrap(),
            "pcm_24000"
        );
        assert_eq!(
            elevenlabs_output_format(Some(AudioFormat::OggOpus)).unwrap(),
            "opus_48000_64"
        );
    }

    #[test]
    fn output_format_rejects_unsupported() {
        assert!(matches!(
            elevenlabs_output_format(Some(AudioFormat::Wav)),
            Err(InferenceError::UnsupportedAudioFormat { .. })
        ));
        assert!(matches!(
            elevenlabs_output_format(Some(AudioFormat::Flac)),
            Err(InferenceError::UnsupportedAudioFormat { .. })
        ));
        assert!(matches!(
            elevenlabs_output_format(Some(AudioFormat::PcmF32Le)),
            Err(InferenceError::UnsupportedAudioFormat { .. })
        ));
    }

    #[test]
    fn ws_alignment_converts_to_word_timings() {
        let a = WsAlignment {
            chars: vec!["h".into(), "i".into()],
            char_start_times_ms: vec![0, 50],
            char_durations_ms: vec![50, 60],
        };
        let delta = ws_alignment_to_delta(&a);
        assert_eq!(delta.words.len(), 2);
        assert_eq!(delta.words[0].text, "h");
        assert_eq!(delta.words[0].ts_start_ms, 0);
        assert_eq!(delta.words[0].ts_end_ms, 50);
        assert_eq!(delta.words[1].ts_start_ms, 50);
        assert_eq!(delta.words[1].ts_end_ms, 110);
        assert!(delta.visemes.is_empty());
    }
}
