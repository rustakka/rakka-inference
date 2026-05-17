//! `DeepgramSttRunner` — [`AudioRunner`] implementation against the
//! Deepgram WSS streaming endpoint
//! (`wss://api.deepgram.com/v1/listen`).
//!
//! The runner opens a WebSocket with `Authorization: Token <key>` on
//! the upgrade request, pumps the caller's audio bytes uplink as WS
//! binary frames, and emits one
//! [`atomr_infer_core::audio::TranscriptChunk`] per inbound `Results`
//! envelope on the downlink. When
//! [`atomr_infer_core::audio::TranscribeOptions::interim_results`] is
//! `false`, the runner filters out interim (non-final) chunks
//! provider-side.
//!
//! [`AudioRunner`]: atomr_infer_core::runner::AudioRunner

#[cfg(feature = "stt-deepgram")]
use std::sync::Arc;

use async_trait::async_trait;

use atomr_infer_core::audio::AudioBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{AudioRunHandle, AudioRunner, SessionRebuildCause};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

#[cfg(feature = "stt-deepgram")]
use arc_swap::ArcSwap;
#[cfg(feature = "stt-deepgram")]
use atomr_infer_core::audio::{
    AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscriptChunk, WordTiming,
};
#[cfg(feature = "stt-deepgram")]
use atomr_infer_core::deployment::RateLimits;
#[cfg(feature = "stt-deepgram")]
use atomr_infer_remote_core::session::SessionSnapshot;
#[cfg(feature = "stt-deepgram")]
use bytes::Bytes;
#[cfg(feature = "stt-deepgram")]
use futures::stream::{BoxStream, StreamExt};
#[cfg(feature = "stt-deepgram")]
use secrecy::ExposeSecret;
#[cfg(feature = "stt-deepgram")]
use tokio::sync::mpsc;
#[cfg(feature = "stt-deepgram")]
use tokio_stream::wrappers::ReceiverStream;

#[cfg(feature = "stt-deepgram")]
use crate::config::DeepgramSttConfig;
#[cfg(feature = "stt-deepgram")]
use crate::error::DeepgramError;
#[cfg(feature = "stt-deepgram")]
use crate::wire::{CloseStream, InboundEnvelope, ResultsEnvelope};

#[cfg(feature = "stt-deepgram")]
use atomr_infer_runtime_ws_core::{Frame as WsFrame, WsClient};

/// `AudioRunner` implementation against Deepgram's WSS streaming
/// endpoint.
///
/// One instance is reusable across batches; per-batch state lives in
/// the returned [`AudioRunHandle`].
pub struct DeepgramSttRunner {
    #[cfg(feature = "stt-deepgram")]
    config: DeepgramSttConfig,
    #[cfg(feature = "stt-deepgram")]
    session: Arc<ArcSwap<SessionSnapshot>>,
    #[cfg(not(feature = "stt-deepgram"))]
    _stub: (),
}

#[cfg(feature = "stt-deepgram")]
impl DeepgramSttRunner {
    /// Construct a runner. `session` carries the shared
    /// `inference-remote-core` snapshot — when the session actor
    /// rotates credentials, the next `execute_audio` call picks up
    /// the fresh API key automatically.
    pub fn new(config: DeepgramSttConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> InferenceResult<Self> {
        Ok(Self { config, session })
    }

    /// Borrow the runtime configuration.
    pub fn config(&self) -> &DeepgramSttConfig {
        &self.config
    }
}

#[cfg(not(feature = "stt-deepgram"))]
impl DeepgramSttRunner {
    /// Stub constructor — accepts no arguments so callers can still
    /// link without pulling the feature in.
    pub fn new_stub() -> Self {
        Self { _stub: () }
    }
}

#[async_trait]
impl AudioRunner for DeepgramSttRunner {
    #[cfg(feature = "stt-deepgram")]
    async fn execute_audio(&mut self, batch: AudioBatch) -> InferenceResult<AudioRunHandle> {
        let opts = match &batch.options {
            AudioOptions::Transcribe(t) => t.clone(),
            AudioOptions::Audio2Face(_) => {
                return Err(DeepgramError::BadRequest {
                    message: "deepgram: AudioOptions::Audio2Face is not supported".into(),
                }
                .into());
            }
            _ => {
                return Err(DeepgramError::BadRequest {
                    message: "deepgram: unknown AudioOptions variant".into(),
                }
                .into());
            }
        };

        let (params, input_stream) = open_audio_input(batch.input).await?;
        let encoding = deepgram_encoding(params.format)?;

        let mut url = self
            .config
            .listen_url()
            .map_err(|e| InferenceError::Internal(format!("deepgram listen url: {e}")))?;
        {
            let mut q = url.query_pairs_mut();
            q.append_pair("model", &batch.model);
            q.append_pair("encoding", encoding);
            q.append_pair("sample_rate", &params.sample_rate_hz.to_string());
            q.append_pair("channels", &params.channels.to_string());
            q.append_pair(
                "interim_results",
                if opts.interim_results { "true" } else { "false" },
            );
            if opts.diarize {
                q.append_pair("diarize", "true");
            }
            if self.config.smart_format {
                q.append_pair("smart_format", "true");
            }
            if let Some(lang) = &opts.language {
                q.append_pair("language", lang.as_str());
            }
            // Deepgram's endpointing is independent of `interim_results`;
            // we always ask for it so `speech_final` arrives.
            q.append_pair("endpointing", "300");
        }

        let api_key = {
            let snap = self.session.load();
            snap.credential.expose_secret().to_string()
        };
        let auth_value = format!("Token {api_key}");
        let headers: &[(&str, &str)] = &[("Authorization", auth_value.as_str())];

        let (mut tx, mut rx) =
            WsClient::connect_with_headers(url.as_str(), headers, self.config.ws_connect_timeout)
                .await
                .map_err(|e| InferenceError::NetworkError(format!("deepgram ws connect: {e}")))?;

        let request_id = batch.request_id.clone();
        let emit_interim = opts.interim_results;
        let want_diarize = opts.diarize;
        let want_words = opts.word_timestamps;

        let (out_tx, out_rx) = mpsc::channel::<InferenceResult<TranscriptChunk>>(16);

        // Uplink task: drain `input_stream` and send each chunk as a
        // binary WS frame. On end-of-stream, send the JSON
        // `CloseStream` flush marker so Deepgram emits the final
        // transcript and tears the connection down cleanly.
        let mut input_stream = input_stream;
        let uplink_err_tx = out_tx.clone();
        let uplink_close_handle = tokio::spawn(async move {
            while let Some(chunk) = input_stream.recv().await {
                if chunk.is_empty() {
                    continue;
                }
                if let Err(e) = tx.send(WsFrame::Binary(chunk)).await {
                    let _ = uplink_err_tx
                        .send(Err(InferenceError::NetworkError(format!(
                            "deepgram ws send: {e}"
                        ))))
                        .await;
                    return;
                }
            }
            let close_json = serde_json::to_string(&CloseStream::new())
                .unwrap_or_else(|_| r#"{"type":"CloseStream"}"#.to_string());
            let _ = tx.send(WsFrame::Text(close_json)).await;
            // Don't actively close the socket here; let the downlink
            // task observe Deepgram's own close after the final
            // transcript flushes.
        });

        // Downlink task: decode Results envelopes into TranscriptChunks
        // and surface them through `out_tx`. The last chunk before the
        // socket closes is rewritten to `is_final = true` if Deepgram
        // didn't already mark one (e.g. very short utterances may not
        // carry `speech_final`).
        tokio::spawn(async move {
            let mut last_idx: Option<u64> = None;
            let mut produced: Vec<TranscriptChunk> = Vec::new();
            loop {
                match rx.next().await {
                    Ok(Some(WsFrame::Text(text))) => {
                        let parsed: Result<InboundEnvelope, _> = serde_json::from_str(&text);
                        match parsed {
                            Ok(InboundEnvelope::Results(r)) => {
                                if !emit_interim && !r.is_final {
                                    continue;
                                }
                                let chunk = results_to_chunk(&request_id, r, want_diarize, want_words);
                                if chunk.is_final {
                                    last_idx = Some(produced.len() as u64);
                                }
                                produced.push(chunk);
                            }
                            Ok(InboundEnvelope::Metadata(_) | InboundEnvelope::Other) => continue,
                            Err(e) => {
                                let _ = out_tx
                                    .send(Err(InferenceError::Internal(format!(
                                        "deepgram envelope decode: {e}"
                                    ))))
                                    .await;
                                break;
                            }
                        }
                    }
                    Ok(Some(WsFrame::Binary(_) | WsFrame::Ping(_) | WsFrame::Pong(_))) => continue,
                    Ok(Some(WsFrame::Close { code, reason })) => {
                        if code != 1000 && code != 1005 && code != 1006 {
                            let _ = out_tx
                                .send(Err(InferenceError::NetworkError(format!(
                                    "deepgram closed code={code} reason={reason}"
                                ))))
                                .await;
                            return;
                        }
                        break;
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = out_tx
                            .send(Err(InferenceError::NetworkError(format!(
                                "deepgram ws recv: {e}"
                            ))))
                            .await;
                        return;
                    }
                }
            }

            // If we never observed `speech_final`, promote the last
            // produced chunk to terminal. Otherwise, ensure there's
            // exactly one terminal chunk — if Deepgram already marked
            // one, leave it; if not and we produced none, emit a
            // synthetic empty terminal.
            if last_idx.is_none() {
                if let Some(last) = produced.last_mut() {
                    last.is_final = true;
                } else {
                    produced.push(TranscriptChunk {
                        request_id: request_id.clone(),
                        is_final: true,
                        text: String::new(),
                        ts_start_ms: 0,
                        ts_end_ms: 0,
                        speaker_id: None,
                        words: Vec::new(),
                        usage: None,
                    });
                }
            }
            for chunk in produced {
                if out_tx.send(Ok(chunk)).await.is_err() {
                    break;
                }
            }
        });

        // Surface the uplink task's lifetime so test panics propagate.
        drop(uplink_close_handle);

        let s: BoxStream<'static, InferenceResult<TranscriptChunk>> = ReceiverStream::new(out_rx).boxed();
        Ok(AudioRunHandle::streaming(s))
    }

    #[cfg(not(feature = "stt-deepgram"))]
    async fn execute_audio(&mut self, _batch: AudioBatch) -> InferenceResult<AudioRunHandle> {
        Err(InferenceError::Internal(
            "stt-deepgram feature disabled at build time".into(),
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
            provider: ProviderKind::Deepgram,
        }
    }

    #[cfg(feature = "stt-deepgram")]
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }
}

/// Open the caller's [`AudioInput`] as a uniform mpsc receiver of
/// [`Bytes`] chunks plus the underlying [`AudioParams`]. For
/// `AudioInput::Static` we re-chunk the payload into 4 KiB binary
/// frames (Deepgram recommends 50–250 ms chunks; 4 KiB @ 16 kHz mono
/// PCM16 ≈ 128 ms).
#[cfg(feature = "stt-deepgram")]
async fn open_audio_input(input: AudioInput) -> InferenceResult<(AudioParams, mpsc::Receiver<Bytes>)> {
    const STATIC_CHUNK_BYTES: usize = 4_096;
    match input {
        AudioInput::Static(payload) => {
            let (bytes, params) = materialize_payload(payload).await?;
            let (tx, rx) = mpsc::channel::<Bytes>(8);
            tokio::spawn(async move {
                let mut offset = 0usize;
                let total = bytes.len();
                while offset < total {
                    let end = (offset + STATIC_CHUNK_BYTES).min(total);
                    let slice = bytes.slice(offset..end);
                    if tx.send(slice).await.is_err() {
                        return;
                    }
                    offset = end;
                }
            });
            Ok((params, rx))
        }
        AudioInput::Stream { params, rx } => Ok((params, rx)),
    }
}

#[cfg(feature = "stt-deepgram")]
async fn materialize_payload(payload: AudioPayload) -> InferenceResult<(Bytes, AudioParams)> {
    match payload {
        AudioPayload::Bytes { data, params } => Ok((data, params)),
        AudioPayload::Path { path, params } => {
            let data = tokio::fs::read(&path)
                .await
                .map_err(|e| InferenceError::BadRequest {
                    message: format!("deepgram: read {}: {e}", path.display()),
                })?;
            Ok((Bytes::from(data), params))
        }
        AudioPayload::Url { .. } => Err(DeepgramError::Unsupported {
            method: "execute_audio: URL audio payloads (use Stream or Bytes)",
        }
        .into()),
        _ => Err(DeepgramError::BadRequest {
            message: "deepgram: unknown AudioPayload variant".into(),
        }
        .into()),
    }
}

/// Map our [`AudioFormat`] onto Deepgram's `encoding` query parameter.
#[cfg(feature = "stt-deepgram")]
fn deepgram_encoding(format: AudioFormat) -> InferenceResult<&'static str> {
    Ok(match format {
        AudioFormat::Pcm16Le => "linear16",
        AudioFormat::PcmF32Le => "linear32",
        AudioFormat::OggOpus => "opus",
        AudioFormat::Flac => "flac",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Wav => "wav",
        AudioFormat::Pcm24Le => {
            return Err(DeepgramError::UnsupportedFormat {
                message: "Deepgram does not accept 24-bit PCM directly; resample to 16-bit".into(),
            }
            .into());
        }
        _ => {
            return Err(DeepgramError::UnsupportedFormat {
                message: "unsupported AudioFormat variant".into(),
            }
            .into());
        }
    })
}

/// Convert a [`ResultsEnvelope`] into the project's
/// [`TranscriptChunk`].
///
/// - `text` comes from the first alternative's transcript (or its
///   `punctuated_word`s if `smart_format` was on — we just take the
///   alternative's `.transcript` field since Deepgram already
///   concatenates the punctuated form there).
/// - `ts_start_ms` / `ts_end_ms` come from `start` / `start + duration`.
/// - `is_final` follows `speech_final` (utterance-final), not
///   `is_final` (segment-final) — multi-segment utterances would
///   otherwise emit multiple "finals" per turn.
/// - `speaker_id` is the first word's `speaker` field stringified
///   when diarization is on.
/// - `words` is converted from `DeepgramWord` to [`WordTiming`] when
///   the caller asked for `word_timestamps`.
#[cfg(feature = "stt-deepgram")]
fn results_to_chunk(
    request_id: &str,
    r: ResultsEnvelope,
    want_diarize: bool,
    want_words: bool,
) -> TranscriptChunk {
    let alt =
        r.channel
            .alternatives
            .into_iter()
            .next()
            .unwrap_or_else(|| crate::wire::TranscriptAlternative {
                transcript: String::new(),
                confidence: None,
                words: Vec::new(),
            });
    let speaker_id = if want_diarize {
        alt.words.iter().find_map(|w| w.speaker.map(|s| s.to_string()))
    } else {
        None
    };
    let words = if want_words {
        alt.words
            .iter()
            .map(|w| WordTiming {
                text: w.punctuated_word.clone().unwrap_or_else(|| w.word.clone()),
                ts_start_ms: seconds_to_ms(w.start),
                ts_end_ms: seconds_to_ms(w.end),
                confidence: w.confidence,
            })
            .collect()
    } else {
        Vec::new()
    };
    TranscriptChunk {
        request_id: request_id.to_string(),
        is_final: r.speech_final,
        text: alt.transcript,
        ts_start_ms: seconds_to_ms(r.start),
        ts_end_ms: seconds_to_ms(r.start + r.duration),
        speaker_id,
        words,
        usage: None,
    }
}

#[cfg(feature = "stt-deepgram")]
fn seconds_to_ms(s: f32) -> u32 {
    (s * 1_000.0).max(0.0) as u32
}

#[cfg(all(test, feature = "stt-deepgram"))]
mod tests {
    use super::*;
    use crate::wire::{DeepgramWord, ResultsChannel, TranscriptAlternative};

    #[test]
    fn encoding_maps_known_variants() {
        assert_eq!(deepgram_encoding(AudioFormat::Pcm16Le).unwrap(), "linear16");
        assert_eq!(deepgram_encoding(AudioFormat::OggOpus).unwrap(), "opus");
        assert_eq!(deepgram_encoding(AudioFormat::Mp3).unwrap(), "mp3");
        assert_eq!(deepgram_encoding(AudioFormat::Flac).unwrap(), "flac");
    }

    #[test]
    fn encoding_rejects_pcm24() {
        assert!(matches!(
            deepgram_encoding(AudioFormat::Pcm24Le),
            Err(InferenceError::UnsupportedAudioFormat { .. })
        ));
    }

    #[test]
    fn results_to_chunk_no_diarize_no_words() {
        let r = ResultsEnvelope {
            start: 0.0,
            duration: 1.0,
            is_final: true,
            speech_final: true,
            channel: ResultsChannel {
                alternatives: vec![TranscriptAlternative {
                    transcript: "hello".into(),
                    confidence: Some(0.95),
                    words: vec![DeepgramWord {
                        word: "hello".into(),
                        punctuated_word: None,
                        start: 0.0,
                        end: 0.5,
                        confidence: Some(0.95),
                        speaker: Some(2),
                    }],
                }],
            },
        };
        let c = results_to_chunk("req", r, false, false);
        assert_eq!(c.request_id, "req");
        assert!(c.is_final);
        assert_eq!(c.text, "hello");
        assert_eq!(c.ts_end_ms, 1_000);
        assert!(c.speaker_id.is_none());
        assert!(c.words.is_empty());
    }

    #[test]
    fn results_to_chunk_with_diarize_and_words() {
        let r = ResultsEnvelope {
            start: 1.0,
            duration: 2.0,
            is_final: true,
            speech_final: false,
            channel: ResultsChannel {
                alternatives: vec![TranscriptAlternative {
                    transcript: "alpha beta".into(),
                    confidence: None,
                    words: vec![
                        DeepgramWord {
                            word: "alpha".into(),
                            punctuated_word: Some("Alpha".into()),
                            start: 1.0,
                            end: 1.5,
                            confidence: Some(0.9),
                            speaker: Some(0),
                        },
                        DeepgramWord {
                            word: "beta".into(),
                            punctuated_word: None,
                            start: 2.0,
                            end: 2.8,
                            confidence: Some(0.8),
                            speaker: Some(1),
                        },
                    ],
                }],
            },
        };
        let c = results_to_chunk("r", r, true, true);
        assert!(!c.is_final); // speech_final was false
        assert_eq!(c.ts_start_ms, 1_000);
        assert_eq!(c.ts_end_ms, 3_000);
        assert_eq!(c.speaker_id.as_deref(), Some("0"));
        assert_eq!(c.words.len(), 2);
        assert_eq!(c.words[0].text, "Alpha");
        assert_eq!(c.words[0].ts_start_ms, 1_000);
        assert_eq!(c.words[1].text, "beta");
        assert_eq!(c.words[1].confidence, Some(0.8));
    }

    #[test]
    fn results_to_chunk_handles_missing_alternative() {
        let r = ResultsEnvelope {
            start: 0.0,
            duration: 0.0,
            is_final: false,
            speech_final: false,
            channel: ResultsChannel { alternatives: vec![] },
        };
        let c = results_to_chunk("r", r, false, false);
        assert_eq!(c.text, "");
        assert!(!c.is_final);
    }
}
