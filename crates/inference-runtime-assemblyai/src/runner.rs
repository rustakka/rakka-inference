//! `AssemblyAiSttRunner` — [`AudioRunner`] implementation against the
//! AssemblyAI Universal-Streaming v3 endpoint
//! (`wss://streaming.assemblyai.com/v3/ws`).
//!
//! The runner opens a WebSocket with `Authorization: <key>` on the
//! upgrade request, pumps the caller's audio bytes uplink as WS
//! binary frames, and emits one
//! [`atomr_infer_core::audio::TranscriptChunk`] per inbound `Turn`
//! envelope on the downlink. When
//! [`atomr_infer_core::audio::TranscribeOptions::interim_results`] is
//! `false`, the runner filters out non-`end_of_turn` updates
//! provider-side.
//!
//! Contrast with Deepgram: AssemblyAI v3 delivers exactly one
//! `end_of_turn=true` update per spoken turn — there is no
//! segment-final vs utterance-final distinction at the wire level.
//!
//! [`AudioRunner`]: atomr_infer_core::runner::AudioRunner

#[cfg(feature = "stt-assemblyai")]
use std::sync::Arc;

use async_trait::async_trait;

use atomr_infer_core::audio::AudioBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{AudioRunHandle, AudioRunner, SessionRebuildCause};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

#[cfg(feature = "stt-assemblyai")]
use arc_swap::ArcSwap;
#[cfg(feature = "stt-assemblyai")]
use atomr_infer_core::audio::{
    AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscriptChunk, WordTiming,
};
#[cfg(feature = "stt-assemblyai")]
use atomr_infer_core::deployment::RateLimits;
#[cfg(feature = "stt-assemblyai")]
use atomr_infer_remote_core::session::SessionSnapshot;
#[cfg(feature = "stt-assemblyai")]
use bytes::Bytes;
#[cfg(feature = "stt-assemblyai")]
use futures::stream::{BoxStream, StreamExt};
#[cfg(feature = "stt-assemblyai")]
use secrecy::ExposeSecret;
#[cfg(feature = "stt-assemblyai")]
use tokio::sync::mpsc;
#[cfg(feature = "stt-assemblyai")]
use tokio_stream::wrappers::ReceiverStream;

#[cfg(feature = "stt-assemblyai")]
use crate::config::AssemblyAiSttConfig;
#[cfg(feature = "stt-assemblyai")]
use crate::error::AssemblyAiError;
#[cfg(feature = "stt-assemblyai")]
use crate::wire::{InboundEnvelope, Terminate, TurnEnvelope};

#[cfg(feature = "stt-assemblyai")]
use atomr_infer_runtime_ws_core::{Frame as WsFrame, WsClient};

/// `AudioRunner` implementation against AssemblyAI's WSS streaming
/// endpoint.
///
/// One instance is reusable across batches; per-batch state lives in
/// the returned [`AudioRunHandle`].
pub struct AssemblyAiSttRunner {
    #[cfg(feature = "stt-assemblyai")]
    config: AssemblyAiSttConfig,
    #[cfg(feature = "stt-assemblyai")]
    session: Arc<ArcSwap<SessionSnapshot>>,
    #[cfg(not(feature = "stt-assemblyai"))]
    _stub: (),
}

#[cfg(feature = "stt-assemblyai")]
impl AssemblyAiSttRunner {
    /// Construct a runner. `session` carries the shared
    /// `inference-remote-core` snapshot — when the session actor
    /// rotates credentials, the next `execute_audio` call picks up
    /// the fresh API key automatically.
    pub fn new(config: AssemblyAiSttConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> InferenceResult<Self> {
        Ok(Self { config, session })
    }

    /// Borrow the runtime configuration.
    pub fn config(&self) -> &AssemblyAiSttConfig {
        &self.config
    }
}

#[cfg(not(feature = "stt-assemblyai"))]
impl AssemblyAiSttRunner {
    /// Stub constructor — accepts no arguments so callers can still
    /// link without pulling the feature in.
    pub fn new_stub() -> Self {
        Self { _stub: () }
    }
}

#[async_trait]
impl AudioRunner for AssemblyAiSttRunner {
    #[cfg(feature = "stt-assemblyai")]
    async fn execute_audio(&mut self, batch: AudioBatch) -> InferenceResult<AudioRunHandle> {
        let opts = match &batch.options {
            AudioOptions::Transcribe(t) => t.clone(),
            AudioOptions::Audio2Face(_) => {
                return Err(AssemblyAiError::BadRequest {
                    message: "assemblyai: AudioOptions::Audio2Face is not supported".into(),
                }
                .into());
            }
            _ => {
                return Err(AssemblyAiError::BadRequest {
                    message: "assemblyai: unknown AudioOptions variant".into(),
                }
                .into());
            }
        };

        let (params, input_stream) = open_audio_input(batch.input).await?;
        // AssemblyAI v3 only accepts 16-bit PCM mono.
        assert_format_supported(params.format)?;
        if params.channels != 1 {
            return Err(AssemblyAiError::UnsupportedFormat {
                message: format!(
                    "assemblyai: only mono audio is supported (got {} channels)",
                    params.channels
                ),
            }
            .into());
        }

        let mut url = self
            .config
            .listen_url()
            .map_err(|e| InferenceError::Internal(format!("assemblyai listen url: {e}")))?;
        {
            let mut q = url.query_pairs_mut();
            q.append_pair("sample_rate", &params.sample_rate_hz.to_string());
            if self.config.format_turns {
                q.append_pair("format_turns", "true");
            }
        }

        let api_key = {
            let snap = self.session.load();
            snap.credential.expose_secret().to_string()
        };
        let headers: &[(&str, &str)] = &[("Authorization", api_key.as_str())];

        let (mut tx, mut rx) =
            WsClient::connect_with_headers(url.as_str(), headers, self.config.ws_connect_timeout)
                .await
                .map_err(|e| InferenceError::NetworkError(format!("assemblyai ws connect: {e}")))?;

        let request_id = batch.request_id.clone();
        let emit_interim = opts.interim_results;
        let want_words = opts.word_timestamps;
        // AssemblyAI v3 doesn't expose explicit speaker labels on
        // Streaming (that lives in their async API); ignore `diarize`.
        let _ = opts.diarize;

        let (out_tx, out_rx) = mpsc::channel::<InferenceResult<TranscriptChunk>>(16);

        // Uplink task: drain `input_stream` and send each chunk as a
        // binary WS frame. On end-of-stream, send the JSON `Terminate`
        // flush marker so AssemblyAI emits the final turn and tears
        // the connection down cleanly.
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
                            "assemblyai ws send: {e}"
                        ))))
                        .await;
                    return;
                }
            }
            let close_json = serde_json::to_string(&Terminate::new())
                .unwrap_or_else(|_| r#"{"type":"Terminate"}"#.to_string());
            let _ = tx.send(WsFrame::Text(close_json)).await;
            // Don't actively close the socket here; let the downlink
            // task observe AssemblyAI's own close after the final turn
            // flushes.
        });

        // Downlink task: decode Turn envelopes into TranscriptChunks
        // and surface them through `out_tx`. The last chunk before the
        // socket closes is rewritten to `is_final = true` if AssemblyAI
        // didn't already mark one (very short utterances may not
        // produce an `end_of_turn=true` update before close).
        tokio::spawn(async move {
            let mut produced: Vec<TranscriptChunk> = Vec::new();
            let mut saw_final = false;
            loop {
                match rx.next().await {
                    Ok(Some(WsFrame::Text(text))) => {
                        let parsed: Result<InboundEnvelope, _> = serde_json::from_str(&text);
                        match parsed {
                            Ok(InboundEnvelope::Turn(t)) => {
                                if !emit_interim && !t.end_of_turn {
                                    continue;
                                }
                                let chunk = turn_to_chunk(&request_id, t, want_words);
                                if chunk.is_final {
                                    saw_final = true;
                                }
                                produced.push(chunk);
                            }
                            Ok(
                                InboundEnvelope::Begin(_)
                                | InboundEnvelope::Termination(_)
                                | InboundEnvelope::Other,
                            ) => continue,
                            Err(e) => {
                                let _ = out_tx
                                    .send(Err(InferenceError::Internal(format!(
                                        "assemblyai envelope decode: {e}"
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
                                    "assemblyai closed code={code} reason={reason}"
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
                                "assemblyai ws recv: {e}"
                            ))))
                            .await;
                        return;
                    }
                }
            }

            // If we never observed `end_of_turn=true`, promote the
            // last produced chunk to terminal. If we produced none,
            // emit a synthetic empty terminal.
            if !saw_final {
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

        drop(uplink_close_handle);

        let s: BoxStream<'static, InferenceResult<TranscriptChunk>> = ReceiverStream::new(out_rx).boxed();
        Ok(AudioRunHandle::streaming(s))
    }

    #[cfg(not(feature = "stt-assemblyai"))]
    async fn execute_audio(&mut self, _batch: AudioBatch) -> InferenceResult<AudioRunHandle> {
        Err(InferenceError::Internal(
            "stt-assemblyai feature disabled at build time".into(),
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
            provider: ProviderKind::AssemblyAi,
        }
    }

    #[cfg(feature = "stt-assemblyai")]
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }
}

/// Open the caller's [`AudioInput`] as a uniform mpsc receiver of
/// [`Bytes`] chunks plus the underlying [`AudioParams`]. For
/// `AudioInput::Static` we re-chunk the payload into 4 KiB binary
/// frames (AssemblyAI recommends 50–250 ms chunks; 4 KiB @ 16 kHz
/// mono PCM16 ≈ 128 ms).
#[cfg(feature = "stt-assemblyai")]
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

#[cfg(feature = "stt-assemblyai")]
async fn materialize_payload(payload: AudioPayload) -> InferenceResult<(Bytes, AudioParams)> {
    match payload {
        AudioPayload::Bytes { data, params } => Ok((data, params)),
        AudioPayload::Path { path, params } => {
            let data = tokio::fs::read(&path)
                .await
                .map_err(|e| InferenceError::BadRequest {
                    message: format!("assemblyai: read {}: {e}", path.display()),
                })?;
            Ok((Bytes::from(data), params))
        }
        AudioPayload::Url { .. } => Err(AssemblyAiError::Unsupported {
            method: "execute_audio: URL audio payloads (use Stream or Bytes)",
        }
        .into()),
        _ => Err(AssemblyAiError::BadRequest {
            message: "assemblyai: unknown AudioPayload variant".into(),
        }
        .into()),
    }
}

/// AssemblyAI v3 only accepts 16-bit PCM (`Pcm16Le` over the wire).
/// Anything else is rejected up front so we don't waste a connect.
#[cfg(feature = "stt-assemblyai")]
fn assert_format_supported(format: AudioFormat) -> InferenceResult<()> {
    match format {
        AudioFormat::Pcm16Le => Ok(()),
        other => Err(AssemblyAiError::UnsupportedFormat {
            message: format!(
                "AssemblyAI Streaming v3 only accepts 16-bit PCM mono (got {other:?}); resample upstream"
            ),
        }
        .into()),
    }
}

/// Convert a [`TurnEnvelope`] into the project's [`TranscriptChunk`].
///
/// - `text` comes from `transcript` (already includes the rolling
///   stable + unstable suffix; for `end_of_turn=true` updates the
///   provider may apply Punctuated & Formatted post-processing when
///   `format_turns=true`).
/// - `ts_start_ms` / `ts_end_ms` come from the first and last word's
///   `start` / `end` (already ms). When `words` is empty, both are 0.
/// - `is_final` follows `end_of_turn`.
/// - `words` is converted from `AssemblyWord` to [`WordTiming`] when
///   the caller asked for `word_timestamps`.
#[cfg(feature = "stt-assemblyai")]
fn turn_to_chunk(request_id: &str, t: TurnEnvelope, want_words: bool) -> TranscriptChunk {
    let ts_start_ms = t.words.first().map(|w| w.start).unwrap_or(0);
    let ts_end_ms = t.words.last().map(|w| w.end).unwrap_or(0);
    let words = if want_words {
        t.words
            .iter()
            .map(|w| WordTiming {
                text: w.text.clone(),
                ts_start_ms: w.start,
                ts_end_ms: w.end,
                confidence: w.confidence,
            })
            .collect()
    } else {
        Vec::new()
    };
    TranscriptChunk {
        request_id: request_id.to_string(),
        is_final: t.end_of_turn,
        text: t.transcript,
        ts_start_ms,
        ts_end_ms,
        speaker_id: None,
        words,
        usage: None,
    }
}

#[cfg(all(test, feature = "stt-assemblyai"))]
mod tests {
    use super::*;
    use crate::wire::AssemblyWord;

    #[test]
    fn format_pcm16_is_supported() {
        assert!(assert_format_supported(AudioFormat::Pcm16Le).is_ok());
    }

    #[test]
    fn format_other_is_rejected() {
        for f in [
            AudioFormat::Pcm24Le,
            AudioFormat::PcmF32Le,
            AudioFormat::Mp3,
            AudioFormat::OggOpus,
            AudioFormat::Flac,
            AudioFormat::Wav,
        ] {
            assert!(matches!(
                assert_format_supported(f),
                Err(InferenceError::UnsupportedAudioFormat { .. })
            ));
        }
    }

    #[test]
    fn turn_to_chunk_no_words() {
        let t = TurnEnvelope {
            turn_order: 0,
            turn_is_formatted: false,
            end_of_turn: true,
            end_of_turn_confidence: Some(0.9),
            transcript: "hello world".into(),
            words: vec![
                AssemblyWord {
                    text: "hello".into(),
                    start: 0,
                    end: 300,
                    confidence: Some(0.95),
                    word_is_final: true,
                },
                AssemblyWord {
                    text: "world".into(),
                    start: 400,
                    end: 900,
                    confidence: Some(0.9),
                    word_is_final: true,
                },
            ],
        };
        let c = turn_to_chunk("req", t, false);
        assert_eq!(c.request_id, "req");
        assert!(c.is_final);
        assert_eq!(c.text, "hello world");
        assert_eq!(c.ts_start_ms, 0);
        assert_eq!(c.ts_end_ms, 900);
        assert!(c.speaker_id.is_none());
        assert!(c.words.is_empty());
    }

    #[test]
    fn turn_to_chunk_with_words() {
        let t = TurnEnvelope {
            turn_order: 0,
            turn_is_formatted: false,
            end_of_turn: false,
            end_of_turn_confidence: None,
            transcript: "hi".into(),
            words: vec![AssemblyWord {
                text: "hi".into(),
                start: 100,
                end: 250,
                confidence: Some(0.88),
                word_is_final: false,
            }],
        };
        let c = turn_to_chunk("r", t, true);
        assert!(!c.is_final);
        assert_eq!(c.words.len(), 1);
        assert_eq!(c.words[0].text, "hi");
        assert_eq!(c.words[0].ts_start_ms, 100);
        assert_eq!(c.words[0].ts_end_ms, 250);
        assert_eq!(c.words[0].confidence, Some(0.88));
    }

    #[test]
    fn turn_to_chunk_empty_words_zeros_timing() {
        let t = TurnEnvelope {
            turn_order: 0,
            turn_is_formatted: false,
            end_of_turn: true,
            end_of_turn_confidence: None,
            transcript: "".into(),
            words: vec![],
        };
        let c = turn_to_chunk("r", t, true);
        assert_eq!(c.ts_start_ms, 0);
        assert_eq!(c.ts_end_ms, 0);
        assert!(c.words.is_empty());
        assert!(c.is_final);
    }
}
