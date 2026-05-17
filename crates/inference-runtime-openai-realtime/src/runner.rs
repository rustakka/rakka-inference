//! `OpenAiRealtimeRunner` — `RealtimeRunner` impl for the OpenAI Realtime API.
//!
//! When the `tts-openai-realtime` feature is **off**, every method returns
//! `InferenceError::Internal("tts-openai-realtime feature disabled at build time")`.
//!
//! When the feature is **on**, `open_session`:
//! 1. Resolves the API key from `config.api_key`.
//! 2. Connects to `wss://api.openai.com/v1/realtime?model=<model>` via
//!    `WsClient::connect_with_headers` with the two required headers:
//!    `Authorization: Bearer <key>` and `OpenAI-Beta: realtime=v1`.
//! 3. Sends `session.update` to configure voice, modalities, and audio formats.
//! 4. Spawns an adapter task (combined uplink + downlink) wrapped in
//!    `futures::future::abortable`.
//! 5. Returns a [`RealtimeSession`] carrying the abort handle.

use async_trait::async_trait;

use atomr_infer_core::audio::{RealtimeBatch, VoiceRef};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{RealtimeRunner, RealtimeSession, SessionRebuildCause};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

use crate::config::OpenAiRealtimeConfig;

/// Runner that opens bidirectional realtime sessions against the OpenAI
/// Realtime API (`wss://api.openai.com/v1/realtime`).
pub struct OpenAiRealtimeRunner {
    #[allow(dead_code)] // used only when the `tts-openai-realtime` feature is on
    config: OpenAiRealtimeConfig,
}

impl OpenAiRealtimeRunner {
    /// Construct a new runner from `config`.
    pub fn new(config: OpenAiRealtimeConfig) -> Self {
        Self { config }
    }
}

/// Resolve the `VoiceRef` to a provider voice string.
///
/// # Errors
///
/// Returns [`InferenceError::BadRequest`] for `VoiceRef::ClonedFrom` — the
/// OpenAI Realtime API does not support voice cloning.
#[allow(dead_code)] // used only when the `tts-openai-realtime` feature is on
pub(crate) fn resolve_voice(voice: &VoiceRef) -> InferenceResult<String> {
    match voice {
        VoiceRef::Named(s) | VoiceRef::Id(s) => Ok(s.clone()),
        VoiceRef::ClonedFrom(_) => Err(InferenceError::BadRequest {
            message: "OpenAI Realtime does not support voice cloning".into(),
        }),
        _ => Err(InferenceError::BadRequest {
            message: "unsupported VoiceRef variant".into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Feature-off stub
// ---------------------------------------------------------------------------

#[cfg(not(feature = "tts-openai-realtime"))]
#[async_trait]
impl RealtimeRunner for OpenAiRealtimeRunner {
    async fn open_session(&mut self, _batch: RealtimeBatch) -> InferenceResult<RealtimeSession> {
        Err(InferenceError::Internal(
            "tts-openai-realtime feature disabled at build time".into(),
        ))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::RealtimeSpeech
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork {
            provider: ProviderKind::OpenAi,
        }
    }
}

// ---------------------------------------------------------------------------
// Full implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "tts-openai-realtime")]
mod full {
    use super::*;

    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine as _;
    use bytes::Bytes;
    use futures::future::abortable;

    use atomr_infer_core::audio::{AudioFormat, AudioParams, RealtimeIn, RealtimeOut, TranscriptRole};
    use atomr_infer_runtime_ws_core::{Frame, WsClient, WsReceiver, WsSender};

    use crate::wire::{
        ConversationItemCreate, InboundEvent, InputAudioAppend, InputAudioCommit, InputAudioTranscription,
        ResponseCancel, ResponseCreate, SessionConfig, SessionUpdate,
    };

    /// 24 kHz mono PCM16-LE — the format OpenAI Realtime emits.
    fn realtime_out_params() -> AudioParams {
        AudioParams::new(24_000, 1, AudioFormat::Pcm16Le)
    }

    #[async_trait]
    impl RealtimeRunner for OpenAiRealtimeRunner {
        #[tracing::instrument(
            skip(self, batch),
            fields(request_id = %batch.request_id, model = %batch.model)
        )]
        async fn open_session(&mut self, batch: RealtimeBatch) -> InferenceResult<RealtimeSession> {
            let voice_str = resolve_voice(&batch.voice)?;
            let api_key = self
                .config
                .resolve_api_key()
                .map_err(|e| InferenceError::Unauthorized { message: e })?;

            let ws_url = self.config.ws_url(&batch.model);
            let timeout = self.config.handshake_timeout();

            let auth_value = format!("Bearer {api_key}");
            let headers: &[(&str, &str)] = &[
                ("Authorization", auth_value.as_str()),
                ("OpenAI-Beta", "realtime=v1"),
            ];
            let (mut sink, stream) = WsClient::connect_with_headers(&ws_url, headers, timeout)
                .await
                .map_err(|e| InferenceError::NetworkError(format!("openai realtime ws connect: {e}")))?;

            // Send session.update immediately after connect.
            let modalities = &["audio", "text"];
            let session_update = SessionUpdate::new(SessionConfig {
                voice: &voice_str,
                modalities,
                input_audio_format: "pcm16",
                output_audio_format: "pcm16",
                input_audio_transcription: InputAudioTranscription::default(),
            });
            let su_json = serde_json::to_string(&session_update)
                .map_err(|e| InferenceError::Internal(format!("serialize session.update: {e}")))?;
            sink.send(Frame::Text(su_json))
                .await
                .map_err(|e| InferenceError::NetworkError(e.to_string()))?;

            let request_id = batch.request_id.clone();

            let (abortable_fut, abort_handle) =
                abortable(adapter_task(sink, stream, batch.inbound, batch.outbound));

            tokio::spawn(async move {
                let _ = abortable_fut.await;
            });

            Ok(RealtimeSession::new(request_id, abort_handle))
        }

        async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
            Ok(())
        }

        fn runtime_kind(&self) -> RuntimeKind {
            RuntimeKind::RealtimeSpeech
        }

        fn transport_kind(&self) -> TransportKind {
            TransportKind::RemoteNetwork {
                provider: ProviderKind::OpenAi,
            }
        }
    }

    /// Combined adapter task — drives both the uplink (caller → WS) and the
    /// downlink (WS → caller) until the connection closes or is aborted.
    async fn adapter_task(
        mut sink: WsSender,
        mut stream: WsReceiver,
        mut inbound: tokio::sync::mpsc::Receiver<RealtimeIn>,
        outbound: tokio::sync::mpsc::Sender<RealtimeOut>,
    ) {
        let params = realtime_out_params();

        loop {
            tokio::select! {
                // Uplink: translate caller events → WS frames
                maybe_in = inbound.recv() => {
                    match maybe_in {
                        None | Some(RealtimeIn::Close) => {
                            let _ = sink
                                .send(Frame::Close { code: 1000, reason: String::new() })
                                .await;
                            break;
                        }
                        Some(msg) => {
                            let frames = translate_inbound_to_frames(msg);
                            for frame in frames {
                                if sink.send(frame).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }

                // Downlink: translate WS frames → caller events
                maybe_frame = stream.next() => {
                    match maybe_frame {
                        Ok(None) | Err(_) => break,
                        Ok(Some(Frame::Text(text))) => {
                            if let Ok(ev) = serde_json::from_str::<InboundEvent>(text.as_str()) {
                                if let Some(out) = translate_outbound(ev, &params) {
                                    let _ = outbound.send(out).await;
                                }
                            }
                        }
                        Ok(Some(Frame::Close { .. })) => break,
                        Ok(Some(_)) => {}
                    }
                }
            }
        }
    }

    /// Translate a single [`RealtimeIn`] to zero, one, or two WS frames.
    ///
    /// Text turns produce two frames: `conversation.item.create` then
    /// `response.create`.
    fn translate_inbound_to_frames(msg: RealtimeIn) -> Vec<Frame> {
        match msg {
            RealtimeIn::AudioFrame { pcm, .. } => {
                let encoded = B64.encode(&pcm);
                serde_json::to_string(&InputAudioAppend::new(&encoded))
                    .ok()
                    .map(|s| vec![Frame::Text(s)])
                    .unwrap_or_default()
            }
            RealtimeIn::Text(text) => {
                let item = ConversationItemCreate::user_text(&text);
                let item_json = serde_json::to_string(&item).ok();
                let resp_json = serde_json::to_string(&ResponseCreate::new()).ok();
                match (item_json, resp_json) {
                    (Some(i), Some(r)) => {
                        vec![Frame::Text(i), Frame::Text(r)]
                    }
                    _ => vec![],
                }
            }
            RealtimeIn::Commit => serde_json::to_string(&InputAudioCommit::new())
                .ok()
                .map(|s| vec![Frame::Text(s)])
                .unwrap_or_default(),
            RealtimeIn::Interrupt => serde_json::to_string(&ResponseCancel::new())
                .ok()
                .map(|s| vec![Frame::Text(s)])
                .unwrap_or_default(),
            RealtimeIn::Close => vec![],
            _ => vec![],
        }
    }

    /// Translate an [`InboundEvent`] to a [`RealtimeOut`] for the caller.
    fn translate_outbound(ev: InboundEvent, params: &AudioParams) -> Option<RealtimeOut> {
        match ev {
            InboundEvent::ResponseAudioDelta(d) => {
                let pcm = B64.decode(&d.delta).ok()?;
                Some(RealtimeOut::AudioFrame {
                    pcm: Bytes::from(pcm),
                    params: *params,
                })
            }
            InboundEvent::ResponseAudioTranscriptDelta(d) => Some(RealtimeOut::Transcript {
                role: TranscriptRole::Assistant,
                text: d.delta,
                is_final: false,
            }),
            InboundEvent::ResponseAudioTranscriptDone(d) => Some(RealtimeOut::Transcript {
                role: TranscriptRole::Assistant,
                text: d.transcript,
                is_final: true,
            }),
            InboundEvent::InputAudioTranscriptionCompleted(d) => Some(RealtimeOut::Transcript {
                role: TranscriptRole::User,
                text: d.transcript,
                is_final: true,
            }),
            InboundEvent::ResponseDone => Some(RealtimeOut::Done),
            InboundEvent::Error(e) => Some(RealtimeOut::Error(crate::error::classify_realtime_error(Some(
                e.error.message,
            )))),
            InboundEvent::Other => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::audio::VoiceRef;

    #[test]
    fn resolve_voice_named() {
        let v = VoiceRef::Named("alloy".into());
        assert_eq!(resolve_voice(&v).unwrap(), "alloy");
    }

    #[test]
    fn resolve_voice_id() {
        let v = VoiceRef::Id("voice-abc".into());
        assert_eq!(resolve_voice(&v).unwrap(), "voice-abc");
    }

    #[test]
    fn resolve_voice_cloned_from_errors() {
        use atomr_infer_core::audio::{AudioFormat, AudioParams, AudioPayload};
        let v = VoiceRef::ClonedFrom(AudioPayload::Bytes {
            data: bytes::Bytes::new(),
            params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
        });
        let e = resolve_voice(&v).unwrap_err();
        assert!(matches!(e, InferenceError::BadRequest { .. }));
    }

    #[test]
    fn runtime_kind_is_openai_realtime() {
        let runner = OpenAiRealtimeRunner::new(crate::config::OpenAiRealtimeConfig::new_with_env_key("K"));
        assert_eq!(runner.runtime_kind(), RuntimeKind::RealtimeSpeech);
    }

    #[test]
    fn transport_kind_is_remote_openai() {
        let runner = OpenAiRealtimeRunner::new(crate::config::OpenAiRealtimeConfig::new_with_env_key("K"));
        assert!(matches!(
            runner.transport_kind(),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::OpenAi
            }
        ));
    }
}
