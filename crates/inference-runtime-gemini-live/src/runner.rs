//! `GeminiLiveRunner` — [`RealtimeRunner`] implementation against the
//! Gemini Live BidiGenerateContent WebSocket endpoint.
//!
//! The runner opens a WebSocket with the API key embedded in the URL
//! (`?key=<api_key>`), performs the setup handshake, then pumps the
//! caller's `RealtimeIn` messages to the server (uplink) and translates
//! server `serverContent` envelopes to `RealtimeOut` messages (downlink).
//!
//! The session adapter task is wrapped in a `futures::future::Abortable` so
//! `RealtimeSession::cancel()` can tear it down from the outside.
//!
//! [`RealtimeRunner`]: atomr_infer_core::runner::RealtimeRunner

#[cfg(feature = "tts-gemini-live")]
use std::sync::Arc;

use async_trait::async_trait;

use atomr_infer_core::audio::RealtimeBatch;
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{RealtimeRunner, RealtimeSession, SessionRebuildCause};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

#[cfg(feature = "tts-gemini-live")]
use arc_swap::ArcSwap;
#[cfg(feature = "tts-gemini-live")]
use atomr_infer_core::audio::{AudioFormat, AudioParams, RealtimeIn, RealtimeOut, TranscriptRole, VoiceRef};
#[cfg(feature = "tts-gemini-live")]
use atomr_infer_core::deployment::RateLimits;
#[cfg(feature = "tts-gemini-live")]
use atomr_infer_remote_core::session::SessionSnapshot;
#[cfg(feature = "tts-gemini-live")]
use bytes::Bytes;
#[cfg(feature = "tts-gemini-live")]
use futures::future::abortable;

#[cfg(feature = "tts-gemini-live")]
use crate::config::GeminiLiveConfig;
#[cfg(feature = "tts-gemini-live")]
use crate::error::GeminiLiveError;
#[cfg(feature = "tts-gemini-live")]
use crate::wire::{
    ClientContent, ClientContentInner, ContentPart, ContentTurn, GenerationConfig, Inbound, MediaChunk,
    RealtimeInput, RealtimeInputInner, Setup, SetupConfig,
};

#[cfg(feature = "tts-gemini-live")]
use atomr_infer_runtime_ws_core::{Frame as WsFrame, WsClient};
#[cfg(feature = "tts-gemini-live")]
use base64::Engine as _;
#[cfg(feature = "tts-gemini-live")]
use secrecy::ExposeSecret;

/// `RealtimeRunner` implementation against Gemini Live BidiGenerateContent.
///
/// One instance is reusable across sessions; per-session state lives in the
/// spawned adapter task.
pub struct GeminiLiveRunner {
    #[cfg(feature = "tts-gemini-live")]
    config: GeminiLiveConfig,
    #[cfg(feature = "tts-gemini-live")]
    session: Arc<ArcSwap<SessionSnapshot>>,
    #[cfg(not(feature = "tts-gemini-live"))]
    _stub: (),
}

#[cfg(feature = "tts-gemini-live")]
impl GeminiLiveRunner {
    /// Construct a runner. `session` carries the shared
    /// `inference-remote-core` snapshot — when the session actor
    /// rotates credentials, the next `open_session` call picks up the
    /// fresh API key automatically.
    pub fn new(config: GeminiLiveConfig, session: Arc<ArcSwap<SessionSnapshot>>) -> InferenceResult<Self> {
        Ok(Self { config, session })
    }

    /// Borrow the runtime configuration.
    pub fn config(&self) -> &GeminiLiveConfig {
        &self.config
    }
}

#[cfg(not(feature = "tts-gemini-live"))]
impl GeminiLiveRunner {
    /// Stub constructor — accepts no arguments so callers can still
    /// link without pulling the feature in.
    pub fn new_stub() -> Self {
        Self { _stub: () }
    }
}

#[async_trait]
impl RealtimeRunner for GeminiLiveRunner {
    #[cfg(feature = "tts-gemini-live")]
    async fn open_session(&mut self, batch: RealtimeBatch) -> InferenceResult<RealtimeSession> {
        // Reject ClonedFrom voice — Gemini Live doesn't support voice cloning.
        if matches!(batch.voice, VoiceRef::ClonedFrom(_)) {
            return Err(GeminiLiveError::BadRequest {
                message: "Gemini Live does not support VoiceRef::ClonedFrom".into(),
            }
            .into());
        }

        let api_key = {
            let snap = self.session.load();
            snap.credential.expose_secret().to_string()
        };

        let url = self
            .config
            .live_url(&api_key)
            .map_err(|e| InferenceError::Internal(format!("gemini live url: {e}")))?;

        let (mut tx, mut rx) = WsClient::connect(url.as_str(), self.config.ws_connect_timeout)
            .await
            .map_err(|e| InferenceError::NetworkError(format!("gemini live ws connect: {e}")))?;

        let model = format!("models/{}", batch.model);
        let request_id = batch.request_id.clone();

        // Send setup message.
        let setup_msg = Setup {
            setup: SetupConfig {
                model: &model,
                generation_config: GenerationConfig {
                    response_modalities: &["AUDIO"],
                },
            },
        };
        let setup_json = serde_json::to_string(&setup_msg)
            .map_err(|e| InferenceError::Internal(format!("setup serialize: {e}")))?;
        tx.send(WsFrame::Text(setup_json))
            .await
            .map_err(|e| InferenceError::NetworkError(format!("gemini live setup send: {e}")))?;

        // Wait for setupComplete before forwarding user input.
        loop {
            match rx.next().await {
                Ok(Some(WsFrame::Text(text))) => {
                    if let Ok(Inbound::SetupComplete(_)) = serde_json::from_str::<Inbound>(&text) {
                        break;
                    }
                    // Check for error envelope.
                    if let Ok(env) = serde_json::from_str::<serde_json::Value>(&text) {
                        if env.get("error").is_some() {
                            return Err(GeminiLiveError::ServerError { body: text }.into());
                        }
                    }
                    // Any other non-setup envelope: keep waiting.
                }
                Ok(Some(WsFrame::Close { code, reason })) => {
                    return Err(GeminiLiveError::SessionClosed {
                        reason: format!("closed during setup: code={code} reason={reason}"),
                    }
                    .into());
                }
                Ok(None) => {
                    return Err(GeminiLiveError::SessionClosed {
                        reason: "connection closed before setupComplete".into(),
                    }
                    .into());
                }
                Err(e) => {
                    return Err(InferenceError::NetworkError(format!(
                        "gemini live setup recv: {e}"
                    )));
                }
                _ => continue,
            }
        }

        let inbound = batch.inbound;
        let outbound = batch.outbound;

        // Spawn the session adapter inside an Abortable so `cancel()` works.
        let (adapter_fut, abort_handle) =
            abortable(session_adapter(request_id.clone(), tx, rx, inbound, outbound));
        tokio::spawn(adapter_fut);

        Ok(RealtimeSession::new(request_id, abort_handle))
    }

    #[cfg(not(feature = "tts-gemini-live"))]
    async fn open_session(&mut self, _batch: RealtimeBatch) -> InferenceResult<RealtimeSession> {
        Err(InferenceError::Internal(
            "tts-gemini-live feature disabled at build time".into(),
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
            provider: ProviderKind::Gemini,
        }
    }

    #[cfg(feature = "tts-gemini-live")]
    fn rate_limits(&self) -> Option<&RateLimits> {
        Some(&self.config.rate_limits)
    }
}

/// The session adapter task: concurrently drives the uplink (caller →
/// server) and downlink (server → caller) until either side closes.
#[cfg(feature = "tts-gemini-live")]
async fn session_adapter(
    request_id: String,
    tx: atomr_infer_runtime_ws_core::WsSender,
    rx: atomr_infer_runtime_ws_core::WsReceiver,
    inbound: tokio::sync::mpsc::Receiver<RealtimeIn>,
    outbound: tokio::sync::mpsc::Sender<RealtimeOut>,
) {
    // Uplink task: translate RealtimeIn → Gemini Live JSON frames.
    // Runs independently; the downlink task drives the session lifetime.
    let uplink_out = outbound.clone();
    tokio::spawn(uplink_task(inbound, tx, uplink_out));

    // Downlink task: translate server envelopes → RealtimeOut.
    // Terminates when the WS closes, not when the uplink drains, so
    // server responses that arrive after the client turn completes are
    // still surfaced.
    downlink_task(request_id, rx, outbound).await;
}

/// Drain the `inbound` channel and translate each [`RealtimeIn`] into the
/// appropriate Gemini Live JSON message.
#[cfg(feature = "tts-gemini-live")]
async fn uplink_task(
    mut inbound: tokio::sync::mpsc::Receiver<RealtimeIn>,
    mut tx: atomr_infer_runtime_ws_core::WsSender,
    outbound: tokio::sync::mpsc::Sender<RealtimeOut>,
) {
    while let Some(msg) = inbound.recv().await {
        match msg {
            RealtimeIn::Text(text) => {
                let payload = ClientContent {
                    client_content: ClientContentInner {
                        turns: Some(vec![ContentTurn {
                            role: "user",
                            parts: vec![ContentPart { text }],
                        }]),
                        turn_complete: true,
                    },
                };
                if let Ok(json) = serde_json::to_string(&payload) {
                    if tx.send(WsFrame::Text(json)).await.is_err() {
                        break;
                    }
                }
            }
            RealtimeIn::AudioFrame { pcm, params } => {
                if params.format != AudioFormat::Pcm16Le {
                    let _ = outbound
                        .send(RealtimeOut::Error(InferenceError::UnsupportedAudioFormat {
                            message: format!("Gemini Live requires Pcm16Le; got {:?}", params.format),
                        }))
                        .await;
                    break;
                }
                let encoded = base64::engine::general_purpose::STANDARD.encode(&pcm);
                let mime = format!("audio/pcm;rate={}", params.sample_rate_hz);
                let payload = RealtimeInput {
                    realtime_input: RealtimeInputInner {
                        media_chunks: vec![MediaChunk {
                            mime_type: mime,
                            data: encoded,
                        }],
                    },
                };
                if let Ok(json) = serde_json::to_string(&payload) {
                    if tx.send(WsFrame::Text(json)).await.is_err() {
                        break;
                    }
                }
            }
            RealtimeIn::Commit => {
                // Send a clientContent with just turnComplete=true.
                let payload = ClientContent {
                    client_content: ClientContentInner {
                        turns: None,
                        turn_complete: true,
                    },
                };
                if let Ok(json) = serde_json::to_string(&payload) {
                    if tx.send(WsFrame::Text(json)).await.is_err() {
                        break;
                    }
                }
            }
            RealtimeIn::Interrupt => {
                // Gemini Live doesn't support mid-session interrupt.
                let _ = outbound
                    .send(RealtimeOut::Error(InferenceError::Unsupported {
                        method: "interrupt".into(),
                        runtime: RuntimeKind::RealtimeSpeech,
                    }))
                    .await;
                break;
            }
            RealtimeIn::Close => {
                break;
            }
            _ => {
                // Unknown variant (non_exhaustive) — ignore.
            }
        }
    }
}

/// Receive server frames and translate them into `RealtimeOut` messages.
/// Terminates when the WS closes cleanly. Emits `Done` before returning.
#[cfg(feature = "tts-gemini-live")]
async fn downlink_task(
    _request_id: String,
    mut rx: atomr_infer_runtime_ws_core::WsReceiver,
    outbound: tokio::sync::mpsc::Sender<RealtimeOut>,
) {
    let mut pending_text: String = String::new();
    loop {
        match rx.next().await {
            Ok(Some(WsFrame::Text(text))) => {
                // Check for top-level error first.
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                    if val.get("error").is_some() {
                        let _ = outbound
                            .send(RealtimeOut::Error(
                                GeminiLiveError::ServerError { body: text.clone() }.into(),
                            ))
                            .await;
                        break;
                    }
                }
                let parsed: Result<Inbound, _> = serde_json::from_str(&text);
                match parsed {
                    Ok(Inbound::ServerContent(sc)) => {
                        // Emit audio parts.
                        if let Some(turn) = &sc.model_turn {
                            for part in &turn.parts {
                                if let Some(inline) = &part.inline_data {
                                    if let Ok(pcm_bytes) =
                                        base64::engine::general_purpose::STANDARD.decode(&inline.data)
                                    {
                                        let params = AudioParams::new(24_000, 1, AudioFormat::Pcm16Le);
                                        if outbound
                                            .send(RealtimeOut::AudioFrame {
                                                pcm: Bytes::from(pcm_bytes),
                                                params,
                                            })
                                            .await
                                            .is_err()
                                        {
                                            return;
                                        }
                                    }
                                }
                                if let Some(text_val) = &part.text {
                                    pending_text.push_str(text_val);
                                    if outbound
                                        .send(RealtimeOut::Transcript {
                                            role: TranscriptRole::Assistant,
                                            text: text_val.clone(),
                                            is_final: false,
                                        })
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                        }
                        // Emit turn-complete signal.
                        if sc.turn_complete == Some(true) {
                            let _ = outbound
                                .send(RealtimeOut::Transcript {
                                    role: TranscriptRole::Assistant,
                                    text: String::new(),
                                    is_final: true,
                                })
                                .await;
                            pending_text.clear();
                        }
                        // Interrupted: no-op (server already stopped streaming).
                    }
                    Ok(Inbound::SetupComplete(_) | Inbound::ToolCall(_) | Inbound::Other) => {
                        // Ignore.
                    }
                    Err(_e) => {
                        // Unparseable frame — ignore (e.g. unknown future API additions).
                    }
                }
            }
            Ok(Some(WsFrame::Close { .. })) | Ok(None) => break,
            Ok(Some(_)) => continue,
            Err(e) => {
                let _ = outbound
                    .send(RealtimeOut::Error(InferenceError::NetworkError(format!(
                        "gemini live ws recv: {e}"
                    ))))
                    .await;
                break;
            }
        }
    }
    let _ = outbound.send(RealtimeOut::Done).await;
}
