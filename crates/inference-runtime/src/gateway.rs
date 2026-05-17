//! `ApiGatewayActor` — HTTP gateway. Doc §4, §6.
//!
//! Exposes an OpenAI-compatible `/v1/chat/completions` endpoint plus a
//! `/healthz`. Each incoming request becomes one `RequestActor`. The
//! response body is streamed back via the per-request `mpsc` channel.
//!
//! The gateway-actor itself is small — it owns the listener task and
//! the wiring to the deployment manager; the per-request handler runs
//! in axum's task pool.
//!
//! # Audio routes (`FR-TTS-001`, `FR-STT-001`, `FR-A2F-001`)
//!
//! - `POST /v1/audio/transcriptions` — STT unary (multipart form upload).
//! - `GET  /v1/audio/transcriptions/stream` — STT streaming (WebSocket;
//!   client pushes PCM frames, server sends back `TranscriptChunk` JSON).
//! - `POST /v1/audio/speech` — TTS batch (chunked WAV response).
//! - `POST /v1/audio/speech/stream` — TTS streaming via SSE for browser
//!   `fetch`/`EventSource` consumers (base64-encoded PCM in the data field).
//! - `GET  /v1/realtime` — bidirectional realtime session (WebSocket).
//! - `GET  /v1/audio2face` — A2F session (WebSocket; client pushes PCM,
//!   server emits binary blendshape frames).
//!
//! At M2 these handlers are wired but stubbed — they validate the
//! route resolves to an audio-shaped deployment and return a
//! placeholder response. Per-runtime bridging lands in M4–M11 as each
//! provider crate plumbs its runner into the gateway shim.

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, ActorRef, Context};
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Multipart, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use atomr_infer_core::batch::{ExecuteBatch, Message, MessageContent, Role, SamplingParams};

use crate::dp_coordinator::DpCoordinatorMsg;

#[derive(Clone)]
pub struct GatewayConfig {
    pub bind: SocketAddr,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            bind: SocketAddr::from(([127, 0, 0, 1], 8080)),
        }
    }
}

pub enum ApiGatewayMsg {
    Stop,
}

#[derive(Clone)]
struct AppState {
    coordinator: ActorRef<DpCoordinatorMsg>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatErrorResponse {
    error: ChatError,
}

#[derive(Debug, Serialize)]
struct ChatError {
    message: String,
    #[serde(rename = "type")]
    kind: String,
}

pub struct ApiGatewayActor {
    config: GatewayConfig,
    coordinator: ActorRef<DpCoordinatorMsg>,
    /// Shutdown channel handed to the listener task in `pre_start`.
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl ApiGatewayActor {
    pub fn new(config: GatewayConfig, coordinator: ActorRef<DpCoordinatorMsg>) -> Self {
        Self {
            config,
            coordinator,
            shutdown_tx: None,
        }
    }
}

#[async_trait]
impl Actor for ApiGatewayActor {
    type Msg = ApiGatewayMsg;

    async fn pre_start(&mut self, _ctx: &mut Context<Self>) {
        let bind = self.config.bind;
        let state = AppState {
            coordinator: self.coordinator.clone(),
        };
        let app = Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/audio/transcriptions", post(audio_transcriptions))
            .route(
                "/v1/audio/transcriptions/stream",
                get(audio_transcriptions_stream),
            )
            .route("/v1/audio/speech", post(audio_speech))
            .route("/v1/audio/speech/stream", post(audio_speech_stream))
            .route("/v1/realtime", get(realtime_session))
            .route("/v1/audio2face", get(audio2face_session))
            .with_state(state);
        let listener = match TcpListener::bind(bind).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(?e, "gateway bind failed");
                return;
            }
        };
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);
        tokio::spawn(async move {
            tracing::info!(%bind, "gateway listening");
            let server = axum::serve(listener, app);
            let _ = tokio::select! {
                r = server => r,
                _ = rx => Ok(()),
            };
        });
    }

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            ApiGatewayMsg::Stop => {
                if let Some(tx) = self.shutdown_tx.take() {
                    let _ = tx.send(());
                }
                ctx.stop_self();
            }
        }
    }

    async fn post_stop(&mut self, _ctx: &mut Context<Self>) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Convenience to start the gateway as a top-level actor.
pub fn spawn_gateway(
    sys: &atomr_core::actor::ActorSystem,
    config: GatewayConfig,
    coordinator: ActorRef<DpCoordinatorMsg>,
) -> Result<ActorRef<ApiGatewayMsg>, atomr_core::actor::ActorSystemError> {
    use atomr_core::actor::Props;
    let coord = Arc::new(coordinator);
    let cfg = Arc::new(config);
    let props = Props::create(move || ApiGatewayActor::new((*cfg).clone(), (*coord).clone()));
    sys.actor_of(props, "gateway")
}

async fn chat_completions(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let messages = req
        .messages
        .into_iter()
        .map(|m| Message {
            role: parse_role(&m.role),
            content: MessageContent::Text(m.content),
        })
        .collect();
    let batch = ExecuteBatch {
        request_id: format!("req-{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)),
        model: req.model.clone(),
        messages,
        sampling: SamplingParams {
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            ..Default::default()
        },
        stream: req.stream,
        estimated_tokens: 256,
    };

    // Look up the route via the coordinator.
    let route = state
        .coordinator
        .ask_with(
            |reply| DpCoordinatorMsg::RouteTo {
                deployment: req.model.clone(),
                reply,
            },
            std::time::Duration::from_secs(2),
        )
        .await;
    match route {
        Ok(Ok(_target)) => {
            // v0: route is resolved but the gateway → engine plumbing
            // for the actual request lives in the per-runtime crates'
            // sample servers. Return a JSON envelope acknowledging
            // route resolution so smoke-tests pass; full SSE bridging
            // is added in the demo example (`examples/remote_only_demo`).
            let body = serde_json::json!({
                "id": batch.request_id,
                "model": batch.model,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop"
                }],
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Ok(Err(e)) => bad_request(e.to_string(), "no_route"),
        Err(_) => bad_request("coordinator timeout".into(), "internal_error"),
    }
}

fn bad_request(msg: String, kind: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ChatErrorResponse {
            error: ChatError {
                message: msg,
                kind: kind.into(),
            },
        }),
    )
        .into_response()
}

fn parse_role(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio modality routes (FR-TTS-001 / FR-STT-001 / FR-A2F-001)
// ─────────────────────────────────────────────────────────────────────────────

/// JSON envelope returned by the OpenAI-shaped STT unary endpoint.
#[derive(Debug, Serialize)]
struct TranscriptionResponse {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ms: Option<u32>,
}

/// JSON request body for `/v1/audio/speech` and its streaming sibling.
#[derive(Debug, Deserialize)]
#[allow(dead_code)] // Stubbed fields consumed by per-runtime bridging in M4–M11.
struct SpeechRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(default = "default_response_format")]
    response_format: String,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    emotion: Option<String>,
}

fn default_response_format() -> String {
    "wav".into()
}

/// `POST /v1/audio/transcriptions` — STT unary.
///
/// Accepts a `multipart/form-data` upload with a `file` field carrying
/// the audio and a `model` field naming the deployment. Returns an
/// OpenAI-compatible JSON response.
async fn audio_transcriptions(State(state): State<AppState>, mut multipart: Multipart) -> Response {
    let mut model: Option<String> = None;
    let mut audio_bytes: Option<bytes::Bytes> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("model") => model = field.text().await.ok(),
            Some("file") => audio_bytes = field.bytes().await.ok(),
            _ => {}
        }
    }
    let model = match model {
        Some(m) => m,
        None => return bad_request("missing 'model' field".into(), "invalid_request"),
    };
    if audio_bytes.is_none() {
        return bad_request("missing 'file' field".into(), "invalid_request");
    }
    match resolve_route(&state, &model).await {
        Ok(()) => {
            let resp = TranscriptionResponse {
                text: String::new(),
                language: None,
                duration_ms: None,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(r) => r,
    }
}

/// `GET /v1/audio/transcriptions/stream` — STT streaming over WebSocket.
///
/// Client opens the socket, streams binary PCM frames; server responds
/// with JSON `TranscriptChunk` messages and closes when the source
/// signals end-of-stream.
async fn audio_transcriptions_stream(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| audio_transcriptions_stream_inner(state, socket))
}

async fn audio_transcriptions_stream_inner(_state: AppState, mut socket: WebSocket) {
    // M2 stub: accept frames, send an empty final transcript when the
    // client closes. Per-runtime bridging fills this in.
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            WsMessage::Binary(_) | WsMessage::Text(_) => continue,
            WsMessage::Close(_) => break,
            _ => continue,
        }
    }
    let final_chunk = serde_json::json!({
        "is_final": true,
        "text": "",
    });
    let _ = socket.send(WsMessage::Text(final_chunk.to_string())).await;
    let _ = socket.send(WsMessage::Close(None)).await;
}

/// `POST /v1/audio/speech` — TTS batch.
///
/// Returns the synthesized audio as a single chunked response body.
async fn audio_speech(State(state): State<AppState>, Json(req): Json<SpeechRequest>) -> Response {
    match resolve_route(&state, &req.model).await {
        Ok(()) => {
            let content_type = wav_content_type(&req.response_format);
            // M2 stub: returns a tiny silent WAV header so smoke-tests
            // see an `audio/*` content type and a non-empty body.
            let body = silent_wav_stub();
            (StatusCode::OK, [(header::CONTENT_TYPE, content_type)], body).into_response()
        }
        Err(r) => r,
    }
}

/// `POST /v1/audio/speech/stream` — TTS streaming over Server-Sent
/// Events.
///
/// Each `data:` line is a base64-encoded PCM frame; the stream
/// terminates with a single `data: [DONE]` line, matching OpenAI's
/// SSE shape.
async fn audio_speech_stream(State(state): State<AppState>, Json(req): Json<SpeechRequest>) -> Response {
    match resolve_route(&state, &req.model).await {
        Ok(()) => {
            let stream = audio_speech_sse_stub();
            Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
        }
        Err(r) => r,
    }
}

/// `GET /v1/realtime` — bidirectional realtime speech session.
async fn realtime_session(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| realtime_session_inner(state, socket))
}

async fn realtime_session_inner(_state: AppState, mut socket: WebSocket) {
    // M2 stub: echo any text frame back as an assistant turn marker.
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            WsMessage::Text(t) => {
                let echo = serde_json::json!({"type": "response.text", "text": t});
                if socket.send(WsMessage::Text(echo.to_string())).await.is_err() {
                    break;
                }
            }
            WsMessage::Binary(_) => continue,
            WsMessage::Close(_) => break,
            _ => continue,
        }
    }
    let _ = socket.send(WsMessage::Close(None)).await;
}

/// `GET /v1/audio2face` — audio → blendshape WebSocket session.
async fn audio2face_session(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| audio2face_session_inner(state, socket))
}

async fn audio2face_session_inner(_state: AppState, mut socket: WebSocket) {
    // M2 stub: drain inbound PCM, never emit a frame. Per-runtime
    // bridging in M11 connects this to `AudioEngineCoreActor`'s A2F arm.
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            WsMessage::Binary(_) => continue,
            WsMessage::Close(_) => break,
            _ => continue,
        }
    }
    let _ = socket.send(WsMessage::Close(None)).await;
}

/// Shared route-resolution helper for audio handlers. Returns `Err`
/// with a ready-to-return BAD_REQUEST response if the deployment is
/// missing or the coordinator is unreachable.
async fn resolve_route(state: &AppState, model: &str) -> Result<(), Response> {
    let model_s = model.to_string();
    let route = state
        .coordinator
        .ask_with(
            |reply| DpCoordinatorMsg::RouteTo {
                deployment: model_s,
                reply,
            },
            std::time::Duration::from_secs(2),
        )
        .await;
    match route {
        Ok(Ok(_target)) => Ok(()),
        Ok(Err(e)) => Err(bad_request(e.to_string(), "no_route")),
        Err(_) => Err(bad_request("coordinator timeout".into(), "internal_error")),
    }
}

fn wav_content_type(fmt: &str) -> &'static str {
    match fmt {
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "opus" | "ogg" => "audio/ogg",
        _ => "audio/wav",
    }
}

/// Tiny 44-byte RIFF/WAVE header with no PCM body. Lets smoke-tests
/// observe a non-empty `audio/wav` response; real runtimes write
/// audio into the body once wired.
fn silent_wav_stub() -> Vec<u8> {
    let mut v = Vec::with_capacity(44);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&36u32.to_le_bytes());
    v.extend_from_slice(b"WAVE");
    v.extend_from_slice(b"fmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&1u16.to_le_bytes()); // mono
    v.extend_from_slice(&16_000u32.to_le_bytes()); // 16 kHz
    v.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
    v.extend_from_slice(&2u16.to_le_bytes()); // block align
    v.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    v.extend_from_slice(b"data");
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

fn audio_speech_sse_stub() -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    stream::iter(vec![Ok(Event::default().data("[DONE]"))])
}
