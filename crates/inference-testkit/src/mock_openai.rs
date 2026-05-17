//! `wiremock`-backed OpenAI mock.
//!
//! Spins up a local HTTP server that speaks the Chat Completions wire
//! format (SSE + JSON envelope). Tests pass the server's URL into
//! `OpenAiConfig::with_endpoint(...)` and exercise rate-limiting,
//! circuit-breaker and retry semantics against deterministic responses.

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

pub struct MockOpenAi {
    pub server: MockServer,
}

impl MockOpenAi {
    pub async fn start() -> Self {
        Self {
            server: MockServer::start().await,
        }
    }

    pub fn url(&self) -> String {
        self.server.uri()
    }
}

/// Mount a happy-path Chat Completions handler returning a single SSE
/// chunk + `[DONE]`. Useful as a baseline for gateway integration
/// tests that don't care about the model's output beyond
/// "did anything come back".
pub async fn mount_chat_happy_path(server: &MockServer, content: &str) {
    let body = format!(
        "data: {{\"id\":\"mock\",\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":1,\"completion_tokens\":1}}}}\n\ndata: [DONE]\n\n",
        serde_json::Value::String(content.to_string())
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(server)
        .await;
}

/// Mount a one-shot 429 — first request returns 429 with a 1-second
/// `Retry-After`, subsequent requests fall through to the previously
/// mounted handlers.
pub async fn inject_429_once(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "1")
                .set_body_string("rate limited"),
        )
        .up_to_n_times(1)
        .mount(server)
        .await;
}

/// Mount N consecutive 5xx responses — used to drive the circuit
/// breaker into Open in tests.
pub async fn inject_5xx_once(server: &MockServer, n: u64) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("upstream busy"))
        .up_to_n_times(n)
        .mount(server)
        .await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio helpers — `FR-TTS-001`, `FR-STT-001`
// ─────────────────────────────────────────────────────────────────────────────

/// Mount a happy-path TTS handler returning a fixed WAV byte body for
/// `POST /v1/audio/speech`. The response uses
/// `Content-Type: audio/wav`.
pub async fn mount_audio_speech_happy_path(server: &MockServer, audio_bytes: Vec<u8>) {
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/wav")
                .set_body_bytes(audio_bytes),
        )
        .mount(server)
        .await;
}

/// Mount a happy-path STT handler returning a fixed transcript JSON
/// envelope for `POST /v1/audio/transcriptions` (multipart upload).
pub async fn mount_audio_transcriptions_happy_path(server: &MockServer, transcript: &str) {
    let body = serde_json::json!({ "text": transcript });
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(body),
        )
        .mount(server)
        .await;
}

/// Mount a one-shot 429 — first audio-modality request returns 429
/// with a 1-second `Retry-After`, subsequent requests fall through to
/// the previously mounted handlers. Matches both speech and
/// transcription paths.
pub async fn inject_audio_429(server: &MockServer) {
    Mock::given(method("POST"))
        .and(wiremock::matchers::path_regex(
            "^/v1/audio/(speech|transcriptions)$",
        ))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "1")
                .set_body_string("audio rate limited"),
        )
        .up_to_n_times(1)
        .mount(server)
        .await;
}
