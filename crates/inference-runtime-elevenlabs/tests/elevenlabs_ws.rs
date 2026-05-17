//! WebSocket integration tests for the ElevenLabs streaming path.
//!
//! Drives `ElevenLabsTtsRunner` against `MockWsServer` to verify:
//! - The runner connects, sends the `WsInitMessage` with the
//!   `xi-api-key` and `model_id` populated, and follows with an empty
//!   `WsTextMessage` flush.
//! - Inbound `audio` frames are base64-decoded into `SpeechChunk`s.
//! - Inbound `alignment` payloads round-trip into `AlignmentDelta`
//!   when `emit_alignment = true`.
//! - The terminal `is_final = true` frame closes the stream cleanly.
//!
//! Source: `FR-TTS-001` M7 acceptance row.

#![cfg(feature = "tts-elevenlabs")]

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
use atomr_infer_core::runner::SpeechRunner;
use atomr_infer_remote_core::http::build_client;
use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_runtime_elevenlabs::{ElevenLabsSecret, ElevenLabsTtsConfig, ElevenLabsTtsRunner};
use atomr_infer_testkit::MockWsServer;
use base64::Engine;
use futures::StreamExt;
use secrecy::SecretString;
use tokio_tungstenite::tungstenite::Message;
use url::Url;

const VOICE_ID: &str = "21m00Tcm4TlvDq8ikWAM";

fn build_runner(ws_url: &str) -> ElevenLabsTtsRunner {
    // The WS endpoint base is derived from `ws_url` by stripping the
    // mock server's trailing `/` and appending `/v1/`. The runner
    // joins `text-to-speech/{voice}/stream-input` onto it; because the
    // mock listens for *any* path on its ephemeral port, the actual
    // tail does not matter for routing.
    let trimmed = ws_url.trim_end_matches('/');
    let base = format!("{trimmed}/v1/");
    let cfg = ElevenLabsTtsConfig::defaults_for_elevenlabs(ElevenLabsSecret::Env {
        name: "ELEVEN_API_KEY".into(),
    })
    .with_endpoint(Url::parse("http://127.0.0.1:1/v1/").expect("placeholder"))
    .with_ws_endpoint(Url::parse(&base).expect("valid ws base"));
    let client = build_client(&Default::default(), "atomr-infer-test/0").expect("build client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("sk-elevenlabs-ws-fixture".to_string()),
    }));
    ElevenLabsTtsRunner::new(cfg, snap).expect("construct runner")
}

fn streaming_batch(emit_alignment: bool) -> SpeechBatch {
    SpeechBatch {
        request_id: "req-ws".into(),
        model: "eleven_turbo_v2_5".into(),
        text: "hi there".into(),
        voice: VoiceRef::Id(VOICE_ID.into()),
        options: SynthOptions::default(),
        stream: true,
        emit_alignment,
        estimated_characters: 8,
    }
}

#[tokio::test]
async fn ws_init_message_carries_credentials_and_model() {
    let server = MockWsServer::start().await.expect("ws server start");
    let url = server.url().to_string();

    // Spawn the speak call in the background; it will connect, send
    // the init + flush frames, and then await audio that we will
    // inject below.
    let speak_handle = tokio::spawn(async move {
        let mut runner = build_runner(&url);
        let handle = runner.speak(streaming_batch(false)).await.expect("speak ok");
        let mut stream = handle.into_stream();
        let mut all = Vec::new();
        while let Some(c) = stream.next().await {
            all.push(c.expect("ok chunk"));
        }
        all
    });

    // First inbound: the init JSON.
    let init = server.recv(Duration::from_secs(2)).await.expect("init frame");
    match init {
        Message::Text(t) => {
            assert!(t.contains("\"xi_api_key\":\"sk-elevenlabs-ws-fixture\""), "{t}");
            assert!(t.contains("\"model_id\":\"eleven_turbo_v2_5\""), "{t}");
            assert!(t.contains("\"text\":\"hi there\""), "{t}");
        }
        other => panic!("expected text init, got {other:?}"),
    }

    // Second inbound: the flush message.
    let flush = server.recv(Duration::from_secs(2)).await.expect("flush frame");
    match flush {
        Message::Text(t) => {
            assert!(t.contains("\"text\":\"\""), "{t}");
            assert!(t.contains("\"try_trigger_generation\":true"), "{t}");
        }
        other => panic!("expected text flush, got {other:?}"),
    }

    // Push a single final audio frame and let the runner finish.
    let b64 = base64::engine::general_purpose::STANDARD.encode(b"\x01\x02\x03\x04");
    let _ = server.send_json(serde_json::json!({
        "audio": b64,
        "is_final": true,
    }));

    let chunks = speak_handle.await.expect("join");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].audio_pcm_chunk.as_ref(), b"\x01\x02\x03\x04");
    assert!(chunks[0].is_final);
}

#[tokio::test]
async fn ws_alignment_round_trips_into_word_timings() {
    let server = MockWsServer::start().await.expect("ws server start");
    let url = server.url().to_string();

    let speak_handle = tokio::spawn(async move {
        let mut runner = build_runner(&url);
        let handle = runner.speak(streaming_batch(true)).await.expect("speak ok");
        let mut stream = handle.into_stream();
        let mut all = Vec::new();
        while let Some(c) = stream.next().await {
            all.push(c.expect("ok chunk"));
        }
        all
    });

    // Drain init + flush so the runner is ready to receive.
    let _ = server.recv(Duration::from_secs(2)).await.expect("init");
    let _ = server.recv(Duration::from_secs(2)).await.expect("flush");

    let b64_first = base64::engine::general_purpose::STANDARD.encode(b"\x10\x20");
    let _ = server.send_json(serde_json::json!({
        "audio": b64_first,
        "alignment": {
            "chars": ["h", "i"],
            "char_start_times_ms": [0, 50],
            "char_durations_ms": [50, 60],
        },
        "is_final": false,
    }));

    let b64_final = base64::engine::general_purpose::STANDARD.encode(b"\x30\x40");
    let _ = server.send_json(serde_json::json!({
        "audio": b64_final,
        "alignment": {
            "chars": [" ", "t"],
            "char_start_times_ms": [110, 130],
            "char_durations_ms": [20, 70],
        },
        "is_final": true,
    }));

    let chunks = speak_handle.await.expect("join");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].audio_pcm_chunk.as_ref(), b"\x10\x20");
    let a0 = chunks[0].alignment.as_ref().expect("alignment");
    assert_eq!(a0.words.len(), 2);
    assert_eq!(a0.words[0].text, "h");
    assert_eq!(a0.words[0].ts_start_ms, 0);
    assert_eq!(a0.words[0].ts_end_ms, 50);
    assert_eq!(a0.words[1].text, "i");
    assert_eq!(a0.words[1].ts_start_ms, 50);
    assert_eq!(a0.words[1].ts_end_ms, 110);

    assert_eq!(chunks[1].audio_pcm_chunk.as_ref(), b"\x30\x40");
    assert!(chunks[1].is_final);
    let a1 = chunks[1].alignment.as_ref().expect("alignment");
    assert_eq!(a1.words.len(), 2);
    assert_eq!(a1.words[1].text, "t");
    assert_eq!(a1.words[1].ts_start_ms, 130);
    assert_eq!(a1.words[1].ts_end_ms, 200);
}

#[tokio::test]
async fn ws_empty_ping_frames_are_ignored() {
    let server = MockWsServer::start().await.expect("ws server start");
    let url = server.url().to_string();

    let speak_handle = tokio::spawn(async move {
        let mut runner = build_runner(&url);
        let handle = runner.speak(streaming_batch(false)).await.expect("speak ok");
        let mut stream = handle.into_stream();
        let mut all = Vec::new();
        while let Some(c) = stream.next().await {
            all.push(c.expect("ok chunk"));
        }
        all
    });

    let _ = server.recv(Duration::from_secs(2)).await.expect("init");
    let _ = server.recv(Duration::from_secs(2)).await.expect("flush");

    // Push two empty pings, then the final audio frame.
    let _ = server.send_json(serde_json::json!({}));
    let _ = server.send_json(serde_json::json!({"audio": null}));
    let b64 = base64::engine::general_purpose::STANDARD.encode(b"\xAA");
    let _ = server.send_json(serde_json::json!({"audio": b64, "is_final": true}));

    let chunks = speak_handle.await.expect("join");
    assert_eq!(chunks.len(), 1, "pings filtered, only final chunk surfaced");
    assert_eq!(chunks[0].audio_pcm_chunk.as_ref(), b"\xAA");
    assert!(chunks[0].is_final);
}
