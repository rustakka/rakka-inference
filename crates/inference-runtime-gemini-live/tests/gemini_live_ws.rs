//! `MockWsServer`-driven integration tests for the Gemini Live runner.
//!
//! These cover:
//!
//! 1. Setup handshake: server observes `setup` envelope, sends
//!    `setupComplete`, runner only forwards user input after that.
//! 2. Text turn round-trip: client sends `RealtimeIn::Text("hi")`, server
//!    observes `clientContent` with `turnComplete:true`, server emits
//!    `serverContent` with inline audio + text in `modelTurn`, then
//!    `serverContent` with `turnComplete:true`. Client observes
//!    `RealtimeOut::AudioFrame` + `RealtimeOut::Transcript` + final
//!    transcript.
//! 3. Audio frame round-trip: client sends `RealtimeIn::AudioFrame`, server
//!    observes `realtimeInput` with `mediaChunks` carrying base64-encoded
//!    audio.
//! 4. Cancellation via `RealtimeSession::cancel()` — adapter shuts down,
//!    `outbound` closes.
//!
//! Each test follows the `MockWsServer::start()` → spawn-runner →
//! drive-server pattern from
//! `crates/inference-runtime-deepgram/tests/deepgram_ws.rs`.

#![cfg(feature = "tts-gemini-live")]

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use atomr_infer_core::audio::{
    AudioFormat, AudioParams, RealtimeBatch, RealtimeIn, RealtimeOut, SynthOptions, VoiceRef,
};
use atomr_infer_core::runner::RealtimeRunner;
use atomr_infer_remote_core::http::build_client;
use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_runtime_gemini_live::{GeminiLiveApiKey, GeminiLiveConfig, GeminiLiveRunner};
use atomr_infer_testkit::MockWsServer;
use base64::Engine as _;
use bytes::Bytes;
use secrecy::SecretString;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use url::Url;

fn build_runner(ws_url: &str) -> GeminiLiveRunner {
    let mut endpoint = Url::parse(ws_url).expect("ws url");
    // Ensure the base URL has a trailing slash so join() works correctly.
    if !endpoint.path().ends_with('/') {
        endpoint.set_path(&format!("{}/", endpoint.path()));
    }
    let cfg = GeminiLiveConfig::defaults_for_gemini_live(GeminiLiveApiKey::Env {
        name: "GEMINI_API_KEY".into(),
    })
    .with_ws_endpoint(endpoint);
    let client = build_client(&Default::default(), "gemini-live-test/0").expect("client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("fake-api-key".to_string()),
    }));
    GeminiLiveRunner::new(cfg, snap).expect("runner")
}

#[allow(dead_code)]
fn small_batch(
    tx_in: mpsc::Sender<RealtimeIn>,
    rx_in: mpsc::Receiver<RealtimeIn>,
    tx_out: mpsc::Sender<RealtimeOut>,
) -> (RealtimeBatch, mpsc::Receiver<RealtimeOut>) {
    let (_, rx_out_consumer) = {
        // We want a receiver on the outbound side for assertions.
        // Build a relay channel for the test consumer.
        mpsc::channel::<RealtimeOut>(32)
    };
    let _ = tx_in; // not used directly here; caller passes rx_in
    let batch = RealtimeBatch {
        request_id: "req-1".into(),
        model: "gemini-2.0-flash-exp".into(),
        voice: VoiceRef::Named("default".into()),
        options: SynthOptions::default(),
        inbound: rx_in,
        outbound: tx_out,
    };
    (batch, rx_out_consumer)
}

/// Convenience: drain the mock server inbound until we see the setup
/// envelope, reply with setupComplete, return the setup JSON.
async fn handshake(server: &MockWsServer) -> String {
    for _ in 0..10 {
        match server.recv(Duration::from_secs(2)).await {
            Some(Message::Text(t)) => {
                if t.contains("\"setup\"") {
                    server
                        .send_json(serde_json::json!({"setupComplete": {}}))
                        .expect("send setupComplete");
                    return t;
                }
            }
            Some(Message::Close(_)) | None => panic!("connection closed before setup"),
            _ => continue,
        }
    }
    panic!("never saw setup message");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Setup handshake
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn setup_handshake_observed_before_user_input() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let (tx_in, rx_in) = mpsc::channel::<RealtimeIn>(4);
    let (tx_out, mut rx_out) = mpsc::channel::<RealtimeOut>(16);

    let batch = RealtimeBatch {
        request_id: "req-setup".into(),
        model: "gemini-2.0-flash-exp".into(),
        voice: VoiceRef::Named("default".into()),
        options: SynthOptions::default(),
        inbound: rx_in,
        outbound: tx_out,
    };

    // Runner connects; server handles handshake.
    let server_task = tokio::spawn({
        let server = server;
        async move {
            let setup_text = handshake(&server).await;
            // Verify the setup message structure.
            let v: serde_json::Value = serde_json::from_str(&setup_text).unwrap();
            assert!(v.get("setup").is_some(), "setup key missing: {v}");
            assert!(
                v["setup"]["generationConfig"]["responseModalities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|m| m.as_str() == Some("AUDIO")),
                "AUDIO missing from responseModalities"
            );
            // Let the session close cleanly.
            server.close_with(1000, "done").expect("close");
        }
    });

    // open_session blocks until setupComplete is received.
    let session = runner
        .open_session(batch)
        .await
        .expect("open_session after handshake");

    // Send Close so the uplink drains.
    tx_in.send(RealtimeIn::Close).await.ok();
    session.cancel();

    // Drain outbound.
    while rx_out.try_recv().is_ok() {}
    server_task.await.expect("server task");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Text turn round-trip
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn text_turn_produces_audio_frame_and_transcript() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let (tx_in, rx_in) = mpsc::channel::<RealtimeIn>(4);
    let (tx_out, mut rx_out) = mpsc::channel::<RealtimeOut>(32);

    let batch = RealtimeBatch {
        request_id: "req-text".into(),
        model: "gemini-2.0-flash-exp".into(),
        voice: VoiceRef::Named("default".into()),
        options: SynthOptions::default(),
        inbound: rx_in,
        outbound: tx_out,
    };

    // Small 4-byte PCM frame (2 samples at 16-bit) — base64: "AQIDBA=="
    let fake_pcm = [1u8, 0, 2, 0];
    let fake_b64 = base64::engine::general_purpose::STANDARD.encode(fake_pcm);

    let server_task = tokio::spawn({
        let server = server;
        let fake_b64 = fake_b64.clone();
        async move {
            // Handshake.
            handshake(&server).await;

            // Wait for the clientContent message.
            let mut saw_client_content = false;
            for _ in 0..10 {
                match server.recv(Duration::from_secs(2)).await {
                    Some(Message::Text(t)) if t.contains("\"clientContent\"") => {
                        let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                        assert_eq!(
                            v["clientContent"]["turnComplete"].as_bool(),
                            Some(true),
                            "turnComplete missing"
                        );
                        saw_client_content = true;
                        break;
                    }
                    Some(Message::Text(_)) | Some(Message::Binary(_)) | Some(Message::Ping(_)) => continue,
                    _ => break,
                }
            }
            assert!(saw_client_content, "never saw clientContent from runner");

            // Server emits: modelTurn with audio + text.
            server
                .send_json(serde_json::json!({
                    "serverContent": {
                        "modelTurn": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/pcm;rate=24000",
                                        "data": fake_b64
                                    }
                                },
                                {
                                    "text": "Hello!"
                                }
                            ]
                        }
                    }
                }))
                .expect("send modelTurn");

            // Server emits: turnComplete.
            server
                .send_json(serde_json::json!({
                    "serverContent": {
                        "turnComplete": true
                    }
                }))
                .expect("send turnComplete");

            // Close.
            server.close_with(1000, "done").expect("close");
        }
    });

    let session = runner.open_session(batch).await.expect("open_session");

    // Send text turn.
    tx_in
        .send(RealtimeIn::Text("hi".into()))
        .await
        .expect("send text");
    tx_in.send(RealtimeIn::Close).await.ok();

    // Collect outbound messages.
    let mut saw_audio = false;
    let mut saw_transcript = false;
    let mut saw_final = false;
    let mut saw_done = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(1), rx_out.recv()).await {
            Ok(Some(RealtimeOut::AudioFrame { pcm, params })) => {
                assert_eq!(&*pcm, &fake_pcm[..]);
                assert_eq!(params, AudioParams::new(24_000, 1, AudioFormat::Pcm16Le));
                saw_audio = true;
            }
            Ok(Some(RealtimeOut::Transcript { text, is_final, .. })) => {
                if is_final {
                    saw_final = true;
                } else {
                    assert_eq!(text, "Hello!");
                    saw_transcript = true;
                }
            }
            Ok(Some(RealtimeOut::Done)) => {
                saw_done = true;
                break;
            }
            Ok(Some(_)) => continue,
            Ok(None) | Err(_) => break,
        }
    }

    session.cancel();
    server_task.await.expect("server task");

    assert!(saw_audio, "expected AudioFrame");
    assert!(saw_transcript, "expected non-final Transcript");
    assert!(saw_final, "expected final Transcript");
    assert!(saw_done, "expected Done");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Audio frame round-trip
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn audio_frame_arrives_at_server_as_realtime_input() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let pcm_bytes = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
    let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&pcm_bytes);

    let (tx_in, rx_in) = mpsc::channel::<RealtimeIn>(4);
    let (tx_out, mut rx_out) = mpsc::channel::<RealtimeOut>(16);

    let batch = RealtimeBatch {
        request_id: "req-audio".into(),
        model: "gemini-2.0-flash-exp".into(),
        voice: VoiceRef::Named("default".into()),
        options: SynthOptions::default(),
        inbound: rx_in,
        outbound: tx_out,
    };

    let server_task = tokio::spawn({
        let server = server;
        let expected_b64 = expected_b64.clone();
        async move {
            handshake(&server).await;

            // Wait for the realtimeInput message.
            let mut saw_realtime_input = false;
            for _ in 0..10 {
                match server.recv(Duration::from_secs(2)).await {
                    Some(Message::Text(t)) if t.contains("\"realtimeInput\"") => {
                        let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                        let chunks = v["realtimeInput"]["mediaChunks"]
                            .as_array()
                            .expect("mediaChunks array");
                        assert!(!chunks.is_empty(), "expected at least one mediaChunk");
                        let chunk = &chunks[0];
                        assert!(
                            chunk["mimeType"].as_str().unwrap_or("").starts_with("audio/pcm"),
                            "mimeType: {}",
                            chunk["mimeType"]
                        );
                        assert_eq!(
                            chunk["data"].as_str().unwrap_or(""),
                            expected_b64,
                            "base64 data mismatch"
                        );
                        saw_realtime_input = true;
                        break;
                    }
                    Some(_) => continue,
                    None => break,
                }
            }
            assert!(saw_realtime_input, "never saw realtimeInput from runner");
            server.close_with(1000, "done").expect("close");
        }
    });

    let session = runner.open_session(batch).await.expect("open_session");

    // Send an audio frame.
    tx_in
        .send(RealtimeIn::AudioFrame {
            pcm: Bytes::from(pcm_bytes),
            params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
        })
        .await
        .expect("send audio frame");
    tx_in.send(RealtimeIn::Close).await.ok();

    // Drain outbound.
    tokio::time::timeout(Duration::from_secs(3), async {
        while rx_out.recv().await.is_some() {}
    })
    .await
    .ok();

    session.cancel();
    server_task.await.expect("server task");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Cancellation via RealtimeSession::cancel()
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn cancel_shuts_down_adapter_and_closes_outbound() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let (_tx_in, rx_in) = mpsc::channel::<RealtimeIn>(4);
    let (tx_out, mut rx_out) = mpsc::channel::<RealtimeOut>(16);

    let batch = RealtimeBatch {
        request_id: "req-cancel".into(),
        model: "gemini-2.0-flash-exp".into(),
        voice: VoiceRef::Named("default".into()),
        options: SynthOptions::default(),
        inbound: rx_in,
        outbound: tx_out,
    };

    let server_task = tokio::spawn({
        let server = server;
        async move {
            handshake(&server).await;
            // Keep the server alive for a moment; the runner will be cancelled.
            tokio::time::sleep(Duration::from_millis(200)).await;
            server.close_with(1000, "done").expect("close");
        }
    });

    let session = runner.open_session(batch).await.expect("open_session");

    // Cancel immediately after opening.
    session.cancel();

    // After cancellation the outbound channel should drain / close.
    let result = tokio::time::timeout(Duration::from_secs(3), rx_out.recv()).await;
    // Either the channel is closed (None) or we get Done — both are acceptable.
    match result {
        Ok(None) | Ok(Some(RealtimeOut::Done)) => {} // expected
        Ok(Some(other)) => {
            // Could see an error or Done before channel closes — acceptable.
            let _ = other;
        }
        Err(_timeout) => {
            // Timeout is unexpected — the adapter should have torn down quickly.
            panic!("timeout waiting for outbound to close after cancel");
        }
    }

    server_task.await.expect("server task");
}
