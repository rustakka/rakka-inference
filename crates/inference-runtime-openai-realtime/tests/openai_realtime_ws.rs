//! Integration tests for `atomr-infer-runtime-openai-realtime`.
//!
//! These tests use [`MockWsServer`] to simulate the OpenAI Realtime API
//! endpoint without hitting the network.  All tests are compiled only when
//! the `tts-openai-realtime` feature is enabled.

#![cfg(feature = "tts-openai-realtime")]

use std::time::Duration;

use atomr_infer_core::audio::{
    AudioFormat, AudioParams, RealtimeBatch, RealtimeIn, RealtimeOut, SynthOptions, TranscriptRole, VoiceRef,
};
use atomr_infer_core::runner::RealtimeRunner;
use atomr_infer_runtime_openai_realtime::{
    config::{ApiKeySource, OpenAiRealtimeConfig},
    runner::OpenAiRealtimeRunner,
};
use atomr_infer_testkit::MockWsServer;
use tokio::sync::mpsc;

fn inline_key_config(ws_url: &str) -> OpenAiRealtimeConfig {
    OpenAiRealtimeConfig {
        api_key: ApiKeySource::Inline {
            value: "sk-test".into(),
        },
        // The mock URL is plain ws:// — no ?model= needed, MockWsServer ignores it
        endpoint: ws_url.to_string(),
        handshake_timeout_ms: 5_000,
    }
}

// ---------------------------------------------------------------------------
// Test 1: Session-update handshake
// ---------------------------------------------------------------------------

/// After `open_session`, the server should observe a `session.update` event
/// before any input from the caller.
#[tokio::test]
async fn test_session_update_handshake() {
    let srv = MockWsServer::start().await.expect("start mock ws server");
    let ws_url = srv.url();

    let cfg = inline_key_config(ws_url);
    let mut runner = OpenAiRealtimeRunner::new(cfg);

    let (_in_tx, in_rx) = mpsc::channel(64);
    let (out_tx, _out_rx) = mpsc::channel(64);

    let batch = RealtimeBatch::new(
        "req-handshake",
        "gpt-4o-realtime-preview",
        VoiceRef::Named("alloy".into()),
        SynthOptions::default(),
        in_rx,
        out_tx,
    );

    let session = runner.open_session(batch).await.unwrap();

    // The very first frame sent by the client should be session.update
    let first = srv.recv_json(Duration::from_secs(2)).await.unwrap();
    assert_eq!(
        first["type"], "session.update",
        "first frame must be session.update"
    );
    assert_eq!(first["session"]["voice"], "alloy");
    assert_eq!(first["session"]["input_audio_format"], "pcm16");
    assert_eq!(first["session"]["output_audio_format"], "pcm16");

    // Clean up
    session.cancel();
}

// ---------------------------------------------------------------------------
// Test 2: Text turn round-trip
// ---------------------------------------------------------------------------

/// Client sends `RealtimeIn::Text("hi")`, server observes
/// `conversation.item.create` + `response.create`, server emits
/// `response.audio.delta` + `response.audio_transcript.done` +
/// `response.done`, client observes `AudioFrame` + `Transcript` + `Done`.
#[tokio::test]
async fn test_text_turn_round_trip() {
    let srv = MockWsServer::start().await.expect("start mock ws server");
    let ws_url = srv.url();

    let cfg = inline_key_config(ws_url);
    let mut runner = OpenAiRealtimeRunner::new(cfg);

    let (in_tx, in_rx) = mpsc::channel(64);
    let (out_tx, mut out_rx) = mpsc::channel(64);

    let batch = RealtimeBatch::new(
        "req-text",
        "gpt-4o-realtime-preview",
        VoiceRef::Named("alloy".into()),
        SynthOptions::default(),
        in_rx,
        out_tx,
    );

    let session = runner.open_session(batch).await.unwrap();

    // Consume the session.update handshake frame
    let _su = srv.recv_json(Duration::from_secs(2)).await.unwrap();

    // Send text turn from caller
    in_tx.send(RealtimeIn::Text("hi".into())).await.unwrap();

    // Server observes conversation.item.create
    let item_frame = srv.recv_json(Duration::from_secs(2)).await.unwrap();
    assert_eq!(item_frame["type"], "conversation.item.create");
    assert_eq!(item_frame["item"]["type"], "message");
    assert_eq!(item_frame["item"]["role"], "user");
    assert_eq!(item_frame["item"]["content"][0]["text"], "hi");

    // Server observes response.create
    let resp_frame = srv.recv_json(Duration::from_secs(2)).await.unwrap();
    assert_eq!(resp_frame["type"], "response.create");

    // Server sends back audio delta (base64 of [0x01, 0x02])
    use base64::Engine as _;
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode([0x01u8, 0x02]);
    srv.send_json(serde_json::json!({
        "type": "response.audio.delta",
        "delta": audio_b64,
    }))
    .unwrap();

    // Server sends transcript done
    srv.send_json(serde_json::json!({
        "type": "response.audio_transcript.done",
        "transcript": "hello there",
    }))
    .unwrap();

    // Server sends response.done
    srv.send_json(serde_json::json!({"type": "response.done"}))
        .unwrap();

    // Client observes AudioFrame
    let out1 = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for AudioFrame")
        .unwrap();
    assert!(matches!(out1, RealtimeOut::AudioFrame { .. }));
    if let RealtimeOut::AudioFrame { pcm, params } = out1 {
        assert_eq!(pcm.as_ref(), &[0x01u8, 0x02]);
        assert_eq!(params.sample_rate_hz, 24_000);
    }

    // Client observes Transcript (is_final=true, role=Assistant)
    let out2 = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for Transcript")
        .unwrap();
    assert!(matches!(
        out2,
        RealtimeOut::Transcript {
            role: TranscriptRole::Assistant,
            is_final: true,
            ..
        }
    ));
    if let RealtimeOut::Transcript { text, .. } = out2 {
        assert_eq!(text, "hello there");
    }

    // Client observes Done
    let out3 = tokio::time::timeout(Duration::from_secs(2), out_rx.recv())
        .await
        .expect("timeout waiting for Done")
        .unwrap();
    assert!(matches!(out3, RealtimeOut::Done));

    session.cancel();
}

// ---------------------------------------------------------------------------
// Test 3: Audio frame round-trip
// ---------------------------------------------------------------------------

/// Client sends `RealtimeIn::AudioFrame`, server observes
/// `input_audio_buffer.append` with base64-encoded PCM.
#[tokio::test]
async fn test_audio_frame_round_trip() {
    let srv = MockWsServer::start().await.expect("start mock ws server");
    let ws_url = srv.url();

    let cfg = inline_key_config(ws_url);
    let mut runner = OpenAiRealtimeRunner::new(cfg);

    let (in_tx, in_rx) = mpsc::channel(64);
    let (out_tx, _out_rx) = mpsc::channel(64);

    let batch = RealtimeBatch::new(
        "req-audio",
        "gpt-4o-realtime-preview",
        VoiceRef::Named("alloy".into()),
        SynthOptions::default(),
        in_rx,
        out_tx,
    );

    let session = runner.open_session(batch).await.unwrap();

    // Consume the session.update handshake
    let _su = srv.recv_json(Duration::from_secs(2)).await.unwrap();

    // Send audio frame
    let pcm = bytes::Bytes::from(vec![0xABu8, 0xCD, 0xEF]);
    let params = AudioParams::new(16_000, 1, AudioFormat::Pcm16Le);
    in_tx
        .send(RealtimeIn::AudioFrame {
            pcm: pcm.clone(),
            params,
        })
        .await
        .unwrap();

    // Server observes input_audio_buffer.append with correct base64
    let frame = srv.recv_json(Duration::from_secs(2)).await.unwrap();
    assert_eq!(frame["type"], "input_audio_buffer.append");

    use base64::Engine as _;
    let expected_b64 = base64::engine::general_purpose::STANDARD.encode(&pcm);
    assert_eq!(frame["audio"], expected_b64);

    session.cancel();
}

// ---------------------------------------------------------------------------
// Test 4: Interrupt
// ---------------------------------------------------------------------------

/// Client sends `RealtimeIn::Interrupt`, server observes `response.cancel`.
#[tokio::test]
async fn test_interrupt() {
    let srv = MockWsServer::start().await.expect("start mock ws server");
    let ws_url = srv.url();

    let cfg = inline_key_config(ws_url);
    let mut runner = OpenAiRealtimeRunner::new(cfg);

    let (in_tx, in_rx) = mpsc::channel(64);
    let (out_tx, _out_rx) = mpsc::channel(64);

    let batch = RealtimeBatch::new(
        "req-interrupt",
        "gpt-4o-realtime-preview",
        VoiceRef::Named("alloy".into()),
        SynthOptions::default(),
        in_rx,
        out_tx,
    );

    let session = runner.open_session(batch).await.unwrap();
    let _su = srv.recv_json(Duration::from_secs(2)).await.unwrap();

    in_tx.send(RealtimeIn::Interrupt).await.unwrap();

    let frame = srv.recv_json(Duration::from_secs(2)).await.unwrap();
    assert_eq!(frame["type"], "response.cancel");

    session.cancel();
}

// ---------------------------------------------------------------------------
// Test 5: Cancellation via RealtimeSession::cancel()
// ---------------------------------------------------------------------------

/// `RealtimeSession::cancel()` aborts the adapter task; the session is
/// shut down cleanly.
#[tokio::test]
async fn test_cancellation() {
    let srv = MockWsServer::start().await.expect("start mock ws server");
    let ws_url = srv.url();

    let cfg = inline_key_config(ws_url);
    let mut runner = OpenAiRealtimeRunner::new(cfg);

    let (_in_tx, in_rx) = mpsc::channel(64);
    let (out_tx, _out_rx) = mpsc::channel(64);

    let batch = RealtimeBatch::new(
        "req-cancel",
        "gpt-4o-realtime-preview",
        VoiceRef::Named("alloy".into()),
        SynthOptions::default(),
        in_rx,
        out_tx,
    );

    let session = runner.open_session(batch).await.unwrap();
    let _su = srv.recv_json(Duration::from_secs(2)).await.unwrap();

    // Cancel the session — the abort handle is consumed, task is aborted.
    session.cancel();

    // Give the task time to notice the abort
    tokio::time::sleep(Duration::from_millis(100)).await;
    // Test passes: no panic; session was cancelled cleanly.
}

// ---------------------------------------------------------------------------
// Test 6: Voice cloning rejection
// ---------------------------------------------------------------------------

/// `VoiceRef::ClonedFrom` must produce `InferenceError::BadRequest` before
/// the session even connects.
#[tokio::test]
async fn test_voice_clone_rejected() {
    let srv = MockWsServer::start().await.expect("start mock ws server");
    let ws_url = srv.url();

    let cfg = inline_key_config(ws_url);
    let mut runner = OpenAiRealtimeRunner::new(cfg);

    let (_in_tx, in_rx) = mpsc::channel(64);
    let (out_tx, _out_rx) = mpsc::channel(64);

    let batch = RealtimeBatch::new(
        "req-clone",
        "gpt-4o-realtime-preview",
        VoiceRef::ClonedFrom(atomr_infer_core::audio::AudioPayload::Bytes {
            data: bytes::Bytes::new(),
            params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
        }),
        SynthOptions::default(),
        in_rx,
        out_tx,
    );

    let err = runner.open_session(batch).await.unwrap_err();
    assert!(matches!(
        err,
        atomr_infer_core::error::InferenceError::BadRequest { .. }
    ));
}

#[tokio::test]
async fn test_ws_client_direct() {
    use atomr_infer_runtime_ws_core::WsClient;
    let srv = MockWsServer::start().await.expect("start mock ws server");
    let url = srv.url();

    let result =
        WsClient::connect_with_headers(url, &[("Authorization", "Bearer test")], Duration::from_secs(5))
            .await;

    assert!(result.is_ok(), "WsClient failed: {:?}", result.err());

    srv.send_text("hello").unwrap();
    let msg = srv.recv(Duration::from_secs(2)).await;
    drop(msg); // just check we got here
}

#[tokio::test]
async fn test_ws_client_with_model_url_and_beta_header() {
    use atomr_infer_runtime_ws_core::WsClient;
    let srv = MockWsServer::start().await.expect("start mock ws server");
    // Use model URL format like the runner does — note the / before ? is required
    let url = format!("{}/?model=gpt-4o-realtime-preview", srv.url());

    let result = WsClient::connect_with_headers(
        &url,
        &[
            ("Authorization", "Bearer sk-test"),
            ("OpenAI-Beta", "realtime=v1"),
        ],
        Duration::from_secs(5),
    )
    .await;

    assert!(
        result.is_ok(),
        "WsClient with model URL failed: {:?}",
        result.err()
    );

    srv.send_text(r#"{"type":"test"}"#).unwrap();
    let _msg = srv.recv(Duration::from_secs(2)).await;
}
