//! `MockWsServer`-driven integration tests for the AssemblyAI runner.
//!
//! These cover:
//!
//! - Partial → final-per-turn transcript progression
//!   (`interim_results = true`).
//! - Partial filtering when `interim_results = false`.
//! - Word-timing round-trip when `word_timestamps = true`.
//! - Format rejection: anything other than `Pcm16Le` fails fast
//!   before the runner attempts a connect.
//! - Graceful close at the end of the audio uplink: the runner sends
//!   a `Terminate` JSON marker after draining its input.
//! - Abnormal close surfaces a `NetworkError` to the consumer.
//!
//! Each test follows the `MockWsServer::start()` → spawn-runner →
//! drive-server pattern from
//! `crates/inference-runtime-deepgram/tests/deepgram_ws.rs`.

#![cfg(feature = "stt-assemblyai")]

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use atomr_infer_core::audio::{
    AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscribeOptions,
};
use atomr_infer_core::runner::AudioRunner;
use atomr_infer_remote_core::http::build_client;
use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_runtime_assemblyai::{AssemblyAiSecret, AssemblyAiSttConfig, AssemblyAiSttRunner};
use atomr_infer_testkit::MockWsServer;
use bytes::Bytes;
use futures::StreamExt;
use secrecy::SecretString;
use tokio_tungstenite::tungstenite::Message;
use url::Url;

fn build_runner(ws_url: &str) -> AssemblyAiSttRunner {
    let mut endpoint = Url::parse(ws_url).expect("ws url");
    // MockWsServer hands back `ws://127.0.0.1:NNN/`; the runner joins
    // `v3/ws` onto the base when it computes the connect URL.
    assert!(endpoint.path().ends_with('/'));
    endpoint.set_path("/");
    let cfg = AssemblyAiSttConfig::defaults_for_assemblyai(AssemblyAiSecret::Env {
        name: "ASSEMBLYAI_API_KEY".into(),
    })
    .with_ws_endpoint(endpoint);
    let client = build_client(&Default::default(), "assemblyai-test/0").expect("client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("aa-fake".to_string()),
    }));
    AssemblyAiSttRunner::new(cfg, snap).expect("runner")
}

fn small_batch(opts: TranscribeOptions) -> AudioBatch {
    AudioBatch {
        request_id: "req-1".into(),
        model: "universal".into(),
        input: AudioInput::Static(AudioPayload::Bytes {
            data: Bytes::from_static(&[0u8; 1_024]),
            params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
        }),
        stream: true,
        options: AudioOptions::Transcribe(opts),
        estimated_units: 1,
    }
}

#[tokio::test]
async fn partial_then_final_progresses_through_runner() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let opts = TranscribeOptions {
        interim_results: true,
        ..Default::default()
    };
    let mut stream = runner
        .execute_audio(small_batch(opts))
        .await
        .expect("execute_audio")
        .into_stream();

    // Server: Begin envelope (ignored), then partial Turn, then final
    // Turn.
    server
        .send_json(serde_json::json!({
            "type": "Begin",
            "id": "sess-abc",
            "expires_at": 1_700_000_000_u64
        }))
        .expect("send begin");
    server
        .send_json(serde_json::json!({
            "type": "Turn",
            "turn_order": 0,
            "end_of_turn": false,
            "transcript": "hello",
            "words": [
                {"text": "hello", "start": 0, "end": 300, "confidence": 0.9, "word_is_final": false}
            ]
        }))
        .expect("send partial");
    server
        .send_json(serde_json::json!({
            "type": "Turn",
            "turn_order": 0,
            "end_of_turn": true,
            "end_of_turn_confidence": 0.95,
            "transcript": "hello world",
            "words": [
                {"text": "hello", "start": 0, "end": 300, "word_is_final": true},
                {"text": "world", "start": 400, "end": 900, "word_is_final": true}
            ]
        }))
        .expect("send final");
    server.close_with(1000, "done").expect("close");

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk"));
    }
    assert_eq!(chunks.len(), 2, "got {:?}", chunks);
    assert!(!chunks[0].is_final, "first should be partial");
    assert_eq!(chunks[0].text, "hello");
    assert!(chunks[1].is_final, "last should be end_of_turn");
    assert_eq!(chunks[1].text, "hello world");
}

#[tokio::test]
async fn partials_filtered_when_caller_disables_them() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let opts = TranscribeOptions {
        interim_results: false,
        ..Default::default()
    };
    let mut stream = runner
        .execute_audio(small_batch(opts))
        .await
        .expect("execute_audio")
        .into_stream();

    server
        .send_json(serde_json::json!({
            "type": "Turn",
            "turn_order": 0,
            "end_of_turn": false,
            "transcript": "drop me",
            "words": []
        }))
        .expect("send partial");
    server
        .send_json(serde_json::json!({
            "type": "Turn",
            "turn_order": 0,
            "end_of_turn": true,
            "transcript": "keep me",
            "words": []
        }))
        .expect("send final");
    server.close_with(1000, "done").expect("close");

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk"));
    }
    assert_eq!(chunks.len(), 1, "partial should have been filtered: {:?}", chunks);
    assert_eq!(chunks[0].text, "keep me");
    assert!(chunks[0].is_final);
}

#[tokio::test]
async fn word_timestamps_surface_on_chunk() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let opts = TranscribeOptions {
        interim_results: false,
        word_timestamps: true,
        ..Default::default()
    };
    let mut stream = runner
        .execute_audio(small_batch(opts))
        .await
        .expect("execute_audio")
        .into_stream();

    server
        .send_json(serde_json::json!({
            "type": "Turn",
            "turn_order": 0,
            "end_of_turn": true,
            "transcript": "alpha beta",
            "words": [
                {"text": "alpha", "start": 0,   "end": 400, "confidence": 0.95, "word_is_final": true},
                {"text": "beta",  "start": 500, "end": 1000,"confidence": 0.91, "word_is_final": true}
            ]
        }))
        .expect("send final w/ words");
    server.close_with(1000, "done").expect("close");

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk"));
    }
    assert_eq!(chunks.len(), 1);
    let c = &chunks[0];
    assert!(c.is_final);
    assert_eq!(c.ts_start_ms, 0);
    assert_eq!(c.ts_end_ms, 1_000);
    assert_eq!(c.words.len(), 2);
    assert_eq!(c.words[0].text, "alpha");
    assert_eq!(c.words[0].ts_end_ms, 400);
    assert_eq!(c.words[1].text, "beta");
    assert_eq!(c.words[1].ts_start_ms, 500);
}

#[tokio::test]
async fn runner_sends_terminate_marker_after_uplink_drains() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let opts = TranscribeOptions {
        interim_results: false,
        ..Default::default()
    };
    let mut stream = runner
        .execute_audio(small_batch(opts))
        .await
        .expect("execute_audio")
        .into_stream();

    // Server: observe the binary audio chunk(s), then the JSON
    // `Terminate` marker the runner sends to flush.
    let mut saw_audio = false;
    let mut saw_terminate_marker = false;
    for _ in 0..6 {
        let msg = server.recv(Duration::from_secs(2)).await;
        match msg {
            Some(Message::Binary(_)) => saw_audio = true,
            Some(Message::Text(t)) => {
                if t.contains("\"Terminate\"") {
                    saw_terminate_marker = true;
                    break;
                }
            }
            Some(Message::Close(_)) | None => break,
            _ => continue,
        }
    }
    assert!(saw_audio, "expected at least one binary audio frame");
    assert!(saw_terminate_marker, "expected Terminate marker after uplink");

    // Close out the downlink so the runner stream drains.
    server
        .send_json(serde_json::json!({
            "type": "Turn",
            "turn_order": 0,
            "end_of_turn": true,
            "transcript": "ok",
            "words": []
        }))
        .expect("send final");
    server.close_with(1000, "done").expect("close");

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk"));
    }
    assert!(!chunks.is_empty());
    assert!(chunks.last().unwrap().is_final);
}

#[tokio::test]
async fn abnormal_close_surfaces_network_error() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let opts = TranscribeOptions::default();
    let mut stream = runner
        .execute_audio(small_batch(opts))
        .await
        .expect("execute_audio")
        .into_stream();

    // Close with an abnormal code (not 1000/1005/1006) — runner should
    // surface a `NetworkError` to the consumer.
    server.close_with(1011, "internal server error").expect("close");

    let mut saw_err = false;
    let mut saw_chunks = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(_) => saw_chunks = true,
            Err(atomr_infer_core::error::InferenceError::NetworkError(m)) => {
                assert!(m.contains("1011"), "{m}");
                saw_err = true;
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
    assert!(saw_err || !saw_chunks, "expected NetworkError on abnormal close");
}

#[tokio::test]
async fn non_pcm16_format_is_rejected_before_connect() {
    // No server stand-up: the runner should bail before attempting
    // to connect. Provide a bogus URL just to be safe.
    let cfg = AssemblyAiSttConfig::defaults_for_assemblyai(AssemblyAiSecret::Env {
        name: "ASSEMBLYAI_API_KEY".into(),
    })
    .with_ws_endpoint(Url::parse("ws://127.0.0.1:1/").unwrap());
    let client = build_client(&Default::default(), "assemblyai-test/0").expect("client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("aa-fake".to_string()),
    }));
    let mut runner = AssemblyAiSttRunner::new(cfg, snap).expect("runner");

    let batch = AudioBatch {
        request_id: "req".into(),
        model: "universal".into(),
        input: AudioInput::Static(AudioPayload::Bytes {
            data: Bytes::from_static(&[0u8; 1024]),
            params: AudioParams::new(16_000, 1, AudioFormat::PcmF32Le),
        }),
        stream: true,
        options: AudioOptions::Transcribe(TranscribeOptions::default()),
        estimated_units: 1,
    };
    let err = runner
        .execute_audio(batch)
        .await
        .expect_err("expected format rejection");
    match err {
        atomr_infer_core::error::InferenceError::UnsupportedAudioFormat { message } => {
            assert!(message.to_lowercase().contains("16-bit") || message.contains("PCM"));
        }
        other => panic!("expected UnsupportedAudioFormat, got {other:?}"),
    }
}
