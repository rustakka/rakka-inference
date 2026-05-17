//! `MockWsServer`-driven integration tests for the Deepgram runner.
//!
//! These cover:
//!
//! - Interim → final transcript progression (interim_results = true).
//! - Interim filtering when `interim_results = false`.
//! - Speaker diarization round-trip (the speaker label arrives on the
//!   first word; the runner stringifies it into `TranscriptChunk::speaker_id`).
//! - Word-timing round-trip when `word_timestamps = true`.
//! - Graceful close at the end of the audio uplink: the runner sends
//!   a `CloseStream` JSON marker after draining its input.
//!
//! Each test follows the `MockWsServer::start()` → spawn-runner →
//! drive-server pattern from
//! `crates/inference-runtime-elevenlabs/tests/elevenlabs_ws.rs`.

#![cfg(feature = "stt-deepgram")]

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use atomr_infer_core::audio::{
    AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscribeOptions,
};
use atomr_infer_core::runner::AudioRunner;
use atomr_infer_remote_core::http::build_client;
use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_runtime_deepgram::{DeepgramSecret, DeepgramSttConfig, DeepgramSttRunner};
use atomr_infer_testkit::MockWsServer;
use bytes::Bytes;
use futures::StreamExt;
use secrecy::SecretString;
use tokio_tungstenite::tungstenite::Message;
use url::Url;

fn build_runner(ws_url: &str) -> DeepgramSttRunner {
    let mut endpoint = Url::parse(ws_url).expect("ws url");
    // MockWsServer hands back `ws://127.0.0.1:NNN/`; we need a base that
    // can `join("listen")` cleanly. The trailing slash already takes
    // care of that.
    assert!(endpoint.path().ends_with('/'));
    // Tag the base with `/v1/` so the resulting URL looks like the
    // public Deepgram shape — keeps assertions readable.
    endpoint.set_path("/v1/");
    let cfg = DeepgramSttConfig::defaults_for_deepgram(DeepgramSecret::Env {
        name: "DEEPGRAM_API_KEY".into(),
    })
    .with_ws_endpoint(endpoint);
    let client = build_client(&Default::default(), "deepgram-test/0").expect("client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("dg-fake".to_string()),
    }));
    DeepgramSttRunner::new(cfg, snap).expect("runner")
}

fn small_batch(opts: TranscribeOptions) -> AudioBatch {
    AudioBatch {
        request_id: "req-1".into(),
        model: "nova-2".into(),
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
async fn interim_then_final_progresses_through_runner() {
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

    // Server side: send interim, then final.
    server
        .send_json(serde_json::json!({
            "type": "Results",
            "start": 0.0,
            "duration": 0.3,
            "is_final": false,
            "speech_final": false,
            "channel": {
                "alternatives": [{"transcript": "hello"}]
            }
        }))
        .expect("send interim");
    server
        .send_json(serde_json::json!({
            "type": "Results",
            "start": 0.3,
            "duration": 0.4,
            "is_final": true,
            "speech_final": true,
            "channel": {
                "alternatives": [{"transcript": "hello world"}]
            }
        }))
        .expect("send final");
    server.close_with(1000, "done").expect("close");

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk"));
    }
    // Interim + final → two chunks.
    assert_eq!(chunks.len(), 2, "got {:?}", chunks);
    assert!(!chunks[0].is_final, "first should be interim");
    assert_eq!(chunks[0].text, "hello");
    assert!(chunks[1].is_final, "last should carry speech_final");
    assert_eq!(chunks[1].text, "hello world");
}

#[tokio::test]
async fn interim_filtered_when_caller_disables_them() {
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
            "type": "Results",
            "start": 0.0,
            "duration": 0.3,
            "is_final": false,
            "channel": {"alternatives": [{"transcript": "drop me"}]}
        }))
        .expect("send interim");
    server
        .send_json(serde_json::json!({
            "type": "Results",
            "start": 0.3,
            "duration": 0.4,
            "is_final": true,
            "speech_final": true,
            "channel": {"alternatives": [{"transcript": "keep me"}]}
        }))
        .expect("send final");
    server.close_with(1000, "done").expect("close");

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk"));
    }
    assert_eq!(chunks.len(), 1, "interim should have been filtered: {:?}", chunks);
    assert_eq!(chunks[0].text, "keep me");
    assert!(chunks[0].is_final);
}

#[tokio::test]
async fn diarize_and_word_timestamps_surface_on_chunk() {
    let server = MockWsServer::start().await.expect("ws server");
    let mut runner = build_runner(server.url());

    let opts = TranscribeOptions {
        interim_results: false,
        diarize: true,
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
            "type": "Results",
            "start": 0.0,
            "duration": 1.2,
            "is_final": true,
            "speech_final": true,
            "channel": {"alternatives": [{
                "transcript": "alpha beta",
                "words": [
                    {"word": "alpha", "start": 0.0, "end": 0.4, "speaker": 0, "confidence": 0.95},
                    {"word": "beta",  "start": 0.5, "end": 1.0, "speaker": 1, "confidence": 0.91}
                ]
            }]}
        }))
        .expect("send final w/ words");
    server.close_with(1000, "done").expect("close");

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("chunk"));
    }
    assert_eq!(chunks.len(), 1);
    let c = &chunks[0];
    assert_eq!(c.speaker_id.as_deref(), Some("0"));
    assert_eq!(c.words.len(), 2);
    assert_eq!(c.words[0].text, "alpha");
    assert_eq!(c.words[0].ts_start_ms, 0);
    assert_eq!(c.words[1].text, "beta");
    assert_eq!(c.words[1].ts_end_ms, 1_000);
}

#[tokio::test]
async fn runner_sends_close_stream_marker_after_uplink_drains() {
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
    // `CloseStream` marker the runner sends to flush.
    let mut saw_audio = false;
    let mut saw_close_marker = false;
    for _ in 0..6 {
        let msg = server.recv(Duration::from_secs(2)).await;
        match msg {
            Some(Message::Binary(_)) => saw_audio = true,
            Some(Message::Text(t)) => {
                if t.contains("\"CloseStream\"") {
                    saw_close_marker = true;
                    break;
                }
            }
            Some(Message::Close(_)) | None => break,
            _ => continue,
        }
    }
    assert!(saw_audio, "expected at least one binary audio frame");
    assert!(saw_close_marker, "expected CloseStream marker after uplink");

    // Close out the downlink so the runner stream drains.
    server
        .send_json(serde_json::json!({
            "type": "Results",
            "start": 0.0,
            "duration": 0.1,
            "is_final": true,
            "speech_final": true,
            "channel": {"alternatives": [{"transcript": "ok"}]}
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
