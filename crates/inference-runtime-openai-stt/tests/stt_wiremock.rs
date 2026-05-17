//! Wiremock-driven integration tests for `OpenAiSttRunner`.
//!
//! Covers the M6 acceptance criteria:
//! - Auth header is forwarded
//! - Multipart body carries `model` text part + `file` part
//! - 429 responses are classified into `InferenceError::RateLimited`
//! - `verbose_json` responses split into per-segment chunks with the
//!   last marked `is_final = true`
//!
//! Source: `FR-STT-001`.

#![cfg(feature = "stt-openai")]

use std::sync::Arc;

use arc_swap::ArcSwap;
use atomr_infer_core::audio::{
    AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscribeOptions,
};
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::AudioRunner;
use atomr_infer_core::runtime::ProviderKind;
use atomr_infer_remote_core::http::build_client;
use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_runtime_openai::config::SecretRef;
use atomr_infer_runtime_openai_stt::{OpenAiSttConfig, OpenAiSttRunner};
use atomr_infer_testkit::mock_openai::{inject_audio_429, mount_audio_transcriptions_happy_path, MockOpenAi};
use bytes::Bytes;
use futures::StreamExt;
use secrecy::SecretString;
use url::Url;
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, ResponseTemplate};

fn make_runner(server_url: &str) -> OpenAiSttRunner {
    let base = format!("{server_url}/v1/");
    let cfg = OpenAiSttConfig::defaults_for_openai(SecretRef::Env {
        name: "OPENAI_API_KEY".into(),
    })
    .with_endpoint(Url::parse(&base).expect("valid base"));
    let client = build_client(&Default::default(), "atomr-infer-test/0").expect("build client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("sk-test-fixture".to_string()),
    }));
    OpenAiSttRunner::new(cfg, snap).expect("construct runner")
}

fn pcm_batch(opts: TranscribeOptions) -> AudioBatch {
    let payload = AudioPayload::Bytes {
        data: Bytes::from_static(&[0u8; 256]),
        params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
    };
    AudioBatch {
        request_id: "req-stt-1".into(),
        model: "whisper-1".into(),
        input: AudioInput::Static(payload),
        stream: false,
        options: AudioOptions::Transcribe(opts),
        estimated_units: 1,
    }
}

#[tokio::test]
async fn happy_path_emits_single_final_chunk() {
    let server = MockOpenAi::start().await;
    mount_audio_transcriptions_happy_path(&server.server, "hello there").await;

    let mut runner = make_runner(&server.url());
    let handle = runner
        .execute_audio(pcm_batch(TranscribeOptions::default()))
        .await
        .expect("execute_audio ok");
    let chunks: Vec<_> = handle.into_stream().collect().await;

    assert_eq!(chunks.len(), 1);
    let chunk = chunks[0].as_ref().expect("chunk ok");
    assert_eq!(chunk.text, "hello there");
    assert!(chunk.is_final);
    assert!(chunk.words.is_empty(), "plain json carries no word timing");
}

#[tokio::test]
async fn auth_header_and_multipart_body_forwarded() {
    let server = MockOpenAi::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("authorization", "Bearer sk-test-fixture"))
        // reqwest's multipart writer prefaces each part with
        // `Content-Disposition: form-data; name="..."`. Both the model
        // and file parts must appear.
        .and(body_string_contains("name=\"model\""))
        .and(body_string_contains("name=\"file\""))
        .and(body_string_contains("whisper-1"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({"text": "matched"})),
        )
        .mount(&server.server)
        .await;

    let mut runner = make_runner(&server.url());
    let handle = runner
        .execute_audio(pcm_batch(TranscribeOptions::default()))
        .await
        .expect("execute_audio ok");
    let chunks: Vec<_> = handle.into_stream().collect().await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].as_ref().unwrap().text, "matched");
}

#[tokio::test]
async fn verbose_json_splits_into_per_segment_chunks() {
    let server = MockOpenAi::start().await;
    let verbose = serde_json::json!({
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 0.5, "text": "hello"},
            {"start": 0.5, "end": 1.0, "text": " world"},
        ],
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ],
    });
    // The runner asks for `verbose_json` whenever word_timestamps is
    // set; assert the body carries that field too.
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(body_string_contains("verbose_json"))
        .and(body_string_contains("timestamp_granularities[]"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(verbose),
        )
        .mount(&server.server)
        .await;

    let opts = TranscribeOptions {
        word_timestamps: true,
        ..Default::default()
    };
    let mut runner = make_runner(&server.url());
    let handle = runner
        .execute_audio(pcm_batch(opts))
        .await
        .expect("execute_audio ok");
    let chunks: Vec<_> = handle.into_stream().collect().await;

    assert_eq!(chunks.len(), 2);
    let c0 = chunks[0].as_ref().unwrap();
    let c1 = chunks[1].as_ref().unwrap();
    assert!(!c0.is_final);
    assert!(c1.is_final);
    assert_eq!(c0.text, "hello");
    assert_eq!(c1.text, " world");
    assert_eq!(c0.words.len(), 1);
    assert_eq!(c0.words[0].text, "hello");
    assert_eq!(c1.words[0].text, "world");
}

#[tokio::test]
async fn rate_limit_classified_as_openai_provider() {
    let server = MockOpenAi::start().await;
    inject_audio_429(&server.server).await;

    let mut runner = make_runner(&server.url());
    let err = runner
        .execute_audio(pcm_batch(TranscribeOptions::default()))
        .await
        .expect_err("429 should bubble up");
    match err {
        InferenceError::RateLimited { provider, .. } => {
            assert_eq!(provider, ProviderKind::OpenAi);
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}
