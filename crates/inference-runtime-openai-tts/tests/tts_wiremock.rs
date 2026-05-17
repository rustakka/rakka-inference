//! Wiremock-driven integration tests for `OpenAiTtsRunner`.
//!
//! Covers the M6 acceptance criteria:
//! - Auth header is forwarded
//! - 429 responses are classified into `InferenceError::RateLimited`
//! - Audio body bytes round-trip through the runner unmodified
//! - The body is re-chunked at `chunk_bytes` boundaries
//!
//! Source: `FR-TTS-001`.

#![cfg(feature = "tts-openai")]

use std::sync::Arc;

use arc_swap::ArcSwap;
use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::SpeechRunner;
use atomr_infer_core::runtime::ProviderKind;
use atomr_infer_remote_core::http::build_client;
use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_runtime_openai::config::SecretRef;
use atomr_infer_runtime_openai_tts::{OpenAiTtsConfig, OpenAiTtsRunner};
use atomr_infer_testkit::mock_openai::{inject_audio_429, mount_audio_speech_happy_path, MockOpenAi};
use futures::StreamExt;
use secrecy::SecretString;
use url::Url;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, ResponseTemplate};

fn make_runner(server_url: &str) -> OpenAiTtsRunner {
    let base = format!("{server_url}/v1/");
    let cfg = OpenAiTtsConfig::defaults_for_openai(SecretRef::Env {
        name: "OPENAI_API_KEY".into(),
    })
    .with_endpoint(Url::parse(&base).expect("valid base"));
    let client = build_client(&Default::default(), "atomr-infer-test/0").expect("build client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("sk-test-fixture".to_string()),
    }));
    OpenAiTtsRunner::new(cfg, snap).expect("construct runner")
}

fn make_runner_with_chunks(server_url: &str, chunk_bytes: usize) -> OpenAiTtsRunner {
    let base = format!("{server_url}/v1/");
    let mut cfg = OpenAiTtsConfig::defaults_for_openai(SecretRef::Env {
        name: "OPENAI_API_KEY".into(),
    })
    .with_endpoint(Url::parse(&base).expect("valid base"));
    cfg.chunk_bytes = chunk_bytes;
    let client = build_client(&Default::default(), "atomr-infer-test/0").expect("build client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("sk-test-fixture".to_string()),
    }));
    OpenAiTtsRunner::new(cfg, snap).expect("construct runner")
}

fn sample_batch() -> SpeechBatch {
    SpeechBatch {
        request_id: "req-1".into(),
        model: "tts-1".into(),
        text: "hello world".into(),
        voice: VoiceRef::Named("alloy".into()),
        options: SynthOptions::default(),
        stream: true,
        emit_alignment: false,
        estimated_characters: 11,
    }
}

#[tokio::test]
async fn happy_path_returns_full_audio_body() {
    let server = MockOpenAi::start().await;
    let fixture: Vec<u8> = (0u8..200).cycle().take(8_192 * 3 + 17).collect();
    mount_audio_speech_happy_path(&server.server, fixture.clone()).await;

    let mut runner = make_runner(&server.url());
    let handle = runner.speak(sample_batch()).await.expect("speak ok");
    let mut stream = handle.into_stream();

    let mut bytes_out: Vec<u8> = Vec::with_capacity(fixture.len());
    let mut last_is_final = false;
    while let Some(chunk) = stream.next().await {
        let c = chunk.expect("chunk ok");
        bytes_out.extend_from_slice(&c.audio_pcm_chunk);
        last_is_final = c.is_final;
    }

    assert_eq!(bytes_out, fixture, "concatenated bytes match fixture");
    assert!(last_is_final, "last chunk has is_final=true");
}

#[tokio::test]
async fn auth_header_forwarded_to_upstream() {
    let server = MockOpenAi::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(header("authorization", "Bearer sk-test-fixture"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/wav")
                .set_body_bytes(b"OKBYTES".to_vec()),
        )
        .mount(&server.server)
        .await;

    let mut runner = make_runner(&server.url());
    let handle = runner.speak(sample_batch()).await.expect("speak ok");
    let mut stream = handle.into_stream();
    let chunk = stream.next().await.expect("at least one chunk").unwrap();
    assert_eq!(chunk.audio_pcm_chunk.as_ref(), b"OKBYTES");
}

#[tokio::test]
async fn chunk_bytes_boundary_respected() {
    let server = MockOpenAi::start().await;
    // 25 bytes split at chunk_bytes=10 → chunks of [10, 10, 5].
    let fixture: Vec<u8> = (0u8..25).collect();
    mount_audio_speech_happy_path(&server.server, fixture.clone()).await;

    let mut runner = make_runner_with_chunks(&server.url(), 10);
    let handle = runner.speak(sample_batch()).await.expect("speak ok");
    let mut stream = handle.into_stream();

    let mut sizes = Vec::new();
    let mut finals = Vec::new();
    while let Some(chunk) = stream.next().await {
        let c = chunk.expect("chunk ok");
        sizes.push(c.audio_pcm_chunk.len());
        finals.push(c.is_final);
    }
    assert_eq!(sizes, vec![10, 10, 5]);
    assert_eq!(finals, vec![false, false, true]);
}

#[tokio::test]
async fn rate_limit_classified_as_openai_provider() {
    let server = MockOpenAi::start().await;
    // 429 is one-shot, but we don't fall through to a happy path here —
    // the runner has no built-in retry loop (that lives in the engine
    // actor), so the first 429 must surface to the caller.
    inject_audio_429(&server.server).await;
    // No happy-path mock: anything else 404s, but with a one-shot 429
    // mounted first the runner sees the 429.

    let mut runner = make_runner(&server.url());
    let err = runner
        .speak(sample_batch())
        .await
        .expect_err("429 should bubble up");
    match err {
        InferenceError::RateLimited { provider, .. } => {
            assert_eq!(provider, ProviderKind::OpenAi);
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}
