//! Wiremock-driven integration tests for the ElevenLabs HTTPS path.
//!
//! Covers the M7 acceptance criteria for one-shot synthesis:
//! - `xi-api-key` auth header is forwarded
//! - The runner POSTs to `/v1/text-to-speech/{voice_id}` with the
//!   correct JSON body (`model_id` + `text`)
//! - The audio body bytes round-trip through the runner unmodified
//! - The body is re-chunked at `chunk_bytes` boundaries
//! - 429 responses are classified into
//!   `InferenceError::RateLimited { provider: ElevenLabs, .. }`
//! - The voice-cloning multipart upload at `/v1/voices/add` returns
//!   the new voice id
//!
//! Source: `FR-TTS-001`.

#![cfg(feature = "tts-elevenlabs")]

use std::sync::Arc;

use arc_swap::ArcSwap;
use atomr_infer_core::audio::{AudioFormat, AudioParams, AudioPayload, SpeechBatch, SynthOptions, VoiceRef};
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::SpeechRunner;
use atomr_infer_core::runtime::ProviderKind;
use atomr_infer_remote_core::http::build_client;
use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_runtime_elevenlabs::{ElevenLabsSecret, ElevenLabsTtsConfig, ElevenLabsTtsRunner};
use bytes::Bytes;
use futures::StreamExt;
use secrecy::SecretString;
use url::Url;
use wiremock::matchers::{body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const VOICE_ID: &str = "21m00Tcm4TlvDq8ikWAM";

fn build_runner(server_url: &str) -> ElevenLabsTtsRunner {
    let base = format!("{server_url}/v1/");
    let cfg = ElevenLabsTtsConfig::defaults_for_elevenlabs(ElevenLabsSecret::Env {
        name: "ELEVEN_API_KEY".into(),
    })
    .with_endpoint(Url::parse(&base).expect("valid base"))
    .with_ws_endpoint(Url::parse("ws://127.0.0.1:1/v1/").expect("valid ws base"));
    let client = build_client(&Default::default(), "atomr-infer-test/0").expect("build client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("sk-elevenlabs-fixture".to_string()),
    }));
    ElevenLabsTtsRunner::new(cfg, snap).expect("construct runner")
}

fn build_runner_with_chunks(server_url: &str, chunk_bytes: usize) -> ElevenLabsTtsRunner {
    let base = format!("{server_url}/v1/");
    let mut cfg = ElevenLabsTtsConfig::defaults_for_elevenlabs(ElevenLabsSecret::Env {
        name: "ELEVEN_API_KEY".into(),
    })
    .with_endpoint(Url::parse(&base).expect("valid base"))
    .with_ws_endpoint(Url::parse("ws://127.0.0.1:1/v1/").expect("valid ws base"));
    cfg.chunk_bytes = chunk_bytes;
    let client = build_client(&Default::default(), "atomr-infer-test/0").expect("build client");
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from("sk-elevenlabs-fixture".to_string()),
    }));
    ElevenLabsTtsRunner::new(cfg, snap).expect("construct runner")
}

fn https_batch() -> SpeechBatch {
    SpeechBatch {
        request_id: "req-eleven".into(),
        model: "eleven_turbo_v2_5".into(),
        text: "hello world".into(),
        voice: VoiceRef::Id(VOICE_ID.into()),
        options: SynthOptions {
            format: Some(AudioFormat::Mp3),
            ..Default::default()
        },
        stream: false,
        emit_alignment: false,
        estimated_characters: 11,
    }
}

async fn mount_speech_happy_path(server: &MockServer, body: Vec<u8>) {
    Mock::given(method("POST"))
        .and(path(format!("/v1/text-to-speech/{VOICE_ID}")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg")
                .set_body_bytes(body),
        )
        .mount(server)
        .await;
}

#[tokio::test]
async fn happy_path_returns_full_audio_body() {
    let server = MockServer::start().await;
    let fixture: Vec<u8> = (0u8..200).cycle().take(8_192 * 2 + 113).collect();
    mount_speech_happy_path(&server, fixture.clone()).await;

    let mut runner = build_runner(&server.uri());
    let handle = runner.speak(https_batch()).await.expect("speak ok");
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
async fn xi_api_key_header_and_body_forwarded() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("/v1/text-to-speech/{VOICE_ID}")))
        .and(header("xi-api-key", "sk-elevenlabs-fixture"))
        .and(body_string_contains("\"model_id\":\"eleven_turbo_v2_5\""))
        .and(body_string_contains("\"text\":\"hello world\""))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg")
                .set_body_bytes(b"AUDIO".to_vec()),
        )
        .mount(&server)
        .await;

    let mut runner = build_runner(&server.uri());
    let handle = runner.speak(https_batch()).await.expect("speak ok");
    let mut stream = handle.into_stream();
    let chunk = stream.next().await.expect("at least one chunk").unwrap();
    assert_eq!(chunk.audio_pcm_chunk.as_ref(), b"AUDIO");
    assert!(chunk.is_final);
}

#[tokio::test]
async fn chunk_bytes_boundary_respected() {
    let server = MockServer::start().await;
    let fixture: Vec<u8> = (0u8..25).collect();
    mount_speech_happy_path(&server, fixture.clone()).await;

    let mut runner = build_runner_with_chunks(&server.uri(), 10);
    let handle = runner.speak(https_batch()).await.expect("speak ok");
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
async fn rate_limit_classified_as_elevenlabs_provider() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("/v1/text-to-speech/{VOICE_ID}")))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "3")
                .set_body_string("{\"detail\":\"rate limited\"}"),
        )
        .mount(&server)
        .await;

    let mut runner = build_runner(&server.uri());
    let err = runner
        .speak(https_batch())
        .await
        .expect_err("429 should bubble up");
    match err {
        InferenceError::RateLimited {
            provider,
            retry_after,
        } => {
            assert_eq!(provider, ProviderKind::ElevenLabs);
            assert_eq!(retry_after, Some(std::time::Duration::from_secs(3)));
        }
        other => panic!("expected RateLimited, got {other:?}"),
    }
}

#[tokio::test]
async fn voice_cloning_multipart_upload_returns_new_voice_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/voices/add"))
        .and(header("xi-api-key", "sk-elevenlabs-fixture"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_string("{\"voice_id\":\"newly_cloned_voice_99\"}"),
        )
        .mount(&server)
        .await;

    let runner = build_runner(&server.uri());
    let sample = AudioPayload::Bytes {
        data: Bytes::from_static(b"RIFF....WAVEfake"),
        params: AudioParams::new(16_000, 1, AudioFormat::Wav),
    };
    let id = runner
        .clone_voice("My Test Voice", sample, Some("integration test"))
        .await
        .expect("clone ok");
    assert_eq!(id, "newly_cloned_voice_99");
}
