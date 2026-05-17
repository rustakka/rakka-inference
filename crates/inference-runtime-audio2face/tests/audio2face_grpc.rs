//! Integration tests for `atomr-infer-runtime-audio2face`.
//!
//! These tests exercise the public surface of [`Audio2FaceRunner`] against
//! the in-memory generator (no live NVIDIA server required). They verify:
//!
//! * A 1-second audio batch at 30 fps yields exactly 30 [`BlendshapeChunk`]s.
//! * Each chunk carries exactly 52 weights.
//! * Timestamps increment by ~33 ms (1 000 / 30).
//! * The final chunk has `is_final == true`; all others have `is_final == false`.
//! * The feature-off stub returns an error with the expected shape.

use atomr_infer_core::audio::{
    A2FOptions, AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload,
};
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::A2FRunner;
use atomr_infer_runtime_audio2face::{Audio2FaceConfig, Audio2FaceRunner};
#[cfg(feature = "audio2face")]
use futures::StreamExt;

fn make_batch(request_id: &str, frames: u32, fps: u32) -> AudioBatch {
    AudioBatch {
        request_id: request_id.into(),
        model: "audio2face-3d".into(),
        input: AudioInput::Static(AudioPayload::Bytes {
            data: bytes::Bytes::new(),
            params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
        }),
        stream: false,
        options: AudioOptions::Audio2Face(A2FOptions {
            fps: Some(fps),
            ..Default::default()
        }),
        estimated_units: frames,
    }
}

/// With the `audio2face` feature enabled the runner emits exactly N frames
/// at the requested fps.
#[cfg(feature = "audio2face")]
#[tokio::test]
async fn one_second_at_30fps_emits_30_chunks() {
    let mut runner = Audio2FaceRunner::new(Audio2FaceConfig::defaults_for_a2f());

    let batch = make_batch("req-30fps", 30, 30);
    let handle = runner
        .execute_audio2face(batch)
        .await
        .expect("execute_audio2face should succeed");

    let chunks: Vec<_> = handle.into_stream().collect().await;
    assert_eq!(chunks.len(), 30, "expected exactly 30 chunks at 30 fps");

    for (i, result) in chunks.iter().enumerate() {
        let chunk = result.as_ref().expect("each chunk should be Ok");
        assert_eq!(chunk.request_id, "req-30fps");
        assert_eq!(chunk.weights.len(), 52, "each chunk must carry 52 weights");

        let is_last = i == 29;
        assert_eq!(
            chunk.is_final, is_last,
            "is_final should be true only on the last chunk (index {i})"
        );

        // Timestamp should increment by ~33 ms per frame (1000/30).
        let expected_ts: u32 = i as u32 * 33;
        assert_eq!(
            chunk.timestamp_ms, expected_ts,
            "timestamp_ms at index {i} should be {expected_ts}"
        );
    }
}

/// Final chunk has `is_final == true`.
#[cfg(feature = "audio2face")]
#[tokio::test]
async fn final_chunk_has_is_final_flag() {
    let mut runner = Audio2FaceRunner::new(Audio2FaceConfig::defaults_for_a2f());

    let handle = runner
        .execute_audio2face(make_batch("req-final", 5, 30))
        .await
        .expect("should succeed");

    let chunks: Vec<_> = handle.into_stream().collect().await;
    assert_eq!(chunks.len(), 5);

    let last = chunks.last().unwrap().as_ref().unwrap();
    assert!(last.is_final, "last chunk must have is_final == true");

    for c in &chunks[..chunks.len() - 1] {
        assert!(
            !c.as_ref().unwrap().is_final,
            "non-last chunks must have is_final == false"
        );
    }
}

/// All 52 weights are in range `[0.0, 1.0]`.
#[cfg(feature = "audio2face")]
#[tokio::test]
async fn weights_in_range() {
    let mut runner = Audio2FaceRunner::new(Audio2FaceConfig::defaults_for_a2f());

    let handle = runner
        .execute_audio2face(make_batch("req-range", 60, 30))
        .await
        .expect("should succeed");

    let chunks: Vec<_> = handle.into_stream().collect().await;
    for result in chunks {
        let chunk = result.unwrap();
        for (i, &w) in chunk.weights.iter().enumerate() {
            assert!((0.0..=1.0).contains(&w), "weight[{i}] = {w} is out of [0.0, 1.0]");
        }
    }
}

/// Wrong `AudioOptions` variant yields a `BadRequest` error.
#[cfg(feature = "audio2face")]
#[tokio::test]
async fn wrong_options_variant_returns_bad_request() {
    let mut runner = Audio2FaceRunner::new(Audio2FaceConfig::defaults_for_a2f());

    let batch = AudioBatch {
        request_id: "req-wrong".into(),
        model: "audio2face-3d".into(),
        input: AudioInput::Static(AudioPayload::Bytes {
            data: bytes::Bytes::new(),
            params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
        }),
        stream: false,
        options: AudioOptions::Transcribe(atomr_infer_core::audio::TranscribeOptions::default()),
        estimated_units: 30,
    };

    let result = runner.execute_audio2face(batch).await;
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), InferenceError::BadRequest { .. }),
        "wrong options variant should yield BadRequest"
    );
}

/// Without the `audio2face` feature the stub returns an Internal error.
#[cfg(not(feature = "audio2face"))]
#[tokio::test]
async fn stub_returns_feature_disabled_error() {
    let mut runner = Audio2FaceRunner::new(Audio2FaceConfig::defaults_for_a2f());
    let result = runner.execute_audio2face(make_batch("req-stub", 30, 30)).await;
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), InferenceError::Internal(_)),
        "feature-off stub should return Internal error"
    );
}
