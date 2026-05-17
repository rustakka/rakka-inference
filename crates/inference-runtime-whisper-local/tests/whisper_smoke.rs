//! Env-gated smoke test: runs real whisper.cpp on a real model + a real
//! WAV. Not in CI by default — requires:
//!
//! - `WHISPER_MODEL_PATH=/path/to/ggml-tiny.en.bin` (or any other ggml
//!   whisper model), and
//! - `WHISPER_FIXTURE_WAV=/path/to/clip.wav` (16 kHz mono PCM-16 WAV).
//!
//! Run with `cargo test -p atomr-infer-runtime-whisper-local
//! --features stt-whisper -- --ignored --nocapture`.

#![cfg(feature = "stt-whisper")]
#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::path::PathBuf;

use atomr_infer_core::audio::{
    AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscribeOptions,
};
use atomr_infer_core::runner::AudioRunner;
use atomr_infer_runtime_whisper_local::{WhisperConfig, WhisperRunner};
use bytes::Bytes;
use futures::StreamExt;

#[ignore = "needs WHISPER_MODEL_PATH + WHISPER_FIXTURE_WAV on disk"]
#[tokio::test]
async fn transcribes_fixture_wav_into_one_or_more_chunks() {
    let model =
        std::env::var("WHISPER_MODEL_PATH").expect("set WHISPER_MODEL_PATH to a ggml whisper model file");
    let wav = std::env::var("WHISPER_FIXTURE_WAV")
        .expect("set WHISPER_FIXTURE_WAV to a 16 kHz mono PCM-16 WAV file");

    let wav_bytes = std::fs::read(&wav).expect("read fixture wav");

    let mut runner = WhisperRunner::new(WhisperConfig {
        model_path: PathBuf::from(model),
        word_timestamps: true,
        ..Default::default()
    });

    let handle = runner
        .execute_audio(AudioBatch {
            request_id: "smoke".into(),
            model: "whisper-local".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from(wav_bytes),
                params: AudioParams::new(16_000, 1, AudioFormat::Wav),
            }),
            stream: false,
            options: AudioOptions::Transcribe(TranscribeOptions::default()),
            estimated_units: 1,
        })
        .await
        .expect("execute_audio");

    let mut stream = handle.into_stream();
    let mut total_chars = 0usize;
    let mut saw_final = false;
    while let Some(item) = stream.next().await {
        let chunk = item.expect("chunk error");
        total_chars += chunk.text.len();
        if chunk.is_final {
            saw_final = true;
        }
        eprintln!(
            "chunk: is_final={} t=[{}..{}]ms text={:?} words={}",
            chunk.is_final,
            chunk.ts_start_ms,
            chunk.ts_end_ms,
            chunk.text,
            chunk.words.len(),
        );
    }
    assert!(saw_final, "stream ended without a final chunk");
    assert!(total_chars > 0, "no text emitted by whisper");
}
