//! One-shot whisper.cpp transcription to stdout. M5 example for the
//! audio program of work — see `examples/stt_whisper_local/README.md`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use atomr_infer_core::audio::{
    AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, TranscribeOptions,
};
use atomr_infer_core::runner::AudioRunner;
use atomr_infer_runtime_whisper_local::{WhisperConfig, WhisperRunner};
use bytes::Bytes;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let wav_path = std::env::args()
        .nth(1)
        .context("usage: stt_whisper_local <path-to-wav>")?;
    let model =
        std::env::var("WHISPER_MODEL_PATH").context("set WHISPER_MODEL_PATH to a ggml whisper model")?;
    let want_words = std::env::var("WHISPER_WORD_TIMESTAMPS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let wav_bytes = std::fs::read(&wav_path).context("read wav")?;

    let mut runner = WhisperRunner::new(WhisperConfig {
        model_path: PathBuf::from(model),
        word_timestamps: want_words,
        ..Default::default()
    });

    let handle = runner
        .execute_audio(AudioBatch {
            request_id: "cli".into(),
            model: "whisper-local".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from(wav_bytes),
                params: AudioParams::new(16_000, 1, AudioFormat::Wav),
            }),
            stream: false,
            options: AudioOptions::Transcribe(TranscribeOptions {
                word_timestamps: want_words,
                ..Default::default()
            }),
            estimated_units: 1,
        })
        .await
        .context("execute_audio")?;

    let mut stream = handle.into_stream();
    while let Some(item) = stream.next().await {
        let chunk = item.context("chunk")?;
        println!(
            "[{:>6}..{:>6}ms] {}",
            chunk.ts_start_ms,
            chunk.ts_end_ms,
            chunk.text.trim()
        );
        if want_words {
            for w in &chunk.words {
                println!(
                    "    [{:>6}..{:>6}ms p={:.2}] {}",
                    w.ts_start_ms,
                    w.ts_end_ms,
                    w.confidence.unwrap_or(0.0),
                    w.text
                );
            }
        }
        if chunk.is_final {
            break;
        }
    }
    Ok(())
}
