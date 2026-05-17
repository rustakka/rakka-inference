//! Env-gated end-to-end smoke test. Skipped in default CI.
//!
//! Requires:
//!
//! - the `piper` feature (`cargo test -p atomr-infer-runtime-piper
//!   --features piper --test piper_smoke -- --ignored`),
//! - `PIPER_VOICE_PATH` pointing at a `.onnx` voice (with sibling
//!   `.onnx.json` manifest next to it). Voices live at
//!   <https://github.com/rhasspy/piper/blob/master/VOICES.md>.
//! - optionally `PIPER_PROMPT` (defaults to a short canned IPA
//!   phrase that uses common phonemes; if your voice rejects it,
//!   pass your own pre-phonemized text via this env var).
//!
//! The test asserts: voice loads, synthesis returns a non-empty
//! PCM stream, and at least one chunk arrives flagged `is_final`.
//!
//! All synthesis happens behind `#[ignore]` so default
//! `cargo test --workspace` skips this file entirely.

#![cfg(feature = "piper")]

use std::path::PathBuf;
use std::time::Duration;

use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
use atomr_infer_core::runner::SpeechRunner;
use atomr_infer_runtime_piper::{PiperConfig, PiperRunner};
use futures::StreamExt;

#[tokio::test]
#[ignore = "needs PIPER_VOICE_PATH + a real voice on disk"]
async fn synth_returns_pcm_with_terminal_chunk() {
    let voice_path = match std::env::var("PIPER_VOICE_PATH") {
        Ok(v) => PathBuf::from(v),
        Err(_) => panic!("set PIPER_VOICE_PATH to a Piper .onnx voice"),
    };
    let prompt = std::env::var("PIPER_PROMPT").unwrap_or_else(|_| "hello".into());

    let mut runner = PiperRunner::new(PiperConfig {
        voice_path,
        voice_manifest_path: None,
        speaker_id: None,
        length_scale: 1.0,
        noise_scale: 0.667,
        noise_w: 0.8,
        chunk_samples: 4096,
        intra_threads: None,
    });

    let handle = runner
        .speak(SpeechBatch {
            request_id: "smoke".into(),
            model: "piper".into(),
            text: prompt,
            voice: VoiceRef::Named("default".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        })
        .await
        .expect("speak");

    let mut stream = handle.into_stream();
    let mut total_bytes = 0usize;
    let mut saw_final = false;
    while let Some(item) = tokio::time::timeout(Duration::from_secs(30), stream.next())
        .await
        .unwrap()
    {
        let chunk = item.expect("chunk");
        total_bytes += chunk.audio_pcm_chunk.len();
        if chunk.is_final {
            saw_final = true;
            break;
        }
    }
    assert!(total_bytes > 0, "no PCM produced");
    assert!(saw_final, "no terminal chunk");
}
