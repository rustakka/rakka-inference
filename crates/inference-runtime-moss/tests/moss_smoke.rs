//! Env-gated end-to-end smoke test. Skipped in default CI.
//!
//! Requires:
//!
//! - the `tts-moss` feature (`cargo test -p atomr-infer-runtime-moss
//!   --features tts-moss --test moss_smoke -- --ignored`),
//! - Linux host (MOSS-TTSD is Linux-only),
//! - `MOSS_MODEL_PATH` pointing at a MOSS-TTSD model directory.
//!
//! When the env var is absent the test prints a skip notice and returns
//! immediately.
//!
//! All synthesis happens behind `#[ignore]` so default
//! `cargo test --workspace` skips this file entirely.

#![cfg(feature = "tts-moss")]

use std::path::PathBuf;
use std::time::Duration;

use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
use atomr_infer_core::runner::SpeechRunner;
use atomr_infer_runtime_moss::{MossConfig, MossRunner};
use futures::StreamExt;

#[tokio::test]
#[ignore = "needs MOSS_MODEL_PATH + a real MOSS model on disk (Linux only)"]
async fn synth_returns_pcm_with_terminal_chunk() {
    let model_path = match std::env::var("MOSS_MODEL_PATH") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            eprintln!("skipping: set MOSS_MODEL_PATH to run");
            return;
        }
    };

    let mut runner = MossRunner::new(MossConfig {
        model_dir: model_path,
        default_voice: "default".into(),
        chunk_samples: 4096,
    });

    let handle = runner
        .speak(SpeechBatch {
            request_id: "smoke".into(),
            model: "moss-tts".into(),
            text: "hello".into(),
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
    while let Some(item) = tokio::time::timeout(Duration::from_secs(60), stream.next())
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
