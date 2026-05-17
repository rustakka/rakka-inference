//! Env-gated end-to-end smoke test. Skipped in default CI.
//!
//! Requires:
//!
//! - the `tts-kokoro` feature (`cargo test -p atomr-infer-runtime-kokoro
//!   --features tts-kokoro --test kokoro_smoke -- --ignored`),
//! - `KOKORO_VOICE_PATH` pointing at a pre-converted Kokoro `.onnx` voice.
//!   Upstream `.pt` files can be exported via `torch.onnx.export` or the
//!   official conversion scripts.
//!
//! When the env var is absent the test prints a skip notice and returns
//! immediately.
//!
//! All synthesis happens behind `#[ignore]` so default
//! `cargo test --workspace` skips this file entirely.

#![cfg(feature = "tts-kokoro")]

use std::path::PathBuf;
use std::time::Duration;

use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
use atomr_infer_core::runner::SpeechRunner;
use atomr_infer_runtime_kokoro::{KokoroConfig, KokoroRunner};
use futures::StreamExt;

#[tokio::test]
#[ignore = "needs KOKORO_VOICE_PATH + a real voice on disk"]
async fn synth_returns_pcm_with_terminal_chunk() {
    let voice_path = match std::env::var("KOKORO_VOICE_PATH") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            eprintln!("skipping: set KOKORO_VOICE_PATH to run");
            return;
        }
    };

    let voice_dir = voice_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    let voice_stem = voice_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    let mut runner = KokoroRunner::new(KokoroConfig {
        voice_pack_dir: voice_dir,
        default_voice: voice_stem,
        chunk_samples: 4096,
        intra_threads: None,
    });

    let handle = runner
        .speak(SpeechBatch {
            request_id: "smoke".into(),
            model: "kokoro-82m".into(),
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
