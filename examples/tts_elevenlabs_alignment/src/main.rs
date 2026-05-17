//! ElevenLabs WS streaming + alignment demo. M7 example for the audio
//! program of work — see `examples/tts_elevenlabs_alignment/README.md`.
//!
//! Reads:
//!   * `ELEVEN_API_KEY` — required, the ElevenLabs bearer token.
//!   * arg 1 — text to speak (defaults to "Hello from atomr-infer.").
//!   * arg 2 — voice id (defaults to ElevenLabs' "Rachel" voice).
//!   * `ELEVEN_OUT_PATH` — write the concatenated MP3 bytes here
//!     (defaults to `elevenlabs_out.mp3`).
//!
//! Streams `SpeechChunk`s as the WS server emits them and prints each
//! alignment delta to stderr so the user can see per-character timing
//! interleave with the audio bytes.

use std::sync::Arc;

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
use atomr_infer_core::runner::SpeechRunner;
use atomr_infer_remote_core::http::build_client;
use atomr_infer_remote_core::session::SessionSnapshot;
use atomr_infer_runtime_elevenlabs::{ElevenLabsSecret, ElevenLabsTtsConfig, ElevenLabsTtsRunner};
use futures::StreamExt;
use secrecy::SecretString;

const DEFAULT_VOICE: &str = "21m00Tcm4TlvDq8ikWAM";

#[tokio::main]
async fn main() -> Result<()> {
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello from atomr-infer.".into());
    let voice = std::env::args().nth(2).unwrap_or_else(|| DEFAULT_VOICE.into());
    let out_path = std::env::var("ELEVEN_OUT_PATH").unwrap_or_else(|_| "elevenlabs_out.mp3".into());

    let api_key = std::env::var("ELEVEN_API_KEY").context("set ELEVEN_API_KEY to your bearer token")?;

    let cfg = ElevenLabsTtsConfig::defaults_for_elevenlabs(ElevenLabsSecret::Env {
        name: "ELEVEN_API_KEY".into(),
    });
    let client = build_client(&Default::default(), "tts-elevenlabs-example/0")?;
    let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
        client,
        credential: SecretString::from(api_key),
    }));
    let mut runner = ElevenLabsTtsRunner::new(cfg, snap)?;

    let batch = SpeechBatch {
        request_id: "cli".into(),
        model: "eleven_turbo_v2_5".into(),
        text: text.clone(),
        voice: VoiceRef::Id(voice),
        options: SynthOptions::default(),
        stream: true,
        emit_alignment: true,
        estimated_characters: text.chars().count() as u32,
    };

    let handle = runner.speak(batch).await.context("speak")?;
    let mut stream = handle.into_stream();
    let mut audio: Vec<u8> = Vec::new();
    let mut frame_idx = 0u32;
    while let Some(item) = stream.next().await {
        let chunk = item.context("chunk")?;
        audio.extend_from_slice(&chunk.audio_pcm_chunk);
        if let Some(alignment) = chunk.alignment {
            eprintln!(
                "frame {:>3}: {} chars, {} bytes audio (is_final={})",
                frame_idx,
                alignment.words.len(),
                chunk.audio_pcm_chunk.len(),
                chunk.is_final,
            );
            for w in alignment.words.iter().take(8) {
                eprintln!(
                    "    char={:?} start={}ms end={}ms",
                    w.text, w.ts_start_ms, w.ts_end_ms
                );
            }
        }
        if chunk.is_final {
            break;
        }
        frame_idx += 1;
    }

    std::fs::write(&out_path, &audio).context("write output")?;
    eprintln!(
        "wrote {} bytes of MP3 to {} ({} alignment frames observed)",
        audio.len(),
        out_path,
        frame_idx
    );
    Ok(())
}
