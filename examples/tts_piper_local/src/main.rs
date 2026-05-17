//! One-shot Piper TTS synthesis to a WAV file. M4 example for the
//! audio program of work — see `examples/tts_piper_local/README.md`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
use atomr_infer_core::runner::SpeechRunner;
use atomr_infer_runtime_piper::{PiperConfig, PiperRunner};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let text = std::env::args().nth(1).unwrap_or_else(|| "həloʊ".into());
    let voice_path =
        std::env::var("PIPER_VOICE_PATH").context("set PIPER_VOICE_PATH to a Piper .onnx voice")?;
    let voice_path = PathBuf::from(voice_path);
    let out_path = std::env::var("PIPER_OUT_PATH").unwrap_or_else(|_| "piper_out.wav".into());

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
            request_id: "cli".into(),
            model: "piper".into(),
            text: text.clone(),
            voice: VoiceRef::Named("default".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: text.chars().count() as u32,
        })
        .await
        .context("speak")?;

    let mut stream = handle.into_stream();
    let mut pcm: Vec<u8> = Vec::new();
    let mut sample_rate = 22_050u32;
    while let Some(item) = stream.next().await {
        let chunk = item.context("chunk")?;
        sample_rate = chunk.params.sample_rate_hz;
        pcm.extend_from_slice(&chunk.audio_pcm_chunk);
        if chunk.is_final {
            break;
        }
    }

    write_wav(&out_path, &pcm, sample_rate, 1)?;
    eprintln!(
        "wrote {} bytes of PCM at {} Hz → {}",
        pcm.len(),
        sample_rate,
        out_path
    );
    Ok(())
}

/// Write a minimal 16-bit mono PCM WAV file.
fn write_wav(path: &str, pcm: &[u8], sample_rate: u32, channels: u16) -> Result<()> {
    use std::io::Write;

    let byte_rate = sample_rate * channels as u32 * 2;
    let data_len = pcm.len() as u32;
    let riff_len = 36 + data_len;

    let mut f = std::fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&riff_len.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // fmt chunk size
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&(channels * 2).to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    f.write_all(pcm)?;
    Ok(())
}
