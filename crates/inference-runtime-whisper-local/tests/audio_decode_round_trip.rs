//! Integration tests for the public decode entry point. Exercises the
//! `AudioPayload::Path` branch (which the inline unit tests can't easily
//! cover because they shouldn't write to disk on every run) plus the
//! `Bytes` round-trip.

use std::io::Write;

use atomr_infer_core::audio::{AudioFormat, AudioParams, AudioPayload};
use atomr_infer_runtime_whisper_local::audio_decode::payload_to_f32_pcm;
use bytes::Bytes;
use tempfile::NamedTempFile;

fn write_canonical_pcm16_wav(out: &mut impl Write, pcm: &[u8]) -> std::io::Result<()> {
    out.write_all(b"RIFF")?;
    out.write_all(&(36u32 + pcm.len() as u32).to_le_bytes())?;
    out.write_all(b"WAVE")?;
    out.write_all(b"fmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&1u16.to_le_bytes())?; // PCM
    out.write_all(&1u16.to_le_bytes())?; // mono
    out.write_all(&16_000u32.to_le_bytes())?;
    out.write_all(&32_000u32.to_le_bytes())?; // byte rate
    out.write_all(&2u16.to_le_bytes())?; // block align
    out.write_all(&16u16.to_le_bytes())?; // bits per sample
    out.write_all(b"data")?;
    out.write_all(&(pcm.len() as u32).to_le_bytes())?;
    out.write_all(pcm)?;
    Ok(())
}

#[test]
fn path_payload_decodes_pcm16_from_tempfile() {
    let pcm: Vec<u8> = (0..200)
        .flat_map(|i: i16| (i.wrapping_mul(100)).to_le_bytes())
        .collect();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&pcm).unwrap();
    tmp.flush().unwrap();

    let payload = AudioPayload::Path {
        path: tmp.path().to_path_buf(),
        params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
    };
    let pcm_f32 = payload_to_f32_pcm(&payload, 16_000).unwrap();
    assert_eq!(pcm_f32.len(), 200);
}

#[test]
fn path_payload_decodes_wav_from_tempfile() {
    let pcm: Vec<u8> = (0..8000)
        .flat_map(|i: i16| (i.wrapping_mul(4)).to_le_bytes())
        .collect();
    let mut tmp = NamedTempFile::new().unwrap();
    write_canonical_pcm16_wav(&mut tmp, &pcm).unwrap();
    tmp.flush().unwrap();

    let payload = AudioPayload::Path {
        path: tmp.path().to_path_buf(),
        params: AudioParams::new(16_000, 1, AudioFormat::Wav),
    };
    let pcm_f32 = payload_to_f32_pcm(&payload, 16_000).unwrap();
    assert_eq!(pcm_f32.len(), 8000);
}

#[test]
fn missing_path_payload_returns_unsupported_audio() {
    let payload = AudioPayload::Path {
        path: std::path::PathBuf::from("/this/path/definitely/does/not/exist.pcm"),
        params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
    };
    let err = payload_to_f32_pcm(&payload, 16_000).unwrap_err();
    assert!(format!("{err}").contains("not found"));
}

#[test]
fn bytes_payload_with_silence_decodes_to_zeros() {
    let payload = AudioPayload::Bytes {
        data: Bytes::from(vec![0u8; 32_000]), // 1 sec of 16 kHz mono silence
        params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
    };
    let pcm = payload_to_f32_pcm(&payload, 16_000).unwrap();
    assert_eq!(pcm.len(), 16_000);
    assert!(pcm.iter().all(|&s| s.abs() < 1e-9));
}
