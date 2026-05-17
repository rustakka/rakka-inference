//! Convert an [`AudioPayload`] into the `Vec<f32>` mono 16 kHz format
//! whisper.cpp expects.
//!
//! Out-of-scope for M5: resampling and stereo→mono mixdown. The runner
//! returns [`WhisperError::UnsupportedAudio`] for inputs that aren't
//! already shaped right.
//!
//! [`AudioPayload`]: atomr_infer_core::audio::AudioPayload

use std::path::Path;

use atomr_infer_core::audio::{AudioFormat, AudioParams, AudioPayload};

use crate::error::WhisperError;

/// Materialize the payload into a contiguous f32 mono 16 kHz PCM
/// buffer.
///
/// Accepts:
/// - `AudioFormat::Pcm16Le` mono @ `cfg_sample_rate` → converts each
///   i16 sample to `f32 / i16::MAX`
/// - `AudioFormat::PcmF32Le` mono @ `cfg_sample_rate` → reinterprets
///   the bytes as `f32` LE
/// - `AudioFormat::Wav` → reads a minimal RIFF header and decodes the
///   `data` chunk according to the embedded sample format (Pcm16Le or
///   PcmF32Le mono only).
pub fn payload_to_f32_pcm(payload: &AudioPayload, cfg_sample_rate: u32) -> Result<Vec<f32>, WhisperError> {
    match payload {
        AudioPayload::Bytes { data, params } => {
            check_params(params, cfg_sample_rate)?;
            decode_bytes(data, params)
        }
        AudioPayload::Path { path, params } => {
            check_params(params, cfg_sample_rate)?;
            let bytes = read_file(path)?;
            decode_bytes(&bytes, params)
        }
        AudioPayload::Url { .. } => Err(WhisperError::UnsupportedAudio {
            message: "AudioPayload::Url not implemented for local whisper.cpp; fetch upstream".into(),
        }),
        // `AudioPayload` is `#[non_exhaustive]` — any new variant added
        // upstream should be rejected explicitly until this crate is
        // updated.
        _ => Err(WhisperError::UnsupportedAudio {
            message: "whisper-local: unknown AudioPayload variant".into(),
        }),
    }
}

fn read_file(path: &Path) -> Result<Vec<u8>, WhisperError> {
    std::fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            WhisperError::UnsupportedAudio {
                message: format!("audio file not found at {}", path.display()),
            }
        } else {
            WhisperError::UnsupportedAudio {
                message: format!("audio file io error at {}: {e}", path.display()),
            }
        }
    })
}

fn check_params(params: &AudioParams, cfg_sample_rate: u32) -> Result<(), WhisperError> {
    if params.sample_rate_hz != cfg_sample_rate {
        return Err(WhisperError::UnsupportedAudio {
            message: format!(
                "expected {} Hz, got {} Hz (resample upstream — whisper.cpp is hard-coded to 16 kHz)",
                cfg_sample_rate, params.sample_rate_hz
            ),
        });
    }
    if params.channels != 1 {
        return Err(WhisperError::UnsupportedAudio {
            message: format!(
                "expected mono, got {} channels (mix down upstream)",
                params.channels
            ),
        });
    }
    Ok(())
}

fn decode_bytes(data: &[u8], params: &AudioParams) -> Result<Vec<f32>, WhisperError> {
    match params.format {
        AudioFormat::Pcm16Le => decode_pcm16(data),
        AudioFormat::PcmF32Le => decode_pcm_f32(data),
        AudioFormat::Wav => decode_wav(data),
        other => Err(WhisperError::UnsupportedAudio {
            message: format!(
                "format {other:?} not handled by whisper-local; supply Pcm16Le, PcmF32Le or Wav"
            ),
        }),
    }
}

fn decode_pcm16(data: &[u8]) -> Result<Vec<f32>, WhisperError> {
    if data.len() % 2 != 0 {
        return Err(WhisperError::UnsupportedAudio {
            message: format!(
                "Pcm16Le payload byte length {} is not a multiple of 2",
                data.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(s as f32 / i16::MAX as f32);
    }
    Ok(out)
}

fn decode_pcm_f32(data: &[u8]) -> Result<Vec<f32>, WhisperError> {
    if data.len() % 4 != 0 {
        return Err(WhisperError::UnsupportedAudio {
            message: format!(
                "PcmF32Le payload byte length {} is not a multiple of 4",
                data.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let s = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(s);
    }
    Ok(out)
}

/// Minimal canonical-WAV decoder. Walks RIFF chunks looking for `fmt `
/// and `data`. Supports PCM (format-tag 1, 16-bit mono) and IEEE-FLOAT
/// (format-tag 3, 32-bit mono) only — anything else is rejected as
/// `UnsupportedAudio` and the caller is expected to resample upstream.
fn decode_wav(data: &[u8]) -> Result<Vec<f32>, WhisperError> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(WhisperError::UnsupportedAudio {
            message: "WAV: missing RIFF/WAVE header".into(),
        });
    }
    let mut pos = 12;
    let mut fmt_tag: Option<u16> = None;
    let mut channels: Option<u16> = None;
    let mut sample_rate: Option<u32> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut audio: Option<&[u8]> = None;

    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;
        let body_start = pos + 8;
        let body_end = body_start
            .checked_add(size)
            .ok_or_else(|| WhisperError::UnsupportedAudio {
                message: "WAV: chunk size overflow".into(),
            })?;
        if body_end > data.len() {
            return Err(WhisperError::UnsupportedAudio {
                message: format!("WAV: chunk `{}` extends past EOF", String::from_utf8_lossy(id)),
            });
        }
        let body = &data[body_start..body_end];
        match id {
            b"fmt " if body.len() >= 16 => {
                fmt_tag = Some(u16::from_le_bytes([body[0], body[1]]));
                channels = Some(u16::from_le_bytes([body[2], body[3]]));
                sample_rate = Some(u32::from_le_bytes([body[4], body[5], body[6], body[7]]));
                bits_per_sample = Some(u16::from_le_bytes([body[14], body[15]]));
            }
            b"data" => {
                audio = Some(body);
            }
            _ => {}
        }
        // RIFF chunks are word-aligned.
        let advance = if size % 2 == 0 { size } else { size + 1 };
        pos = body_start + advance;
    }

    let fmt_tag = fmt_tag.ok_or_else(|| WhisperError::UnsupportedAudio {
        message: "WAV: missing fmt chunk".into(),
    })?;
    let channels = channels.unwrap_or(0);
    let sample_rate = sample_rate.unwrap_or(0);
    let bits = bits_per_sample.unwrap_or(0);
    let audio = audio.ok_or_else(|| WhisperError::UnsupportedAudio {
        message: "WAV: missing data chunk".into(),
    })?;

    if channels != 1 {
        return Err(WhisperError::UnsupportedAudio {
            message: format!("WAV: expected mono, got {channels} channels"),
        });
    }
    if sample_rate != 16_000 {
        return Err(WhisperError::UnsupportedAudio {
            message: format!("WAV: expected 16000 Hz, got {sample_rate} Hz"),
        });
    }
    match (fmt_tag, bits) {
        (1, 16) => decode_pcm16(audio),
        (3, 32) => decode_pcm_f32(audio),
        _ => Err(WhisperError::UnsupportedAudio {
            message: format!(
                "WAV: only PCM-16 (fmt 1, 16-bit) or IEEE-FLOAT (fmt 3, 32-bit) supported; got fmt {fmt_tag}/{bits}-bit"
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    fn params_16k_mono(format: AudioFormat) -> AudioParams {
        AudioParams::new(16_000, 1, format)
    }

    #[test]
    fn pcm16_decodes_to_normalised_floats() {
        let payload = AudioPayload::Bytes {
            data: Bytes::from_static(&[
                0x00, 0x00, // 0
                0xff, 0x7f, // i16::MAX → ~+1.0
                0x00, 0x80, // i16::MIN → ~-1.0 (-1.0000305)
            ]),
            params: params_16k_mono(AudioFormat::Pcm16Le),
        };
        let pcm = payload_to_f32_pcm(&payload, 16_000).unwrap();
        assert_eq!(pcm.len(), 3);
        assert!((pcm[0] - 0.0).abs() < 1e-6);
        assert!((pcm[1] - 1.0).abs() < 1e-4);
        assert!(pcm[2] < -0.99);
    }

    #[test]
    fn pcm_f32_decodes_verbatim() {
        let mut bytes = Vec::new();
        for &v in &[-1.0_f32, 0.0, 0.5, 1.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let payload = AudioPayload::Bytes {
            data: Bytes::from(bytes),
            params: params_16k_mono(AudioFormat::PcmF32Le),
        };
        let pcm = payload_to_f32_pcm(&payload, 16_000).unwrap();
        assert_eq!(pcm, vec![-1.0, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let payload = AudioPayload::Bytes {
            data: Bytes::from_static(&[0u8; 4]),
            params: AudioParams::new(48_000, 1, AudioFormat::Pcm16Le),
        };
        let err = payload_to_f32_pcm(&payload, 16_000).unwrap_err();
        assert!(matches!(err, WhisperError::UnsupportedAudio { .. }));
        assert!(format!("{err}").contains("48000"));
    }

    #[test]
    fn rejects_stereo() {
        let payload = AudioPayload::Bytes {
            data: Bytes::from_static(&[0u8; 4]),
            params: AudioParams::new(16_000, 2, AudioFormat::Pcm16Le),
        };
        let err = payload_to_f32_pcm(&payload, 16_000).unwrap_err();
        assert!(matches!(err, WhisperError::UnsupportedAudio { .. }));
        assert!(format!("{err}").contains("mono"));
    }

    #[test]
    fn rejects_url_payload() {
        let payload = AudioPayload::Url {
            url: "https://example.com/clip.wav".parse().unwrap(),
            params: params_16k_mono(AudioFormat::Wav),
        };
        let err = payload_to_f32_pcm(&payload, 16_000).unwrap_err();
        assert!(matches!(err, WhisperError::UnsupportedAudio { .. }));
    }

    #[test]
    fn pcm16_rejects_odd_byte_length() {
        let payload = AudioPayload::Bytes {
            data: Bytes::from_static(&[0x00]),
            params: params_16k_mono(AudioFormat::Pcm16Le),
        };
        let err = payload_to_f32_pcm(&payload, 16_000).unwrap_err();
        assert!(format!("{err}").contains("not a multiple of 2"));
    }

    #[test]
    fn wav_pcm16_round_trip() {
        // Hand-roll a 4-sample 16 kHz mono PCM16 WAV.
        let pcm_bytes: Vec<u8> = [0x00, 0x00, 0xff, 0x7f, 0x00, 0x80, 0xff, 0xff].to_vec();
        let mut wav: Vec<u8> = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36u32 + pcm_bytes.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(pcm_bytes.len() as u32).to_le_bytes());
        wav.extend_from_slice(&pcm_bytes);

        let payload = AudioPayload::Bytes {
            data: Bytes::from(wav),
            params: params_16k_mono(AudioFormat::Wav),
        };
        let pcm = payload_to_f32_pcm(&payload, 16_000).unwrap();
        assert_eq!(pcm.len(), 4);
    }

    #[test]
    fn wav_rejects_bad_header() {
        let payload = AudioPayload::Bytes {
            data: Bytes::from_static(b"NOTAWAV12345"),
            params: params_16k_mono(AudioFormat::Wav),
        };
        let err = payload_to_f32_pcm(&payload, 16_000).unwrap_err();
        assert!(format!("{err}").contains("RIFF/WAVE"));
    }
}
