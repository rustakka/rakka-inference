//! Piper runner configuration + on-disk voice manifest.
//!
//! Two structs live here:
//!
//! - [`PiperConfig`] — operator-facing runner config (path to .onnx,
//!   inference scales, optional speaker id, sample-rate override).
//! - [`PiperVoiceManifest`] — the JSON sidecar that ships next to
//!   every Piper voice (`.onnx.json`). We deserialize the subset we
//!   need: `audio.sample_rate`, `inference.{noise_scale, length_scale,
//!   noise_w}`, `phoneme_id_map`, `num_symbols`, `num_speakers`,
//!   optional `espeak.voice`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Operator-facing config for [`crate::PiperRunner`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiperConfig {
    /// Path to the `.onnx` voice model. The sibling `.onnx.json`
    /// manifest is auto-resolved by appending `.json` unless
    /// [`Self::voice_manifest_path`] is set.
    pub voice_path: PathBuf,
    /// Override path for the JSON manifest. Defaults to
    /// `format!("{voice_path}.json")`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_manifest_path: Option<PathBuf>,
    /// Speaker id for multi-speaker voices. `None` means single-speaker
    /// (the runner passes 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<i64>,
    /// Length scale (higher = slower speech). Default 1.0.
    #[serde(default = "default_length_scale")]
    pub length_scale: f32,
    /// Noise scale (timbre variability). Default 0.667 per Piper.
    #[serde(default = "default_noise_scale")]
    pub noise_scale: f32,
    /// Noise W (phoneme duration variability). Default 0.8 per Piper.
    #[serde(default = "default_noise_w")]
    pub noise_w: f32,
    /// PCM chunk size in samples for the output stream. Lower =
    /// finer-grained streaming, higher = less overhead. Default 4096.
    #[serde(default = "default_chunk_samples")]
    pub chunk_samples: usize,
    /// Pin the runner to N CPU threads inside the ORT session. `None`
    /// = let ORT decide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_threads: Option<u32>,
}

fn default_length_scale() -> f32 {
    1.0
}
fn default_noise_scale() -> f32 {
    0.667
}
fn default_noise_w() -> f32 {
    0.8
}
fn default_chunk_samples() -> usize {
    4096
}

impl PiperConfig {
    /// Resolve the manifest path: explicit override wins, otherwise
    /// `{voice_path}.json`.
    pub fn resolved_manifest_path(&self) -> PathBuf {
        if let Some(p) = &self.voice_manifest_path {
            return p.clone();
        }
        let mut s = self.voice_path.clone().into_os_string();
        s.push(".json");
        PathBuf::from(s)
    }
}

/// Subset of the Piper voice manifest we read at session-build time.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PiperVoiceManifest {
    pub audio: PiperAudio,
    #[serde(default)]
    pub inference: PiperInference,
    /// Maps a phoneme (one Unicode grapheme, sometimes a multi-char
    /// token) to its list of input ids. Multi-id mappings happen with
    /// e.g. punctuation that maps to `[id, pad]`.
    pub phoneme_id_map: BTreeMap<String, Vec<i64>>,
    pub num_symbols: u32,
    #[serde(default = "default_num_speakers")]
    pub num_speakers: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub espeak: Option<PiperEspeak>,
}

fn default_num_speakers() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PiperAudio {
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PiperInference {
    #[serde(default)]
    pub noise_scale: Option<f32>,
    #[serde(default)]
    pub length_scale: Option<f32>,
    #[serde(default)]
    pub noise_w: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PiperEspeak {
    /// e.g. `"en-us"`. Not yet consumed — espeak-ng FFI is a follow-up.
    pub voice: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest_json() -> &'static str {
        r#"{
            "audio": { "sample_rate": 22050 },
            "inference": { "noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8 },
            "phoneme_id_map": { "_": [0], "h": [10], "e": [11], "l": [12], "o": [13] },
            "num_symbols": 256,
            "num_speakers": 1,
            "espeak": { "voice": "en-us" }
        }"#
    }

    #[test]
    fn manifest_deserializes() {
        let m: PiperVoiceManifest = serde_json::from_str(sample_manifest_json()).unwrap();
        assert_eq!(m.audio.sample_rate, 22050);
        assert_eq!(m.num_speakers, 1);
        assert_eq!(m.phoneme_id_map.get("h"), Some(&vec![10]));
        assert_eq!(m.espeak.unwrap().voice, "en-us");
    }

    #[test]
    fn manifest_defaults_num_speakers_to_one() {
        let json = r#"{
            "audio": { "sample_rate": 16000 },
            "phoneme_id_map": {},
            "num_symbols": 0
        }"#;
        let m: PiperVoiceManifest = serde_json::from_str(json).unwrap();
        assert_eq!(m.num_speakers, 1);
        assert!(m.inference.noise_scale.is_none());
    }

    #[test]
    fn config_round_trips_through_serde() {
        let cfg = PiperConfig {
            voice_path: "/tmp/voice.onnx".into(),
            voice_manifest_path: None,
            speaker_id: Some(2),
            length_scale: 1.1,
            noise_scale: 0.667,
            noise_w: 0.8,
            chunk_samples: 2048,
            intra_threads: Some(2),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: PiperConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.voice_path, cfg.voice_path);
        assert_eq!(back.speaker_id, cfg.speaker_id);
        assert_eq!(back.length_scale, cfg.length_scale);
        assert_eq!(back.chunk_samples, cfg.chunk_samples);
    }

    #[test]
    fn resolved_manifest_path_appends_json_by_default() {
        let cfg = PiperConfig {
            voice_path: "/tmp/voice.onnx".into(),
            voice_manifest_path: None,
            speaker_id: None,
            length_scale: 1.0,
            noise_scale: 0.667,
            noise_w: 0.8,
            chunk_samples: 4096,
            intra_threads: None,
        };
        assert_eq!(
            cfg.resolved_manifest_path(),
            PathBuf::from("/tmp/voice.onnx.json")
        );
    }

    #[test]
    fn explicit_manifest_path_wins() {
        let cfg = PiperConfig {
            voice_path: "/tmp/voice.onnx".into(),
            voice_manifest_path: Some("/etc/sidecar.json".into()),
            speaker_id: None,
            length_scale: 1.0,
            noise_scale: 0.667,
            noise_w: 0.8,
            chunk_samples: 4096,
            intra_threads: None,
        };
        assert_eq!(cfg.resolved_manifest_path(), PathBuf::from("/etc/sidecar.json"));
    }
}
