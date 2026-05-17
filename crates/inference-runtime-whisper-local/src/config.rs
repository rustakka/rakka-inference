//! `WhisperConfig` — the user-facing configuration blob handed to
//! [`WhisperRunner::new`](crate::WhisperRunner).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Configuration for a local whisper.cpp transcription runner.
///
/// All defaults match the upstream whisper.cpp CLI behavior:
/// auto-detect language, full-thread inference, no word-timestamps,
/// no translation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// Path to a ggml whisper model file (e.g. `ggml-tiny.en.bin`,
    /// `ggml-base.bin`, `ggml-large-v3.bin`).
    pub model_path: PathBuf,

    /// ISO-639-1 language hint. `None` (the default) lets whisper.cpp
    /// auto-detect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Number of inference threads. `None` lets whisper.cpp pick.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_threads: Option<u32>,

    /// Translate non-English speech to English at decode time.
    #[serde(default)]
    pub translate: bool,

    /// Emit per-word timestamps inline on the [`TranscriptChunk`].
    ///
    /// [`TranscriptChunk`]: atomr_infer_core::audio::TranscriptChunk
    #[serde(default)]
    pub word_timestamps: bool,

    /// Required input sample rate. whisper.cpp is hard-coded to 16 kHz;
    /// callers must resample upstream. Default = `16_000`.
    #[serde(default = "default_sample_rate")]
    pub sample_rate_hz: u32,
}

fn default_sample_rate() -> u32 {
    16_000
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            language: None,
            n_threads: None,
            translate: false,
            word_timestamps: false,
            sample_rate_hz: default_sample_rate(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_track_whisper_cpp_cli() {
        let c = WhisperConfig::default();
        assert!(c.language.is_none());
        assert!(c.n_threads.is_none());
        assert!(!c.translate);
        assert!(!c.word_timestamps);
        assert_eq!(c.sample_rate_hz, 16_000);
    }

    #[test]
    fn serde_round_trip_minimal() {
        let c = WhisperConfig {
            model_path: PathBuf::from("/models/ggml-tiny.en.bin"),
            ..Default::default()
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: WhisperConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_path, c.model_path);
        assert_eq!(back.sample_rate_hz, 16_000);
    }

    #[test]
    fn serde_round_trip_full() {
        let c = WhisperConfig {
            model_path: PathBuf::from("/models/ggml-large-v3.bin"),
            language: Some("ja".into()),
            n_threads: Some(8),
            translate: true,
            word_timestamps: true,
            sample_rate_hz: 16_000,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: WhisperConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.language.as_deref(), Some("ja"));
        assert_eq!(back.n_threads, Some(8));
        assert!(back.translate);
        assert!(back.word_timestamps);
    }
}
