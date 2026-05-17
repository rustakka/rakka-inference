//! XTTS runner configuration.
//!
//! XTTS-v2 is a cross-lingual TTS model. The primary voice-selection
//! mechanism is `VoiceRef::ClonedFrom` — pass a reference audio clip and
//! the runner conditions synthesis on the extracted speaker embedding.
//!
//! [`XttsConfig`] holds the operator-facing settings: the ONNX model
//! directory and the default synthesis language.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Operator-facing config for [`crate::XttsRunner`].
///
/// # Examples
///
/// ```
/// use atomr_infer_runtime_xtts::XttsConfig;
///
/// let cfg = XttsConfig {
///     model_path: "/models/xtts-v2".into(),
///     default_language: "en".into(),
///     chunk_samples: 4096,
///     intra_threads: None,
/// };
/// assert_eq!(cfg.default_language, "en");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XttsConfig {
    /// Path to the XTTS-v2 ONNX model directory. Expected to contain
    /// `model.onnx` (and optionally `config.json`).
    pub model_path: PathBuf,
    /// ISO 639-1 language code used when none is specified in
    /// `SpeechBatch::options.language`. XTTS-v2 supports >17 languages;
    /// default is `"en"`.
    #[serde(default = "default_language")]
    pub default_language: String,
    /// PCM chunk size in samples for the output stream. Default 4096.
    #[serde(default = "default_chunk_samples")]
    pub chunk_samples: usize,
    /// Pin the ORT session to N CPU threads. `None` = let ORT decide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_threads: Option<u32>,
}

fn default_language() -> String {
    "en".into()
}

fn default_chunk_samples() -> usize {
    4096
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> XttsConfig {
        XttsConfig {
            model_path: "/models/xtts-v2".into(),
            default_language: "en".into(),
            chunk_samples: 4096,
            intra_threads: None,
        }
    }

    #[test]
    fn config_defaults() {
        let cfg = sample_config();
        assert_eq!(cfg.default_language, "en");
        assert_eq!(cfg.chunk_samples, 4096);
        assert!(cfg.intra_threads.is_none());
    }

    #[test]
    fn language_defaults_to_en_when_missing() {
        let json = r#"{"model_path": "/models/xtts-v2"}"#;
        let cfg: XttsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.default_language, "en");
        assert_eq!(cfg.chunk_samples, 4096);
    }

    #[test]
    fn config_round_trips_through_serde() {
        let cfg = XttsConfig {
            model_path: "/tmp/xtts".into(),
            default_language: "fr".into(),
            chunk_samples: 2048,
            intra_threads: Some(2),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: XttsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_path, cfg.model_path);
        assert_eq!(back.default_language, "fr");
        assert_eq!(back.intra_threads, Some(2));
    }
}
