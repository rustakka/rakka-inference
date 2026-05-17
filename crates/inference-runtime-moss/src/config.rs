//! MOSS TTS runner configuration.
//!
//! [`MossConfig`] is the operator-facing config. `model_dir` points to the
//! MOSS-TTSD model directory (containing checkpoints and config files).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Operator-facing config for [`crate::MossRunner`].
///
/// # Examples
///
/// ```
/// use atomr_infer_runtime_moss::MossConfig;
///
/// let cfg = MossConfig {
///     model_dir: "/models/moss-tts".into(),
///     default_voice: "default".into(),
///     chunk_samples: 4096,
/// };
/// assert_eq!(cfg.default_voice, "default");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MossConfig {
    /// Directory containing the MOSS-TTSD model checkpoints and config.
    pub model_dir: PathBuf,
    /// Default voice/speaker name. MOSS-TTSD may support multiple
    /// built-in speaker ids; `"default"` selects the single-speaker
    /// configuration.
    #[serde(default = "default_voice")]
    pub default_voice: String,
    /// PCM chunk size in samples for the output stream. Default 4096.
    #[serde(default = "default_chunk_samples")]
    pub chunk_samples: usize,
}

fn default_voice() -> String {
    "default".into()
}

fn default_chunk_samples() -> usize {
    4096
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> MossConfig {
        MossConfig {
            model_dir: "/models/moss-tts".into(),
            default_voice: "default".into(),
            chunk_samples: 4096,
        }
    }

    #[test]
    fn config_defaults() {
        let cfg = sample_config();
        assert_eq!(cfg.default_voice, "default");
        assert_eq!(cfg.chunk_samples, 4096);
    }

    #[test]
    fn default_voice_and_chunk_samples_filled_by_serde_default() {
        let json = r#"{"model_dir": "/models/moss"}"#;
        let cfg: MossConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.default_voice, "default");
        assert_eq!(cfg.chunk_samples, 4096);
    }

    #[test]
    fn config_round_trips_through_serde() {
        let cfg = MossConfig {
            model_dir: "/tmp/moss".into(),
            default_voice: "speaker_a".into(),
            chunk_samples: 2048,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: MossConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_dir, cfg.model_dir);
        assert_eq!(back.default_voice, "speaker_a");
        assert_eq!(back.chunk_samples, 2048);
    }
}
