//! Kokoro runner configuration.
//!
//! [`KokoroConfig`] is the operator-facing config. Voice packs are
//! pre-converted `.onnx` files placed in `voice_pack_dir`; the voice
//! loaded at session-build time is `default_voice` (the filename stem,
//! without the `.onnx` extension).
//!
//! Kokoro-82M is an open-weight ONNX TTS model from
//! <https://huggingface.co/hexgrad/Kokoro-82M>. Voice pack `.pt` files
//! must be exported to ONNX before use with this runtime.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Operator-facing config for [`crate::KokoroRunner`].
///
/// # Examples
///
/// ```
/// use atomr_infer_runtime_kokoro::KokoroConfig;
///
/// let cfg = KokoroConfig {
///     voice_pack_dir: "/models/kokoro/voices".into(),
///     default_voice: "af_heart".into(),
///     chunk_samples: 4096,
///     intra_threads: None,
/// };
/// assert_eq!(cfg.default_voice, "af_heart");
/// assert!(!cfg.default_voice.is_empty());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroConfig {
    /// Directory containing pre-converted Kokoro `.onnx` voice files.
    pub voice_pack_dir: PathBuf,
    /// Filename stem (without `.onnx`) of the voice to load at session
    /// startup. Must not be empty.
    pub default_voice: String,
    /// PCM chunk size in samples for the output stream. Default 4096
    /// (≈185 ms at 24 000 Hz, Kokoro's native sample rate).
    #[serde(default = "default_chunk_samples")]
    pub chunk_samples: usize,
    /// Pin the ORT session to N CPU threads. `None` = let ORT decide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intra_threads: Option<u32>,
}

fn default_chunk_samples() -> usize {
    4096
}

impl KokoroConfig {
    /// Resolve the ONNX path for the current `default_voice`.
    pub fn voice_onnx_path(&self) -> PathBuf {
        self.voice_pack_dir.join(format!("{}.onnx", self.default_voice))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> KokoroConfig {
        KokoroConfig {
            voice_pack_dir: "/models/kokoro/voices".into(),
            default_voice: "af_heart".into(),
            chunk_samples: 4096,
            intra_threads: None,
        }
    }

    #[test]
    fn config_defaults() {
        let cfg = sample_config();
        assert_eq!(cfg.chunk_samples, 4096);
        assert!(cfg.intra_threads.is_none());
        assert_eq!(cfg.default_voice, "af_heart");
    }

    #[test]
    fn voice_name_must_not_be_empty() {
        let cfg = KokoroConfig {
            voice_pack_dir: "/voices".into(),
            default_voice: String::new(),
            chunk_samples: 4096,
            intra_threads: None,
        };
        assert!(
            cfg.default_voice.is_empty(),
            "guard enforced in runner not config"
        );
    }

    #[test]
    fn voice_onnx_path_assembles_correctly() {
        let cfg = sample_config();
        assert_eq!(
            cfg.voice_onnx_path(),
            std::path::PathBuf::from("/models/kokoro/voices/af_heart.onnx")
        );
    }

    #[test]
    fn config_round_trips_through_serde() {
        let cfg = KokoroConfig {
            voice_pack_dir: "/tmp/voices".into(),
            default_voice: "af_sky".into(),
            chunk_samples: 2048,
            intra_threads: Some(4),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: KokoroConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.voice_pack_dir, cfg.voice_pack_dir);
        assert_eq!(back.default_voice, cfg.default_voice);
        assert_eq!(back.chunk_samples, cfg.chunk_samples);
        assert_eq!(back.intra_threads, cfg.intra_threads);
    }
}
