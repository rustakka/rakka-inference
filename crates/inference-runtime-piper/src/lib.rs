//! # inference-runtime-piper
//!
//! Local Piper TTS runtime for `atomr-infer`. Implements
//! [`atomr_infer_core::runner::SpeechRunner`] against the
//! [rhasspy/piper](https://github.com/rhasspy/piper) voice-pack
//! format: a `.onnx` graph plus a sibling `.onnx.json` manifest
//! shipping the sample rate, phoneme→id map, inference scales, and
//! optional `espeak` voice id.
//!
//! ## Build profiles
//!
//! | Build                                                                  | Result                                                |
//! |------------------------------------------------------------------------|-------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-piper`                              | Stub — `ort` not in dep graph.                        |
//! | `cargo build -p atomr-infer-runtime-piper --features piper`             | Real path with `ort` crate (CPU EP).                  |
//! | `cargo build -p atomr-infer-runtime-piper --features piper-cuda`        | Adds the CUDA EP — needs a working CUDA toolkit.      |
//! | `cargo build -p atomr-infer-runtime-piper --features piper-load-dynamic`| Loads `libonnxruntime` at runtime via `ORT_DYLIB_PATH`. |
//!
//! ## Phonemization scope (M4)
//!
//! M4 wires the [`SpeechRunner`] surface and the ONNX session. Text
//! → phoneme conversion is char-level: each Unicode grapheme of
//! [`atomr_infer_core::audio::SpeechBatch::text`] is looked up
//! directly in the voice's `phoneme_id_map`. This is the right
//! behavior when the caller has already fed text through
//! `espeak-ng -q -x --ipa`. Real text → IPA via espeak-ng FFI is a
//! follow-up — the seam is [`phoneme::PhonemeMap::ids_for_text`].
//!
//! ## Output shape
//!
//! `PiperRunner::speak` streams a sequence of
//! [`atomr_infer_core::audio::SpeechChunk`]s carrying PCM16-LE mono
//! audio at the voice's native sample rate. The chunk size in
//! samples is configurable via [`PiperConfig::chunk_samples`]
//! (default 4096 — about 185 ms at 22 050 Hz).
//!
//! [`SpeechRunner`]: atomr_infer_core::runner::SpeechRunner
//!
//! ## Source
//!
//! `FR-TTS-001`. See `docs/audio-modalities.md`.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod error;
pub mod phoneme;
mod runner;

#[cfg(feature = "piper")]
mod session;

pub use config::{PiperAudio, PiperConfig, PiperEspeak, PiperInference, PiperVoiceManifest};
pub use error::PiperError;
pub use phoneme::{PhonemeMap, BOS, EOS, PAD};
pub use runner::PiperRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::SpeechRunner;
    use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

    fn runner() -> PiperRunner {
        PiperRunner::new(PiperConfig {
            voice_path: "/tmp/does-not-exist.onnx".into(),
            voice_manifest_path: None,
            speaker_id: None,
            length_scale: 1.0,
            noise_scale: 0.667,
            noise_w: 0.8,
            chunk_samples: 4096,
            intra_threads: None,
        })
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::TextToSpeech);
        assert_eq!(r.transport_kind(), TransportKind::LocalCpu);
    }

    #[cfg(not(feature = "piper"))]
    #[tokio::test]
    async fn speak_without_feature_returns_internal_error() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "m".into(),
            text: "hi".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 2,
        };
        match r.speak(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("piper feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[cfg(feature = "piper")]
    #[tokio::test]
    async fn speak_with_feature_but_missing_voice_returns_bad_request() {
        use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;

        let mut r = runner();
        let batch = SpeechBatch {
            request_id: "t".into(),
            model: "m".into(),
            text: "hi".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 2,
        };
        match r.speak(batch).await {
            Err(InferenceError::BadRequest { message }) => {
                assert!(message.contains("voice file not found"), "{message}");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }
}
