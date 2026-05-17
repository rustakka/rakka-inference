//! Piper ONNX session lifecycle.
//!
//! Only compiled with the `piper` feature. The session loads:
//!
//! 1. The `.onnx` model file.
//! 2. The sibling `.onnx.json` manifest (sample rate, scales,
//!    phoneme id map, speaker count).
//!
//! Inference is `i64[1,T] phoneme_ids + i64[1] input_lengths +
//! f32[3] scales + i64[1] sid` → `f32[1,1,samples]` PCM, matching
//! the contract every rhasspy/piper-published voice ships against.

use std::path::Path;
use std::sync::Arc;

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::config::{PiperConfig, PiperVoiceManifest};
use crate::error::PiperError;
use crate::phoneme::PhonemeMap;

pub(crate) struct PiperState {
    pub(crate) session: parking_lot::Mutex<Session>,
    pub(crate) manifest: PiperVoiceManifest,
    pub(crate) phoneme_map: PhonemeMap,
    pub(crate) sample_rate: u32,
}

pub(crate) fn load_manifest(path: &Path) -> Result<PiperVoiceManifest, PiperError> {
    if !path.exists() {
        return Err(PiperError::ManifestNotFound {
            path: path.to_path_buf(),
        });
    }
    let raw = std::fs::read(path)?;
    let m: PiperVoiceManifest = serde_json::from_slice(&raw)?;
    Ok(m)
}

pub(crate) fn build_state(cfg: &PiperConfig) -> Result<Arc<PiperState>, PiperError> {
    if !cfg.voice_path.exists() {
        return Err(PiperError::VoiceNotFound {
            path: cfg.voice_path.clone(),
        });
    }
    let manifest_path = cfg.resolved_manifest_path();
    let manifest = load_manifest(&manifest_path)?;
    let sample_rate = manifest.audio.sample_rate;
    let phoneme_map = PhonemeMap::new(manifest.phoneme_id_map.clone());

    let mut builder = Session::builder().map_err(|e| PiperError::Ort(e.to_string()))?;
    if let Some(t) = cfg.intra_threads {
        builder = builder
            .with_intra_threads(t as usize)
            .map_err(|e| PiperError::Ort(e.to_string()))?;
    }
    let session = builder
        .commit_from_file(&cfg.voice_path)
        .map_err(|e| PiperError::Ort(e.to_string()))?;

    Ok(Arc::new(PiperState {
        session: parking_lot::Mutex::new(session),
        manifest,
        phoneme_map,
        sample_rate,
    }))
}

/// Run one synthesis pass. Returns the f32 PCM audio at the voice's
/// native sample rate.
pub(crate) fn run_synthesis(
    state: &PiperState,
    text: &str,
    cfg: &PiperConfig,
) -> Result<Vec<f32>, PiperError> {
    use std::borrow::Cow;

    let ids = state.phoneme_map.ids_for_text(text)?;
    let length = ids.len() as i64;

    // Validate speaker id against the voice's declared range.
    let sid = cfg.speaker_id.unwrap_or(0);
    if sid < 0 || sid as u32 >= state.manifest.num_speakers {
        return Err(PiperError::SpeakerOutOfRange {
            requested: sid,
            num_speakers: state.manifest.num_speakers,
        });
    }

    // Inference-time overrides: per-call config wins over manifest
    // defaults.
    let noise_scale = cfg.noise_scale;
    let length_scale = cfg.length_scale;
    let noise_w = cfg.noise_w;

    let input_ids = Tensor::from_array(([1_i64, length], ids)).map_err(|e| PiperError::Ort(e.to_string()))?;
    let input_lengths =
        Tensor::from_array(([1_i64], vec![length])).map_err(|e| PiperError::Ort(e.to_string()))?;
    let scales = Tensor::from_array(([3_i64], vec![noise_scale, length_scale, noise_w]))
        .map_err(|e| PiperError::Ort(e.to_string()))?;
    let sid_tensor = Tensor::from_array(([1_i64], vec![sid])).map_err(|e| PiperError::Ort(e.to_string()))?;

    let mut entries: Vec<(Cow<'static, str>, SessionInputValue<'_>)> = vec![
        (Cow::Borrowed("input"), input_ids.into()),
        (Cow::Borrowed("input_lengths"), input_lengths.into()),
        (Cow::Borrowed("scales"), scales.into()),
    ];
    // Multi-speaker voices want `sid`; single-speaker voices don't
    // declare it as an input.
    if state.manifest.num_speakers > 1 {
        entries.push((Cow::Borrowed("sid"), sid_tensor.into()));
    }

    let mut session = state.session.lock();
    // Snapshot output names before run() so the immutable borrow does
    // not collide with the mutable borrow `session.run` takes.
    let output_names: Vec<String> = session.outputs().iter().map(|o| o.name().to_owned()).collect();
    let outputs = session.run(entries).map_err(|e| PiperError::Ort(e.to_string()))?;

    // Piper emits a single `output` tensor. Use the first available
    // f32 tensor regardless of name to stay tolerant of voice-pack
    // variants.
    for name in output_names {
        if let Some(v) = outputs.get(name.as_str()) {
            if let Ok((_shape, data)) = v.try_extract_tensor::<f32>() {
                return Ok(data.to_vec());
            }
        }
    }
    Err(PiperError::Ort("no f32 output tensor".into()))
}
