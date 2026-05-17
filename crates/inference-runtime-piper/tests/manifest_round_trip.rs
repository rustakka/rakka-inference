//! Voice-manifest + phoneme-map round trip against a tiny in-test
//! fixture. Exercises the load → map-build → ids-for-text path
//! without needing a real `.onnx` voice on disk.

use std::collections::BTreeMap;
use std::path::PathBuf;

use atomr_infer_runtime_piper::{PhonemeMap, PiperConfig, PiperVoiceManifest};

const SAMPLE_MANIFEST: &str = r#"{
    "audio": { "sample_rate": 22050 },
    "inference": { "noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8 },
    "phoneme_id_map": {
        "_": [0],
        "^": [1],
        "$": [2],
        "h": [10],
        "e": [11],
        "l": [12],
        "o": [13],
        " ": [3],
        "w": [14],
        "r": [15],
        "d": [16]
    },
    "num_symbols": 256,
    "num_speakers": 1,
    "espeak": { "voice": "en-us" }
}"#;

#[test]
fn manifest_loads_from_tempfile() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = dir.path().join("voice.onnx.json");
    std::fs::write(&manifest, SAMPLE_MANIFEST).unwrap();

    let cfg = PiperConfig {
        voice_path: dir.path().join("voice.onnx"),
        voice_manifest_path: Some(manifest.clone()),
        speaker_id: None,
        length_scale: 1.0,
        noise_scale: 0.667,
        noise_w: 0.8,
        chunk_samples: 1024,
        intra_threads: None,
    };
    assert_eq!(cfg.resolved_manifest_path(), manifest);

    let raw = std::fs::read(cfg.resolved_manifest_path()).unwrap();
    let m: PiperVoiceManifest = serde_json::from_slice(&raw).unwrap();
    assert_eq!(m.audio.sample_rate, 22050);
    assert_eq!(m.num_speakers, 1);
    assert_eq!(m.phoneme_id_map.get("h"), Some(&vec![10]));
}

#[test]
fn phoneme_map_ids_for_text_matches_expected_envelope() {
    let m: PiperVoiceManifest = serde_json::from_str(SAMPLE_MANIFEST).unwrap();
    let map = PhonemeMap::new(m.phoneme_id_map.clone());

    let ids = map.ids_for_text("hello").unwrap();
    // ^ _ h _ e _ l _ l _ o _ $
    assert_eq!(ids, vec![1, 0, 10, 0, 11, 0, 12, 0, 12, 0, 13, 0, 2]);
}

#[test]
fn phoneme_map_handles_word_with_space() {
    let m: PiperVoiceManifest = serde_json::from_str(SAMPLE_MANIFEST).unwrap();
    let map = PhonemeMap::new(m.phoneme_id_map.clone());

    let ids = map.ids_for_text("hello world").unwrap();
    let pad = 0;
    let bos = 1;
    let eos = 2;
    let count_pad = ids.iter().filter(|&&i| i == pad).count();
    assert!(ids.first() == Some(&bos));
    assert!(ids.last() == Some(&eos));
    // One leading pad after BOS, then one pad after each of 11
    // graphemes, so 12 total pad slots.
    assert_eq!(count_pad, 12);
}

#[test]
fn phoneme_map_constructed_directly_skips_envelope_when_keys_absent() {
    let mut raw = BTreeMap::new();
    raw.insert("a".into(), vec![1]);
    raw.insert("b".into(), vec![2]);
    let map = PhonemeMap::new(raw);
    assert_eq!(map.ids_for_text("ab").unwrap(), vec![1, 2]);
}

#[test]
fn missing_manifest_path_resolves_predictably() {
    let cfg = PiperConfig {
        voice_path: PathBuf::from("/tmp/voice.onnx"),
        voice_manifest_path: None,
        speaker_id: None,
        length_scale: 1.0,
        noise_scale: 0.667,
        noise_w: 0.8,
        chunk_samples: 1024,
        intra_threads: None,
    };
    assert_eq!(
        cfg.resolved_manifest_path(),
        PathBuf::from("/tmp/voice.onnx.json")
    );
}
