//! Config-layer tests that work without the `ort` feature.

use atomr_infer_runtime_ort::{ExecutionProvider, OrtConfig};

#[test]
fn execution_provider_default_is_cpu() {
    assert_eq!(ExecutionProvider::default(), ExecutionProvider::Cpu);
}

#[test]
fn config_round_trip_minimal() {
    let cfg = OrtConfig {
        onnx_path: "/tmp/x.onnx".into(),
        execution_provider: ExecutionProvider::Cpu,
        device_id: 0,
        tokenizer_path: None,
        hf_repo: None,
        intra_threads: None,
        default_max_new_tokens: 256,
    };
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: OrtConfig = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.onnx_path, cfg.onnx_path);
    assert_eq!(back.execution_provider, cfg.execution_provider);
}

#[test]
fn config_accepts_partial_json() {
    // Only `onnx_path` is required; the rest should default.
    let cfg: OrtConfig =
        serde_json::from_str(r#"{"onnx_path":"/etc/models/m.onnx"}"#).expect("deserialize");
    assert_eq!(cfg.onnx_path.to_str(), Some("/etc/models/m.onnx"));
    assert_eq!(cfg.execution_provider, ExecutionProvider::Cpu);
    assert_eq!(cfg.device_id, 0);
    assert_eq!(cfg.default_max_new_tokens, 256);
}

#[test]
fn execution_provider_serde_uses_snake_case() {
    let json = serde_json::to_string(&ExecutionProvider::TensorRt).expect("serialize");
    assert_eq!(json, "\"tensor_rt\"");
    let json = serde_json::to_string(&ExecutionProvider::DirectMl).expect("serialize");
    assert_eq!(json, "\"direct_ml\"");
}
