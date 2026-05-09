//! End-to-end smoke test for the low-level `infer()` path on CPU.
//!
//! Skipped unless `ATOMR_INFER_ORT_TEST_INFER_MODEL` is set to a
//! single-input single-f32-output ONNX file (an identity / passthrough
//! / linear projection works). We don't bundle a fixture in-tree
//! because hand-rolling ONNX protobuf bytes adds opaque binary to the
//! repo and a generated fixture would need a build script with a
//! Python dep.
//!
//! How to run locally:
//!
//! ```sh
//! python -c "import onnx, onnx.helper as h; \
//!   x = h.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [None, 4]); \
//!   y = h.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [None, 4]); \
//!   node = h.make_node('Identity', ['x'], ['y']); \
//!   g = h.make_graph([node], 't', [x], [y]); \
//!   m = h.make_model(g, opset_imports=[h.make_opsetid('', 13)]); \
//!   onnx.save(m, '/tmp/identity.onnx')"
//! ATOMR_INFER_ORT_TEST_INFER_MODEL=/tmp/identity.onnx \
//!   cargo test -p atomr-infer-runtime-ort --features ort cpu_smoke
//! ```

#![cfg(feature = "ort")]

use std::collections::HashMap;

use atomr_infer_runtime_ort::{ExecutionProvider, InferTensor, OrtConfig, OrtRunner};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn infer_identity_round_trip() {
    let Ok(path) = std::env::var("ATOMR_INFER_ORT_TEST_INFER_MODEL") else {
        eprintln!("skip: set ATOMR_INFER_ORT_TEST_INFER_MODEL to enable");
        return;
    };

    let mut runner = OrtRunner::new(OrtConfig {
        onnx_path: path.into(),
        execution_provider: ExecutionProvider::Cpu,
        device_id: 0,
        tokenizer_path: None,
        hf_repo: None,
        intra_threads: Some(1),
        default_max_new_tokens: 0,
    });

    let mut inputs: HashMap<String, InferTensor> = HashMap::new();
    inputs.insert(
        "x".into(),
        InferTensor::F32 {
            shape: vec![1, 4],
            data: vec![1.0, 2.0, 3.0, 4.0],
        },
    );

    let outputs = runner.infer(inputs).await.expect("infer");
    let (shape, data) = outputs.f32.get("y").expect("y output present");
    assert_eq!(shape, &[1, 4]);
    assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
}
