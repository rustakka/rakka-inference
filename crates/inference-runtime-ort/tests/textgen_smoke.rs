//! Chat-style `execute()` smoke test against a real ONNX-exported
//! causal LM.
//!
//! Skipped unless `ATOMR_INFER_ORT_TEST_GEN_MODEL` is set to a
//! directory containing `model.onnx` (HF Optimum-ONNX layout — KV
//! cache + f32 logits) and `tokenizer.json`.
//!
//! How to run locally:
//!
//! ```sh
//! optimum-cli export onnx --model gpt2 /tmp/gpt2-onnx
//! ATOMR_INFER_ORT_TEST_GEN_MODEL=/tmp/gpt2-onnx \
//!   cargo test -p atomr-infer-runtime-ort --features ort textgen_smoke
//! ```

#![cfg(feature = "ort")]

use atomr_infer_core::batch::{ExecuteBatch, Message, MessageContent, Role, SamplingParams};
use atomr_infer_core::runner::ModelRunner;
use atomr_infer_runtime_ort::{ExecutionProvider, OrtConfig, OrtRunner};
use futures::StreamExt;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn five_token_greedy_generation() {
    let Ok(dir) = std::env::var("ATOMR_INFER_ORT_TEST_GEN_MODEL") else {
        eprintln!("skip: set ATOMR_INFER_ORT_TEST_GEN_MODEL to enable");
        return;
    };
    let dir: std::path::PathBuf = dir.into();

    let mut runner = OrtRunner::new(OrtConfig {
        onnx_path: dir.join("model.onnx"),
        execution_provider: ExecutionProvider::Cpu,
        device_id: 0,
        tokenizer_path: Some(dir.join("tokenizer.json")),
        hf_repo: None,
        intra_threads: Some(2),
        default_max_new_tokens: 5,
    });

    let batch = ExecuteBatch {
        request_id: "smoke-1".into(),
        model: "test".into(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello,".into()),
        }],
        sampling: SamplingParams {
            temperature: Some(0.0), // greedy
            max_tokens: Some(5),
            seed: Some(42),
            ..Default::default()
        },
        stream: true,
        estimated_tokens: 16,
    };

    let handle = runner.execute(batch).await.expect("execute");
    let mut stream = handle.into_stream();

    let mut chunks = 0usize;
    let mut total_text = String::new();
    let mut final_chunk = None;
    while let Some(item) = stream.next().await {
        let chunk = item.expect("chunk");
        total_text.push_str(&chunk.text_delta);
        if chunk.finish_reason.is_some() {
            final_chunk = Some(chunk);
        }
        chunks += 1;
    }
    assert!(chunks > 0, "expected at least one chunk");
    let final_chunk = final_chunk.expect("expected a final chunk with finish_reason");
    assert!(
        final_chunk.usage.is_some(),
        "final chunk should carry usage stats"
    );
}
