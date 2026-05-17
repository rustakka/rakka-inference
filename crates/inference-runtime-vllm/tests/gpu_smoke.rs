//! GPU smoke tests — local-only, **never run in CI**.
//!
//! Every test in this file is `#[ignore]` so `cargo test` skips them
//! by default. Run them locally on a workstation with a CUDA GPU,
//! Python 3.10+, vLLM installed, and a HuggingFace token that has
//! accepted the Gemma 4 ToS:
//!
//! ```sh
//! cargo test -p atomr-infer-runtime-vllm --features gemma-default \
//!     -- --ignored --test-threads=1
//! ```
//!
//! `--test-threads=1` matters: each test spins up a vLLM engine
//! that owns the GPU, and parallel runs would OOM.
//!
//! ## What's covered
//!
//! - `probe_runs` — env probe returns *something* coherent (Ready /
//!   Skipped / Error). Cheap, no model load.
//! - `gemma_4_e4b_smoke` — load Gemma 4 E4B-it (with cpu_offload_gb=4
//!   for 16 GB cards), run one completion end-to-end, assert non-empty
//!   output. ~75 s cold.
//! - `concurrent_requests_dont_corrupt` — two concurrent generations
//!   on one engine instance return distinct, non-empty outputs.
//!
//! Tests default to `gemma-4-E4B-it` paired with `cpu_offload_gb=4`
//! so they fit on a 16 GB card. Override via
//! `ATOMR_INFER_GEMMA_SMOKE_MODEL=...` for ≥24 GB hardware (drop the
//! offload too) or for the smaller `gemma-4-E2B-it` variant.

#![cfg(feature = "gemma-default")]

use std::time::Duration;

use atomr_infer_core::batch::{ExecuteBatch, Message, MessageContent, Role, SamplingParams};
use atomr_infer_core::runner::ModelRunner;

use atomr_infer_runtime_vllm::{
    defaults::{validate_variant, GemmaDefaults},
    probe::{probe, ProbeResult},
    VllmConfig, VllmRunner,
};

fn smoke_model() -> String {
    std::env::var("ATOMR_INFER_GEMMA_SMOKE_MODEL").unwrap_or_else(|_| "google/gemma-4-E4B-it".into())
}

/// Build a `VllmConfig` for smoke testing. Conservative defaults so
/// we don't accidentally OOM a small dev GPU. `cpu_offload_gb=4`
/// pairs with E4B-it on 16 GB cards.
fn smoke_config() -> VllmConfig {
    VllmConfig {
        model: smoke_model(),
        tensor_parallel_size: 1,
        dtype: "auto".into(),
        gpu_memory_utilization: Some(0.85),
        max_model_len: Some(2048),
        hf_cache_dir: None,
        enforce_eager: Some(true),
        enable_prefix_caching: None,
        enable_chunked_prefill: None,
        max_num_seqs: Some(8),
        block_size: None,
        quantization: None,
        limit_mm_per_prompt: None,
        cpu_offload_gb: Some(4),
    }
}

/// Build a single-message `ExecuteBatch` asking Gemma to count.
fn count_batch(request_id: &str) -> ExecuteBatch {
    ExecuteBatch {
        request_id: request_id.into(),
        model: smoke_model(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(
                "Count from one to five. Just the numbers, comma-separated.".into(),
            ),
        }],
        sampling: SamplingParams {
            temperature: Some(0.0),
            max_tokens: Some(32),
            ..Default::default()
        },
        stream: true,
        estimated_tokens: 64,
    }
}

/// `cargo test ... -- --ignored probe_runs` — sanity-check that the
/// env probe returns coherent output. Doesn't require a GPU; doesn't
/// require vLLM. Always cheap.
#[test]
fn probe_runs() {
    let cfg = GemmaDefaults::default();
    validate_variant(&cfg.model_id).expect("default variant valid");
    let result = probe(&cfg.model_id, 4.5, 7.0, Some("google/gemma-4-E2B-it"));
    println!("probe outcome: {result:?}");
}

/// `cargo test ... -- --ignored gemma_4_e4b_smoke` — full
/// end-to-end: load the model, drive a single completion, assert
/// non-empty output. Skips if the env probe returns Skipped (no GPU
/// / no vLLM / no token).
#[ignore = "GPU + vLLM required; run locally with --ignored"]
#[tokio::test]
async fn gemma_4_e4b_smoke() {
    let cfg = smoke_config();

    // Probe first so a missing prereq surfaces as a clean SKIP rather
    // than a multi-minute hang.
    match probe(&cfg.model, 2.5, 4.0, None) {
        ProbeResult::Ready { vram_free_gb, .. } => {
            println!("probe ok: {vram_free_gb:.1} GB free VRAM");
        }
        ProbeResult::Skipped { reason, hint } => {
            eprintln!("smoke skipped: {reason}\n  hint: {hint}");
            return;
        }
        ProbeResult::Error(e) => panic!("probe error: {e}"),
    }

    let mut runner = VllmRunner::new(cfg);
    let handle = runner
        .execute(count_batch("smoke-1"))
        .await
        .expect("execute returned error");

    let mut stream = handle.into_stream();
    let mut total = String::new();
    let mut chunks = 0usize;
    let deadline = std::time::Instant::now() + Duration::from_secs(120);

    use futures::StreamExt;
    while let Some(chunk_result) = stream.next().await {
        if std::time::Instant::now() >= deadline {
            panic!("smoke test exceeded 120s — engine appears hung");
        }
        let chunk = chunk_result.expect("stream emitted error");
        chunks += 1;
        total.push_str(&chunk.text_delta);
        if chunk.finish_reason.is_some() {
            break;
        }
    }

    println!("smoke output ({chunks} chunks): {total:?}");
    assert!(chunks > 0, "no chunks received");
    assert!(!total.is_empty(), "empty output text");
    assert!(
        total.contains('1') || total.to_lowercase().contains("one"),
        "expected the digit 1 / word 'one' in counting response, got: {total:?}"
    );
}

/// `cargo test ... -- --ignored concurrent_requests_dont_corrupt`
/// — two simultaneous generations on the same engine return
/// distinct outputs. Validates that `request_seq` + the per-request
/// abort don't cross-contaminate.
#[ignore = "GPU + vLLM required; run locally with --ignored"]
#[tokio::test]
async fn concurrent_requests_dont_corrupt() {
    let cfg = smoke_config();
    match probe(&cfg.model, 2.5, 4.0, None) {
        ProbeResult::Ready { .. } => {}
        ProbeResult::Skipped { reason, hint } => {
            eprintln!("concurrent smoke skipped: {reason}\n  hint: {hint}");
            return;
        }
        ProbeResult::Error(e) => panic!("probe error: {e}"),
    }

    let mut runner = VllmRunner::new(cfg);

    let h1 = runner
        .execute(ExecuteBatch {
            request_id: "concur-A".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Reply with just the word ALPHA.".into()),
            }],
            sampling: SamplingParams {
                temperature: Some(0.0),
                max_tokens: Some(8),
                ..Default::default()
            },
            stream: true,
            estimated_tokens: 16,
            model: smoke_model(),
        })
        .await
        .expect("h1 launch");

    let h2 = runner
        .execute(ExecuteBatch {
            request_id: "concur-B".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("Reply with just the word BRAVO.".into()),
            }],
            sampling: SamplingParams {
                temperature: Some(0.0),
                max_tokens: Some(8),
                ..Default::default()
            },
            stream: true,
            estimated_tokens: 16,
            model: smoke_model(),
        })
        .await
        .expect("h2 launch");

    use futures::StreamExt;
    let collect = |handle: atomr_infer_core::runner::RunHandle| async move {
        let mut s = handle.into_stream();
        let mut text = String::new();
        while let Some(c) = s.next().await {
            let c = c.expect("chunk error");
            text.push_str(&c.text_delta);
            if c.finish_reason.is_some() {
                break;
            }
        }
        text
    };

    let (a, b) = tokio::join!(collect(h1), collect(h2));
    println!("concurrent A: {a:?}\nconcurrent B: {b:?}");
    assert!(!a.is_empty() && !b.is_empty());
    assert_ne!(a, b, "concurrent requests returned identical outputs");
}
