//! Local performance harness for Gemma 4 via vLLM.
//!
//! This binary is **not** run in CI. It's the experimentation tool
//! for "I have a discrete GPU and want to measure something." Every
//! subcommand probes the env first and exits with a clean message
//! when the GPU / vLLM / HF token isn't ready, so it's safe to run
//! on a CPU-only laptop too — it just won't measure anything.
//!
//! ```sh
//! cargo run -p gemma_bench --release --features gemma-default -- smoke
//! cargo run -p gemma_bench --release --features gemma-default -- latency
//! cargo run -p gemma_bench --release --features gemma-default -- sweep gpu-util
//! cargo run -p gemma_bench --release --features gemma-default -- experiments
//! ```
//!
//! Output is human-readable by default; pipe `--format jsonl` into
//! `jq` for analysis. Results are also written to
//! `target/gemma-bench/<subcommand>-<unix-ts>.jsonl` for archiving.

#![cfg(feature = "gemma-default")]

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use futures::StreamExt;
use serde::Serialize;

use atomr_infer_core::batch::{ExecuteBatch, Message, MessageContent, Role, SamplingParams};
use atomr_infer_core::runner::ModelRunner;
use atomr_infer_runtime_vllm::{
    defaults::{validate_variant, SUPPORTED_VARIANTS},
    probe::{probe, ProbeResult},
    VllmConfig, VllmRunner,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// HuggingFace model id. Validated against the supported Gemma 4
    /// variants unless `--allow-any-model` is passed. E4B-it is the
    /// default; on 16 GB cards the bench's `base_config` sets
    /// `cpu_offload_gb=4` to keep peak GPU memory under the limit.
    /// Operators with ≥24 GB GPUs can drop the offload.
    #[arg(long, default_value = "google/gemma-4-E4B-it", global = true)]
    model: String,

    /// Skip the supported-variant allow-list (useful when probing a
    /// new Gemma release before the harness is updated).
    #[arg(long, default_value_t = false, global = true)]
    allow_any_model: bool,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Human, global = true)]
    format: OutputFormat,

    /// Skip writing the JSONL archive under `target/gemma-bench/`.
    #[arg(long, default_value_t = false, global = true)]
    no_archive: bool,

    /// Per-request token budget. Larger ⇒ more decode time per
    /// sample, more stable measurements, longer wall time.
    #[arg(long, default_value_t = 64, global = true)]
    max_tokens: u32,

    /// Reduce the iteration count of every sweep so each invocation
    /// finishes in seconds rather than minutes. Use during dev.
    #[arg(long, default_value_t = false, global = true)]
    quick: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutputFormat {
    Human,
    Jsonl,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Single completion against a default config — verifies the
    /// bridge boots and emits non-empty output.
    Smoke,

    /// TTFT + tokens/sec for one request at a few prompt lengths.
    Latency,

    /// Aggregate tokens/sec across N concurrent requests.
    Throughput {
        /// Number of concurrent requests.
        #[arg(long, default_value_t = 4)]
        concurrency: u32,
        /// Total requests to dispatch.
        #[arg(long, default_value_t = 16)]
        total: u32,
    },

    /// Sweep over a single perf knob.
    Sweep {
        #[command(subcommand)]
        knob: SweepKnob,
    },

    /// Curated battery: cuda-graphs on/off, prefix-caching on/off,
    /// chunked-prefill on/off, gpu_util ∈ {0.4, 0.6}. Useful baseline
    /// for "is this faster than vLLM defaults on my box?".
    Experiments,

    /// Head-to-head E4B-it vs E2B-it on the same prompt.
    Compare,
}

#[derive(Subcommand, Debug)]
enum SweepKnob {
    /// `gpu_memory_utilization` in [0.3, 0.5, 0.7, 0.9].
    GpuUtil,
    /// `dtype` in [auto, float16, bfloat16].
    Dtype,
    /// `enforce_eager` ∈ {false (graphs), true (eager)}. Quantifies
    /// CUDA-graph speedup.
    CudaGraphs,
    /// `enable_prefix_caching` ∈ {false, true}. Useful when prompts
    /// share a long system prefix.
    PrefixCache,
    /// `enable_chunked_prefill` ∈ {false, true}. Helps TTFT under
    /// concurrent load.
    ChunkedPrefill,
    /// Concurrency ∈ [1, 2, 4, 8] (or [1, 2] in `--quick`).
    Concurrency,
    /// `block_size` ∈ [16, 32].
    BlockSize,
    /// `max_num_seqs` ∈ [16, 64, 256].
    MaxNumSeqs,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    if !cli.allow_any_model {
        validate_variant(&cli.model)
            .map_err(|e| anyhow!("{e}; pass --allow-any-model to bypass"))?;
    } else if !SUPPORTED_VARIANTS.contains(&cli.model.as_str()) {
        eprintln!(
            "warning: --allow-any-model in effect — model {:?} not in supported list",
            cli.model
        );
    }

    let writer = ResultWriter::new(&cli)?;
    match cli.cmd {
        Cmd::Smoke => cmd_smoke(&cli, &writer).await?,
        Cmd::Latency => cmd_latency(&cli, &writer).await?,
        Cmd::Throughput { concurrency, total } => {
            cmd_throughput(&cli, &writer, concurrency, total).await?
        }
        Cmd::Sweep { ref knob } => cmd_sweep(&cli, &writer, knob).await?,
        Cmd::Experiments => cmd_experiments(&cli, &writer).await?,
        Cmd::Compare => cmd_compare(&cli, &writer).await?,
    }
    writer.finish();
    Ok(())
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

async fn cmd_smoke(cli: &Cli, writer: &ResultWriter) -> Result<()> {
    let cfg = base_config(&cli.model);
    if !pass_probe(&cfg, "smoke") {
        return Ok(());
    }
    let mut runner = VllmRunner::new(cfg);
    let stats = run_one(
        &mut runner,
        &cli.model,
        "Reply with just the word OK.",
        cli.max_tokens.min(8),
    )
    .await?;
    writer.emit("smoke", &stats);
    println!(
        "smoke OK: ttft={:.1}ms total={:.1}ms tokens={} text={:?}",
        stats.ttft_ms, stats.total_ms, stats.output_tokens, stats.text
    );
    Ok(())
}

async fn cmd_latency(cli: &Cli, writer: &ResultWriter) -> Result<()> {
    let cfg = base_config(&cli.model);
    if !pass_probe(&cfg, "latency") {
        return Ok(());
    }
    let mut runner = VllmRunner::new(cfg);

    let prompts = if cli.quick {
        vec![("short", short_prompt())]
    } else {
        vec![
            ("short", short_prompt()),
            ("medium", medium_prompt()),
            ("long", long_prompt()),
        ]
    };

    println!("{:<10} {:>10} {:>12} {:>12} {:>10}", "size", "ttft_ms", "decode_ms", "tok/s_dec", "tokens");
    for (label, prompt) in prompts {
        let stats = run_one(&mut runner, &cli.model, &prompt, cli.max_tokens).await?;
        writer.emit_with("latency", label, &stats);
        println!(
            "{:<10} {:>10.1} {:>12.1} {:>12.2} {:>10}",
            label,
            stats.ttft_ms,
            stats.decode_ms,
            stats.tokens_per_sec_decode(),
            stats.output_tokens
        );
    }
    Ok(())
}

async fn cmd_throughput(
    cli: &Cli,
    writer: &ResultWriter,
    concurrency: u32,
    total: u32,
) -> Result<()> {
    let cfg = base_config(&cli.model);
    if !pass_probe(&cfg, "throughput") {
        return Ok(());
    }
    let mut runner = VllmRunner::new(cfg);

    let total = if cli.quick { total.min(8) } else { total };
    let agg = run_concurrent(
        &mut runner,
        &cli.model,
        concurrency,
        total,
        cli.max_tokens,
        &medium_prompt(),
    )
    .await?;
    writer.emit("throughput", &agg);
    println!(
        "throughput: concurrency={} total={} wall_ms={:.1} tokens={} aggregate_tok/s={:.2} \
         per_req_ttft_p50={:.1}ms p95={:.1}ms",
        agg.concurrency,
        agg.total_requests,
        agg.wall_ms,
        agg.total_output_tokens,
        agg.tokens_per_sec_aggregate,
        agg.ttft_p50,
        agg.ttft_p95
    );
    Ok(())
}

async fn cmd_sweep(cli: &Cli, writer: &ResultWriter, knob: &SweepKnob) -> Result<()> {
    let base = base_config(&cli.model);
    if !pass_probe(&base, "sweep") {
        return Ok(());
    }

    let n_requests = if cli.quick { 4 } else { 8 };
    let prompt = medium_prompt();

    match knob {
        SweepKnob::GpuUtil => {
            let values: &[f32] = if cli.quick { &[0.45, 0.7] } else { &[0.3, 0.5, 0.7, 0.9] };
            sweep_runs(
                "gpu_util",
                values,
                |v| {
                    let mut c = base.clone();
                    c.gpu_memory_utilization = Some(*v);
                    c
                },
                cli,
                writer,
                n_requests,
                &prompt,
            )
            .await?;
        }
        SweepKnob::Dtype => {
            let values = if cli.quick {
                vec!["auto", "bfloat16"]
            } else {
                vec!["auto", "float16", "bfloat16"]
            };
            sweep_runs(
                "dtype",
                &values,
                |v| {
                    let mut c = base.clone();
                    c.dtype = (*v).to_string();
                    c
                },
                cli,
                writer,
                n_requests,
                &prompt,
            )
            .await?;
        }
        SweepKnob::CudaGraphs => {
            let values: &[bool] = &[false, true];
            sweep_runs(
                "enforce_eager",
                values,
                |v| {
                    let mut c = base.clone();
                    c.enforce_eager = Some(*v);
                    c
                },
                cli,
                writer,
                n_requests,
                &prompt,
            )
            .await?;
        }
        SweepKnob::PrefixCache => {
            let values: &[bool] = &[false, true];
            sweep_runs(
                "enable_prefix_caching",
                values,
                |v| {
                    let mut c = base.clone();
                    c.enable_prefix_caching = Some(*v);
                    c
                },
                cli,
                writer,
                n_requests,
                &prompt,
            )
            .await?;
        }
        SweepKnob::ChunkedPrefill => {
            let values: &[bool] = &[false, true];
            sweep_runs(
                "enable_chunked_prefill",
                values,
                |v| {
                    let mut c = base.clone();
                    c.enable_chunked_prefill = Some(*v);
                    c
                },
                cli,
                writer,
                n_requests,
                &prompt,
            )
            .await?;
        }
        SweepKnob::Concurrency => {
            // Concurrency sweep doesn't change the engine config — we
            // build one engine and re-run with different concurrency.
            let cs: Vec<u32> = if cli.quick { vec![1, 2] } else { vec![1, 2, 4, 8] };
            let mut runner = VllmRunner::new(base.clone());
            println!(
                "{:<12} {:>10} {:>14} {:>10}",
                "concurrency", "wall_ms", "agg_tok/s", "ttft_p50"
            );
            for c in cs {
                let agg = run_concurrent(
                    &mut runner,
                    &cli.model,
                    c,
                    c * 2,
                    cli.max_tokens,
                    &prompt,
                )
                .await?;
                writer.emit_with("sweep_concurrency", &c.to_string(), &agg);
                println!(
                    "{:<12} {:>10.1} {:>14.2} {:>10.1}",
                    c, agg.wall_ms, agg.tokens_per_sec_aggregate, agg.ttft_p50
                );
            }
        }
        SweepKnob::BlockSize => {
            let values: &[u32] = &[16, 32];
            sweep_runs(
                "block_size",
                values,
                |v| {
                    let mut c = base.clone();
                    c.block_size = Some(*v);
                    c
                },
                cli,
                writer,
                n_requests,
                &prompt,
            )
            .await?;
        }
        SweepKnob::MaxNumSeqs => {
            let values: &[u32] = if cli.quick { &[16, 64] } else { &[16, 64, 256] };
            sweep_runs(
                "max_num_seqs",
                values,
                |v| {
                    let mut c = base.clone();
                    c.max_num_seqs = Some(*v);
                    c
                },
                cli,
                writer,
                n_requests,
                &prompt,
            )
            .await?;
        }
    }
    Ok(())
}

async fn cmd_experiments(cli: &Cli, writer: &ResultWriter) -> Result<()> {
    let base = base_config(&cli.model);
    if !pass_probe(&base, "experiments") {
        return Ok(());
    }
    let n = if cli.quick { 4 } else { 8 };
    let prompt = medium_prompt();

    println!("== experiments ({n} requests each, prompt=medium) ==");
    let configs: Vec<(&str, VllmConfig)> = vec![
        ("vllm_defaults", base.clone()),
        ("eager", VllmConfig { enforce_eager: Some(true), ..base.clone() }),
        ("graphs", VllmConfig { enforce_eager: Some(false), ..base.clone() }),
        (
            "prefix_cache",
            VllmConfig {
                enable_prefix_caching: Some(true),
                ..base.clone()
            },
        ),
        (
            "chunked_prefill",
            VllmConfig {
                enable_chunked_prefill: Some(true),
                ..base.clone()
            },
        ),
        (
            "high_gpu_util",
            VllmConfig {
                gpu_memory_utilization: Some(0.85),
                ..base.clone()
            },
        ),
        (
            "low_gpu_util",
            VllmConfig {
                gpu_memory_utilization: Some(0.4),
                ..base.clone()
            },
        ),
    ];

    println!(
        "{:<20} {:>10} {:>12} {:>14} {:>10}",
        "label", "ttft_ms", "decode_ms", "agg_tok/s", "wall_ms"
    );
    for (label, cfg) in configs {
        let mut runner = VllmRunner::new(cfg);
        match run_concurrent(&mut runner, &cli.model, 2, n, cli.max_tokens, &prompt).await {
            Ok(agg) => {
                writer.emit_with("experiments", label, &agg);
                println!(
                    "{:<20} {:>10.1} {:>12.1} {:>14.2} {:>10.1}",
                    label,
                    agg.ttft_p50,
                    agg.decode_p50,
                    agg.tokens_per_sec_aggregate,
                    agg.wall_ms
                );
            }
            Err(e) => {
                let err = e.to_string();
                writer.emit_with(
                    "experiments",
                    label,
                    &serde_json::json!({ "error": err }),
                );
                eprintln!("{:<20} ERROR: {}", label, err.lines().next().unwrap_or(""));
            }
        }
        // Engine teardown happens on drop. vLLM V1 doesn't always
        // release VRAM cleanly across drops in a single process — if
        // a later config OOMs, run experiments individually instead.
        drop(runner);
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }
    Ok(())
}

async fn cmd_compare(cli: &Cli, writer: &ResultWriter) -> Result<()> {
    let prompt = medium_prompt();
    let n = if cli.quick { 3 } else { 6 };
    println!("== compare E4B-it vs E2B-it ({n} requests each) ==");

    for variant in ["google/gemma-4-E4B-it", "google/gemma-4-E2B-it"] {
        let cfg = base_config(variant);
        if !pass_probe(&cfg, &format!("compare:{variant}")) {
            continue;
        }
        let mut runner = VllmRunner::new(cfg);
        let agg = run_concurrent(&mut runner, variant, 1, n, cli.max_tokens, &prompt).await?;
        writer.emit_with("compare", variant, &agg);
        println!(
            "{:<28} ttft_p50={:>7.1}ms decode_p50={:>7.1}ms tok/s={:>6.2}",
            variant, agg.ttft_p50, agg.decode_p50, agg.tokens_per_sec_aggregate
        );
        drop(runner);
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let _ = cli;
    Ok(())
}

// ---------------------------------------------------------------------------
// Sweep helpers
// ---------------------------------------------------------------------------

async fn sweep_runs<T: std::fmt::Display + Clone>(
    knob_name: &str,
    values: &[T],
    mk_config: impl Fn(&T) -> VllmConfig,
    cli: &Cli,
    writer: &ResultWriter,
    n_requests: u32,
    prompt: &str,
) -> Result<()> {
    println!(
        "{:<24} {:>10} {:>12} {:>14}",
        knob_name, "ttft_p50", "decode_p50", "agg_tok/s"
    );
    for v in values {
        let cfg = mk_config(v);
        let mut runner = VllmRunner::new(cfg);
        let label = format!("{knob_name}={v}");
        match run_concurrent(&mut runner, &cli.model, 1, n_requests, cli.max_tokens, prompt).await
        {
            Ok(agg) => {
                writer.emit_with(&format!("sweep_{knob_name}"), &v.to_string(), &agg);
                println!(
                    "{:<24} {:>10.1} {:>12.1} {:>14.2}",
                    label, agg.ttft_p50, agg.decode_p50, agg.tokens_per_sec_aggregate
                );
            }
            Err(e) => {
                let err = e.to_string();
                writer.emit_with(
                    &format!("sweep_{knob_name}"),
                    &v.to_string(),
                    &serde_json::json!({ "error": err }),
                );
                eprintln!("{:<24} ERROR: {}", label, err.lines().next().unwrap_or(""));
            }
        }
        drop(runner);
        // vLLM V1 doesn't always release VRAM cleanly across drops in
        // a single process — give the allocator a moment before the
        // next config tries to grab it.
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Measurement primitives
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct RunStats {
    request_id: String,
    ttft_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    output_tokens: u32,
    text: String,
}

impl RunStats {
    fn tokens_per_sec_decode(&self) -> f64 {
        if self.decode_ms <= 0.0 {
            0.0
        } else {
            self.output_tokens as f64 / (self.decode_ms / 1000.0)
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ConcurrentStats {
    concurrency: u32,
    total_requests: u32,
    wall_ms: f64,
    total_output_tokens: u32,
    tokens_per_sec_aggregate: f64,
    ttft_p50: f64,
    ttft_p95: f64,
    decode_p50: f64,
    decode_p95: f64,
    per_request_ttft_ms: Vec<f64>,
    per_request_decode_ms: Vec<f64>,
    per_request_tokens: Vec<u32>,
}

async fn run_one(
    runner: &mut VllmRunner,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<RunStats> {
    let request_id = format!("bench-{}", random_id());
    let started = Instant::now();
    let handle = runner
        .execute(make_batch(&request_id, model, prompt, max_tokens))
        .await
        .with_context(|| format!("execute({prompt:?})"))?;

    let mut stream = handle.into_stream();
    let mut total_text = String::new();
    let mut output_tokens: u32 = 0;
    let mut ttft = None;
    // Generous deadline for cold-start + Triton-attention JIT +
    // CPU-offload models. Gemma 4 E4B-it on a 16 GB card with
    // `cpu_offload_gb=4` can take 5–10 min for a multi-token reply
    // because every forward pass shuffles ~4 GB GPU↔CPU.
    let deadline = started + Duration::from_secs(900);

    while let Some(chunk_result) = stream.next().await {
        if Instant::now() >= deadline {
            return Err(anyhow!("run exceeded 900s — engine appears hung"));
        }
        let chunk = chunk_result.with_context(|| "stream chunk error")?;
        if !chunk.text_delta.is_empty() && ttft.is_none() {
            ttft = Some(started.elapsed());
        }
        total_text.push_str(&chunk.text_delta);
        if let Some(usage) = chunk.usage.as_ref() {
            output_tokens = usage.output_tokens;
        }
        if chunk.finish_reason.is_some() {
            break;
        }
    }

    let total = started.elapsed();
    let ttft = ttft.unwrap_or(total);
    let decode = total.saturating_sub(ttft);
    Ok(RunStats {
        request_id,
        ttft_ms: ttft.as_secs_f64() * 1000.0,
        decode_ms: decode.as_secs_f64() * 1000.0,
        total_ms: total.as_secs_f64() * 1000.0,
        output_tokens,
        text: total_text,
    })
}

async fn run_concurrent(
    runner: &mut VllmRunner,
    model: &str,
    concurrency: u32,
    total: u32,
    max_tokens: u32,
    prompt: &str,
) -> Result<ConcurrentStats> {
    let started = Instant::now();
    let mut next: u32 = 0;
    let mut in_flight: Vec<tokio::task::JoinHandle<Result<RunStats>>> = Vec::new();
    let mut completed: Vec<RunStats> = Vec::new();

    while completed.len() < total as usize {
        // Top up to `concurrency` in-flight.
        while in_flight.len() < concurrency as usize && next < total {
            // We can't move `runner` into multiple tasks, so we
            // serialize the `execute` calls but let the resulting
            // streams run concurrently.
            let request_id = format!("bench-{}-{}", concurrency, next);
            let batch = make_batch(&request_id, model, prompt, max_tokens);
            let handle = runner.execute(batch).await?;
            let req_started = started;
            let task: tokio::task::JoinHandle<Result<RunStats>> =
                tokio::spawn(async move { drain_stream(request_id, handle, req_started).await });
            in_flight.push(task);
            next += 1;
        }
        // Await any one to complete.
        if let Some(idx) = first_ready(&in_flight).await {
            let task = in_flight.swap_remove(idx);
            match task.await {
                Ok(Ok(stats)) => completed.push(stats),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(anyhow!("task join: {e}")),
            }
        } else {
            // No tasks ready and no more to dispatch — break to avoid
            // tight-loop spin.
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    let wall = started.elapsed();
    let total_output_tokens: u32 = completed.iter().map(|s| s.output_tokens).sum();
    let mut ttfts: Vec<f64> = completed.iter().map(|s| s.ttft_ms).collect();
    let mut decodes: Vec<f64> = completed.iter().map(|s| s.decode_ms).collect();
    let tokens: Vec<u32> = completed.iter().map(|s| s.output_tokens).collect();
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    decodes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok(ConcurrentStats {
        concurrency,
        total_requests: total,
        wall_ms: wall.as_secs_f64() * 1000.0,
        total_output_tokens,
        tokens_per_sec_aggregate: total_output_tokens as f64 / wall.as_secs_f64().max(1e-6),
        ttft_p50: percentile(&ttfts, 50.0),
        ttft_p95: percentile(&ttfts, 95.0),
        decode_p50: percentile(&decodes, 50.0),
        decode_p95: percentile(&decodes, 95.0),
        per_request_ttft_ms: ttfts,
        per_request_decode_ms: decodes,
        per_request_tokens: tokens,
    })
}

async fn drain_stream(
    request_id: String,
    handle: atomr_infer_core::runner::RunHandle,
    started: Instant,
) -> Result<RunStats> {
    let req_started = Instant::now();
    let mut stream = handle.into_stream();
    let mut total_text = String::new();
    let mut output_tokens: u32 = 0;
    let mut ttft = None;
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.with_context(|| "stream chunk error")?;
        if !chunk.text_delta.is_empty() && ttft.is_none() {
            ttft = Some(req_started.elapsed());
        }
        total_text.push_str(&chunk.text_delta);
        if let Some(u) = chunk.usage.as_ref() {
            output_tokens = u.output_tokens;
        }
        if chunk.finish_reason.is_some() {
            break;
        }
    }
    let total = req_started.elapsed();
    let ttft = ttft.unwrap_or(total);
    let decode = total.saturating_sub(ttft);
    let _ = started;
    Ok(RunStats {
        request_id,
        ttft_ms: ttft.as_secs_f64() * 1000.0,
        decode_ms: decode.as_secs_f64() * 1000.0,
        total_ms: total.as_secs_f64() * 1000.0,
        output_tokens,
        text: total_text,
    })
}

async fn first_ready(handles: &[tokio::task::JoinHandle<Result<RunStats>>]) -> Option<usize> {
    if handles.is_empty() {
        return None;
    }
    for (i, h) in handles.iter().enumerate() {
        if h.is_finished() {
            return Some(i);
        }
    }
    None
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn base_config(model: &str) -> VllmConfig {
    // Conservative defaults sized for a single 16 GB consumer GPU
    // running Gemma 4 E4B (~6 GB weights + vision tower + KV cache).
    // vLLM 0.20 captures CUDA graphs at 19 batch sizes by default,
    // which OOMs even with `gpu_memory_utilization=0.5` — hence
    // `enforce_eager=true` here. The `sweep cuda-graphs` subcommand
    // explicitly exercises eager-vs-graphs so this default doesn't
    // hide that comparison.
    // Gemma 4 E4B-it weights are 10.2 GiB; the multimodal profile
    // pass needs another ~5 GB on top, which OOMs a 16 GB card. We
    // can't use `limit_mm_per_prompt` because vLLM 0.20's text-only
    // path for Gemma 4 is buggy (per-layer embeddings share the
    // mm pipeline). The reliable fix on small GPUs is to offload
    // weights to CPU RAM via vLLM's `cpu_offload_gb`. 4 GB offload
    // brings peak GPU usage to ~12 GB and lets E4B fit. Operators
    // on ≥24 GB hardware can drop this and re-run for full GPU-side
    // numbers.
    VllmConfig {
        model: model.into(),
        tensor_parallel_size: 1,
        dtype: "auto".into(),
        gpu_memory_utilization: Some(0.85),
        max_model_len: Some(2048),
        hf_cache_dir: None,
        enforce_eager: Some(true),
        enable_prefix_caching: None,
        enable_chunked_prefill: None,
        max_num_seqs: Some(16),
        block_size: None,
        quantization: None,
        limit_mm_per_prompt: None,
        cpu_offload_gb: Some(4),
    }
}

fn make_batch(id: &str, model: &str, prompt: &str, max_tokens: u32) -> ExecuteBatch {
    ExecuteBatch {
        request_id: id.into(),
        model: model.into(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(prompt.into()),
        }],
        sampling: SamplingParams {
            temperature: Some(0.0),
            max_tokens: Some(max_tokens),
            ..Default::default()
        },
        stream: true,
        estimated_tokens: max_tokens.max(8),
    }
}

fn short_prompt() -> String {
    "Reply with the word OK.".into()
}

fn medium_prompt() -> String {
    "Summarise the difference between an actor system and a thread pool in three sentences.".into()
}

fn long_prompt() -> String {
    // ~300 tokens. Used to surface TTFT scaling with prompt length.
    let para = "The actor model is a mathematical model of concurrent computation that treats `actors` as the universal primitives of concurrent computation. \
                In response to a message it receives, an actor can: make local decisions, create more actors, send more messages, and determine how to respond to the next message received. \
                Actors may modify their own private state, but can only affect each other indirectly through messaging (removing the need for lock-based synchronisation). ";
    let mut out = String::new();
    for _ in 0..6 {
        out.push_str(para);
    }
    out.push_str("\nGiven this background, briefly summarise three production trade-offs of the actor model versus a thread pool. Keep it under 150 words.");
    out
}

fn random_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}", now & 0xFFFFFFFF)
}

fn pass_probe(cfg: &VllmConfig, label: &str) -> bool {
    let min_vram = match cfg.model.as_str() {
        "google/gemma-4-E2B" | "google/gemma-4-E2B-it" => 2.5,
        _ => 4.5,
    };
    let min_disk = if cfg.model.contains("E4B") { 7.0 } else { 4.0 };
    match probe(&cfg.model, min_vram, min_disk, None) {
        ProbeResult::Ready { vram_free_gb, .. } => {
            tracing::info!(label, vram_free_gb, "probe ok");
            true
        }
        ProbeResult::Skipped { reason, hint } => {
            eprintln!("[{label}] skipped: {reason}\n  hint: {hint}");
            false
        }
        ProbeResult::Error(e) => {
            eprintln!("[{label}] probe error: {e}");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// JSONL archive writer
// ---------------------------------------------------------------------------

struct ResultWriter {
    format: OutputFormat,
    archive: Option<std::fs::File>,
    archive_path: Option<PathBuf>,
}

impl ResultWriter {
    fn new(cli: &Cli) -> Result<Self> {
        let archive = if cli.no_archive {
            None
        } else {
            let dir = PathBuf::from("target").join("gemma-bench");
            std::fs::create_dir_all(&dir)?;
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let subcmd = match &cli.cmd {
                Cmd::Smoke => "smoke",
                Cmd::Latency => "latency",
                Cmd::Throughput { .. } => "throughput",
                Cmd::Sweep { .. } => "sweep",
                Cmd::Experiments => "experiments",
                Cmd::Compare => "compare",
            };
            let path = dir.join(format!("{subcmd}-{ts}.jsonl"));
            let file = std::fs::File::create(&path)?;
            Some((file, path))
        };
        Ok(Self {
            format: cli.format,
            archive: archive.as_ref().map(|(f, _)| f.try_clone().expect("clone fd")),
            archive_path: archive.map(|(_, p)| p),
        })
    }

    fn emit<T: Serialize>(&self, label: &str, value: &T) {
        self.emit_with(label, "", value);
    }

    fn emit_with<T: Serialize>(&self, label: &str, sub_label: &str, value: &T) {
        let line = serde_json::json!({
            "label": label,
            "sub_label": sub_label,
            "value": value,
        });
        let s = line.to_string();
        if matches!(self.format, OutputFormat::Jsonl) {
            println!("{s}");
        }
        if let Some(mut f) = self.archive.as_ref().and_then(|f| f.try_clone().ok()) {
            let _ = writeln!(f, "{s}");
        }
    }

    fn finish(&self) {
        if let Some(p) = &self.archive_path {
            eprintln!("results archived to {}", p.display());
        }
    }
}
