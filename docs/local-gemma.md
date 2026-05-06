# Zero-config local Gemma 4

The `gemma-default` feature on the rollup auto-provisions a
`gemma-local` deployment when the host has a workable GPU, Python
3.10+, vLLM installed, and a HuggingFace token with Gemma access.
The intent is a **fast first run** for developers who want a local
LLM without writing a project-file deployment by hand.

## Three-line setup

```sh
pip install 'vllm>=0.6.4' timm
hf auth login    # then accept the ToS at https://huggingface.co/google/gemma-4-E4B-it
cargo run -p atomr-infer-cli --features gemma-default -- serve --config demo.toml
```

Why `timm`: every Gemma 4 variant is multimodal (text + vision +
audio). vLLM loads the vision tower through `transformers`'
`TimmWrapperModel`, which imports `timm` lazily. Pinning it in your
install command up front avoids a 4 GB-into-failure on first run.

On boot, `atomr-infer serve` runs the env probe; if every gate
passes, it registers a deployment named `gemma-local` backed by
`google/gemma-4-E4B-it`. If any gate fails, the boot logs a single
`info!` line with the reason and a fix-it hint, and the server keeps
running with whatever deployments came from the project file.

## Supported variants

All four Gemma 4 variants are supported. E4B-it is the default
because it gives the best chat quality on the 4GB-VRAM workstations
this feature targets.

| Repo                          | Variant            | Approx. VRAM (fp16) | Approx. disk | When to pick                                            |
|-------------------------------|--------------------|---------------------|--------------|---------------------------------------------------------|
| `google/gemma-4-E2B`         | base               | ~2 GB               | ~4 GB        | Continued pretraining / domain-specific fine-tuning     |
| `google/gemma-4-E2B-it`      | instruction-tuned  | ~2 GB               | ~4 GB        | Chat / tools on hardware with < 4 GB free VRAM          |
| `google/gemma-4-E4B`         | base               | ~4 GB               | ~7 GB        | Continued pretraining where a stronger base helps       |
| `google/gemma-4-E4B-it`      | instruction-tuned  | ~4 GB               | ~7 GB        | **Default.** Chat / tools on a typical dev workstation  |

The bare `E2B` / `E4B` variants are pretraining checkpoints intended
for fine-tuners — they're not chat-template-aware. Use the `-it`
variants for actual agent / chat work.

To switch variants without editing the project file:

```sh
ATOMR_INFER_GEMMA_MODEL=google/gemma-4-E2B-it \
  cargo run -p atomr-infer-cli --features gemma-default -- serve --config demo.toml
```

## Probe outcomes

The probe runs in this order. The first failure short-circuits with
a `Skipped { reason, hint }` outcome.

| Step              | Failure → message                                                          | Fix                                                                                         |
|-------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| GPU detected      | `no CUDA GPU detected (cudarc: ...)`                                       | Set `ATOMR_INFER_GEMMA_AUTO=skip-quietly` to suppress, or use a remote backend.             |
| Sufficient VRAM   | `free VRAM X.X GB < Y.Y GB needed for <model>; try ATOMR_INFER_GEMMA_MODEL=...-E2B-it` | Switch to the smaller E2B variant, or free VRAM (close other GPU consumers).      |
| Python 3.10+      | `python3 not on PATH` / `Python 3.X.Y on PATH; vLLM requires 3.10+`        | Install Python 3.10 or newer in your active environment.                                    |
| vLLM importable   | `vLLM not importable in active python3`                                    | `pip install 'vllm>=0.6.4'` in the same venv.                                               |
| HF token present  | `no HuggingFace token found`                                               | `hf auth login`, then accept the Gemma ToS at the model's HF page.                  |
| Disk space        | `insufficient disk for <model>: X.X GB free at <path>`                     | Free up space, or set `HF_HUB_CACHE` to a different mountpoint.                             |

The probe does **not** verify HF access (it doesn't make a network
call). If the token doesn't have Gemma access, the probe passes but
vLLM's first model load surfaces a 403 from HF with a clear message.

## Configuration

### Environment variables

| Variable                          | Effect                                                                                                |
|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| `ATOMR_INFER_GEMMA_AUTO`          | `0` / `false` / `no` / `off` / `skip` / `skip-quietly` ⇒ disable auto-provision. Anything else ⇒ on.  |
| `ATOMR_INFER_GEMMA_MODEL`         | Override the model id. Validated against the four supported variants — typo fails fast.               |
| `ATOMR_INFER_GEMMA_DEPLOYMENT`    | Override the deployment name (default `gemma-local`).                                                 |
| `ATOMR_INFER_GEMMA_GPU_UTIL`      | Float, fraction of GPU memory vLLM pre-allocates. Default `0.5` to leave room for dev tools.          |
| `ATOMR_INFER_GEMMA_MAX_LEN`       | Maximum sequence length the engine schedules. Defaults to vLLM's choice from the model config.        |
| `HF_HOME`                         | Root of the HuggingFace cache hierarchy. Defaults to `$XDG_CACHE_HOME/huggingface` or `~/.cache/huggingface`. |
| `HF_HUB_CACHE`                    | Where downloaded models live. Defaults to `$HF_HOME/hub`.                                             |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | HF auth. Falls back to `$HF_HOME/token` (the file `hf auth login` writes).                |

### Project-file overrides

Same fields, surfaced as a `[defaults.gemma]` block in the project
file, take precedence over env vars (planned for v0.5; today the
env vars are the only override path).

## Cache layout

The probe and engine respect the standard HF cache precedence so a
workstation that already runs `huggingface-cli` "just works" without
re-downloading models:

```
HF_HUB_CACHE                      explicit override
HF_HOME/hub                       if HF_HOME is set
XDG_CACHE_HOME/huggingface/hub    if XDG_CACHE_HOME is set
~/.cache/huggingface/hub          default
```

Multi-instance unification falls out of pointing every atomr-infer
process at the same `$HF_HOME`. Pre-downloaded models from
`huggingface-cli download` or other tools (transformers, llama.cpp,
etc.) are reused without re-downloading.

## Disabling

Three ways:

```sh
# Per-invocation
ATOMR_INFER_GEMMA_AUTO=skip-quietly cargo run -p atomr-infer-cli --features gemma-default -- serve ...

# Build without the feature (cleanest)
cargo run -p atomr-infer-cli -- serve --config demo.toml

# Build with the feature but never trigger
ATOMR_INFER_GEMMA_AUTO=0 cargo run -p atomr-infer-cli --features gemma-default -- serve ...
```

The feature is **deliberately not in `default-prod`** — production
builds shouldn't surprise-download a multi-GB model on first boot.

## Why vLLM

vLLM is the canonical local-LLM backend in atomr-infer's architecture
and gives the best throughput on Gemma 4 once warmed up. The
trade-off is cold-start time: vLLM takes ~30–60 seconds to load the
model and warm the KV cache. For genuinely fast dev iteration, the
[mistral.rs runner](../crates/inference-runtime-mistralrs/) ships
with similar coverage and cold-starts in seconds — set up a deployment
manually if you need that path.

## Local performance experiments

The `examples/gemma_bench/` binary is a local-only harness for
TTFT / tokens-per-second measurements and perf experiments
(CUDA graphs on/off, prefix caching, chunked prefill,
`gpu_memory_utilization` sweeps, etc.). It runs **outside CI**;
every subcommand probes the host first and exits cleanly when no
GPU is present.

```sh
cargo run -p gemma_bench --release --features gemma-default -- experiments
```

See [`examples/gemma_bench/README.md`](../examples/gemma_bench/README.md)
for the full subcommand list and experiment hypotheses.

GPU-required pass/fail tests live alongside as `#[ignore]`'d
integration tests in
[`crates/inference-runtime-vllm/tests/gpu_smoke.rs`](../crates/inference-runtime-vllm/tests/gpu_smoke.rs);
run via `cargo test -p atomr-infer-runtime-vllm --features gemma-default -- --ignored`.

## See also

- [`crates/inference-runtime-vllm/src/defaults.rs`](../crates/inference-runtime-vllm/src/defaults.rs) — `GemmaDefaults`, `provision_if_ready`, `SUPPORTED_VARIANTS`
- [`crates/inference-runtime-vllm/src/probe.rs`](../crates/inference-runtime-vllm/src/probe.rs) — env probe sequence
- [`crates/inference-runtime-vllm/src/hf_cache.rs`](../crates/inference-runtime-vllm/src/hf_cache.rs) — cache + token resolution
- [`examples/gemma_bench/`](../examples/gemma_bench/) — local perf harness
- [`docs/feature-matrix.md`](feature-matrix.md) — every feature flag in atomr-infer
- [`docs/architecture.md`](architecture.md) — the actor topology this deployment plugs into
