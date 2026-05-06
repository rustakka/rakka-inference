# `gemma_bench`

Local-only performance harness for Gemma 4 through the
`inference-runtime-vllm` PyO3 bridge. **Not run in CI** — every
subcommand probes the host first and exits with a clean message
when the GPU / vLLM / HF token isn't ready.

## Setup

The same prereqs as `--features gemma-default`:

```sh
pip install 'vllm>=0.6.4' timm
hf auth login    # accept the ToS at https://huggingface.co/google/gemma-4-E4B-it
```

The bench binary requires `gemma-default` to compile:

```sh
cargo build -p gemma_bench --release --features gemma-default
```

`--release` is intentional — debug builds add ~10 % overhead in the
hot loop and skew tokens/sec measurements.

## Subcommands

```sh
# Verify the bridge boots and emits one chunk.
cargo run -p gemma_bench --release --features gemma-default -- smoke

# TTFT + decode rate at three prompt lengths.
cargo run -p gemma_bench --release --features gemma-default -- latency

# Aggregate tokens/sec across N concurrent requests.
cargo run -p gemma_bench --release --features gemma-default -- \
    throughput --concurrency 4 --total 16

# Sweep one perf knob.
cargo run -p gemma_bench --release --features gemma-default -- sweep gpu-util
cargo run -p gemma_bench --release --features gemma-default -- sweep cuda-graphs
cargo run -p gemma_bench --release --features gemma-default -- sweep dtype
cargo run -p gemma_bench --release --features gemma-default -- sweep prefix-cache
cargo run -p gemma_bench --release --features gemma-default -- sweep chunked-prefill
cargo run -p gemma_bench --release --features gemma-default -- sweep concurrency
cargo run -p gemma_bench --release --features gemma-default -- sweep block-size
cargo run -p gemma_bench --release --features gemma-default -- sweep max-num-seqs

# Curated battery of seven configs against vLLM defaults.
cargo run -p gemma_bench --release --features gemma-default -- experiments

# Head-to-head: E4B-it vs E2B-it on the same prompt.
cargo run -p gemma_bench --release --features gemma-default -- compare
```

Every subcommand accepts:

| Flag                  | Default                       | Purpose                                                   |
|-----------------------|-------------------------------|-----------------------------------------------------------|
| `--model`             | `google/gemma-4-E4B-it`     | Override the variant. Validated against the allow-list.  |
| `--allow-any-model`   | off                           | Bypass the allow-list (probing a new release).            |
| `--max-tokens`        | 64                            | Per-request token budget.                                 |
| `--quick`             | off                           | Halve every iteration count for fast dev iteration.       |
| `--format human|jsonl`| human                         | `jsonl` for piping into `jq`.                             |
| `--no-archive`        | off                           | Skip writing `target/gemma-bench/<subcmd>-<ts>.jsonl`.    |

## Output

Two paths:

1. **Stdout.** Human-readable summary by default; `--format jsonl`
   prints one JSON object per measurement.
2. **Archive.** Every run also writes
   `target/gemma-bench/<subcmd>-<unix-ts>.jsonl` so historical
   results are inspectable. Skip with `--no-archive`.

A typical sweep prints a column-aligned summary like:

```
gpu_util                  ttft_p50   decode_p50      agg_tok/s
gpu_util=0.3                  342.1        511.4          39.78
gpu_util=0.5                  301.2        467.8          43.12
gpu_util=0.7                  289.7        452.1          44.31
gpu_util=0.9                  281.5        449.9          44.55
```

## Experiments worth running

The `experiments` subcommand runs seven configs against vLLM defaults
on the same prompt, two concurrent requests, eight total. Quick
hypothesis check on a fresh GPU:

| Knob | Hypothesis | Expected outcome |
|---|---|---|
| `cuda-graphs` (off → on) | Big speedup on small models | 1.5–2× decode tok/s with graphs on |
| `prefix_cache` | Helps when many requests share a system prompt | 5–20 % TTFT improvement on shared-prefix workloads only |
| `chunked_prefill` | Helps TTFT under concurrent load | TTFT p95 improves; aggregate tok/s roughly flat |
| `gpu_memory_utilization` 0.4 → 0.85 | Larger KV cache ⇒ better continuous batching | Modest agg tok/s improvement, big decrease at very high util if dev tools steal VRAM |
| `block_size` 16 → 32 | Larger blocks = better throughput | Typically a few % improvement |
| `max_num_seqs` 16 → 256 | More concurrent slots | Aggregate tok/s up, per-request latency unchanged at low concurrency |
| `dtype` auto → bf16 | bf16 has better numerical range than fp16 | Throughput parity; fewer NaN edge cases |

Things to try **after** the harness baseline:

- **Quantization.** Add `--quantization awq` (or other supported
  schemes) on a quantized checkpoint; expect ~2× decode tok/s on
  small Gemma 4 at minor quality cost. The `quantization` knob is
  already wired through `VllmConfig`.
- **Speculative decoding.** vLLM supports a draft-model-based
  speculative decoder. Not yet wired — see `crates/inference-pipeline`
  for the higher-level pattern.
- **Multi-GPU `tensor_parallel_size`.** Only matters at E4B; E2B fits
  on one card. Add a flag if you have a multi-GPU box.
- **Prefix-cache with a fat system prompt.** The current `medium_prompt`
  has no shared prefix; to actually exercise prefix caching, hardcode
  a 2KB system message and run the same user prompt N times.

## Repeatability

vLLM warms a CUDA-graphs cache on first launch. Cold and warm runs
will differ; for stable measurements:

- Use `--release` builds.
- Drop the first run's measurement (cold). The harness already excludes
  the first chunk's TTFT from sustained tokens/sec calculations.
- Pin the GPU clock if your driver supports it
  (`nvidia-smi -lgc <freq>,<freq>`).
- Close other GPU consumers (browsers can hold ~500 MB VRAM).

## Running the GPU smoke tests

For pass/fail rather than measurement, run the `#[ignore]`'d
integration tests:

```sh
cargo test -p atomr-infer-runtime-vllm --release \
    --features gemma-default -- --ignored --test-threads=1
```

`--test-threads=1` is required because each test owns the GPU.
