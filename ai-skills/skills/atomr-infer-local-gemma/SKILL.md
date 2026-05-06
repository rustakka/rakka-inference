---
name: atomr-infer-local-gemma
description: Use when standing up zero-config local Gemma 4 on a workstation with a working GPU + Python + vLLM, picking a Gemma 4 variant (E2B / E4B, base vs `-it`), or wiring the `gemma-default` feature. Triggers on enabling `--features gemma-default`, setting `ATOMR_INFER_GEMMA_*` env vars, asking "how do I run Gemma locally", or seeing a `gemma-local` deployment auto-appear.
---

# Zero-config local Gemma 4

The `gemma-default` feature on the `atomr-infer` rollup auto-provisions
a `gemma-local` deployment backed by `google/gemma-4-E4B-it` through
the vLLM runner — but only when an env probe passes (NVIDIA GPU
visible, Python 3.10+ on `PATH`, `vllm` importable, `HF_TOKEN` set,
enough free VRAM and disk). When the probe fails the deployment is
*skipped*, not errored, so the same binary works on laptops, CI, and
production hosts without conditional config.

## Variant matrix

| Model | VRAM | Use for |
|---|---|---|
| `google/gemma-4-E2B` | ~2 GB | Base weights, fine-tuners |
| `google/gemma-4-E2B-it` | ~2 GB | Low-VRAM chat / agents |
| `google/gemma-4-E4B` | ~4 GB | Base weights, fine-tuners |
| `google/gemma-4-E4B-it` | ~4 GB | **Default.** Balanced chat / agents |

`-it` variants are instruction-tuned for chat/agent loops; the bare
variants are for fine-tuners that want clean base weights. Switch with
`ATOMR_INFER_GEMMA_MODEL=...`, never by editing TOML — the whole point
of `gemma-default` is that it's TOML-free.

## Three-line setup

```sh
pip install 'vllm>=0.6.4' timm
hf auth login   # then accept ToS at https://huggingface.co/google/gemma-4-E4B-it
cargo run -p atomr-infer-cli --features gemma-default -- serve --config demo.toml
```

The `demo.toml` doesn't need to mention Gemma at all — the auto-deployment
is injected before the file is parsed if the probe passes.

## Probe outcomes

| Skipped reason | What it means | Fix |
|---|---|---|
| `no GPU` | No NVIDIA GPU visible (`nvidia-smi` failed or absent) | Install drivers, or accept the skip on CPU-only hosts |
| `vLLM not installed` | `python3 -c 'import vllm'` failed | `pip install 'vllm>=0.6.4' timm` in the same Python on `PATH` |
| `no HF token` | `HF_TOKEN` env var unset and no `~/.cache/huggingface/token` | `hf auth login` or export `HF_TOKEN=hf_...` |
| `ToS not accepted` | HF returns 403 fetching the model card | Open the model URL while logged in and accept the gated-model ToS |
| `insufficient VRAM` | Free VRAM under variant requirement | Free the GPU, pick a smaller variant via `ATOMR_INFER_GEMMA_MODEL`, or accept the skip |
| `insufficient disk` | `$HF_HOME` partition has less than weights-size + headroom | Point `HF_HOME` at a larger volume |

A skip emits one INFO-level log line with the reason; nothing else
fails. Production hosts that *want* Gemma should treat any skip as
fatal at deploy time and gate it in their own startup checks.

## Env vars

| Var | Default | Effect |
|---|---|---|
| `ATOMR_INFER_GEMMA_AUTO` | `1` | Set `0` to disable auto-provisioning entirely |
| `ATOMR_INFER_GEMMA_MODEL` | `google/gemma-4-E4B-it` | Override the variant |
| `ATOMR_INFER_GEMMA_DEPLOYMENT` | `gemma-local` | Override the deployment name (the `model=` value clients send) |
| `ATOMR_INFER_GEMMA_GPU_UTIL` | `0.9` | Forwarded to vLLM `gpu_memory_utilization` |
| `ATOMR_INFER_GEMMA_MAX_LEN` | model default | Forwarded to vLLM `max_model_len` |
| `HF_TOKEN` | — | Required; checked by the probe |
| `HF_HOME` | `~/.cache/huggingface` | Root cache dir |
| `HF_HUB_CACHE` | `$HF_HOME/hub` | Where weights actually land |

## Cache layout

The runner respects `$HF_HOME` / `$HF_HUB_CACHE` verbatim and never
writes anywhere else. Multiple `atomr-infer` instances on the same
workstation share one on-disk copy of the model — useful for
side-by-side dev gateways, agent harnesses, and notebooks. Default is
`~/.cache/huggingface`, matching every other Hugging Face tool.

## Common mistakes

- **Running with `--features gemma-default` on a CPU-only box.** The
  probe skips, the gateway boots without `gemma-local`, and the
  feature flag costs you nothing at runtime. This is by design — not
  a bug, not a problem.
- **Enabling `gemma-default` for production.** Don't. A first-boot HF
  download on a prod host is a surprise that times out deploys; pin
  the model in TOML and pre-warm the cache instead.
- **Editing TOML to switch variants.** Use
  `ATOMR_INFER_GEMMA_MODEL=google/gemma-4-E2B-it` instead. If you're
  reaching for TOML to configure Gemma, you've outgrown
  `gemma-default` and should drop a hand-written `[[deployment]]`
  block with `runtime = "vllm"`.
- **Forgetting the gated-model ToS step.** `hf auth login`
  alone isn't enough; the Gemma family is gated and needs a one-time
  click-through per account on huggingface.co.

## When to reach beyond this skill

| You need to… | Reach for skill… |
|---|---|
| Configure vLLM by hand (TP, dtype, KV-cache) | `atomr-infer-runtimes` |
| Run Gemma without Python (mistralrs / candle path) | `atomr-infer-runtimes` |
| Pin the model in production TOML | `atomr-infer-quickstart` |
| Diagnose a skip that shouldn't be a skip | `atomr-infer-troubleshooting` |

## Canonical references

- [`docs/local-gemma.md`](https://github.com/rustakka/atomr-infer/blob/main/docs/local-gemma.md) — full feature doc
- [`crates/inference-runtime-vllm/src/defaults.rs`](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-vllm/src/defaults.rs) — auto-deployment construction
- [`crates/inference-runtime-vllm/src/probe.rs`](https://github.com/rustakka/atomr-infer/blob/main/crates/inference-runtime-vllm/src/probe.rs) — env probe and skip reasons
