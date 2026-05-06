# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.6.3] — 2026-05-06

### Added — full workspace publishes to crates.io
- Upstream `atomr` family is now at **0.3.1** and `atomr-accel`
  family at **0.3.3** on crates.io, which means every inference-*
  crate's dep graph resolves cleanly from the registry. The publish
  allowlist is now empty (= publish all 18 crates in dep order).
- `cargo xtask release-checklist` reports 18 / 18 publishable, 0
  gated. Sibling-workspace path deps in `Cargo.toml` remain as
  reference-only for local development; they're stripped at
  publish time.
- `RELEASING.md` documents the new state and the version-pin
  compatibility (we pin `atomr-* = "0.3.1"` and `atomr-accel-* =
  "0.3.0"`; both accept the published 0.3.x lines).

## [0.6.2] — 2026-05-06

### Fixed — crates.io publish allowlist now reflects transitive deps
- `release.yml`'s `DEFAULT_PUBLISH_ALLOWLIST` was overstating what
  could ship. The previous list (7 crates) included
  `atomr-infer-runtime` and the four remote runners, but those
  transitively declare `atomr-*` deps that are not yet on
  crates.io — so `cargo publish` fails on them. The v0.6.1 publish
  job hit this: `atomr-infer-core` shipped, then
  `atomr-infer-runtime` failed with
  `failed to select a version for the requirement
   atomr-accel = "^0.3.0"; candidate versions found: 0.1.0`.
- Trimmed the default allowlist to **just `atomr-infer-core`** —
  the only crate whose entire `[dependencies]` section resolves
  from crates.io alone. Sibling-workspace path deps to `atomr` and
  `atomr-accel` are reference-only for planning and local
  development; they don't change what crates.io accepts.
- `cargo xtask release-checklist` now accounts for transitive
  upstream-`atomr-*` deps and lists only `atomr-infer-core` as
  publishable today; the other 17 crates are gated with a
  per-crate reason.
- `RELEASING.md` updated to match. Expand the allowlist as upstream
  ships 0.3.x crates to crates.io.

## [0.6.1] — 2026-05-06

### Fixed — retry publish that never fired
- The version-bump bot tagged v0.5.0 and v0.6.0 using `GITHUB_TOKEN`,
  which (per GitHub's downstream-workflow security default) does not
  trigger workflows that fire on tag pushes. The `release.yml`
  workflow's publish jobs are gated on
  `github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')`,
  so neither tag actually shipped to crates.io / PyPI / GitHub
  Releases. v0.6.1 is tagged and pushed from a developer machine so
  the publish pipeline actually fires. No source changes vs v0.6.0;
  this is purely a CI-infrastructure retry.

## [0.6.0] — 2026-05-05

### Added — native aarch64-Linux wheels
- PyPI now ships pre-built wheels for `aarch64-unknown-linux-gnu`
  and `aarch64-unknown-linux-musl`, built natively on GitHub-hosted
  ARM runners (`ubuntu-22.04-arm`). Mirrors the upstream atomr
  v0.3.1 pattern. Closes the gap where ARM Linux users had to
  install from sdist; native build avoids the `ring`/`aws-lc-rs`
  cross-compile blocker that previously forced the skip.

  PyPI wheel coverage as of this release:

  | Platform              | Wheel  |
  |-----------------------|--------|
  | linux-gnu x86_64      | ✓      |
  | linux-musl x86_64     | ✓      |
  | linux-gnu aarch64     | ✓ new  |
  | linux-musl aarch64    | ✓ new  |
  | macOS universal2      | ✓      |
  | windows-msvc x86_64   | ✓      |

## [0.5.0] — 2026-05-05

### Added — zero-config local Gemma 4
- `gemma-default` feature on the rollup auto-provisions a
  `gemma-local` deployment through the native PyO3 vLLM runner.
  Default model is `google/gemma-4-E4B-it`; all four Gemma 4
  variants (`E2B`, `E2B-it`, `E4B`, `E4B-it`) are validated by an
  allow-list and reachable via `ATOMR_INFER_GEMMA_MODEL` or a
  `[defaults.gemma]` block. The env probe handles GPU / Python /
  vLLM / HF-token gracefully — missing prereq logs a one-line `info!`
  tip and continues without the deployment; insufficient VRAM hints
  at the matching smaller variant. Cache respects `$HF_HOME` /
  `$HF_HUB_CACHE` so multi-instance deployments share one on-disk
  model.
- New PyO3 `VllmEngine` wrapper (`crates/inference-runtime-vllm/src/engine.rs`)
  bridges vLLM's V1 `AsyncLLMEngine` behind the `ModelRunner` trait.
  Token streaming via `tokio::mpsc`; consumer-drop triggers
  `engine.abort(request_id)`; lazy initialisation so `VllmRunner::new`
  stays cheap.
- New `crates/inference-runtime-vllm/src/{hf_cache,probe,defaults}.rs`
  modules — pure Rust, no PyO3 — for cache resolution, env probe, and
  the `provision_if_ready` adapter that registers the deployment with
  the running `DeploymentManagerActor`.
- New env vars: `ATOMR_INFER_GEMMA_AUTO`,
  `ATOMR_INFER_GEMMA_MODEL`, `ATOMR_INFER_GEMMA_DEPLOYMENT`,
  `ATOMR_INFER_GEMMA_GPU_UTIL`, `ATOMR_INFER_GEMMA_MAX_LEN`. Documented
  in `docs/local-gemma.md`.
- New ai-skill: `atomr-infer-local-gemma`.
- The feature is **deliberately not in `default-prod`** — production
  builds shouldn't surprise-download a multi-GB model on first boot.
  Operators opt in via `--features gemma-default` on the CLI.

### Added — local perf harness (`examples/gemma_bench/`)
- New binary `gemma_bench` (workspace member, `publish = false`,
  required-features `gemma-default`) for TTFT / tokens-per-second
  measurements and perf experiments. Subcommands: `smoke`,
  `latency`, `throughput`, `sweep <knob>` (gpu-util, dtype,
  cuda-graphs, prefix-cache, chunked-prefill, concurrency,
  block-size, max-num-seqs), `experiments`, `compare`.
- New `#[ignore]`'d integration tests in
  `crates/inference-runtime-vllm/tests/gpu_smoke.rs` for GPU pass/
  fail. Run with
  `cargo test -p atomr-infer-runtime-vllm --features gemma-default -- --ignored --test-threads=1`.
- `VllmConfig` extended with the perf knobs the harness sweeps:
  `enforce_eager`, `enable_prefix_caching`, `enable_chunked_prefill`,
  `max_num_seqs`, `block_size`, `quantization`. All forwarded through
  `engine.rs` to `AsyncEngineArgs`.
- `engine::generate` now renders chat through the model's tokenizer
  template (`tokenizer.apply_chat_template`) so Gemma's
  `<start_of_turn>` format is used correctly. Falls back to the
  generic `<|role|>` format on older vLLM versions.

### Aligned with upstream atomr 0.3.1 / atomr-accel 0.3.0
- Bumped every `atomr-*` workspace dep from `version = "0.1.0"` to
  `version = "0.3.1"`, and every `atomr-accel*` dep to
  `version = "0.3.0"`. Path-resolution worked locally before; this
  closes the `cargo publish` / `cargo-semver-checks` gap.
- Migrated to the upstream `atomr-accel-cuda` split. The umbrella
  `atomr-accel` no longer ships a `cuda` feature in 0.3 — CUDA lives
  in its own sibling crate now. `inference-runtime/Cargo.toml`,
  `inference/Cargo.toml`, and the candle / cudarc runners were
  updated accordingly. Source-level paths
  (`atomr_accel::cuda::error::*`) were rewritten to
  `atomr_accel_cuda::error::*` in `worker.rs`.
- Added `atomr-accel-tensorrt` (Phase 8 of atomr-accel) as a
  workspace dep, gated behind the `tensorrt` feature.

### Added — TensorRT runner is no longer a stub
- `inference-runtime-tensorrt` now wires the upstream `TrtRuntime` /
  `TrtEngine` / `ExecutionContext` / `ExecutionBindings` types behind
  the `ModelRunner` trait. Engine plans are loaded eagerly at
  construction; the runtime / engine / context are built lazily on
  the first `execute` call so a runner can be instantiated on hosts
  that don't ship libnvinfer.
- New sub-features forwarded straight to upstream:
  `tensorrt-onnx`, `tensorrt-int8`, `tensorrt-fp8`,
  `tensorrt-plugin`, `tensorrt-link`. All are reachable from the
  rollup with the same names.
- `TensorRtRunner::enqueue(ExecutionBindings)` for callers that own
  the tokenisation / device-pointer staging path. The chat-style
  `ModelRunner::execute` returns a typed `InferenceError::Internal`
  pointing at this entry point until an LLM-aware adapter lands.
- New config fields: `precision: TrtPrecision` (Fp32 / Fp16 / Bf16 /
  Int8 / Fp8 / Best — mirrors `atomr_accel_tensorrt::Precision`)
  and `device_id: u32`.
- `TrtError -> InferenceError` mapping for the full upstream
  variant set (NotLinked / Build / Runtime / Execution / Onnx /
  Calibration / Plugin / Refit / NullEngine / InvalidArg).

### Added — Mistral.rs runner is no longer a stub
- `inference-runtime-mistralrs` now wires `mistralrs::TextModelBuilder`
  and `mistralrs::Model` behind the `ModelRunner` trait. Models load
  lazily on the first `execute` call (so HuggingFace downloads happen
  at request time, not at runner-construction time). Tokens stream
  back through a `tokio::mpsc` channel as `TokenChunk`s.
- New config fields: `model_id`, `quant` (ISQ value parsed via
  `mistralrs::parse_isq_value`), `hf_revision`, `force_cpu`,
  `max_num_seqs`.
- Note: mistralrs 0.8 declares MSRV 1.88. The atomr-infer workspace
  MSRV (1.78) only applies to remote-only / default-features builds;
  operators enabling this runner need a toolchain that satisfies
  mistralrs's own MSRV.

### Added — 1.0-readiness hardening
- `#[non_exhaustive]` on every public enum that callers might match
  on: `RuntimeKind`, `TransportKind`, `ProviderKind`, `JitterKind`,
  `Role`, `MessageContent`, `ContentPart`, `FinishReason`,
  `InferenceError`, `WeightSource`, `SessionRebuildCause`. This is
  a deliberate breaking-style hardening pass before 1.0 — downstream
  matches against these enums will need a `_` arm.
- `deny.toml` and a `cargo-deny` CI job covering the four
  cargo-deny checks (advisories / bans / licenses / sources).
- Per-backend `feature-matrix` CI job — twelve backends checked
  individually so a regression in one feature gate doesn't hide
  behind the workspace build.
- `tracing::instrument` decorators on every remote runner's `execute`
  so structured spans carry `request_id` and `model` automatically.

### Changed
- `inference` rollup re-export of the CUDA backend renamed: callers
  now reach the NVIDIA backend at `atomr_infer::accel_cuda::*` (was
  `atomr_infer::accel::cuda::*`). The old `cuda` / `cuda_patterns`
  back-compat aliases (marked for removal in 0.4) were dropped.
- `DeploymentManagerMsg::Apply` carries the full `Deployment` value
  inline; clippy's `large_enum_variant` lint is suppressed with a
  doc-commented justification (boxing would force every caller to
  wrap a short-lived mailbox message).

### Renamed
- `docs/rustakka-inference-architecture-v4.md` →
  `docs/architecture.md`. All doc cross-references and rustdoc links
  follow.

### Removed
- The legacy "rakka" naming has been swept out of every README,
  source comment, environment variable, sample TOML, ai-skills
  bundle, and architecture doc. The `RAKKA_INFERENCE_*` env vars in
  `xtask` and the release pipeline are now `ATOMR_INFER_*`.

## [0.4.0] — 2026-05-05

### Added
- Re-enabled the `atomr-accel` features after the upstream rename:
  the `accel` and `accel-patterns` features on the rollup pull in
  the upstream substrate again, the `local-gpu` feature on
  `atomr-infer-runtime` is wired, and the candle / cudarc runners
  declare optional `atomr-accel` deps. The atomr-accel version pins
  in `Cargo.toml` were left at `0.1.0` in this release; see the
  Unreleased entry for the corrective bump to `0.3.x`.

## [0.3.1] — 2026-05-05

### Fixed
- CI `release-notes` job greps against the `atomr-infer-` crate
  prefix instead of the legacy name, so version-bump release notes
  attach correctly.

## [0.3.0] — 2026-05-05

### Changed
- README rewritten to match the atomr formatting (top-level "Why...
  in Rust, now" framing + crate table + quick start (Rust) + quick
  start (Python) + layout). Remaining `inference-*` references in
  docs swept to `atomr-infer-*`.
- `xtask` verify steps now point at the `atomr-infer` rollup rather
  than the legacy `inference` crate name.

## [0.2.0] – [0.2.6] — 2026-04 to 2026-05

### Added
- PyPI publish pipeline: real wheels + sdist + OIDC trusted publisher.
- `pyproject.toml` version is now dynamic so PyPI tracks `Cargo.toml`.
- Workspace-wide `version = workspace.package.version` inheritance for
  every member crate; explicit description / metadata on every
  publishable crate.

### Changed
- Renamed publishable crates from `inference-*` to
  `atomr-infer-*` so the user-facing namespace matches the upstream
  atomr / atomr-accel naming.

### Renamed
- Project: `rakka-inference` → `atomr-infer`. Every namespace, every
  import, every doc reference. (See the Unreleased entry above for
  the final sweep of stragglers.)

## [0.1.0] — 2026-04

### Added
- Initial commit — the atomr-infer rollup, the per-backend runners
  (vLLM, TensorRT, ORT, candle, cudarc, mistralrs, OpenAI, Anthropic,
  Gemini, LiteLLM), the actor topology
  (`ApiGatewayActor` / `RequestActor` / `DpCoordinatorActor` /
  `EngineCoreActor` / `WorkerActor` / `ContextActor`), and the
  remote-core primitives (`RateLimiterActor` CRDT,
  `CircuitBreakerActor`, `RetryEngine`, SSE parser).
