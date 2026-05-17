# Feature matrix

`atomr-infer` is layered so you can opt into exactly the runtimes
and infrastructure pieces you need. This page tells you *which feature
to flip* and *what it pulls in*.

The principle: **declaring `inference = { features = [...] }` in your
`Cargo.toml` is a statement of intent**. The feature graph computes the
actual dependency graph for you.

---

## Quick recipes

| You want…                                                | Feature flags                                                |
|----------------------------------------------------------|--------------------------------------------------------------|
| Pure-remote router (no GPU, no Python)                   | `remote-only`                                                |
| Just OpenAI                                              | `openai`                                                     |
| OpenAI + Anthropic, with hybrid pipeline                 | `openai`, `anthropic`, `pipeline`                            |
| Local Candle GPU + remote OpenAI                         | `candle`, `openai`, `pipeline`                               |
| Zero-config local Gemma 4 on a workstation               | `gemma-default` *(opt-in only — auto-provisions a deployment)* |
| The full production preset                               | `default-prod`                                               |
| Everything                                               | `all-runtimes`                                               |
| Mocking + wiremock for tests                             | `testkit` (alongside the runtimes you mock)                  |
| Reach into `atomr_accel::*` directly                     | `accel` (re-exports as `atomr_infer::accel`)                 |
| Reach the NVIDIA CUDA backend                            | `accel` (re-exports as `atomr_infer::accel_cuda`)            |
| Use `DynamicBatchingServer` / `InferenceCascade`         | `accel-patterns` (re-exports as `atomr_infer::accel_patterns`) |
| Embed in Python                                          | `atomr-infer-py-bindings/python` on the bindings crate       |

---

## What each feature pulls in

| Feature                | Adds crate(s)                                                       | System / heavy deps         | Notes |
|------------------------|---------------------------------------------------------------------|-----------------------------|-------|
| `openai`               | `atomr-infer-runtime-openai`                                        | `reqwest`, `hyper`          | Includes the Azure variant. |
| `anthropic`            | `atomr-infer-runtime-anthropic`                                     | `reqwest`, `hyper`          | Tool-use + base64 vision. |
| `gemini`               | `atomr-infer-runtime-gemini`                                        | `reqwest`, `hyper`          | AI Studio + Vertex; OAuth2 via pluggable `CredentialProvider`. |
| `litellm`              | `atomr-infer-runtime-litellm`                                       | (re-uses `openai`)          | LiteLLM proxy with proxy-friendly defaults. |
| `vllm`                 | `atomr-infer-runtime-vllm`                                          | **`pyo3`**, `python`        | Native `LLMEngine` bridge with token-streaming. |
| `tensorrt`             | `atomr-infer-runtime-tensorrt` + `atomr-accel-tensorrt`             | `libnvinfer.so` (link-time) | Real engine builder + runtime wrapper since v0.5. |
| `tensorrt-onnx` / `-int8` / `-fp8` / `-plugin` / `-link` | (sub-features on `tensorrt`)             | per-feature                 | ONNX import / INT8 / FP8 PTQ / `IPluginV3` / actual `libnvinfer` link. |
| `ort`                  | `atomr-infer-runtime-ort`                                           | `ort`, `tokenizers`         | ONNX Runtime via the `ort` crate. CPU EP + tokenizer + autoregressive sampling for ONNX-exported causal LMs; low-level `OrtRunner::infer` for embeddings / encoders / Whisper / vision. |
| `ort-cuda` / `-load-dynamic` / `-hf-hub` | (sub-features on `ort`)                           | per-feature                 | CUDA EP / runtime `libonnxruntime` lookup / HuggingFace `tokenizer.json` fallback. |
| `candle`               | `atomr-infer-runtime-candle` + `accel`                              | `candle-*`, `cudarc`        | Pure-Rust transformer inference. |
| `cudarc`               | `atomr-infer-runtime-cudarc` + `accel`                              | `cudarc`                    | Direct kernel dispatch via `atomr_accel_cuda::kernel::*`. |
| `mistralrs`            | `atomr-infer-runtime-mistralrs`                                     | `mistralrs`                 | Rust-native LLM runtime with token-streaming. |
| `gemma-default`        | `atomr-infer-runtime-vllm/gemma-default` (= `vllm` + env probe)     | `pyo3`, `python`, `hf-hub`  | Auto-provisions a `gemma-local` deployment (`google/gemma-4-E4B-it`) when GPU + Python + vLLM + HF token are present. See [`docs/local-gemma.md`](local-gemma.md). |
| `pipeline`             | `atomr-infer-pipeline`                                              | `atomr-streams`             | Streams DSL adapter. |
| `accel`                | `atomr-accel` (trait surface) + `atomr-accel-cuda` (NVIDIA backend) | `cudarc`                    | Reach `atomr_infer::accel::*` and `atomr_infer::accel_cuda::*`. |
| `accel-patterns`       | `atomr-accel-patterns` re-export, `pipeline`                        | `cudarc`                    | `DynamicBatchingServer`, `InferenceCascade`, `ModelReplicaPool`, `FairShareScheduler`, `ModelHotSwapServer`, `SpeculativeDecoder`, `MoeRouter`. |
| `testkit`              | `atomr-infer-testkit`                                               | `wiremock`                  | `MockRunner`, OpenAI/Anthropic/Gemini wiremock fixtures. |

### Audio transport infrastructure

| Crate                              | Adds                                                                | System / heavy deps | Notes |
|------------------------------------|---------------------------------------------------------------------|----------------------|-------|
| `atomr-infer-runtime-ws-core`      | `WsClient` (`connect` / `connect_with_headers`), `Frame`, `ReconnectEngine`, `Keepalive`, `coalesce_binary` | `tokio-tungstenite` (`rustls-tls-webpki-roots`) | Shared WebSocket transport for the audio provider crates (Deepgram, AssemblyAI, ElevenLabs, OpenAI Realtime, Gemini Live). `connect_with_headers(url, &[(name, value)], timeout)` injects custom HTTP headers (e.g. `Authorization: Token <key>` for Deepgram) onto the WS upgrade request. Always-on dependency of those crates; not user-facing. |

### Audio provider runtimes

| Crate                              | Adds                              | System / heavy deps  | Per-arch / OS    | Notes |
|------------------------------------|-----------------------------------|----------------------|------------------|-------|
| `atomr-infer-runtime-piper`        | `PiperRunner` (`SpeechRunner`)    | `ort` (CPU EP)       | any              | Local TTS over the [rhasspy/piper](https://github.com/rhasspy/piper) ONNX voice pack (`.onnx` + `.onnx.json`). Feature gate `piper`. CUDA EP via `piper-cuda`; runtime lib via `piper-load-dynamic`. **M4 scope**: char-level phoneme lookup against the voice's `phoneme_id_map`. Real text→IPA via `espeak-ng` is a documented follow-up. |
| `atomr-infer-runtime-whisper-local` | `WhisperRunner` (`AudioRunner`) | `whisper-rs` 0.16 + `whisper.cpp` (vendored C bindings) | linux x86_64 / aarch64 (runs); any other arch (builds, runs return `Unsupported`) | Local STT over a ggml whisper model file (`ggml-tiny.en.bin`, `ggml-base.bin`, `ggml-large-v3.bin`, …). Feature gate `stt-whisper`. Accelerator backends via `stt-whisper-cuda` / `stt-whisper-metal` / `stt-whisper-coreml` / `stt-whisper-vulkan` / `stt-whisper-openblas`. **M5 scope**: 16 kHz mono `Pcm16Le` / `PcmF32Le` / canonical `Wav` `AudioPayload::Bytes`/`Path` inputs only; streaming `AudioInput::Stream` returns `Unsupported` (needs a VAD chunker upstream). |
| `atomr-infer-runtime-openai-tts`   | `OpenAiTtsRunner` (`SpeechRunner`) | `reqwest` (rustls)   | any              | Remote TTS over `POST /v1/audio/speech`. Feature gate `tts-openai`. Models: `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`. Default `response_format=pcm` (24 kHz mono `Pcm16Le`); `mp3` / `wav` / `opus` / `flac` selectable via `SynthOptions::format`. Re-uses `OpenAiConfig`'s bearer + retry classification; response body re-chunked at `OpenAiTtsConfig::chunk_bytes` (default 8192 ≈ 170 ms). Without the feature the runner compiles to a stub returning `Internal("tts-openai feature disabled at build time")`. |
| `atomr-infer-runtime-openai-stt`   | `OpenAiSttRunner` (`AudioRunner`)  | `reqwest` (rustls, `multipart`) | any   | Remote STT over `POST /v1/audio/transcriptions`. Feature gate `stt-openai`. Models: `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`. Multipart upload from `AudioPayload::Bytes` or `AudioPayload::Path`; `AudioInput::Stream` returns `Unsupported` (use the realtime endpoint or accumulate upstream). Auto-selects `verbose_json` (per-segment chunks + optional `WordTiming`) when `TranscribeOptions::word_timestamps` or `interim_results` is set; otherwise `json` (one final chunk). Stub fallback identical to `tts-openai`. |
| `atomr-infer-runtime-elevenlabs`   | `ElevenLabsTtsRunner` (`SpeechRunner`) | `reqwest` (rustls, `multipart`) + `tokio-tungstenite` (via `atomr-infer-runtime-ws-core`) | any | Remote TTS over `POST /v1/text-to-speech/{voice_id}` (one-shot HTTPS) and `WSS /v1/text-to-speech/{voice_id}/stream-input` (bidirectional WebSocket with per-character alignment frames). Feature gate `tts-elevenlabs`. Models: `eleven_multilingual_v2`, `eleven_turbo_v2_5`, `eleven_flash_v2_5`. Voices selected via `VoiceRef::Id(21-char_voice_id)` or `VoiceRef::Named(_)`; `VoiceRef::ClonedFrom(_)` requires an explicit `ElevenLabsTtsRunner::clone_voice` round-trip (multipart upload to `/v1/voices/add`) first. `output_format` query string auto-derives from `SynthOptions::format` (`Mp3 → mp3_44100_128`, `Pcm16Le → pcm_24000`, `OggOpus → opus_48000_64`). `SpeechBatch::stream = false` takes the HTTPS path (response re-chunked at `chunk_bytes` boundary); `stream = true` opens the WSS path and emits one `SpeechChunk` per inbound JSON frame, attaching an `AlignmentDelta` (one `WordTiming` per character) when `emit_alignment = true`. 429s classify into `RateLimited { provider: ElevenLabs, .. }`. Stub fallback identical to `tts-openai`. |
| `atomr-infer-runtime-deepgram`     | `DeepgramSttRunner` (`AudioRunner`) | `tokio-tungstenite` (via `atomr-infer-runtime-ws-core`) | any | Remote STT over `WSS /v1/listen` with the runner's `Authorization: Token <key>` header injected on the WS upgrade (via the new `WsClient::connect_with_headers`). Feature gate `stt-deepgram`. Models: `nova-2`, `nova-2-phonecall`, `nova-2-meeting`, `nova`, `enhanced`, `base`. Audio mapping: `Pcm16Le → linear16`, `PcmF32Le → linear32`, `OggOpus → opus`, `Mp3`/`Flac`/`Wav` pass through; `Pcm24Le` returns `UnsupportedAudioFormat`. Uplink re-chunks at ≈4096 B (≈128 ms @ 16 kHz mono PCM16) and flushes with a `{"type":"CloseStream"}` text frame once the source drains. Downlink decodes `Results` envelopes; `TranscriptChunk::is_final` follows the provider's *utterance* boundary (`speech_final`), not the segment boundary. Interim chunks are dropped when `TranscribeOptions::interim_results == false`. `diarize` stringifies the first word's `speaker` label into `TranscriptChunk::speaker_id`; `word_timestamps` surfaces per-word `WordTiming`s (preferring `punctuated_word` when present). Always appends `endpointing=300` so the provider VAD emits `speech_final` after ~300 ms of silence. Abnormal close codes (≠ 1000/1005/1006) surface as `InferenceError::NetworkError`. Stub fallback identical to `tts-openai`. |
| `atomr-infer-runtime-assemblyai`   | `AssemblyAiSttRunner` (`AudioRunner`) | `tokio-tungstenite` (via `atomr-infer-runtime-ws-core`) | any | Remote STT over `WSS /v3/ws` (AssemblyAI Universal-Streaming v3) with the runner's `Authorization: <key>` header (no `Token` prefix — contrast with Deepgram) injected on the WS upgrade. Feature gate `stt-assemblyai`. Model: `universal` (a.k.a. `universal-streaming` / `universal-v3`; the model is implicit at v3, not a query parameter). **Audio gate**: 16 kHz mono `Pcm16Le` only — every other `AudioFormat` is rejected with `UnsupportedAudioFormat` before the connect (resample upstream). Uplink re-chunks at ≈4096 B and flushes with a `{"type":"Terminate"}` text frame once the source drains. Downlink decodes `Begin` / `Turn` / `Termination` envelopes; only `Turn` produces `TranscriptChunk`s. `TranscriptChunk::is_final` follows `end_of_turn` — **exactly one turn-final per spoken turn**, with no Deepgram-style segment-final vs utterance-final distinction. Partial chunks are dropped when `TranscribeOptions::interim_results == false`. `word_timestamps` surfaces per-word `WordTiming`s (v3 already gives `start`/`end` in milliseconds). `TranscribeOptions::diarize` is silently ignored — v3 Streaming has no speaker labels on the wire (their async API does). `AssemblyAiSttConfig::format_turns = true` requests Punctuated & Formatted text on turn-final updates; off by default because it adds latency. Abnormal close codes (≠ 1000/1005/1006) surface as `InferenceError::NetworkError`. Stub fallback identical to `tts-openai`. |
| `atomr-infer-runtime-openai-realtime` | `OpenAiRealtimeRunner` (`RealtimeRunner`) | `tokio-tungstenite` (via `atomr-infer-runtime-ws-core`) | any | Bidirectional realtime over `WSS api.openai.com/v1/realtime?model=<m>` with `Authorization: Bearer <key>` + `OpenAI-Beta: realtime=v1` headers. Feature gate `tts-openai-realtime`. Models: `gpt-4o-realtime-preview`, `gpt-4o-mini-realtime-preview`. Output PCM16-LE @ 24 kHz mono. Inbound: `AudioFrame` → `input_audio_buffer.append` (base64); `Text` → `conversation.item.create` + `response.create`; `Commit` → `input_audio_buffer.commit`; `Interrupt` → `response.cancel`; `Close` → WS close 1000. Outbound: `response.audio.delta` → `AudioFrame`; `response.audio_transcript.done` → `Transcript { role: Assistant, is_final: true }`; `response.done` → `Done`; `error` → `Error`. `VoiceRef::ClonedFrom` rejected with `BadRequest`. Stub fallback identical to `tts-openai`. |
| `atomr-infer-runtime-gemini-live`     | `GeminiLiveRunner` (`RealtimeRunner`)     | `tokio-tungstenite` (via `atomr-infer-runtime-ws-core`) | any | Bidirectional realtime over `WSS generativelanguage.googleapis.com/.../BidiGenerateContent?key=<key>` (API-key-in-URL — no Authorization header). Feature gate `tts-gemini-live`. Models: `gemini-2.0-flash-exp`. Setup handshake: client sends `BidiGenerateContentSetup` then waits for `setupComplete` before forwarding user input. Audio gate: `AudioFrame` requires `Pcm16Le`; other formats → `UnsupportedAudioFormat`. `Interrupt` → `InferenceError::Unsupported` (provider has no interrupt). `VoiceRef::ClonedFrom` rejected with `BadRequest`. Cost: `per_minute_usd` for time-based audio billing (0.0 on flash-exp). Stub fallback identical to `tts-openai`. |
| `atomr-infer-runtime-kokoro`          | `KokoroRunner` (`SpeechRunner`)           | `ort` (CPU EP)                                          | any | Local TTS via Kokoro-82M ONNX voice pack. Feature gate `tts-kokoro`. Output PCM16-LE @ 24 kHz mono. `KokoroConfig { voice_pack_dir, default_voice }`. Sub-features: `tts-kokoro-cuda`, `tts-kokoro-load-dynamic`. Smoke test gated on `KOKORO_VOICE_PATH`. Stub fallback identical to `tts-piper`. |
| `atomr-infer-runtime-xtts`            | `XttsRunner` (`SpeechRunner`)             | `ort` (CPU EP)                                          | any | Local TTS via Coqui XTTS-v2 (cross-lingual voice cloning). Feature gate `tts-xtts`. `XttsConfig { model_path, default_language }`. `VoiceRef::ClonedFrom` is the primary entry point (speaker-embedding computation stubbed in M10 — validation + warning logged, zero embedding used). Sub-features: `tts-xtts-cuda`, `tts-xtts-load-dynamic`. Smoke test gated on `XTTS_MODEL_PATH`. Stub fallback identical to `tts-piper`. |
| `atomr-infer-runtime-moss`            | `MossRunner` (`SpeechRunner`)             | none (CPU)                                              | linux x86_64 / aarch64 | Local TTS via MOSS-TTSD transformer. Feature gate `tts-moss`. Linux-only: on non-Linux hosts the feature compiles cleanly but `speak` returns `Internal("tts-moss requires Linux")`. `MossConfig { model_dir, default_voice }`. Smoke test gated on `MOSS_MODEL_PATH`. Stub fallback identical to `tts-piper`. |
| `atomr-infer-runtime-audio2face`      | `Audio2FaceRunner` (`A2FRunner`)          | `tonic` 0.12 + `prost` 0.13                             | linux x86_64 | NVIDIA Audio2Face-3D over gRPC (transport scaffolded with `TODO(grpc-transport)`; M11 ships an in-memory deterministic generator). Feature gate `audio2face`. Ingests `AudioBatch` with `AudioOptions::Audio2Face`; emits `BlendshapeChunk` stream with 52 ARKit-canonical weights. `arkit::ARKIT_BLENDSHAPE_NAMES` is the 52-element Apple `ARBlendShapeLocation` ordered constant; `arkit::a2f_to_arkit` is the A2F→ARKit index adapter (currently passthrough). Architecture gate: requires Linux x86_64 at runtime; other platforms return `Internal("audio2face requires Linux x86_64")`. Hand-written `prost` message types — no `protoc` / `tonic-build` needed. Stub fallback returns `FeatureDisabled`. |

The `candle` and `cudarc` features automatically imply `accel` because
their bodies use `atomr_accel_cuda::dispatcher::GpuDispatcher` and
`atomr_accel_cuda::kernel::*` for thread pinning and kernel dispatch.

---

## Aggregates

| Aggregate            | Expands to                                                              |
|----------------------|--------------------------------------------------------------------------|
| `all-native`         | `tensorrt`, `ort`, `candle`, `cudarc`, `mistralrs`                       |
| `all-python`         | `vllm`                                                                   |
| `all-local`          | `all-native` + `all-python`                                              |
| `all-remote`         | `openai`, `anthropic`, `gemini`, `litellm`                               |
| `all-runtimes`       | `all-local` + `all-remote` + `accel-patterns`                            |
| `default-prod`       | `vllm`, `tensorrt`, `ort`, `openai`, `anthropic`, `pipeline`             |
| `remote-only`        | `all-remote` + `pipeline` *(deliberately excludes `accel` / `accel-patterns` / `gemma-default`)* |

---

## The remote-only invariant

> `cargo build -p atomr-infer --no-default-features --features remote-only`
> compiles **zero** GPU dependencies.

This is enforced by the feature graph:

```sh
$ cargo tree -p inference --no-default-features --features remote-only \
    | grep -Ec 'cudarc|atomr-accel|candle|pyo3'
0
```

Why this matters: a remote-only deployment (a fleet that fronts OpenAI
/ Anthropic / Gemini with rate limiting, fallback chains, and
observability) doesn't need to drag CUDA toolchains, candle's
ML stack, or a Python interpreter into its container image. The
feature gate guarantees the dep graph reflects intent.

---

## Per-crate features

Some crates expose their own gates so they can be consumed
**independently** without going through the rollup:

### `atomr-infer-runtime`

| Feature      | Adds                                            |
|--------------|-------------------------------------------------|
| `local-gpu`  | `atomr-accel` dep; `WorkerActor` adopts upstream `device_supervisor_strategy()` |

Default builds compile without atomr-accel; useful when you're embedding
the runtime-agnostic actors into a remote-only service.

### `atomr-infer-pipeline`

| Feature              | Adds                                  |
|----------------------|---------------------------------------|
| `cuda-patterns`      | `atomr-accel-patterns` re-export (sub-feature of the rollup's `accel-patterns`) |

Without the feature you still get `request_source`, `HybridGraph`, and
the `atomr-streams` `Source` adapter — useful for remote-only
pipelines.

### `atomr-infer-python-bridge`

| Feature   | Adds                              |
|-----------|-----------------------------------|
| `python`  | `pyo3` + `tokio` + `parking_lot`; real `PythonGpuBridge` |

Off by default so the workspace builds without a Python venv.

### `atomr-infer-py-bindings`

| Feature   | Adds                                       |
|-----------|--------------------------------------------|
| `python`  | `pyo3` + `tracing`; builds the `cdylib`    |

### Per-runtime crates (`atomr-infer-runtime-*`)

Each carries one feature whose name matches the runtime
(`vllm`, `tensorrt`, `ort`, `candle`, `cudarc`, `mistralrs`). Without
the feature, the runner returns
`InferenceError::Internal("<runtime> feature disabled at build time")`
so dependent code links cleanly. With the feature, real bodies pull
their respective system / Rust crates.

---

## Choosing a slice

A few common shapes:

**1. The OpenAI-compatible router.** No hardware. Sits in front of
managed APIs. Adds rate limiting and fallback chains.

```toml
inference = { workspace = true, features = ["remote-only"] }
```

**2. The Rust-native LLM box.** Owns one box of GPUs, runs Candle (or
mistral.rs), no Python.

```toml
inference = { workspace = true, features = ["candle", "mistralrs", "pipeline"] }
```

**3. The hybrid agent.** Local Mistral classifier escalates to GPT-4o
on hard queries; falls back to Claude on saturation.

```toml
inference = { workspace = true, features = ["mistralrs", "openai", "anthropic", "accel-patterns"] }
```

**4. The vLLM cluster.** Production LLM inference on owned hardware.

```toml
inference = { workspace = true, features = ["vllm", "tensorrt", "openai", "pipeline"] }
```

Each shape uses **only** the layers it needs. No dead weight in your
binary.

---

## Adding a new backend

The contract is small: implement `inference_core::ModelRunner`,
provide a `RuntimeConfig`-shaped struct, and add a feature flag in the
rollup to wire it in. The 18-crate layout is *additive*: a third-party
runtime (Bedrock, Cohere, internal proxy, custom CUDA kernel package)
ships as a sibling crate that depends on `atomr-infer-core` and
`atomr-infer-remote-core` (for remote) or `atomr-infer-core` +
`atomr-accel` (for local), without forking the workspace.
