# atomr-infer-runtime-whisper-local

Local speech-to-text via [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp)
through the [`whisper-rs`](https://crates.io/crates/whisper-rs) Rust
bindings.

## What it does

- Implements [`AudioRunner`](../inference-core/src/runner.rs) for STT.
- Decodes a `.bin` ggml Whisper model (`ggml-tiny.en.bin`,
  `ggml-base.bin`, `ggml-large-v3.bin`, …) and runs full-audio
  transcription with optional per-word timings.
- Reports `transport_kind() = LocalCpu` so the placement layer never
  asks for a GPU ordinal for it (CUDA / Metal / Vulkan / CoreML
  backends are opt-in cargo features that flip the underlying
  `whisper-rs` flags; the runner stays on `LocalCpu` from
  `inference-core`'s perspective).

## Supported architectures

The whisper.cpp C bindings build cleanly on Linux x86_64 and Linux
aarch64. On any other host the crate compiles to an architecture stub
whose runner returns [`InferenceError::Unsupported`](../inference-core/src/error.rs)
the first time `execute_audio` is called — the rest of the workspace
still type-checks, which matters for cross-arch CI rows that build
`--features stt-all` without the underlying C toolchain.

| arch / OS              | builds | runs |
|------------------------|--------|------|
| `linux × x86_64`       | yes    | yes  |
| `linux × aarch64`      | yes    | yes  |
| `macos × x86_64/arm64` | yes    | no (returns `Unsupported`) |
| `windows × *`          | yes    | no (returns `Unsupported`) |

## Cargo features

| Feature                  | Pulls in                       |
|--------------------------|--------------------------------|
| `stt-whisper` (off by default) | `whisper-rs` + tokio + tracing |
| `stt-whisper-cuda`       | `whisper-rs/cuda`              |
| `stt-whisper-metal`      | `whisper-rs/metal`             |
| `stt-whisper-coreml`     | `whisper-rs/coreml`            |
| `stt-whisper-vulkan`     | `whisper-rs/vulkan`            |
| `stt-whisper-openblas`   | `whisper-rs/openblas`          |

## Audio input

`execute_audio` accepts an [`AudioInput::Static`](../inference-core/src/audio.rs)
carrying [`AudioPayload::Bytes`] or [`AudioPayload::Path`] in
**16 kHz mono 16-bit PCM** (`AudioFormat::Pcm16Le`) or
**16 kHz mono 32-bit float PCM** (`AudioFormat::PcmF32Le`). Anything
else returns `InferenceError::UnsupportedAudioFormat`; resampling is
the caller's responsibility (or a job for an upstream audio pipeline
crate landing later in the program of work).

## Example

See [`examples/stt_whisper_local`](../../examples/stt_whisper_local).
