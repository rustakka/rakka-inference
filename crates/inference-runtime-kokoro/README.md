# atomr-infer-runtime-kokoro

Local [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS runtime for
`atomr-infer`. Implements [`SpeechRunner`] against an ONNX-exported Kokoro voice
pack and emits PCM16-LE `SpeechChunk`s.

## Build profiles

| Command | Result |
|---|---|
| `cargo build -p atomr-infer-runtime-kokoro` | Stub — `ort` not in dep graph. |
| `cargo build -p atomr-infer-runtime-kokoro --features tts-kokoro` | Real path with `ort` crate (CPU EP). |
| `cargo build -p atomr-infer-runtime-kokoro --features tts-kokoro-cuda` | Adds the CUDA EP. |
| `cargo build -p atomr-infer-runtime-kokoro --features tts-kokoro-load-dynamic` | Loads `libonnxruntime` at runtime via `ORT_DYLIB_PATH`. |

## Voice packs

Kokoro ships weights as PyTorch `.pt` files. For this runtime, pre-convert
them to ONNX (e.g. via `torch.onnx.export` or the upstream export scripts)
and point `KokoroConfig::voice_pack_dir` at the directory containing the
converted `.onnx` files. Set `KokoroConfig::default_voice` to the filename
stem (without `.onnx`) of the voice to load.

## Usage

```rust,ignore
use atomr_infer_runtime_kokoro::{KokoroConfig, KokoroRunner};

let runner = KokoroRunner::new(KokoroConfig {
    voice_pack_dir: "/path/to/voices".into(),
    default_voice: "af_heart".into(),
    chunk_samples: 4096,
    intra_threads: None,
});
```

## Source

`FR-TTS-001`. See `docs/audio-modalities.md`.
