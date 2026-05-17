# atomr-infer-runtime-moss

Local [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD) TTS runtime for
`atomr-infer`. Implements [`SpeechRunner`] against a MOSS transformer-based TTS
model and emits PCM16-LE `SpeechChunk`s.

**Linux-only.** On non-Linux platforms, `speak` returns
`InferenceError::Internal("tts-moss requires Linux")` even when the `tts-moss`
feature is enabled.

## Build profiles

| Command | Result |
|---|---|
| `cargo build -p atomr-infer-runtime-moss` | Stub — feature disabled. |
| `cargo build -p atomr-infer-runtime-moss --features tts-moss` | Real path (Linux) / `requires Linux` error (other OS). |

## Configuration

```rust,ignore
use atomr_infer_runtime_moss::{MossConfig, MossRunner};

let runner = MossRunner::new(MossConfig {
    model_dir: "/models/moss-tts".into(),
    default_voice: "default".into(),
    chunk_samples: 4096,
});
```

## Source

`FR-TTS-001`. See `docs/audio-modalities.md`.
