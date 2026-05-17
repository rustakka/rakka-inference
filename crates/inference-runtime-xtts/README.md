# atomr-infer-runtime-xtts

Local [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) TTS runtime for
`atomr-infer`. Implements [`SpeechRunner`] against an ONNX-exported XTTS-v2
model and emits PCM16-LE `SpeechChunk`s.

XTTS-v2 is a cross-lingual, voice-cloning TTS model. The primary voice
selection mode is `VoiceRef::ClonedFrom(AudioPayload)` — pass a short reference
audio clip and the runner conditions synthesis on the extracted speaker
embedding.

## Build profiles

| Command | Result |
|---|---|
| `cargo build -p atomr-infer-runtime-xtts` | Stub — `ort` not in dep graph. |
| `cargo build -p atomr-infer-runtime-xtts --features tts-xtts` | Real path with `ort` crate (CPU EP). |
| `cargo build -p atomr-infer-runtime-xtts --features tts-xtts-cuda` | Adds the CUDA EP. |
| `cargo build -p atomr-infer-runtime-xtts --features tts-xtts-load-dynamic` | Loads `libonnxruntime` at runtime via `ORT_DYLIB_PATH`. |

## Voice cloning status (M10)

The voice-cloning **embedding computation is stubbed** in this milestone. The
runner validates that the reference audio payload is materializable and logs a
warning, then proceeds with a zero-valued speaker embedding. A follow-up
milestone will wire the real `d-vector` / `speaker-encoder` ONNX model.

## Source

`FR-TTS-001`. See `docs/audio-modalities.md`.
