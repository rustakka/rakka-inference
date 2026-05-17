# `atomr-infer-runtime-piper`

Local Piper TTS runtime for atomr-infer. Implements the
[`SpeechRunner`] trait against the [Piper](https://github.com/rhasspy/piper)
voice-pack format (`.onnx` + `.onnx.json` companion config).

## Quick start

```sh
# Download a small voice from rhasspy/piper-voices, e.g.
#   en_US-amy-low.onnx + en_US-amy-low.onnx.json
export PIPER_VOICE_PATH=/path/to/en_US-amy-low.onnx

cargo run -p tts_piper_local --features piper -- "Hello world"
```

## Build profiles

| Build                                                                  | Result                                                |
|------------------------------------------------------------------------|-------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-piper`                              | Stub — `ort` not in dep graph.                        |
| `cargo build -p atomr-infer-runtime-piper --features piper`             | Real path with `ort` crate (CPU EP).                  |
| `cargo build -p atomr-infer-runtime-piper --features piper-cuda`        | Adds the CUDA EP — needs a working CUDA toolkit.      |
| `cargo build -p atomr-infer-runtime-piper --features piper-load-dynamic`| Loads `libonnxruntime` at runtime via `ORT_DYLIB_PATH`. |

## Phonemization scope (M4)

M4 wires the SpeechRunner trait and the ONNX session. The text →
phoneme step is intentionally minimal: characters in `SpeechBatch.text`
are looked up directly in the voice config's `phoneme_id_map`. This
works for callers that have already passed text through `espeak-ng -x`
(IPA tokens) or that emit text in a script where each char *is* a
phoneme.

Full `espeak-ng` integration is a documented follow-up; the seam is
[`phoneme::PhonemeMap::ids_for_text`], swap in an espeak adapter that
returns IPA before the lookup.

See `docs/audio-modalities.md` for the wider audio program of work.
