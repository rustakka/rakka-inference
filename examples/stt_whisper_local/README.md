# `stt_whisper_local`

One-shot speech-to-text via local whisper.cpp. Drives the
[`atomr-infer-runtime-whisper-local`](../../crates/inference-runtime-whisper-local)
runtime end-to-end on a single WAV file and prints the transcript.

## Usage

```sh
# 1. Grab a ggml whisper model, e.g. tiny.en (75 MB):
#    https://huggingface.co/ggerganov/whisper.cpp/tree/main
export WHISPER_MODEL_PATH=/path/to/ggml-tiny.en.bin

# 2. Transcribe a 16 kHz mono PCM-16 WAV file.
cargo run -p stt_whisper_local --features atomr-infer-runtime-whisper-local/stt-whisper -- /path/to/clip.wav
```

Output: one line per segment, prefixed with `[t_start..t_end]` in
milliseconds. When `WHISPER_WORD_TIMESTAMPS=1` is set, each segment
also prints its per-token timings.

## Supported architectures

Linux `x86_64` and `aarch64` only — anything else returns
`InferenceError::Unsupported`. See the crate README for the rationale.
