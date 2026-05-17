# `tts_piper_local`

One-shot Piper TTS synthesis. Drives the local
[`atomr-infer-runtime-piper`](../../crates/inference-runtime-piper)
runtime end-to-end and writes a 16-bit PCM WAV file.

## Usage

```sh
# 1. Grab a voice from rhasspy/piper-voices, e.g. en_US-amy-low:
#    https://github.com/rhasspy/piper/blob/master/VOICES.md
export PIPER_VOICE_PATH=/path/to/en_US-amy-low.onnx
#    (the sibling en_US-amy-low.onnx.json is auto-resolved)

# 2. Synthesize. Input must already be phonemized (M4 scope — see
#    the crate README). Try the canned IPA below for a default
#    en-US voice:
cargo run -p tts_piper_local -- "həloʊ"
```

Output lands at `./piper_out.wav` at the voice's native sample rate.
