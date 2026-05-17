# tts_elevenlabs_alignment

Stream ElevenLabs WS TTS and print per-character alignment frames as
they arrive. Companion example to milestone **M7** of the audio
program of work (`FR-TTS-001`).

## Run

```bash
ELEVEN_API_KEY=sk-... \
  cargo run -p tts_elevenlabs_alignment -- \
  "Streaming alignment is fun." 21m00Tcm4TlvDq8ikWAM
```

Arguments:

1. **text** — the utterance to speak (default: `"Hello from atomr-infer."`).
2. **voice id** — 21-character ElevenLabs voice id (default:
   `21m00Tcm4TlvDq8ikWAM`, the "Rachel" voice).

Environment:

- `ELEVEN_API_KEY` — **required**, your ElevenLabs bearer token.
- `ELEVEN_OUT_PATH` — where to write the concatenated MP3 output
  (default: `elevenlabs_out.mp3`).

## Expected output

Each inbound JSON frame surfaces a `SpeechChunk` with `audio` + an
`AlignmentDelta`. The example writes the audio bytes to disk and
prints the per-character timings to stderr:

```
frame   0: 5 chars, 824 bytes audio (is_final=false)
    char="H" start=0ms end=44ms
    char="e" start=44ms end=83ms
    char="l" start=83ms end=120ms
    ...
frame   3: 0 chars, 612 bytes audio (is_final=true)
wrote 6_540 bytes of MP3 to elevenlabs_out.mp3 (4 alignment frames observed)
```
