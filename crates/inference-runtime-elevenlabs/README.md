# atomr-infer-runtime-elevenlabs

ElevenLabs text-to-speech runtime for `atomr-infer`. Implements
[`SpeechRunner`] against the ElevenLabs HTTP API
(`POST /v1/text-to-speech/{voice_id}`) and the WebSocket streaming
API (`/v1/text-to-speech/{voice_id}/stream-input`), with shared
WebSocket transport from `atomr-infer-runtime-ws-core`.

## Build profiles

| Build                                                                       | Result                                                |
|-----------------------------------------------------------------------------|-------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-elevenlabs`                             | Stub — `speak` returns `Internal("tts-elevenlabs feature disabled at build time")`. |
| `cargo build -p atomr-infer-runtime-elevenlabs --features tts-elevenlabs`   | Real path — HTTPS one-shot + WSS streaming alignment. |

## Models / voices

ElevenLabs assigns model + voice **identifiers** rather than named
buckets. Pass:

- `SpeechBatch::model` — e.g. `eleven_multilingual_v2`,
  `eleven_turbo_v2_5`, or `eleven_flash_v2_5`.
- `SpeechBatch::voice` — `VoiceRef::Id(...)` carrying the 21-character
  ElevenLabs voice id (e.g. `21m00Tcm4TlvDq8ikWAM` for "Rachel"), or
  `VoiceRef::Named(...)` which the runner forwards verbatim. A
  `VoiceRef::ClonedFrom(_)` payload triggers the cloning multipart
  upload path (`/v1/voices/add`) — see `ElevenLabsTtsRunner::clone_voice`.

## Output shape

The HTTPS path materializes the full audio body and re-chunks it at
`ElevenLabsTtsConfig::chunk_bytes` boundaries before emitting
`SpeechChunk`s. The terminal chunk carries `is_final = true`.

The WS streaming path emits one `SpeechChunk` per inbound JSON frame,
attaching an `AlignmentDelta` (with per-character timings) to the
chunk when the provider includes one.

## Source

`FR-TTS-001`. See `docs/audio-modalities.md` for the architectural
decisions and `docs/feature-matrix.md` for the per-arch / per-feature
support matrix.
