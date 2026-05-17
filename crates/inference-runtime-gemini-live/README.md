# atomr-infer-runtime-gemini-live

Gemini Live bidirectional realtime speech runtime for `atomr-infer`.

Implements `atomr_infer_core::runner::RealtimeRunner` against the
`wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`
endpoint via the shared `atomr-infer-runtime-ws-core` transport.

## Feature gate

| Build                                                                          | Result                                                                  |
|--------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-gemini-live`                               | Stub — `open_session` returns `Internal("tts-gemini-live feature disabled at build time")`. |
| `cargo build -p atomr-infer-runtime-gemini-live --features tts-gemini-live`    | Real path — bidirectional WSS, setup handshake, PCM audio I/O.          |

## Auth

Unlike OpenAI Realtime, Gemini Live embeds the API key as a `?key=<api_key>`
URL query parameter. No `Authorization` header is used.

## Source

`FR-TTS-001` (realtime section). See `docs/audio-modalities.md` and `docs/feature-matrix.md`.
