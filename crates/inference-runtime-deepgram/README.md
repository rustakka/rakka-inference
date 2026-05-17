# atomr-infer-runtime-deepgram

Deepgram WSS streaming speech-to-text runtime for `atomr-infer`.

Implements `atomr_infer_core::runner::AudioRunner` against the
`wss://api.deepgram.com/v1/listen` realtime endpoint via the shared
`atomr-infer-runtime-ws-core` transport.

## Feature gate

| Build                                                                | Result                                                  |
|----------------------------------------------------------------------|---------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-deepgram`                        | Stub — `execute_audio` returns `Internal("stt-deepgram feature disabled at build time")`. |
| `cargo build -p atomr-infer-runtime-deepgram --features stt-deepgram`| Real path — WSS streaming + interim/final progression.   |

## Source

`FR-STT-001`. See `docs/audio-modalities.md` and `docs/feature-matrix.md`.
