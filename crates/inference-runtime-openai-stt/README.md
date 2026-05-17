# `atomr-infer-runtime-openai-stt`

OpenAI speech-to-text runtime for `atomr-infer`. Implements
[`atomr_infer_core::runner::AudioRunner`] against the
`POST /v1/audio/transcriptions` endpoint.

## Build profiles

| Build                                                                | Result                                                       |
|----------------------------------------------------------------------|--------------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-openai-stt`                      | Stub — `reqwest` not in dep graph; `execute_audio` returns `Internal`. |
| `cargo build -p atomr-infer-runtime-openai-stt --features stt-openai` | Real path — multipart upload + JSON / verbose-JSON parsing. |

## Models

Tested against `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`.
The crate is model-agnostic — anything OpenAI accepts on
`/v1/audio/transcriptions` works.

## Response format

The runner asks the API for `response_format=verbose_json` when the
caller has requested either word timestamps or interim segments (any
[`TranscribeOptions::word_timestamps`] / `interim_results`); otherwise
it asks for plain `json` and emits one final [`TranscriptChunk`] for
the full transcript.

With `verbose_json`:

- One [`TranscriptChunk`] per OpenAI segment, in order, with
  `ts_start_ms` / `ts_end_ms` set from the segment's `start` / `end`.
- Word-level timestamps from the API ride on
  [`TranscriptChunk::words`] when the caller requested them
  (`timestamp_granularities` includes `word`).
- The last segment carries `is_final = true`.

## Audio input

The runner materializes [`AudioInput::Static`] payloads into a
multipart `file` part with the filename `audio.wav`. The API
auto-detects most formats; the wire payload's bytes are uploaded
verbatim. Streaming inputs are not yet supported (returns
`Unsupported`).

## Authentication

Reuses the auth surface of the sibling
[`atomr_infer_runtime_openai::OpenAiConfig`]: bearer token, optional
`OpenAI-Organization` / `OpenAI-Project` headers.

## Source

- [`FR-STT-001`](../../docs/audio-modalities.md)
