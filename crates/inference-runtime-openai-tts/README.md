# `atomr-infer-runtime-openai-tts`

OpenAI text-to-speech runtime for `atomr-infer`. Implements
[`atomr_infer_core::runner::SpeechRunner`] against the
`POST /v1/audio/speech` endpoint.

## Build profiles

| Build                                                            | Result                                                       |
|------------------------------------------------------------------|--------------------------------------------------------------|
| `cargo build -p atomr-infer-runtime-openai-tts`                  | Stub — `reqwest` not in dep graph; `speak` returns `Internal`. |
| `cargo build -p atomr-infer-runtime-openai-tts --features tts-openai` | Real path — HTTPS POST + chunked PCM streaming.         |

## Models

Tested against `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`. The crate is
model-agnostic — anything OpenAI accepts on `/v1/audio/speech` works.

## Response format

The runner asks the API for `response_format=pcm` so chunks arrive as
24 kHz signed 16-bit little-endian mono PCM with no container framing
— each `SpeechChunk::audio_pcm_chunk` is a raw byte slice that can be
concatenated end-to-end. The final chunk has `is_final = true`.

Callers that want a different container (Opus, MP3, WAV) should set
[`SynthOptions::format`] explicitly; the runner forwards the format
through and updates `SpeechChunk::params` accordingly.

## Authentication

`OpenAiTtsConfig` reuses the sibling
[`atomr_infer_runtime_openai::OpenAiConfig`] auth surface — bearer
token, optional `OpenAI-Organization` / `OpenAI-Project` headers — and
defers session rebuild to `inference-remote-core::RemoteSessionActor`.

## Source

- [`FR-TTS-001`](../../docs/audio-modalities.md)
