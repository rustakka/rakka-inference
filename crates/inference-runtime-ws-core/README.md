# `atomr-infer-runtime-ws-core`

Shared WebSocket transport used by the audio provider runtimes
(`inference-runtime-deepgram`, `-assemblyai`, `-elevenlabs`,
`-openai-realtime`, `-gemini-live`).

This crate is intentionally provider-agnostic. It carries no JSON
envelopes, no auth headers, no schema knowledge — only the connection
lifecycle: TLS-aware `connect`, split tx/rx, ping/pong keepalive,
exponential-backoff reconnect honoring
[`atomr_infer_remote_core::BackoffPolicy`], and bounded frame
buffering with drop-oldest semantics.

See `docs/audio-modalities.md` for the wider audio program of work.
