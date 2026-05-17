# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added — audio modality type surface (`inference-core`)
- New `audio` module (`inference-core/src/audio.rs`) introducing the
  shared type vocabulary for the audio program of work
  (`FR-TTS-001`, `FR-STT-001`, `FR-A2F-001`):
  - I/O primitives: `AudioFormat`, `AudioParams`, `AudioPayload`,
    `AudioInput`.
  - Per-modality batches: `AudioBatch` (STT + A2F), `SpeechBatch`
    (TTS), `RealtimeBatch` (bidi).
  - Per-modality option blobs: `TranscribeOptions`, `SynthOptions`,
    `A2FOptions`, `AudioOptions`.
  - Output chunks: `TranscriptChunk`, `SpeechChunk`,
    `BlendshapeChunk`.
  - Alignment / timing primitives: `WordTiming`, `AlignmentDelta`,
    `Viseme`.
  - Realtime session protocol: `RealtimeIn`, `RealtimeOut`,
    `TranscriptRole`.
  - Voice selection: `VoiceRef`.
  - `emotion_presets` constants module with common provider preset
    names.
- Sibling traits in `inference-core::runner` next to `ModelRunner`:
  `AudioRunner` (STT), `SpeechRunner` (TTS), `RealtimeRunner` (bidi),
  `A2FRunner` (Audio2Face). Each is object-safe and async; each
  exposes `runtime_kind` / `transport_kind` / optional `rate_limits`.
- Sibling handles: `AudioRunHandle`, `SpeechRunHandle`,
  `A2FRunHandle`, `RealtimeSession`. `RunHandle` stays monomorphic
  over `TokenChunk`.
- New `RuntimeKind` variants — `SpeechToText`, `TextToSpeech`,
  `RealtimeSpeech`, `Audio2Face` — and matching `RuntimeConfig`
  variants carrying opaque per-provider config blobs.
- New `TransportKind::LocalCpu` for whisper.cpp / Piper CPU voices,
  and new `TransportKind::UnknownTransport` for audio-modality kinds
  whose transport depends on the deployment config rather than the
  runtime kind. The `From<&RuntimeKind>` wildcard is replaced with
  explicit arms.
- New `ProviderKind` arms: `Deepgram`, `AssemblyAi`, `ElevenLabs`,
  `NvidiaA2F`.
- New `InferenceError` variants: `Unsupported { method, runtime }`,
  `UnsupportedAudioFormat { message }`, `RealtimeClosed { reason }`.
  None are retryable.
- Inline `#[cfg(test)]` coverage with `proptest` (first use of the
  workspace `proptest` dep): constructor coverage for every new
  struct, serde round-trip for `AudioPayload` / `SpeechBatch` /
  `TranscribeOptions` / `SynthOptions` / `A2FOptions` /
  `BlendshapeChunk`, `WordTiming` ordering invariants, `AudioParams`
  range validation, `RuntimeKind` / `TransportKind` / `ProviderKind`
  exhaustiveness across the audio additions.
- Architecture doc: `docs/audio-modalities.md` captures the
  sibling-trait / per-modality-engine / unified-input decisions and
  the resolution of the four FR open questions.
- `inference-core` deps: `tokio` added with `features = ["sync"]`
  (for `mpsc::Receiver` on the realtime batch). `proptest` added as
  a dev-dependency.

### Added — audio engine actors + gateway routes (`inference-runtime`, `inference-testkit`)

M2 of the audio program of work. Wires the M1 type vocabulary into
the actor topology without changing the text path.

- New per-modality engine actors in `inference-runtime`:
  - `SpeechEngineCoreActor` (`src/speech_engine.rs`) over
    `Box<dyn SpeechRunner>` — TTS, admits on characters.
  - `AudioEngineCoreActor` (`src/audio_engine.rs`) over either
    `Box<dyn AudioRunner>` (STT) or `Box<dyn A2FRunner>` (Audio2Face)
    — `new_stt` / `new_audio2face` constructors pin the modality, and
    cross-modality messages surface as `InferenceError::Unsupported`
    rather than mis-dispatching.
  - `RealtimeEngineCoreActor` (`src/realtime_engine.rs`) over
    `Box<dyn RealtimeRunner>` — admits on concurrent sessions; relays
    outbound frames through a bridge task that releases the admission
    slot when either side of the session drops.
  - Each carries an `AsyncMutex<Box<dyn Runner>>` mirroring the
    existing `EngineCoreActor` lock model; `try_admit` /
    `saturating_sub` admission counter mirrors the text engine.
- New gateway routes in `inference-runtime/src/gateway.rs` —
  `POST /v1/audio/transcriptions`, `GET /v1/audio/transcriptions/stream`,
  `POST /v1/audio/speech`, `POST /v1/audio/speech/stream`,
  `GET /v1/realtime`, `GET /v1/audio2face`. M2 ships stub handlers
  that validate route resolution and return placeholder bodies;
  per-runtime bridging arrives in M4–M11.
- `axum` workspace dep extended with the `ws` and `multipart`
  features required by the new routes.
- `PlacementConstraints` extended with `cpu_nodes` for `LocalCpu`
  deployments; `PlacementError::NoCpuNodes` covers the empty case.
  The `place()` wildcard is replaced with explicit arms for
  `LocalCpu` and `UnknownTransport`.
- New mock runners in `inference-testkit`: `MockTtsRunner`,
  `MockSttRunner`, `MockA2FRunner`, `MockRealtimeRunner`, each with a
  `*Script` value-struct following the existing `MockRunner` /
  `MockScript` shape (inter-chunk delay, deterministic chunk list,
  `fail_with: Option<InferenceError>` for error injection).
- New `MockWsServer` (`src/mock_ws.rs`) — in-process
  `tokio-tungstenite` server with `expect_binary_frames`,
  `send_transcript_chunk`, `close_with` helpers. Used by the WS
  provider crates landing in M7–M9.
- `MockOpenAi` extended with `mount_audio_speech_happy_path`,
  `mount_audio_transcriptions_happy_path`, `inject_audio_429`.
- New cross-cutting integration test:
  `inference-testkit/tests/audio_dispatch.rs` drives each modality
  through its engine actor end-to-end, covers modality-mismatch
  rejection, and exercises the consumer-dropped-mid-stream
  backpressure path.
- Architecture doc: `docs/architecture.md` gains a §5.9 "Audio
  modality dispatch" section describing the per-modality engine
  topology, the `AudioEngineCoreActor` discriminator pattern, and
  the realtime bridge task's lifecycle contract.

### Added — WebSocket transport crate (`atomr-infer-runtime-ws-core`)

M3 of the audio program of work. Carves the connection lifecycle out
of the to-be-built provider crates so Deepgram, AssemblyAI,
ElevenLabs, OpenAI Realtime, and Gemini Live (landing in M7–M9) share
one TLS-aware client, one reconnect policy, and one keepalive
implementation.

- New crate `atomr-infer-runtime-ws-core` at
  `crates/inference-runtime-ws-core/` with five modules:
  - `client.rs` — `WsClient::connect(url, timeout)` returning split
    `WsSender` / `WsReceiver` halves over
    `WebSocketStream<MaybeTlsStream<TcpStream>>`. Time-bounded
    connect, graceful close, no provider state.
  - `error.rs` — `WsError` (`Closed`, `ConnectTimeout`, `IdleTimeout`,
    `Tls`, `Io`, `Protocol`, `BadUrl`, `ReconnectExhausted`) +
    `is_retryable()`. `From<tungstenite::Error>` collapses the
    upstream surface into intent.
  - `frame.rs` — `Frame` (Binary / Text / Ping / Pong / Close) hides
    `tungstenite::Message` from providers, plus `coalesce_binary` for
    drop-oldest binary merging under upstream backpressure. Property
    tests pin relative-order preservation and non-binary frame
    boundary invariants.
  - `reconnect.rs` — `ReconnectEngine::next_delay` non-async backoff
    state machine wrapping `atomr_infer_remote_core::BackoffPolicy`
    with a `max_attempts` ceiling; first attempt is free, subsequent
    attempts consult `compute_backoff`.
  - `keepalive.rs` — `Keepalive::tick(now) → KeepaliveAction::{Idle,
    SendPing, Dead}` with explicit time injection so tests stay
    deterministic.
- `tokio-tungstenite` workspace dep extended with
  `rustls-tls-webpki-roots` so `wss://` URLs work out of the box.
- 22 inline unit tests + 3 integration tests
  (`tests/reconnect_loop.rs`): reconnect-after-one-drop recovery,
  reconnect exhaustion under sustained server-side hangups, and an
  end-to-end ping/pong round-trip with auto-pong from
  `tokio-tungstenite`'s server side.
- Reuses the existing M2 `MockWsServer` from `atomr-infer-testkit` for
  the happy-path round-trip test in `client.rs`.
- Docs: crate-level `//!` header + `# Examples` doctest on
  `WsClient::connect`; `docs/feature-matrix.md` gains an "Audio
  transport infrastructure" subsection.

### Added — Piper TTS runtime (`atomr-infer-runtime-piper`)

M4 of the audio program of work. Validates the M1 `SpeechRunner`
contract against the simplest local provider — a self-contained
char-level pipeline that needs no system libraries beyond ONNX
Runtime.

- New crate `atomr-infer-runtime-piper` at
  `crates/inference-runtime-piper/` with five modules:
  - `config.rs` — `PiperConfig` (voice path + manifest override +
    optional speaker id + length / noise / noise-w scales + chunk
    size + intra-op thread count, all serde-defaulted) and
    `PiperVoiceManifest` deserialiser for the sibling `.onnx.json`
    file rhasspy/piper voices ship with (`audio.sample_rate`,
    `inference.{length,noise,noise_w}_scale`, `phoneme_id_map`,
    `num_speakers`, optional `espeak`).
  - `phoneme.rs` — `PhonemeMap::ids_for_text` performs char-level
    lookup against the voice's `phoneme_id_map`, wraps the
    sequence in BOS (`^`) / EOS (`$`) markers and interleaves PAD
    (`_`) between symbols when those envelope ids are present in
    the voice. Returns `PiperError::UnknownPhoneme` on any
    character the voice doesn't ship an id for.
  - `error.rs` — `PiperError` (`VoiceNotFound`, `ManifestNotFound`,
    `ManifestIo`, `ManifestParse`, `UnknownPhoneme`,
    `SpeakerOutOfRange`, `Ort`, `FeatureDisabled`) with a
    `From<PiperError> for InferenceError` impl that routes user
    errors to `BadRequest` and system errors to `Internal`.
  - `session.rs` *(feature-gated `piper`)* — wraps an
    `ort::session::Session` behind `parking_lot::Mutex`, builds the
    Piper input tensors (`input` `i64[1,T]`, `input_lengths`
    `i64[1]`, `scales` `f32[3]`, optional `sid` `i64[1]`), runs the
    session, and extracts the first f32 output tensor as raw PCM.
  - `runner.rs` — `PiperRunner` implements `SpeechRunner` with
    `runtime_kind() = TextToSpeech` and `transport_kind() =
    LocalCpu`. Without the `piper` feature the runner builds and
    returns `InferenceError::Internal("piper feature disabled at
    build time")`. With the feature, `speak()` lazily initializes a
    `tokio::sync::OnceCell<Arc<PiperState>>`, runs inference under
    `spawn_blocking`, converts f32 PCM to 16-bit LE i16 with
    saturation, and streams `SpeechChunk`s of size
    `cfg.chunk_samples` with `is_final` set on the terminal chunk.
- Cargo features:
  - `piper` — pulls in `ort` (`std`, `download-binaries`,
    `copy-dylibs`, `tls-rustls`, `api-21`), `ndarray`, `tokio`,
    `tokio-stream`, `futures`, `parking_lot`, `tracing`.
  - `piper-cuda` — forwards ort's `cuda` execution provider.
  - `piper-load-dynamic` — forwards ort's `load-dynamic` so the
    binary looks for `libonnxruntime` at runtime instead of linking
    it.
- Workspace `Cargo.toml` gains the crate as a member + workspace
  dep; `examples/tts_piper_local` joins as a workspace member.
- Tests: 8 inline unit tests (config defaults + manifest path
  resolution + 3 manifest fixture parses + runner exposes correct
  kind/transport + feature-gated "missing voice file → BadRequest"
  + feature-disabled "Internal" smoke); 5 integration tests in
  `tests/manifest_round_trip.rs` (tempfile manifest round-trip,
  `hello` → `[BOS, _, h, _, e, _, l, _, l, _, o, _, EOS]` symbol
  expansion against a fixture map, multi-word with whitespace,
  envelope-absent fallback, explicit manifest override); 1
  env-gated `#[ignore]`'d smoke test in `tests/piper_smoke.rs`
  driving a real `.onnx` voice via `PIPER_VOICE_PATH` /
  `PIPER_PROMPT`.
- Example: `examples/tts_piper_local/` — workspace example with
  `Cargo.toml`, `README.md`, and `src/main.rs` that drains the
  `SpeechRunHandle` stream and writes a minimal 16-bit mono PCM
  WAV file using a hand-rolled 44-byte RIFF/WAVE header.
- Docs: crate-level `//!` doc with a build-profile table and an
  explicit M4-scope note that text→IPA via `espeak-ng` is a
  documented follow-up (the `phoneme::PhonemeMap` boundary is the
  seam where that lands); `docs/feature-matrix.md` gains an "Audio
  provider runtimes" subsection with the Piper row.

### Added — whisper.cpp local STT runtime (`atomr-infer-runtime-whisper-local`)

M5 of the audio program of work. First STT provider, first user of
the `AudioRunner` trait, and the validation case for the
`TransportKind::LocalCpu` placement path.

- New crate `atomr-infer-runtime-whisper-local` at
  `crates/inference-runtime-whisper-local/` with five modules:
  - `config.rs` — `WhisperConfig { model_path, language, n_threads,
    translate, word_timestamps, sample_rate_hz }`, serde-defaulted to
    match the upstream whisper.cpp CLI behavior (auto-detect language,
    no translation, no word timestamps, 16 kHz hard-coded).
  - `error.rs` — `WhisperError` (`ModelNotFound`, `ModelIo`,
    `UnsupportedAudio`, `StreamingNotSupported`, `Backend`,
    `FeatureDisabled`, `UnsupportedArch`) with a
    `From<WhisperError> for InferenceError` impl that routes
    user errors to `BadRequest` / `UnsupportedAudioFormat`, missing
    capabilities to `Unsupported`, and backend bugs to `Internal`.
  - `audio_decode.rs` — `payload_to_f32_pcm` materializes an
    `AudioPayload::{Bytes, Path}` into a contiguous `Vec<f32>` mono
    16 kHz buffer. Accepts `Pcm16Le`, `PcmF32Le`, and canonical
    PCM-16 / IEEE-FLOAT mono `Wav`; rejects stereo, off-rate, and
    `AudioPayload::Url` with `UnsupportedAudio`. Hand-rolled minimal
    RIFF walker — no `hound` dep yet to keep the crate's surface tight.
  - `session.rs` *(feature-gated `stt-whisper` + supported arch)* —
    wraps a `whisper_rs::WhisperContext` and `WhisperState` behind
    `parking_lot::Mutex`, runs `state.full(params, samples)`, and
    extracts each segment via the 0.16 `get_segment` / `WhisperSegment`
    API. Per-segment text + centisecond→ms timestamps; per-token
    word timings (with confidence) when `word_timestamps` is set,
    filtered to drop whisper.cpp's special `[_…]` / `<|…|>` tokens.
  - `runner.rs` — `WhisperRunner` implements `AudioRunner` with
    `runtime_kind() = SpeechToText` and `transport_kind() =
    LocalCpu`. Rejects `AudioOptions::Audio2Face` and
    `AudioInput::Stream` synchronously; on supported builds, runs
    inference under `spawn_blocking` and streams `TranscriptChunk`s
    via `futures::stream::iter`.
- Cargo features:
  - `stt-whisper` — pulls in `whisper-rs 0.16` (default features off),
    `tokio`, `tokio-stream`, `futures`, `parking_lot`, `tracing`.
  - `stt-whisper-cuda` / `-metal` / `-coreml` / `-vulkan` /
    `-openblas` — each implies `stt-whisper` and forwards the
    matching `whisper-rs` feature flag, so the C bindings can use
    accelerator backends without surfacing them through the
    placement layer (the runner still reports `LocalCpu`).
- Workspace `Cargo.toml` gains the crate as a member + workspace
  dep; `examples/stt_whisper_local` joins as a workspace member.
- Tests: 15 inline unit tests covering config defaults, serde
  round-trip, audio-decode happy paths and rejection paths for
  every unsupported permutation, runtime/transport-kind reporting,
  A2F-option rejection, `AudioInput::Stream` rejection, the
  feature-disabled "Internal" fallback, an arch-gate stub for
  non-x86_64/aarch64 hosts, and the feature-on "missing model →
  BadRequest" path; 4 integration tests in
  `tests/audio_decode_round_trip.rs` exercising `AudioPayload::Path`
  with PCM and WAV tempfiles plus the not-found and silence-decode
  cases; 1 env-gated `#[ignore]`'d smoke test in
  `tests/whisper_smoke.rs` that drives a real ggml model + WAV via
  `WHISPER_MODEL_PATH` / `WHISPER_FIXTURE_WAV`.
- Example: `examples/stt_whisper_local/` — workspace example with
  `Cargo.toml`, `README.md`, and `src/main.rs` that loads a WAV
  argument, runs the runner, and prints one line per segment with
  optional per-token timings when `WHISPER_WORD_TIMESTAMPS=1`.
- Docs: crate-level `//!` doc with a build-profile table and a
  per-arch support matrix explicitly documenting the
  `target_arch` gate; `docs/feature-matrix.md` "Audio provider
  runtimes" subsection gains the whisper-local row alongside Piper.

### Added — OpenAI TTS + STT runtimes (`atomr-infer-runtime-openai-tts`, `atomr-infer-runtime-openai-stt`)

M6 of the audio program of work. First remote audio providers and the
validation case for the "depend on the sibling OpenAI crate for
classification + auth surface" pattern. Lands as one PR since the two
crates share testkit fixtures and CHANGELOG narrative.

- New crate `atomr-infer-runtime-openai-tts` at
  `crates/inference-runtime-openai-tts/` with five modules:
  - `config.rs` — `OpenAiTtsConfig { endpoint, api_key (`SecretRef`
    re-exported from `atomr-infer-runtime-openai::config`),
    organization, project, chunk_bytes (default 8192 ≈ 170 ms at
    24 kHz mono PCM16), rate_limits, retry, circuit_breaker, timeouts
    }`. `defaults_for_openai`, `with_endpoint`, and `speech_url() ->
    endpoint.join("audio/speech")` helpers.
  - `wire.rs` — `SpeechRequest<'a>` (serializes `model`, `input`,
    `voice`, `response_format`, optional `speed`, optional
    `instructions` from `SynthOptions::emotion`); `voice_name`
    (passthrough for `Named`/`Id`, conservative fallback to `"alloy"`
    for `ClonedFrom` since OpenAI's TTS doesn't support cloning);
    `response_format_str` (maps `AudioFormat` → `"pcm"` /  `"mp3"` /
    `"opus"` / `"flac"` / `"wav"`, `None` for unsupported variants).
  - `cost.rs` — `per_million_chars_usd("tts-1") = 15.0`,
    `tts-1-hd = 30.0`, `gpt-4o-mini-tts = 12.0`; `estimate_usd(model,
    chars)` scales linearly.
  - `error.rs` — `OpenAiTtsError` (`UnsupportedFormat { message }`,
    `FeatureDisabled`) with `From<OpenAiTtsError> for InferenceError`.
  - `runner.rs` — `OpenAiTtsRunner` implements `SpeechRunner` with
    `runtime_kind() = TextToSpeech` and `transport_kind() =
    RemoteNetwork { provider: OpenAi }`. POSTs `SpeechRequest` JSON
    to `speech_url`; classifies non-2xx via
    `atomr_infer_runtime_openai::error::classify_openai_error`;
    materializes the full audio body (OpenAI's `/v1/audio/speech` is
    one-shot, not per-frame chunked) and re-chunks at `chunk_bytes`
    boundaries before emitting `SpeechChunk`s with the terminal one
    marked `is_final = true`.
- New crate `atomr-infer-runtime-openai-stt` at
  `crates/inference-runtime-openai-stt/` with five modules:
  - `config.rs` — symmetric to `OpenAiTtsConfig`;
    `transcriptions_url() -> endpoint.join("audio/transcriptions")`.
  - `wire.rs` — `PlainResponse { text }`, `VerboseResponse { text,
    segments: Vec<VerboseSegment>, words: Vec<VerboseWord> }` with
    optional `avg_logprob` captured for future cost/quality estimation.
  - `cost.rs` — `per_minute_usd("whisper-1") = 0.006`,
    `gpt-4o-transcribe = 0.006`, `gpt-4o-mini-transcribe = 0.003`;
    `estimate_usd(model, audio_seconds)` scales linearly.
  - `error.rs` — `OpenAiSttError` (`FeatureDisabled`,
    `Unsupported { method }`, `BadRequest { message }`) with
    `From<OpenAiSttError> for InferenceError`.
  - `runner.rs` — `OpenAiSttRunner` implements `AudioRunner` with
    `runtime_kind() = SpeechToText` and `transport_kind() =
    RemoteNetwork { provider: OpenAi }`. Builds a `multipart/form-data`
    body via `reqwest::multipart::Form` (model + response_format +
    optional language/prompt/temperature + `file` part with
    `application/octet-stream`); auto-selects `verbose_json` (with
    `timestamp_granularities[]=word&segment`) when
    `TranscribeOptions::word_timestamps` or `interim_results` is set,
    otherwise `json`. `AudioInput::Stream` and `AudioPayload::Url`
    return `Unsupported`; `AudioOptions::Audio2Face` is rejected
    early with `BadRequest`. `verbose_json` responses split into one
    `TranscriptChunk` per segment, with words attributed by
    time-range overlap and the last segment marked `is_final = true`.
- Cargo features: `tts-openai` / `stt-openai` each gate the real HTTP
  bodies; without them both crates compile to stubs whose entry
  points return `Internal("…-openai feature disabled at build time")`,
  so workspace-wide `cargo build` stays clean on no-default-features.
- Workspace `Cargo.toml` gains both crates as members + workspace
  deps; `reqwest`'s workspace features pick up `multipart` so the
  STT crate's form encoder is available.
- Testkit: `MockOpenAi` (`inference-testkit/src/mock_openai.rs`)
  gains three audio helpers — `mount_audio_speech_happy_path(server,
  audio_bytes)`, `mount_audio_transcriptions_happy_path(server,
  transcript)`, and `inject_audio_429(server)` — matching the shape
  of the existing chat-completions helpers and reused by both new
  crates' integration tests.
- Tests: 10 inline unit tests in `atomr-infer-runtime-openai-tts`
  (config defaults / with_endpoint / serde round-trip, cost rate +
  scaling + unknown-model, wire `SpeechRequest` serialization +
  voice-cloning fallback + format-variant coverage, runner kind /
  transport reporting); 13 inline unit tests in
  `atomr-infer-runtime-openai-stt` (config defaults / with_endpoint /
  serde round-trip, cost rate + scaling + unknown-model, wire
  `PlainResponse` / `VerboseResponse` decoding with and without
  segments / words, runner default-filename coverage + verbose →
  per-segment chunk attribution + `is_final` tail-marking); 4
  wiremock-driven integration tests per crate in
  `tests/tts_wiremock.rs` and `tests/stt_wiremock.rs` covering the
  happy path (body bytes round-trip / single final chunk),
  auth-header forwarding, chunk-boundary respect (TTS) / multipart
  body content (STT), `verbose_json` per-segment splitting (STT),
  and 429 → `InferenceError::RateLimited { provider: OpenAi, .. }`
  classification.
- Docs: crate-level `//!` doc on both crates with a build-profile
  table linking to the upstream OpenAI API reference;
  `docs/feature-matrix.md` "Audio provider runtimes" subsection
  gains the `openai-tts` and `openai-stt` rows alongside Piper and
  whisper-local.

### Added — ElevenLabs TTS runtime (`atomr-infer-runtime-elevenlabs`)

M7 of the audio program of work. First real consumer of the
`inference-runtime-ws-core` shared WebSocket transport and the first
provider to expose voice-cloning via a separate multipart-upload
round-trip.

- New crate `atomr-infer-runtime-elevenlabs` (path
  `crates/inference-runtime-elevenlabs`) with the standard per-provider
  module layout (`config.rs`, `cost.rs`, `error.rs`, `wire.rs`,
  `runner.rs`, `lib.rs`):
  - `config.rs` — `ElevenLabsTtsConfig` carries the HTTPS + WSS base
    URLs, the `ElevenLabsSecret { Env { name }, File { path } }`
    credential reference, an optional `default_voice_id`, the
    `chunk_bytes` HTTPS re-chunk boundary (default 8192 ≈ 170 ms at
    24 kHz mono PCM16), a `ws_connect_timeout` (default 5 s), and the
    shared `RateLimits` / `RetryPolicy` / `CircuitBreakerConfig` /
    `Timeouts` blobs. `defaults_for_elevenlabs(secret)` points at the
    public `https://api.elevenlabs.io/v1/` / `wss://api.elevenlabs.io/v1/`
    bases; `with_endpoint` / `with_ws_endpoint` override for tests and
    corporate proxies. `speech_url(voice_id)`,
    `speech_stream_url(voice_id)`, and `add_voice_url()` build the
    three concrete request URLs.
  - `cost.rs` — `per_million_chars_usd(model)` returns approximate
    Creator-tier USD per million characters (`eleven_multilingual_v2 →
    300`, `eleven_turbo_v2_5 → 150`, `eleven_flash_v2_5 → 50`, unknown
    → `0.0` so estimation degrades safely). `estimate_usd(model,
    chars)` scales linearly.
  - `wire.rs` — `SpeechRequest<'a> { model_id, text, voice_settings }`
    for the HTTPS body and `VoiceSettings { stability, similarity_boost,
    style, use_speaker_boost }`; WS protocol types
    `WsInitMessage<'a> { text, model_id, xi_api_key, voice_settings,
    generation_config }` (the init frame ElevenLabs requires up front),
    `WsTextMessage<'a> { text, try_trigger_generation }` (subsequent
    text frames + the empty-text flush), and `WsInboundFrame { audio,
    alignment, is_final }` decoding base64 audio + optional
    `WsAlignment { chars, char_start_times_ms, char_durations_ms }`.
  - `error.rs` — `ElevenLabsError { FeatureDisabled, Unsupported,
    BadRequest, UnsupportedFormat }` with `From<ElevenLabsError> for
    InferenceError`.
  - `runner.rs` — `ElevenLabsTtsRunner` implements `SpeechRunner`
    with `runtime_kind() = TextToSpeech` and `transport_kind() =
    RemoteNetwork { provider: ElevenLabs }`. Routes on
    `SpeechBatch::stream`: `false` → HTTPS one-shot (POST with
    `xi-api-key` header + `output_format=...` query, response body
    re-chunked at `chunk_bytes`, terminal `SpeechChunk` carries
    `is_final = true`); `true` → WSS streaming via
    `WsClient::connect` (init message → empty-text flush → drain
    inbound JSON frames → base64-decode audio → optional
    `AlignmentDelta` from `WsAlignment` when `emit_alignment` is on →
    one `SpeechChunk` per non-ping frame). Silent ping frames are
    filtered out. `output_format` is auto-derived from
    `SynthOptions::format` (`Mp3 → mp3_44100_128`, `Pcm16Le → pcm_24000`,
    `OggOpus → opus_48000_64`); `Wav` / `Flac` / `PcmF32Le` return
    `UnsupportedAudioFormat`. Voice cloning is **explicit**: callers
    invoke `clone_voice(name, sample, description)` which POSTs
    multipart to `/v1/voices/add` and returns the new voice id;
    `VoiceRef::ClonedFrom(_)` passed to `speak` falls back to
    `default_voice_id` (or `BadRequest` if unset) rather than silently
    re-uploading on every batch. 429 responses classify into
    `RateLimited { provider: ElevenLabs, retry_after }`; 401/403 split
    into `Unauthorized` / `Forbidden`; 5xx into `ServerError { status,
    body }`.
- Cargo feature: `tts-elevenlabs` gates the real HTTP / WS client
  deps; without it the crate compiles to a stub whose `speak` returns
  `Internal("tts-elevenlabs feature disabled at build time")`.
- Workspace `Cargo.toml` gains the new crate as both a member and
  workspace dep; new workspace dep `base64 = "0.22"` (the WS path
  decodes `WsInboundFrame::audio` strings).
- Tests: 18 inline unit tests (config defaults / `with_endpoint` /
  serde round-trip, cost rate + scaling + unknown-model, wire
  `SpeechRequest` + `WsInitMessage` + `WsInboundFrame` serialization,
  runner kind / transport reporting, 401 / 429 / 503 classification,
  `output_format` mapping + rejection, `WsAlignment → AlignmentDelta`
  conversion); 5 wiremock-driven integration tests in
  `tests/elevenlabs_wiremock.rs` covering the HTTPS happy path
  (concatenated bytes match fixture), `xi-api-key` + body forwarding,
  `chunk_bytes` boundary respect, 429 → `RateLimited { provider:
  ElevenLabs }`, and voice-cloning multipart upload returning a new
  voice id; 3 `MockWsServer`-driven integration tests in
  `tests/elevenlabs_ws.rs` covering the WS init+flush sequence with
  credentials, the alignment round-trip across two chunks, and the
  silent-ping filter.
- Example: `examples/tts_elevenlabs_alignment/` (new workspace
  member). Streams real ElevenLabs WS TTS and prints per-character
  alignment frames as they arrive. Reads `ELEVEN_API_KEY`; CLI args
  control the text and voice id.
- Docs: crate-level `//!` doc with a build-profile table and the
  voice-id / model-id reference; `docs/feature-matrix.md` "Audio
  provider runtimes" subsection gains the `elevenlabs` row alongside
  the OpenAI audio crates.

### Added — Deepgram STT runtime (`atomr-infer-runtime-deepgram`)

First half of M8 of the audio program of work (`FR-STT-001`). Adds a
WebSocket-streaming Deepgram runner over the shared
`atomr-infer-runtime-ws-core` transport, alongside the first
`Authorization`-header-bearing WSS upgrade path.

- New crate `atomr-infer-runtime-deepgram` (path
  `crates/inference-runtime-deepgram`) with the standard per-provider
  module layout (`config.rs`, `cost.rs`, `error.rs`, `wire.rs`,
  `runner.rs`, `lib.rs`):
  - `config.rs` — `DeepgramSttConfig` carries the WSS base URL, the
    `DeepgramSecret { Env { name }, File { path } }` credential
    reference, a `ws_connect_timeout` (default 5 s), the `smart_format`
    toggle, and the shared `RateLimits` / `RetryPolicy` /
    `CircuitBreakerConfig` / `Timeouts` blobs.
    `defaults_for_deepgram(secret)` points at the public
    `wss://api.deepgram.com/v1/` base; `with_ws_endpoint` overrides
    for tests and corporate proxies. `listen_url()` resolves to
    `<base>/listen`.
  - `cost.rs` — `per_minute_usd(model)` returns approximate
    pay-as-you-go USD per audio-minute (`nova-2`, `nova-2-phonecall`,
    `nova-2-meeting`, `nova`, `enhanced`, `base`; unknown → `0.0` so
    estimation degrades safely). `estimate_usd(model, seconds)`
    scales linearly.
  - `wire.rs` — `InboundEnvelope` tagged enum (`Results` |
    `Metadata` | `Other` for forward-compat) with
    `ResultsEnvelope { start, duration, is_final, speech_final,
    channel: ResultsChannel { alternatives: Vec<TranscriptAlternative
    { transcript, confidence, words: Vec<DeepgramWord { word,
    punctuated_word, start, end, confidence, speaker }> }> } }` for
    the downlink and `CloseStream { type_: "CloseStream" }` for the
    uplink flush marker.
  - `error.rs` — `DeepgramError { FeatureDisabled, Unsupported,
    BadRequest, UnsupportedFormat }` with `From<DeepgramError> for
    InferenceError`.
  - `runner.rs` — `DeepgramSttRunner` implements `AudioRunner` with
    `runtime_kind() = SpeechToText` and `transport_kind() =
    RemoteNetwork { provider: Deepgram }`. Builds the query string
    from `TranscribeOptions` (`model`, `encoding`, `sample_rate`,
    `channels`, `interim_results`, `diarize`, `smart_format`,
    `language`, fixed `endpointing=300`), attaches an `Authorization:
    Token <key>` header via `WsClient::connect_with_headers`, then
    spawns split uplink + downlink tasks. The uplink task drains
    `AudioInput::{Static, Stream}` into ≈128 ms (4096 B) binary
    frames and emits the JSON `{"type":"CloseStream"}` flush marker
    once the source drains. The downlink decodes `Results` envelopes,
    filters interim chunks when `TranscribeOptions::interim_results
    == false`, surfaces per-word `WordTiming`s when `word_timestamps`
    is on, and stringifies the first word's `speaker` label into
    `TranscriptChunk::speaker_id` when `diarize` is on.
    `TranscriptChunk::is_final` follows the provider's *utterance*
    boundary (`speech_final`), not the segment boundary (`is_final`).
    Abnormal close codes (not 1000/1005/1006) surface as
    `InferenceError::NetworkError`.
  - `lib.rs` — crate-level `//!` doc covering the interim / final /
    speech-final progression, the contrast with AssemblyAI's
    one-final-per-turn cadence, and the `endpointing=300` rationale.
- Audio-format mapping: `Pcm16Le → linear16`, `PcmF32Le → linear32`,
  `OggOpus → opus`, `Mp3` / `Flac` / `Wav` pass through;
  `Pcm24Le` rejected as `UnsupportedAudioFormat` (Deepgram requires
  16-bit PCM when the codec is `linear16`).
- Cargo feature: `stt-deepgram` gates the real WS client deps;
  without it the crate compiles to a stub whose `execute_audio`
  returns `Internal("stt-deepgram feature disabled at build
  time")`.
- Workspace `Cargo.toml` gains the new crate as both a member and
  workspace dep.
- `inference-runtime-ws-core` gains a new
  `WsClient::connect_with_headers(url, &[(name, value)], timeout)`
  method (the existing `connect` delegates to it with an empty
  header set). Uses `tungstenite`'s `IntoClientRequest` to inject
  `Authorization` (and any future provider-specific headers) onto
  the WSS upgrade request. Required for Deepgram, AssemblyAI, and
  the upcoming Gemini Live runner.
- Tests: 19 inline unit tests in feature-on mode (config defaults +
  override + serde round-trip, cost rate + scaling + unknown-model,
  wire `InboundEnvelope` interim / final / metadata / unknown-type
  decoding + `CloseStream` serialization, runner kind / transport
  reporting, encoding mapping + `Pcm24Le` rejection, diarize +
  word-timestamps surfacing, missing-alternative graceful fallback)
  plus 7 stub-mode tests; 5 `MockWsServer`-driven integration tests
  in `tests/deepgram_ws.rs` covering the interim → final
  progression, interim filtering when callers disable them,
  diarization + word-timestamp round-trip, the `CloseStream` marker
  emission after the uplink drains, and abnormal-close →
  `NetworkError` surfacing.
- Docs: crate-level `//!` doc with a build-profile table and the
  interim-vs-final / speech-final / endpointing reference;
  `docs/feature-matrix.md` "Audio provider runtimes" subsection
  gains the `deepgram` row alongside the OpenAI / ElevenLabs audio
  crates.

### Added — AssemblyAI STT runtime (`atomr-infer-runtime-assemblyai`)

Second half of M8 of the audio program of work (`FR-STT-001`). Adds a
WebSocket-streaming AssemblyAI runner against the Universal-Streaming
v3 protocol on top of the shared `atomr-infer-runtime-ws-core`
transport.

- New crate `atomr-infer-runtime-assemblyai` (path
  `crates/inference-runtime-assemblyai`) with the standard
  per-provider module layout (`config.rs`, `cost.rs`, `error.rs`,
  `wire.rs`, `runner.rs`, `lib.rs`):
  - `config.rs` — `AssemblyAiSttConfig` carries the WSS base URL, the
    `AssemblyAiSecret { Env { name }, File { path } }` credential
    reference, a `ws_connect_timeout` (default 5 s), the
    `format_turns` toggle (Punctuated & Formatted text on turn-final
    updates; off by default because it adds latency), and the
    shared `RateLimits` / `RetryPolicy` / `CircuitBreakerConfig` /
    `Timeouts` blobs. `defaults_for_assemblyai(secret)` points at
    the public `wss://streaming.assemblyai.com/` base;
    `with_ws_endpoint` overrides for tests and corporate proxies.
    `listen_url()` resolves to `<base>/v3/ws`.
  - `cost.rs` — `per_hour_usd(model)` returns approximate
    Universal-Streaming USD per audio-hour (`universal` /
    `universal-streaming` / `universal-v3` → `0.15`; unknown →
    `0.0`). `estimate_usd(model, audio_seconds)` scales linearly.
  - `wire.rs` — `InboundEnvelope` tagged enum (`Begin` | `Turn` |
    `Termination` | `Other` for forward-compat). `TurnEnvelope`
    carries `turn_order`, `turn_is_formatted`, `end_of_turn`,
    `end_of_turn_confidence`, `transcript`, and per-token
    `AssemblyWord { text, start, end, confidence, word_is_final }`
    (v3 already gives `start`/`end` in milliseconds, not seconds).
    `Terminate { type_: "Terminate" }` for the uplink flush marker.
  - `error.rs` — `AssemblyAiError { FeatureDisabled, Unsupported,
    BadRequest, UnsupportedFormat }` with `From<AssemblyAiError>
    for InferenceError`.
  - `runner.rs` — `AssemblyAiSttRunner` implements `AudioRunner`
    with `runtime_kind() = SpeechToText` and `transport_kind() =
    RemoteNetwork { provider: AssemblyAi }`. Connect-time format
    gate: rejects anything other than `Pcm16Le` mono up front with
    `UnsupportedAudioFormat` (v3 only accepts 16-bit PCM). Builds
    the query string from `TranscribeOptions` and config
    (`sample_rate`, optional `format_turns`), attaches an
    `Authorization: <key>` header (no `Token` prefix, contrast with
    Deepgram) via `WsClient::connect_with_headers`, then spawns
    split uplink + downlink tasks. The uplink task drains
    `AudioInput::{Static, Stream}` into ≈128 ms (4096 B) binary
    frames and emits the JSON `{"type":"Terminate"}` flush marker
    once the source drains. The downlink decodes `Turn` envelopes,
    filters partial chunks when `TranscribeOptions::interim_results
    == false`, surfaces per-word `WordTiming`s when `word_timestamps`
    is on, and ignores `Begin` / `Termination` envelopes.
    `TranscribeOptions::diarize` is silently ignored — v3 Streaming
    has no speaker labels on the wire (their async API does).
    `TranscriptChunk::is_final` follows `end_of_turn`. Abnormal
    close codes (not 1000/1005/1006) surface as
    `InferenceError::NetworkError`.
  - `lib.rs` — crate-level `//!` doc covering the partial / final
    cadence, the contrast with Deepgram's interim / segment-final /
    speech-final, and the Pcm16Le-only constraint.
- Cargo feature: `stt-assemblyai` gates the real WS client deps;
  without it the crate compiles to a stub whose `execute_audio`
  returns `Internal("stt-assemblyai feature disabled at build
  time")`.
- Workspace `Cargo.toml` gains the new crate as both a member and
  workspace dep.
- Tests: 20 inline unit tests in feature-on mode (config defaults +
  override + serde round-trip, cost rate + scaling + unknown-model,
  wire `InboundEnvelope` Begin / Turn-partial / Turn-final-formatted
  / Termination / unknown-type decoding + `Terminate` serialization,
  runner kind / transport reporting, format Pcm16Le accepted +
  every other variant rejected, turn-with-words / turn-without-words
  / empty-words-zeros-timing conversions, `AssemblyAiError`
  mapping) plus 7 stub-mode tests; 6 `MockWsServer`-driven
  integration tests in `tests/assemblyai_ws.rs` covering the
  partial → final progression, partial filtering when callers
  disable them, word-timestamp surfacing, the `Terminate` marker
  emission after the uplink drains, abnormal-close → `NetworkError`
  surfacing, and non-Pcm16 format rejection before connect.
- Docs: crate-level `//!` doc with a build-profile table and the
  partial-vs-final / Pcm16Le-only / format_turns reference;
  `docs/feature-matrix.md` "Audio provider runtimes" subsection
  gains the `assemblyai` row alongside the Deepgram / OpenAI /
  ElevenLabs audio crates.

### Added — OpenAI Realtime runtime (`atomr-infer-runtime-openai-realtime`)

M9-A of the audio program of work (FR-TTS-001). New crate implementing
`RealtimeRunner` against the OpenAI Realtime WebSocket API
(`wss://api.openai.com/v1/realtime?model=<model>`).

- `OpenAiRealtimeRunner::open_session(RealtimeBatch)` connects via WSS
  with `Authorization: Bearer <key>` and `OpenAI-Beta: realtime=v1`,
  sends `session.update` to configure voice / audio format, then spawns
  a bidirectional adapter task (uplink + downlink in one `tokio::select!`
  loop) wrapped in `futures::future::abortable`.
- Inbound translation: `RealtimeIn::AudioFrame` → `input_audio_buffer.append`
  (base64 PCM); `RealtimeIn::Text` → `conversation.item.create` +
  `response.create`; `RealtimeIn::Commit` → `input_audio_buffer.commit`;
  `RealtimeIn::Interrupt` → `response.cancel`; `RealtimeIn::Close` → close.
- Outbound translation: `response.audio.delta` → `RealtimeOut::AudioFrame`
  (decoded base64 PCM16-LE 24 kHz mono); `response.audio_transcript.done`
  → `Transcript { role: Assistant, is_final: true }`; `response.done` →
  `RealtimeOut::Done`; `error` → `RealtimeOut::Error`.
- `VoiceRef::ClonedFrom` returns `BadRequest` immediately (voice cloning
  unsupported on this API).
- Build profiles: no-feature build returns `Internal` from all methods;
  `tts-openai-realtime` enables the full WSS adapter.
- Tests: 30 inline unit tests (config, cost, wire, error, runner kinds)
  + 8 integration tests against `MockWsServer`.
- `MockWsServer` extended with `send_text(&str)` and `recv_json(timeout)`
  helpers to support the new test surface.

### Added — Gemini Live runtime (`atomr-infer-runtime-gemini-live`)

M9-B of the audio program of work (FR-TTS-001). New crate implementing
`RealtimeRunner` against the Google Gemini Live `BidiGenerateContent`
WebSocket endpoint.

- Feature gate `tts-gemini-live`; without it the crate compiles to a stub.
- API-key-in-URL auth (`?key=<api_key>`) — no Authorization header needed,
  contrasting with OpenAI Realtime.
- Setup handshake: runner sends `BidiGenerateContentSetup` (audio response
  modality) and waits for `setupComplete` before forwarding user input.
- Uplink: `RealtimeIn::Text` → `clientContent` turns; `RealtimeIn::AudioFrame`
  → `realtimeInput.mediaChunks` (Pcm16Le only; others → `UnsupportedAudioFormat`);
  `RealtimeIn::Commit` → `clientContent{turnComplete:true}`; `RealtimeIn::Interrupt`
  → `InferenceError::Unsupported` (not supported by this provider).
- Downlink: inline audio PCM → `RealtimeOut::AudioFrame`; text parts → interim
  `Transcript`; `turnComplete:true` → final `Transcript`; top-level `error` →
  `ServerError`.
- `VoiceRef::Named`/`Id` accepted; `VoiceRef::ClonedFrom` → `BadRequest`.
- Cost helpers: `per_minute_usd` (0.0 for `gemini-2.0-flash-exp`, 0.05 placeholder
  for others) and `per_million_tokens_usd` (always 0.0 — audio billing is
  time-based).
- Tests: 24 inline unit tests + 4 `MockWsServer`-driven integration tests
  (setup handshake, text turn round-trip, audio frame round-trip, cancellation).

### Added — Kokoro / XTTS / MOSS local TTS runtimes

M10 of the audio program of work (FR-TTS-001). Three new local TTS
runtime crates. All implement `SpeechRunner` and follow the established
Piper crate pattern: real synthesis path under a feature gate, stub
(returns `InferenceError::Internal`) without the feature, env-gated
smoke test.

- **`atomr-infer-runtime-kokoro`** (`--features tts-kokoro`): Kokoro-82M
  open-weight ONNX TTS. `KokoroConfig { voice_pack_dir, default_voice }`.
  Emits PCM16-LE at 24 000 Hz. Sub-features: `tts-kokoro-cuda`,
  `tts-kokoro-load-dynamic`. Smoke test gated on `KOKORO_VOICE_PATH`.
- **`atomr-infer-runtime-xtts`** (`--features tts-xtts`): Coqui XTTS-v2
  cross-lingual voice-cloning TTS. `XttsConfig { model_path, default_language }`
  — language defaults to `"en"`. `VoiceRef::ClonedFrom` is the primary use
  case; speaker-embedding computation is stubbed in M10 (validation + warning
  logged, zero embedding used). Sub-features: `tts-xtts-cuda`,
  `tts-xtts-load-dynamic`. Smoke test gated on `XTTS_MODEL_PATH`.
- **`atomr-infer-runtime-moss`** (`--features tts-moss`): MOSS-TTSD
  transformer TTS. Linux-only: on non-Linux hosts the feature compiles
  cleanly but `speak` returns `Internal("tts-moss requires Linux")`.
  `MossConfig { model_dir, default_voice }`. Smoke test gated on
  `MOSS_MODEL_PATH`.

All three: `runtime_kind()` → `TextToSpeech`; `transport_kind()` →
`LocalCpu`; `cost::per_million_chars_usd` → `0.0`; inline tests covering
config defaults, error mapping, voice-name validation, non-PCM format
rejection, feature-disabled stub.

### Added — NVIDIA Audio2Face-3D runtime (`atomr-infer-runtime-audio2face`)

M11 of the audio program of work (FR-A2F-001). New crate implementing
`A2FRunner` for NVIDIA Omniverse Audio2Face-3D.

- `Audio2FaceRunner` ingests `AudioBatch` with `AudioOptions::Audio2Face`
  and emits a stream of `BlendshapeChunk`s with 52 ARKit-canonical
  blendshape weights.
- `arkit::ARKIT_BLENDSHAPE_NAMES` — the 52-element Apple
  `ARBlendShapeLocation` ordered constant, with inline tests verifying
  count and uniqueness.
- `arkit::a2f_to_arkit` — A2F-native → ARKit index adapter (currently
  passthrough; `TODO(a2f-normalization)` marker documents the gap for the
  real NVIDIA mapping).
- Architecture gate: the `audio2face` feature requires Linux x86_64 at
  runtime; other platforms return `Internal("audio2face requires Linux
  x86_64")`.
- In-memory blendshape generator (deterministic, sine-driven) stands in
  for the live gRPC transport; `TODO(grpc-transport)` markers document the
  scaffold.
- Hand-written `prost` message types (`PushAudioRequest`,
  `BlendshapeOutput`) — no `protoc` or `tonic-build` needed.
- Workspace deps added: `tonic 0.12` (transport + tls + codegen + prost
  features) and `prost 0.13`.
- Tests: 22 inline unit tests + 4 integration tests + 1 doc test.

### Added — umbrella feature aggregation + Python parity (audio program of work)

M12 of the audio program of work. Wires the per-provider crates into
the umbrella facade, the Python wheel, the verify gate, and CI.

- Umbrella feature flags in `atomr-infer` (`crates/inference/Cargo.toml`):
  one `tts-*` / `stt-*` / `audio2face` per provider, plus aggregates
  `tts-remote-all`, `tts-local-all`, `tts-all`, `stt-remote-all`,
  `stt-local-all`, `stt-all`, `audio-all`. `tts-local-all` and
  `stt-local-all` now roll into `all-native`; `tts-remote-all` and
  `stt-remote-all` roll into `all-remote` and `remote-only`.
  `audio2face` stays out of `all-runtimes` because it carries a hard
  CPU-arch gate (Linux x86_64).
- Feature-gated re-exports in `inference/src/lib.rs`:
  `runtime_tts_piper`, `runtime_tts_openai`, `runtime_tts_elevenlabs`,
  `runtime_tts_gemini_live`, `runtime_tts_kokoro`, `runtime_tts_xtts`,
  `runtime_tts_moss`, `runtime_tts_openai_realtime`,
  `runtime_stt_whisper`, `runtime_stt_openai`, `runtime_stt_deepgram`,
  `runtime_stt_assemblyai`, `runtime_audio2face`. Each downstream
  consumer reaches its provider runner without a second `Cargo.toml`
  dep.
- `inference::prelude` now re-exports the audio type vocabulary
  (`AudioBatch`, `SpeechBatch`, `RealtimeBatch`, the four trait
  surfaces, the chunk and timing types) so callers can declare audio
  deployments with the same one-line `use atomr_infer::prelude::*;`
  as text deployments.
- Py-bindings (`inference-py-bindings`): new `audio` PyO3 submodule
  (`src/audio.rs`) exposing `AudioFormat`, `AudioParams`,
  `AudioPayload`, `VoiceRef`, `TranscribeOptions`, `SynthOptions`,
  `A2FOptions`, `SpeechBatch`, `AudioBatch`, `WordTiming`, `Viseme`,
  `AlignmentDelta`, `TranscriptChunk`, `SpeechChunk`,
  `BlendshapeChunk`. Registered under `atomr_infer._native.audio` and
  re-exported via `python/atomr_infer/audio.py`.
- Python tests: `python/tests/test_audio_dispatch.py` exercises
  construction + getter round-trip for all three modalities (TTS via
  `SpeechBatch`, STT via `AudioBatch.transcribe`, A2F via
  `AudioBatch.audio2face`) plus the seven `AudioFormat` class
  attributes and the facade `__all__`.
- Downstream migration guide: `docs/migrating-audio-modalities.md`
  walks `rustakka/atomr-agents` from in-tree clients onto the
  unified trait surface — `atomr-agents-tts-core`,
  `atomr-agents-stt-core`, `avatar-provider-audio2face` shim shapes,
  before/after diff examples, feature-flag cheatsheet.
- README: "What's in the box" table now lists the four audio runtime
  groups (`-{openai-tts,openai-realtime,elevenlabs,gemini-live}`,
  `-{piper,kokoro,xtts,moss}`, `-{openai-stt,whisper-local,deepgram,assemblyai}`,
  `-audio2face`) and the shared `-ws-core` transport crate.
- `docs/feature-matrix.md`: full audio rowset including supported
  archs, system deps (libonnxruntime, whisper.cpp, protoc), umbrella
  feature flag, and the `LocalCpu` / `RemoteNetwork` transport tag.
- xtask: new `verify-audio` subcommand builds `atomr-infer --features
  audio-all`, runs the `audio_dispatch` integration test, and rebuilds
  with `--features audio2face` to assert the arch-gated crate still
  compiles independently.
- CI (`.github/workflows/ci.yml`): the per-feature compilation matrix
  gains 17 new rows (every individual audio feature + the three
  meta-aggregates); a new `test-audio-fakes` job runs `cargo xtask
  verify-audio` after installing `protobuf-compiler`; the existing
  `verify` gate now depends on `test-audio-fakes`.

## [0.7.1] - 2026-05-09

### Added — full ONNX Runtime adapter (`atomr-infer-runtime-ort`)
- The `ort` runtime is no longer a stub. `ModelRunner::execute` now
  runs a real autoregressive generation loop on ONNX-exported causal
  LMs (HuggingFace Optimum-ONNX layout): tokenizer load, prefill +
  decode with KV cache, temperature / top-k / top-p sampling, stop
  strings, EOS detection, streaming `TokenChunk`s.
- `OrtRunner::infer(HashMap<String, InferTensor>) -> InferOutputs` is
  the low-level entry point for embeddings (BGE / E5), rerankers,
  Whisper encoders, and vision classifiers.
- New crate features: `ort-cuda` (CUDA EP), `ort-load-dynamic`
  (`ORT_DYLIB_PATH`), `ort-hf-hub` (`tokenizer.json` from HuggingFace).
  All forwarded from the `inference` rollup.
- `OrtConfig` extended with `device_id`, `tokenizer_path`, `hf_repo`,
  `intra_threads`, `default_max_new_tokens`.
- Topology probe is tolerant of HF export name variants
  (`past_key_values.0.key` vs `past.0.key`); fails with `BadRequest`
  echoing probed shape when the model isn't a recognised causal LM.
- Workspace deps added: `tokenizers 0.20`, `ndarray 0.16`,
  `tokio-stream 0.1`, `rand 0.8`, `half 2`, `regex 1`.
- Tests: config round-trip (no-feature), inline `runtime_kind` /
  `transport_kind`, `cpu_smoke` and `textgen_smoke` integration
  tests gated on env-var paths to local ONNX fixtures.

### Changed — track upstream atomr 0.6.0 + atomr-accel 0.3.3
- Workspace pins bumped: `atomr-* = "0.3.1"` → `"0.6.0"` and
  `atomr-accel-* = "0.3.0"` → `"0.3.3"`. The path-dep clones at
  `../atomr` and `../atomr-accel` are already at these versions
  upstream; the pin gap was cosmetic (Cargo.lock had already
  resolved `atomr-accel` to 0.3.3 transitively, and 0.3.3 itself
  pulls in `atomr-* = "0.6.0"` for its own deps).
- `RELEASING.md` allowlist paragraph refreshed to match.

### Added — Python parity wave (`inference-py-bindings`)
- Bindings restructured into hierarchical submodules
  (`atomr_infer._native.{core,runtime,config,errors,cluster}`)
  matching upstream `atomr/pycore`'s layout.
- `core` exposes `Deployment` (full fields w/ setters),
  `ExecuteBatch`, `Message`, `MessageContent`, `Role`,
  `ContentPart`, `SamplingParams`, `TokenChunk`, `Tokens`,
  `TokenUsage`, `FinishReason`, `CostEstimate`, `Replica`.
- `runtime` exposes `RuntimeKind`, `RuntimeConfig`,
  `ProviderKind`, `TransportKind`, `CircuitBreakerConfig`,
  `JitterKind` (string-tagged-enum pattern from upstream).
- `config` exposes `Serving`, `RateLimits`, `RetryPolicy`,
  `Timeouts`, `Budget`, `BudgetAction`, `CapacityPolicy`.
- `errors` exposes a Python exception hierarchy mirroring
  `inference_core::error::InferenceError` variants
  (`InferenceError` base ← `RateLimited`, `CircuitOpen`,
  `BadRequest`, …).
- `Cluster.deploy(deployment)` is now real for remote runtimes
  (OpenAI / Anthropic / Gemini / LiteLLM) and the testkit
  `MockRunner`; local-GPU runtimes return a `BadRequest`.
- `Cluster.execute(name, batch)` is async (asyncio interop via
  `pyo3-async-runtimes::tokio::future_into_py`); drains the
  `RunHandle` stream into a single `Tokens`.
- `Cluster.execute_stream(name, batch)` returns an async
  iterator yielding `TokenChunk` per `__anext__`.
- Pure-Python facade re-exports the new surface; `__version__`
  is now sourced from `importlib.metadata`.

## [0.6.5] — 2026-05-06

### Fixed — release pipeline: publish dep order + crates.io rate-limit headroom
- v0.6.4 published 16 of 18 crates (everything from `core` through
  `testkit`); the rollup `atomr-infer` and `atomr-infer-cli` were
  left out:
  1. `release.yml` had a stale `inference` (legacy short name) at
     the end of the dep-order list. Renamed to `atomr-infer`.
  2. `atomr-infer-cli` depends on the `atomr-infer` rollup but
     was listed *before* it. Reordered so the rollup publishes
     first, then the CLI.
- v0.6.4's run also spent ~12 extra minutes on 429-rate-limit
  retries: the 30s inter-publish sleep wasn't enough to stay under
  crates.io's rolling per-minute cap. Bumped to 60s so a
  full-workspace publish (~18 crates) completes without 429
  backoffs in the typical case.

## [0.6.4] — 2026-05-06

### Fixed — crates.io metadata for the remaining 8 crates
- `cargo publish` rejects a crate with `missing or empty metadata
  fields: description`. Eight crates were missing `description`
  (and the standard publish-metadata block) in their own
  `Cargo.toml`: `inference-cli`, `inference-pipeline`,
  `inference-py-bindings`, `inference-python-bridge`,
  `inference-runtime-candle`, `inference-runtime-cudarc`,
  `inference-runtime-ort`, `inference-testkit`.
- Added per-crate `description`, `repository.workspace = true`,
  `homepage.workspace = true`, `authors.workspace = true`,
  `keywords.workspace = true`, `categories.workspace = true`,
  `readme = "README.md"` to each. README files already existed for
  every crate.
- v0.6.3 partially shipped before failing here:
  `atomr-infer-core 0.6.3` and `atomr-infer-runtime 0.6.3` are on
  crates.io. The publish loop's idempotent "already uploaded"
  handler will skip those during the v0.6.4 attempt and continue.

## [0.6.3] — 2026-05-06

### Added — full workspace publishes to crates.io
- Upstream `atomr` family is now at **0.3.1** and `atomr-accel`
  family at **0.3.3** on crates.io, which means every inference-*
  crate's dep graph resolves cleanly from the registry. The publish
  allowlist is now empty (= publish all 18 crates in dep order).
- `cargo xtask release-checklist` reports 18 / 18 publishable, 0
  gated. Sibling-workspace path deps in `Cargo.toml` remain as
  reference-only for local development; they're stripped at
  publish time.
- `RELEASING.md` documents the new state and the version-pin
  compatibility (we pin `atomr-* = "0.3.1"` and `atomr-accel-* =
  "0.3.0"`; both accept the published 0.3.x lines).

## [0.6.2] — 2026-05-06

### Fixed — crates.io publish allowlist now reflects transitive deps
- `release.yml`'s `DEFAULT_PUBLISH_ALLOWLIST` was overstating what
  could ship. The previous list (7 crates) included
  `atomr-infer-runtime` and the four remote runners, but those
  transitively declare `atomr-*` deps that are not yet on
  crates.io — so `cargo publish` fails on them. The v0.6.1 publish
  job hit this: `atomr-infer-core` shipped, then
  `atomr-infer-runtime` failed with
  `failed to select a version for the requirement
   atomr-accel = "^0.3.0"; candidate versions found: 0.1.0`.
- Trimmed the default allowlist to **just `atomr-infer-core`** —
  the only crate whose entire `[dependencies]` section resolves
  from crates.io alone. Sibling-workspace path deps to `atomr` and
  `atomr-accel` are reference-only for planning and local
  development; they don't change what crates.io accepts.
- `cargo xtask release-checklist` now accounts for transitive
  upstream-`atomr-*` deps and lists only `atomr-infer-core` as
  publishable today; the other 17 crates are gated with a
  per-crate reason.
- `RELEASING.md` updated to match. Expand the allowlist as upstream
  ships 0.3.x crates to crates.io.

## [0.6.1] — 2026-05-06

### Fixed — retry publish that never fired
- The version-bump bot tagged v0.5.0 and v0.6.0 using `GITHUB_TOKEN`,
  which (per GitHub's downstream-workflow security default) does not
  trigger workflows that fire on tag pushes. The `release.yml`
  workflow's publish jobs are gated on
  `github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')`,
  so neither tag actually shipped to crates.io / PyPI / GitHub
  Releases. v0.6.1 is tagged and pushed from a developer machine so
  the publish pipeline actually fires. No source changes vs v0.6.0;
  this is purely a CI-infrastructure retry.

## [0.6.0] — 2026-05-05

### Added — native aarch64-Linux wheels
- PyPI now ships pre-built wheels for `aarch64-unknown-linux-gnu`
  and `aarch64-unknown-linux-musl`, built natively on GitHub-hosted
  ARM runners (`ubuntu-22.04-arm`). Mirrors the upstream atomr
  v0.3.1 pattern. Closes the gap where ARM Linux users had to
  install from sdist; native build avoids the `ring`/`aws-lc-rs`
  cross-compile blocker that previously forced the skip.

  PyPI wheel coverage as of this release:

  | Platform              | Wheel  |
  |-----------------------|--------|
  | linux-gnu x86_64      | ✓      |
  | linux-musl x86_64     | ✓      |
  | linux-gnu aarch64     | ✓ new  |
  | linux-musl aarch64    | ✓ new  |
  | macOS universal2      | ✓      |
  | windows-msvc x86_64   | ✓      |

## [0.5.0] — 2026-05-05

### Added — zero-config local Gemma 4
- `gemma-default` feature on the rollup auto-provisions a
  `gemma-local` deployment through the native PyO3 vLLM runner.
  Default model is `google/gemma-4-E4B-it`; all four Gemma 4
  variants (`E2B`, `E2B-it`, `E4B`, `E4B-it`) are validated by an
  allow-list and reachable via `ATOMR_INFER_GEMMA_MODEL` or a
  `[defaults.gemma]` block. The env probe handles GPU / Python /
  vLLM / HF-token gracefully — missing prereq logs a one-line `info!`
  tip and continues without the deployment; insufficient VRAM hints
  at the matching smaller variant. Cache respects `$HF_HOME` /
  `$HF_HUB_CACHE` so multi-instance deployments share one on-disk
  model.
- New PyO3 `VllmEngine` wrapper (`crates/inference-runtime-vllm/src/engine.rs`)
  bridges vLLM's V1 `AsyncLLMEngine` behind the `ModelRunner` trait.
  Token streaming via `tokio::mpsc`; consumer-drop triggers
  `engine.abort(request_id)`; lazy initialisation so `VllmRunner::new`
  stays cheap.
- New `crates/inference-runtime-vllm/src/{hf_cache,probe,defaults}.rs`
  modules — pure Rust, no PyO3 — for cache resolution, env probe, and
  the `provision_if_ready` adapter that registers the deployment with
  the running `DeploymentManagerActor`.
- New env vars: `ATOMR_INFER_GEMMA_AUTO`,
  `ATOMR_INFER_GEMMA_MODEL`, `ATOMR_INFER_GEMMA_DEPLOYMENT`,
  `ATOMR_INFER_GEMMA_GPU_UTIL`, `ATOMR_INFER_GEMMA_MAX_LEN`. Documented
  in `docs/local-gemma.md`.
- New ai-skill: `atomr-infer-local-gemma`.
- The feature is **deliberately not in `default-prod`** — production
  builds shouldn't surprise-download a multi-GB model on first boot.
  Operators opt in via `--features gemma-default` on the CLI.

### Added — local perf harness (`examples/gemma_bench/`)
- New binary `gemma_bench` (workspace member, `publish = false`,
  required-features `gemma-default`) for TTFT / tokens-per-second
  measurements and perf experiments. Subcommands: `smoke`,
  `latency`, `throughput`, `sweep <knob>` (gpu-util, dtype,
  cuda-graphs, prefix-cache, chunked-prefill, concurrency,
  block-size, max-num-seqs), `experiments`, `compare`.
- New `#[ignore]`'d integration tests in
  `crates/inference-runtime-vllm/tests/gpu_smoke.rs` for GPU pass/
  fail. Run with
  `cargo test -p atomr-infer-runtime-vllm --features gemma-default -- --ignored --test-threads=1`.
- `VllmConfig` extended with the perf knobs the harness sweeps:
  `enforce_eager`, `enable_prefix_caching`, `enable_chunked_prefill`,
  `max_num_seqs`, `block_size`, `quantization`. All forwarded through
  `engine.rs` to `AsyncEngineArgs`.
- `engine::generate` now renders chat through the model's tokenizer
  template (`tokenizer.apply_chat_template`) so Gemma's
  `<start_of_turn>` format is used correctly. Falls back to the
  generic `<|role|>` format on older vLLM versions.

### Aligned with upstream atomr 0.3.1 / atomr-accel 0.3.0
- Bumped every `atomr-*` workspace dep from `version = "0.1.0"` to
  `version = "0.3.1"`, and every `atomr-accel*` dep to
  `version = "0.3.0"`. Path-resolution worked locally before; this
  closes the `cargo publish` / `cargo-semver-checks` gap.
- Migrated to the upstream `atomr-accel-cuda` split. The umbrella
  `atomr-accel` no longer ships a `cuda` feature in 0.3 — CUDA lives
  in its own sibling crate now. `inference-runtime/Cargo.toml`,
  `inference/Cargo.toml`, and the candle / cudarc runners were
  updated accordingly. Source-level paths
  (`atomr_accel::cuda::error::*`) were rewritten to
  `atomr_accel_cuda::error::*` in `worker.rs`.
- Added `atomr-accel-tensorrt` (Phase 8 of atomr-accel) as a
  workspace dep, gated behind the `tensorrt` feature.

### Added — TensorRT runner is no longer a stub
- `inference-runtime-tensorrt` now wires the upstream `TrtRuntime` /
  `TrtEngine` / `ExecutionContext` / `ExecutionBindings` types behind
  the `ModelRunner` trait. Engine plans are loaded eagerly at
  construction; the runtime / engine / context are built lazily on
  the first `execute` call so a runner can be instantiated on hosts
  that don't ship libnvinfer.
- New sub-features forwarded straight to upstream:
  `tensorrt-onnx`, `tensorrt-int8`, `tensorrt-fp8`,
  `tensorrt-plugin`, `tensorrt-link`. All are reachable from the
  rollup with the same names.
- `TensorRtRunner::enqueue(ExecutionBindings)` for callers that own
  the tokenisation / device-pointer staging path. The chat-style
  `ModelRunner::execute` returns a typed `InferenceError::Internal`
  pointing at this entry point until an LLM-aware adapter lands.
- New config fields: `precision: TrtPrecision` (Fp32 / Fp16 / Bf16 /
  Int8 / Fp8 / Best — mirrors `atomr_accel_tensorrt::Precision`)
  and `device_id: u32`.
- `TrtError -> InferenceError` mapping for the full upstream
  variant set (NotLinked / Build / Runtime / Execution / Onnx /
  Calibration / Plugin / Refit / NullEngine / InvalidArg).

### Added — Mistral.rs runner is no longer a stub
- `inference-runtime-mistralrs` now wires `mistralrs::TextModelBuilder`
  and `mistralrs::Model` behind the `ModelRunner` trait. Models load
  lazily on the first `execute` call (so HuggingFace downloads happen
  at request time, not at runner-construction time). Tokens stream
  back through a `tokio::mpsc` channel as `TokenChunk`s.
- New config fields: `model_id`, `quant` (ISQ value parsed via
  `mistralrs::parse_isq_value`), `hf_revision`, `force_cpu`,
  `max_num_seqs`.
- Note: mistralrs 0.8 declares MSRV 1.88. The atomr-infer workspace
  MSRV (1.78) only applies to remote-only / default-features builds;
  operators enabling this runner need a toolchain that satisfies
  mistralrs's own MSRV.

### Added — 1.0-readiness hardening
- `#[non_exhaustive]` on every public enum that callers might match
  on: `RuntimeKind`, `TransportKind`, `ProviderKind`, `JitterKind`,
  `Role`, `MessageContent`, `ContentPart`, `FinishReason`,
  `InferenceError`, `WeightSource`, `SessionRebuildCause`. This is
  a deliberate breaking-style hardening pass before 1.0 — downstream
  matches against these enums will need a `_` arm.
- `deny.toml` and a `cargo-deny` CI job covering the four
  cargo-deny checks (advisories / bans / licenses / sources).
- Per-backend `feature-matrix` CI job — twelve backends checked
  individually so a regression in one feature gate doesn't hide
  behind the workspace build.
- `tracing::instrument` decorators on every remote runner's `execute`
  so structured spans carry `request_id` and `model` automatically.

### Changed
- `inference` rollup re-export of the CUDA backend renamed: callers
  now reach the NVIDIA backend at `atomr_infer::accel_cuda::*` (was
  `atomr_infer::accel::cuda::*`). The old `cuda` / `cuda_patterns`
  back-compat aliases (marked for removal in 0.4) were dropped.
- `DeploymentManagerMsg::Apply` carries the full `Deployment` value
  inline; clippy's `large_enum_variant` lint is suppressed with a
  doc-commented justification (boxing would force every caller to
  wrap a short-lived mailbox message).

### Renamed
- `docs/rustakka-inference-architecture-v4.md` →
  `docs/architecture.md`. All doc cross-references and rustdoc links
  follow.

### Removed
- The legacy "rakka" naming has been swept out of every README,
  source comment, environment variable, sample TOML, ai-skills
  bundle, and architecture doc. The `RAKKA_INFERENCE_*` env vars in
  `xtask` and the release pipeline are now `ATOMR_INFER_*`.

## [0.4.0] — 2026-05-05

### Added
- Re-enabled the `atomr-accel` features after the upstream rename:
  the `accel` and `accel-patterns` features on the rollup pull in
  the upstream substrate again, the `local-gpu` feature on
  `atomr-infer-runtime` is wired, and the candle / cudarc runners
  declare optional `atomr-accel` deps. The atomr-accel version pins
  in `Cargo.toml` were left at `0.1.0` in this release; see the
  Unreleased entry for the corrective bump to `0.3.x`.

## [0.3.1] — 2026-05-05

### Fixed
- CI `release-notes` job greps against the `atomr-infer-` crate
  prefix instead of the legacy name, so version-bump release notes
  attach correctly.

## [0.3.0] — 2026-05-05

### Changed
- README rewritten to match the atomr formatting (top-level "Why...
  in Rust, now" framing + crate table + quick start (Rust) + quick
  start (Python) + layout). Remaining `inference-*` references in
  docs swept to `atomr-infer-*`.
- `xtask` verify steps now point at the `atomr-infer` rollup rather
  than the legacy `inference` crate name.

## [0.2.0] – [0.2.6] — 2026-04 to 2026-05

### Added
- PyPI publish pipeline: real wheels + sdist + OIDC trusted publisher.
- `pyproject.toml` version is now dynamic so PyPI tracks `Cargo.toml`.
- Workspace-wide `version = workspace.package.version` inheritance for
  every member crate; explicit description / metadata on every
  publishable crate.

### Changed
- Renamed publishable crates from `inference-*` to
  `atomr-infer-*` so the user-facing namespace matches the upstream
  atomr / atomr-accel naming.

### Renamed
- Project: `rakka-inference` → `atomr-infer`. Every namespace, every
  import, every doc reference. (See the Unreleased entry above for
  the final sweep of stragglers.)

## [0.1.0] — 2026-04

### Added
- Initial commit — the atomr-infer rollup, the per-backend runners
  (vLLM, TensorRT, ORT, candle, cudarc, mistralrs, OpenAI, Anthropic,
  Gemini, LiteLLM), the actor topology
  (`ApiGatewayActor` / `RequestActor` / `DpCoordinatorActor` /
  `EngineCoreActor` / `WorkerActor` / `ContextActor`), and the
  remote-core primitives (`RateLimiterActor` CRDT,
  `CircuitBreakerActor`, `RetryEngine`, SSE parser).
