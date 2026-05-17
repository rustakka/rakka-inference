# Audio modalities — architecture decision record

**Status:** accepted (M1 of the audio program of work). \
**Driver FRs:** `FR-TTS-001`, `FR-STT-001`, `FR-A2F-001`. \
**Scope:** the cross-cutting design of how `atomr-infer` represents
text-to-speech, speech-to-text, bidirectional realtime speech, and
NVIDIA Audio2Face-3D blendshape streaming.

This document records the architectural decisions taken when extending
`inference-core` with the audio modality surface, and the rationale
for each. It is reference material — the implementation lives in
[`crates/inference-core/src/audio.rs`](../crates/inference-core/src/audio.rs)
and the sibling traits live in
[`crates/inference-core/src/runner.rs`](../crates/inference-core/src/runner.rs).

## Goals

- **One control plane.** Every audio call flows through `atomr-infer`'s
  existing actor topology — one rate limiter, one retry stack, one
  telemetry sink, one supervision tree, regardless of modality.
- **Per-modality static typing.** Consumers of TTS know they receive
  audio chunks; consumers of STT know they receive transcript chunks.
  No `match` on chunk type per frame.
- **Shared input vocabulary.** STT and A2F both ingest audio, so they
  share `AudioInput` / `AudioPayload` / `AudioParams`. TTS and
  alignment-emitting STT both produce `WordTiming` sequences.
- **Provider polymorphism.** A `RuntimeKind::SpeechToText` deployment
  can resolve to local whisper.cpp, OpenAI Whisper, Deepgram, or
  AssemblyAI depending on its `RuntimeConfig` body.

## Decisions

### 1. Sibling traits, not sibling methods on `ModelRunner`

The FRs literally propose adding `execute_audio` / `speak` /
`open_session` to `ModelRunner`. We chose the architecturally cleaner
alternative: **sibling traits** that live next to `ModelRunner` —
`AudioRunner`, `SpeechRunner`, `RealtimeRunner`, `A2FRunner`.

**Why:** the existing engine actor (`EngineCoreActor` in
`crates/inference-runtime/src/engine_core.rs`) owns a single
`AsyncMutex<Box<dyn ModelRunner>>` per replica. The runner lock is the
load-bearing serialization point. Extending `ModelRunner` with audio
methods would force the same lock to be held across
text-and-audio-and-realtime calls — a head-of-line blocking surface
with no upside. Sibling traits let each modality's engine actor be
monomorphic over `Box<dyn $Trait>`, preserving the per-replica
admission policy that's already in place for text.

**Cost paid:** the downstream shim (`atomr-agents-tts-core` etc.) must
choose between `Arc<dyn ModelRunner>` and `Arc<dyn SpeechRunner>`
explicitly. Bounded duplication; ~100 lines per shim.

### 2. Monomorphic `RunHandle`; sibling handles per modality

`RunHandle` stays `BoxStream<'static, InferenceResult<TokenChunk>>`.
We add `AudioRunHandle`, `SpeechRunHandle`, `A2FRunHandle`,
`RealtimeSession` as siblings.

**Why:** generalizing to `RunHandle<T>` would cascade into
`EngineCoreActor<T>`, `ActorRef<EngineCoreMsg<T>>`, the dp-coordinator,
placement, and py-bindings — most of which need to remain object-safe
to be stored in heterogeneous registries. The duplication cost (four
handle types) is bounded; the generic cost would be every dispatch
site monomorphized by chunk type forever.

### 3. One engine actor per modality

Four engine actor types in `inference-runtime`:

- `EngineCoreActor` — text (unchanged), `Box<dyn ModelRunner>`, tokens.
- `SpeechEngineCoreActor` — TTS, `Box<dyn SpeechRunner>`, characters.
- `AudioEngineCoreActor` — STT or A2F, `Box<dyn AudioRunner>` /
  `Box<dyn A2FRunner>`, audio-seconds or frames.
- `RealtimeEngineCoreActor` — bidi sessions,
  `Box<dyn RealtimeRunner>`, concurrent sessions.

**Why:** admission policy differs per modality. TTS counts characters
toward provider character quotas; STT counts audio-seconds; realtime
counts concurrent sessions; A2F counts frames. Sharing the engine
would force conditional logic on the hot path; splitting it lets each
actor body be straight-line.

### 4. Unified `AudioInput` / `AudioPayload` / `AudioParams`

STT and A2F both ingest audio, so they share a single
`AudioInput`/`AudioPayload`/`AudioParams` triple. `AudioInput` has two
variants — `Static(AudioPayload)` for one-shot calls and `Stream {
params, rx }` for live mic feeds.

**Cost paid:** `AudioInput::Stream` is not `Serialize`/`Deserialize`
(it owns an `mpsc::Receiver`). Consequently `AudioBatch` is not
serializable either. The static counterpart `AudioPayload` round-trips
through serde so deployments and replays can be persisted; runtime
adapters always go through `AudioPayload` when materializing a
static call.

### 5. `RuntimeKind` discriminants are ungated; provider crates are
feature-gated

We added `RuntimeKind::SpeechToText` / `TextToSpeech` /
`RealtimeSpeech` / `Audio2Face` directly to the enum, with no
`cfg(feature = "...")` gates.

**Why:** `RuntimeKind` is `#[non_exhaustive]` and is matched
exhaustively across most of the workspace and at every JSON config
boundary. Feature-gating its variants would fragment the serde shape
of every config-parsing site by feature combination forever. Instead
we gate the *crates* — `inference-runtime-deepgram`,
`inference-runtime-piper`, etc. — and accept that
`RuntimeKind::TextToSpeech` is part of the type even when no TTS
runtime crate is compiled in.

### 6. `mpsc::Receiver` + `mpsc::Sender` for realtime transport

`RealtimeBatch` carries `inbound: mpsc::Receiver<RealtimeIn>` and
`outbound: mpsc::Sender<RealtimeOut>`. The WebSocket adapter that
talks to OpenAI Realtime / Gemini Live lives in the per-provider
crate. The session bridge task — that owns both channels until either
closes — lives in `RealtimeEngineCoreActor`.

**Resolves:** FR-TTS-001 §8.3 open question on bidi transport.

### 7. Raw provider viseme id, not normalized

`Viseme { id: u8, ts_start_ms, ts_end_ms, weight }` carries the
provider's viseme identifier verbatim. We do **not** normalize across
provider viseme tables.

**Why:** premature normalization locks us to one viseme set before we
have two providers to triangulate against. Downstream consumers that
care about cross-provider portability can build a normalization layer
on top.

**Resolves:** FR-TTS-001 §8.2.

### 8. Word timing inline on `TranscriptChunk.words`

Per-word timestamps arrive on the same `TranscriptChunk` stream as
the transcript text, in `TranscriptChunk.words`. They do **not** travel
on a sibling stream.

**Why:** a single stream = a single backpressure surface. Splitting
into two streams would force consumers to synchronize them at the
sink. Empty `words: Vec<WordTiming>` is cheap; opt-in via
`TranscribeOptions::word_timestamps`.

**Resolves:** FR-STT-001 §7.3.

### 9. `Option<String>` emotion preset with documented common values

`SynthOptions::emotion` and `A2FOptions::emotion` are
`Option<String>`. Common values are documented in
`audio::emotion_presets` (NEUTRAL, HAPPY, SAD, ANGRY, CALM, EXCITED).

**Why:** provider preset lists evolve faster than enum definitions
ship. Adapters accept any string the caller passes; the constants
module is a hint for autocomplete-style UIs, not a constraint.

**Resolves:** FR-A2F-001 §8.1.

## Implementation status

| Milestone | Status | Surface |
|---|---|---|
| M1 — core types, sibling traits, taxonomy | **landed** | `inference-core` |
| M2 — engine actors, gateway scaffolding, mocks | pending | `inference-runtime`, `inference-testkit` |
| M3 — `inference-runtime-ws-core` | pending | new crate |
| M4 — first local TTS (`piper`) | pending | new crate |
| M5 — first local STT (`whisper-local`) | pending | new crate |
| M6 — first remote (`openai-tts` + `openai-stt`) | pending | new crates |
| M7–M11 — parallel provider crates | pending | new crates |
| M12 — umbrella features, py-bindings, migration guide | pending | `inference`, `inference-py-bindings` |
