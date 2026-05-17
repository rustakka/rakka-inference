# Migrating to the audio modality surface

**Audience:** downstream consumers of `atomr-infer` — primarily the
`rustakka/atomr-agents` workspace and any other repository that
embeds `atomr-infer` for inference dispatch and now wants to migrate
its in-tree TTS / STT / Audio2Face clients onto the unified
trait surface.

**Status:** stable. The trait surface, batch types, chunk types,
gateway routes, and feature flags described below ship as of
`atomr-infer` v0.9.0 (the audio program of work — milestones M1
through M12 of the plan that landed FR-TTS-001, FR-STT-001, and
FR-A2F-001).

**See also:**
- [`docs/audio-modalities.md`](audio-modalities.md) — the
  architectural decision record that explains *why* the trait split
  looks the way it does. This guide is the *how*.
- [`docs/feature-matrix.md`](feature-matrix.md) — the canonical list
  of every shipped runtime crate, its feature flag, and its supported
  host archs.

---

## Why migrate

The pre-migration state in `atomr-agents` was: separate in-tree
clients for ElevenLabs (TTS), Deepgram (STT), and the avatar pipeline
returning `Audio2FaceError::Blocked` because no embedded path existed.
Each client managed its own rate limiter, retry stack, connection
pool, and telemetry sink.

After migration:

- **One control plane.** Every audio call — local Piper, remote
  ElevenLabs, NVIDIA A2F — flows through the same
  `EngineCoreActor` (or modality sibling), so rate limits, retries,
  circuit breakers, supervision restart policy, and telemetry are
  centralized.
- **Provider polymorphism without code changes.** A
  `RuntimeKind::TextToSpeech` deployment can resolve to local Piper,
  Kokoro, XTTS, MOSS, OpenAI, ElevenLabs, or Gemini Live based on
  its `RuntimeConfig` body. The consumer code is the same.
- **Static typing per modality.** No `match` on chunk kind per frame
  — `SpeechRunner` yields `SpeechChunk`, `AudioRunner` (STT) yields
  `TranscriptChunk`, `A2FRunner` yields `BlendshapeChunk`. Compile-
  time guarantees.

---

## What changed in the public API

### New traits in `atomr_infer_core::runner`

```rust
#[async_trait] pub trait SpeechRunner   : Send + Sync { /* speak  */ }
#[async_trait] pub trait AudioRunner    : Send + Sync { /* STT    */ }
#[async_trait] pub trait A2FRunner      : Send + Sync { /* A2F    */ }
#[async_trait] pub trait RealtimeRunner : Send + Sync { /* bidi   */ }
```

Each is object-safe, returns a modality-specific `RunHandle`, and
exposes `runtime_kind()` + `transport_kind()` so the placement
manager can route requests to the right replica.

### New batch types in `atomr_infer_core::audio`

```rust
SpeechBatch    // TTS: text → SpeechChunk audio stream
AudioBatch     // STT or A2F: AudioInput → TranscriptChunk / BlendshapeChunk
RealtimeBatch  // bidi: paired mpsc channels for live audio + transcript
```

`AudioInput` is the unified input vocabulary: `Static(AudioPayload)`
for one-shot byte/path/URL inputs, or `Stream { params, rx }` for live
microphone-style inputs. `AudioPayload` is serializable; `AudioInput`
is not (the `Stream` variant owns a channel receiver).

### New chunk types

| Modality | Chunk struct | Key fields |
|---|---|---|
| TTS | `SpeechChunk` | `audio_pcm_chunk: Bytes`, `params: AudioParams`, `alignment: Option<AlignmentDelta>` |
| STT | `TranscriptChunk` | `text: String`, `is_final: bool`, `words: Vec<WordTiming>`, `speaker_id: Option<String>` |
| A2F | `BlendshapeChunk` | `weights: [f32; 52]` (ARKit canonical order), `timestamp_ms: u64` |
| Realtime out | `RealtimeOut` | `AudioFrame { … } | Transcript { … } | Alignment(…) | Error(…) | Done` |

### New `RuntimeKind` variants (ungated)

```rust
RuntimeKind::SpeechToText
RuntimeKind::TextToSpeech
RuntimeKind::RealtimeSpeech
RuntimeKind::Audio2Face
```

Always present in the enum — the provider *crates* are feature-gated,
not the discriminants. Pattern-match exhaustiveness is preserved
across `cfg`s.

### New `TransportKind` variants

```rust
TransportKind::LocalCpu        // whisper.cpp, Piper, Kokoro, XTTS
TransportKind::RemoteNetwork { provider: ProviderKind }
// ProviderKind gains: Deepgram, AssemblyAi, ElevenLabs, NvidiaA2F
```

---

## Migration recipe per downstream crate

### `atomr-agents-tts-core` (TTS consumer)

**Before:** in-tree `TextToSpeech` trait, one impl per provider,
each managing its own HTTP client and retry stack.

**After:** trait becomes a default-impl wrapper over
`Arc<dyn SpeechRunner>`.

```rust
// Cargo.toml
[dependencies]
atomr-infer-core    = { version = "0.9", features = ["audio"] }
atomr-infer         = { version = "0.9", features = ["tts-elevenlabs"] }
# or whichever provider the binary needs

// adapter.rs
use atomr_infer_core::audio::{SpeechBatch, SpeechChunk, SynthOptions, VoiceRef};
use atomr_infer_core::runner::{SpeechRunner, SpeechRunHandle};
use futures::StreamExt;
use std::sync::Arc;

pub struct TtsAdapter {
    runner: Arc<tokio::sync::Mutex<Box<dyn SpeechRunner>>>,
}

impl TtsAdapter {
    pub async fn synth(&self, text: &str, voice: VoiceRef)
        -> Result<Vec<SpeechChunk>, Box<dyn std::error::Error + Send + Sync>>
    {
        let batch = SpeechBatch {
            request_id: uuid::Uuid::new_v4().to_string(),
            model: "default".into(),
            text: text.into(),
            voice,
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: text.chars().count() as u32,
        };
        let mut runner = self.runner.lock().await;
        let handle: SpeechRunHandle = runner.speak(batch).await?;
        drop(runner);
        let mut stream = handle.into_stream();
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await {
            out.push(chunk?);
        }
        Ok(out)
    }
}
```

Provider selection moves up one level — the *binary's* `Cargo.toml`
selects the feature flag (`tts-elevenlabs`, `tts-piper`, etc.). The
shim itself stays provider-agnostic.

### `atomr-agents-stt-core` (STT consumer)

**Before:** in-tree `SpeechToText` trait, one impl per provider.

**After:** adapter accepts the agent's mic source as
`impl Stream<Item = Bytes>` and wraps an
`Arc<dyn AudioRunner>`. WS providers inherit reconnect semantics
from `atomr-infer-runtime-ws-core` automatically — Deepgram and
AssemblyAI both reconnect on transient WS close without consumer
involvement.

```rust
use atomr_infer_core::audio::{
    AudioBatch, AudioInput, AudioOptions, AudioParams, AudioFormat,
    TranscribeOptions, TranscriptChunk,
};
use atomr_infer_core::runner::{AudioRunner, AudioRunHandle};
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct SttAdapter {
    runner: Arc<tokio::sync::Mutex<Box<dyn AudioRunner>>>,
}

impl SttAdapter {
    pub async fn transcribe_stream(
        &self,
        mic_rx: mpsc::Receiver<bytes::Bytes>,
        sample_rate: u32,
        on_chunk: impl Fn(TranscriptChunk) + Send + 'static,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let batch = AudioBatch {
            request_id: uuid::Uuid::new_v4().to_string(),
            model: "default".into(),
            input: AudioInput::Stream {
                params: AudioParams::new(sample_rate, 1, AudioFormat::Pcm16Le),
                rx: mic_rx,
            },
            stream: true,
            options: AudioOptions::Transcribe(TranscribeOptions {
                interim_results: true,
                word_timestamps: true,
                ..Default::default()
            }),
            estimated_units: 0,
        };
        let mut runner = self.runner.lock().await;
        let handle: AudioRunHandle = runner.execute_audio(batch).await?;
        drop(runner);
        let mut stream = handle.into_stream();
        while let Some(chunk) = stream.next().await {
            on_chunk(chunk?);
        }
        Ok(())
    }
}
```

### `avatar-provider-audio2face` (A2F consumer)

**Before:** stub returning `Audio2FaceError::Blocked` — no embedded
path existed.

**After:** ~100-line adapter that constructs an `AudioBatch` with
`A2FOptions`, drives `A2FRunner::execute_audio2face`, and forwards
`BlendshapeChunk`s into the existing `AvatarSink` channel.

```rust
use atomr_infer_core::audio::{
    A2FOptions, AudioBatch, AudioInput, AudioOptions, AudioParams,
    AudioFormat, AudioPayload, BlendshapeChunk,
};
use atomr_infer_core::runner::{A2FRunner, A2FRunHandle};
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct A2FAdapter {
    runner: Arc<tokio::sync::Mutex<Box<dyn A2FRunner>>>,
}

impl A2FAdapter {
    pub async fn drive_avatar(
        &self,
        wav_bytes: bytes::Bytes,
        sink: mpsc::Sender<BlendshapeChunk>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let batch = AudioBatch {
            request_id: uuid::Uuid::new_v4().to_string(),
            model: "claire".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: wav_bytes,
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: true,
            options: AudioOptions::Audio2Face(A2FOptions {
                emotion_preset: None,
                ..Default::default()
            }),
            estimated_units: 0,
        };
        let mut runner = self.runner.lock().await;
        let handle: A2FRunHandle = runner.execute_audio2face(batch).await?;
        drop(runner);
        let mut stream = handle.into_stream();
        while let Some(chunk) = stream.next().await {
            let _ = sink.send(chunk?).await;
        }
        Ok(())
    }
}
```

On aarch64 dev hosts the `audio2face` feature does not compile (Linux
+ x86_64 only). `cfg`-fall-back to `MockA2FRunner` from `atomr-infer-
testkit` so the rest of the avatar pipeline keeps building locally.

---

## Feature flag selection cheatsheet

The umbrella crate (`atomr-infer`) re-exports each provider behind
its own flag. Aggregate flags exist for common groupings.

| Use case | Feature(s) to enable |
|---|---|
| Local TTS only (offline avatar, kiosk) | `tts-local-all` |
| Remote TTS only (low-latency cloud) | `tts-remote-all` |
| Any TTS | `tts-all` |
| Local STT only (privacy-sensitive) | `stt-local-all` |
| Remote STT only (cloud transcripts) | `stt-remote-all` |
| Any STT | `stt-all` |
| TTS + STT + A2F | `audio-all` |
| Specific provider | `tts-elevenlabs`, `stt-deepgram`, `audio2face`, etc. |

`audio2face` is **not** part of `audio-all` on hosts that fail the
arch gate; it remains opt-in there. On Linux x86_64, the umbrella
exposes it transitively through `audio-all`.

---

## Gateway endpoints

If the downstream consumer talks to `atomr-infer` over HTTP/WS
instead of linking the crate, the following gateway routes carry the
audio modalities:

| Route | Method | In | Out |
|---|---|---|---|
| `/v1/audio/transcriptions` | POST | `multipart/form-data` | `application/json` (OpenAI-shape) |
| `/v1/audio/transcriptions/stream` | GET → WS | binary PCM frames | JSON `TranscriptChunk` |
| `/v1/audio/speech` | POST | `application/json` | `audio/wav` chunked |
| `/v1/audio/speech/stream` | POST | `application/json` | `text/event-stream` (base64 PCM) |
| `/v1/realtime` | GET → WS | bidi JSON+binary | bidi JSON+binary |
| `/v1/audio2face` | GET → WS | binary PCM | binary blendshape frames |

The OpenAI-compatible shape on `/v1/audio/transcriptions` and
`/v1/audio/speech` is intentional — drop-in for any existing OpenAI
SDK pointed at the gateway.

---

## Python consumers

Python downstream consumers use `atomr_infer.audio` (added in M12):

```python
import asyncio
from atomr_infer import Cluster, Deployment, RuntimeKind
from atomr_infer.audio import (
    SpeechBatch, AudioBatch, A2FBatch,
    SynthOptions, TranscribeOptions, A2FOptions,
    VoiceRef, AudioParams,
)

cluster = Cluster.connect("inproc://test")
cluster.deploy(Deployment(name="tts", model="claribel-dora",
                          runtime=RuntimeKind.text_to_speech()))

async def speak(text: str):
    batch = SpeechBatch(
        request_id="r1", model="claribel-dora", text=text,
        voice=VoiceRef.named("rachel"),
        options=SynthOptions(), stream=True,
    )
    chunks = []
    async for chunk in cluster.speak_stream("tts", batch):
        chunks.append(chunk.audio_pcm_chunk)
    return b"".join(chunks)

audio_bytes = asyncio.run(speak("hello world"))
```

`Cluster` gains `speak`, `transcribe`, `audio2face`, `open_realtime`
methods mirroring the Rust trait surface. Streams use the
`__aiter__` / `__anext__` protocol — same shape as the existing
`TokenStream`.

---

## Breaking-change checklist

This program of work is **strictly additive** in `inference-core`:
no existing trait method, batch type, chunk type, runtime kind, or
gateway route changes signature. Downstream code that doesn't touch
the new audio surface keeps building.

The one near-breakage to be aware of:
`InferenceError` gains three new variants
(`Unsupported`, `UnsupportedAudioFormat`, `RealtimeClosed`). Any
downstream `match` on `InferenceError` without a `_ =>` wildcard
will need to grow new arms — but `InferenceError` is already
`#[non_exhaustive]`, so the breakage is contained to callers that
opted out of exhaustiveness.

---

## Need a runtime that isn't here yet?

Open an issue describing the provider — STT, TTS, A2F, or realtime —
the protocol (HTTPS, WSS, gRPC, local model), the supported audio
formats, and any rate-limit shape. New providers follow the
`crates/inference-runtime-<provider>/` template established by the
shipped audio crates and rarely take more than a week per provider.
