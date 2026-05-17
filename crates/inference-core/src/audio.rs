//! Audio modality surface — shared types for STT, TTS, realtime
//! bidirectional speech, and Audio2Face blendshape streaming.
//!
//! This module defines the inputs, outputs, and option blobs used by
//! the four audio sibling traits in [`crate::runner`]:
//!
//! - [`AudioRunner`](crate::runner::AudioRunner) — speech-to-text
//! - [`SpeechRunner`](crate::runner::SpeechRunner) — text-to-speech
//! - [`RealtimeRunner`](crate::runner::RealtimeRunner) — bidirectional
//!   sessions (OpenAI Realtime, Gemini Live)
//! - [`A2FRunner`](crate::runner::A2FRunner) — audio → ARKit blendshapes
//!
//! Source request IDs: `FR-TTS-001`, `FR-STT-001`, `FR-A2F-001`.
//!
//! # Why a single module
//!
//! STT and A2F both ingest audio, so they share [`AudioInput`],
//! [`AudioPayload`], [`AudioParams`], and [`AudioBatch`]. TTS and
//! alignment-emitting STT both produce [`WordTiming`] sequences. The
//! same primitive types compose into per-modality batches and chunks.
//!
//! # Serializability
//!
//! Live runtime types that carry channel handles ([`AudioInput::Stream`],
//! [`RealtimeBatch`]) are deliberately not `Serialize`/`Deserialize` —
//! their interpretation only makes sense inside a running actor system.
//! Their static-config counterparts ([`AudioPayload`], [`SpeechBatch`])
//! round-trip through serde so deployments and replays can be persisted.

use std::path::PathBuf;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::cost::CostEstimate;
use crate::error::InferenceError;

// ─────────────────────────────────────────────────────────────────────────────
// I/O primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Wire format of a PCM or compressed audio payload.
///
/// Used by both inbound ([`AudioBatch`]) and outbound ([`SpeechChunk`])
/// audio to describe sample encoding. Compressed formats (Opus, MP3,
/// FLAC) only appear at the gateway edge; runtimes typically convert
/// to a PCM variant before handing audio to a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AudioFormat {
    /// 16-bit signed little-endian PCM.
    Pcm16Le,
    /// 24-bit signed little-endian PCM.
    Pcm24Le,
    /// 32-bit float little-endian PCM.
    PcmF32Le,
    /// Ogg-encapsulated Opus.
    OggOpus,
    /// MPEG-1/2 Audio Layer III.
    Mp3,
    /// Free Lossless Audio Codec.
    Flac,
    /// WAV container (RIFF). Inner PCM variant is implicit from header.
    Wav,
}

impl AudioFormat {
    /// True for the linear PCM variants — runtimes can feed these
    /// directly to a model without decoding.
    pub fn is_pcm(self) -> bool {
        matches!(self, Self::Pcm16Le | Self::Pcm24Le | Self::PcmF32Le)
    }

    /// True for compressed/container formats — runtimes must decode
    /// before processing.
    pub fn requires_decode(self) -> bool {
        !self.is_pcm() && self != Self::Wav
    }
}

/// Audio container parameters — sample rate, channel layout, encoding.
///
/// Common values: 16 kHz mono Pcm16Le for STT, 24 kHz mono PcmF32Le for
/// most neural TTS, 48 kHz stereo OggOpus for browser-side delivery.
///
/// # Examples
///
/// ```
/// use atomr_infer_core::{AudioFormat, AudioParams};
///
/// let p = AudioParams::new(16_000, 1, AudioFormat::Pcm16Le);
/// assert!(p.is_valid());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AudioParams {
    /// Samples per second. Audio engines accept 8000–96000 Hz.
    pub sample_rate_hz: u32,
    /// Channel count. 1 = mono, 2 = stereo. Audio engines accept 1–8.
    pub channels: u8,
    /// Wire format of the underlying payload bytes.
    pub format: AudioFormat,
}

impl AudioParams {
    /// Construct without validation. Use [`is_valid`](Self::is_valid)
    /// to check the values are in the supported range before handing
    /// to a runtime.
    pub const fn new(sample_rate_hz: u32, channels: u8, format: AudioFormat) -> Self {
        Self {
            sample_rate_hz,
            channels,
            format,
        }
    }

    /// True iff `sample_rate_hz ∈ [8000, 96000]` and `channels ∈ [1, 8]`.
    pub fn is_valid(&self) -> bool {
        (8_000..=96_000).contains(&self.sample_rate_hz) && (1..=8).contains(&self.channels)
    }
}

/// Static audio payload — bytes, a file path, or a URL.
///
/// Serializable, suitable for config files and replay logs. The runtime
/// adapter is responsible for materializing path/url variants into bytes
/// before the model call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AudioPayload {
    /// In-memory audio bytes plus the parameters needed to decode them.
    Bytes {
        #[serde(with = "bytes_b64")]
        data: Bytes,
        params: AudioParams,
    },
    /// Filesystem path. The runtime opens and reads this when executed.
    Path { path: PathBuf, params: AudioParams },
    /// HTTP(S) URL. The runtime fetches it (subject to allow-list and
    /// circuit-breaker policy from [`crate::runtime`]).
    Url { url: url::Url, params: AudioParams },
}

impl AudioPayload {
    /// Borrow the audio parameters regardless of payload kind.
    pub fn params(&self) -> &AudioParams {
        match self {
            Self::Bytes { params, .. } | Self::Path { params, .. } | Self::Url { params, .. } => params,
        }
    }
}

/// Runtime-facing audio input — either a static payload or a live
/// channel of audio frames.
///
/// Deliberately **not** `Serialize`/`Deserialize`: the streaming variant
/// owns an [`mpsc::Receiver`] whose interpretation only makes sense
/// inside a running actor system. To persist an audio request, use
/// [`AudioPayload`] (the static counterpart).
pub enum AudioInput {
    /// One-shot, materialized payload.
    Static(AudioPayload),
    /// Live audio frames pushed by the caller (mic feed, network
    /// stream). The receiver yields raw bytes encoded per `params`.
    Stream {
        params: AudioParams,
        rx: mpsc::Receiver<Bytes>,
    },
}

impl AudioInput {
    /// Borrow the audio parameters regardless of input kind.
    pub fn params(&self) -> &AudioParams {
        match self {
            Self::Static(payload) => payload.params(),
            Self::Stream { params, .. } => params,
        }
    }
}

impl std::fmt::Debug for AudioInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(payload) => f.debug_tuple("Static").field(payload).finish(),
            Self::Stream { params, .. } => f
                .debug_struct("Stream")
                .field("params", params)
                .field("rx", &"<mpsc::Receiver<Bytes>>")
                .finish(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-modality option blobs
// ─────────────────────────────────────────────────────────────────────────────

/// Per-modality options carried alongside an [`AudioBatch`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "modality", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AudioOptions {
    /// Speech-to-text options.
    Transcribe(TranscribeOptions),
    /// Audio2Face blendshape generation options.
    Audio2Face(A2FOptions),
}

/// Speech-to-text request options.
///
/// Fields map directly to OpenAI / Whisper / Deepgram / AssemblyAI
/// parameters; not every provider honors every field — adapters drop
/// unsupported keys with a `tracing::warn!`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// ISO-639-1 language hint. `None` lets the provider auto-detect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Whether to emit interim (non-final) chunks during streaming.
    #[serde(default)]
    pub interim_results: bool,
    /// Whether to emit per-word timestamps inline on each
    /// [`TranscriptChunk`].
    #[serde(default)]
    pub word_timestamps: bool,
    /// Diarization — assign `speaker_id`s to transcript chunks. Off by
    /// default to avoid silent extra spend.
    #[serde(default)]
    pub diarize: bool,
    /// Vendor-specific prompt / vocabulary hint passed through verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Sampling temperature for stochastic ASR models (Whisper).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// Text-to-speech request options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynthOptions {
    /// Synthesis speed multiplier (1.0 = natural). Providers clamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Pitch shift in semitones. Providers clamp.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch_semitones: Option<f32>,
    /// Output sample rate hint. Providers may override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Output format hint. Providers may override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<AudioFormat>,
    /// Optional emotion preset (provider-specific). Common values
    /// listed in [`emotion_presets`]. Per FR-A2F-001 §8.1 we
    /// deliberately type this as a free `String` rather than an enum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    /// Optional language hint for multilingual TTS engines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// Audio2Face request options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct A2FOptions {
    /// Output frame rate. Defaults to 30 fps (the A2F-3D convention).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    /// Optional emotion preset (provider-specific). Common values
    /// listed in [`emotion_presets`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emotion: Option<String>,
    /// Smoothing window in milliseconds applied to the blendshape
    /// stream before emission. `None` leaves the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoothing_ms: Option<u32>,
    /// Strength multiplier in `[0, 1]`. `None` = provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
}

/// Documented common emotion preset names. Adapters accept any string
/// the caller passes; this list is a hint for downstream consumers
/// who want autocomplete-style UIs.
pub mod emotion_presets {
    /// `"neutral"` — no emotional inflection.
    pub const NEUTRAL: &str = "neutral";
    /// `"happy"` — bright, energetic.
    pub const HAPPY: &str = "happy";
    /// `"sad"` — subdued, slower.
    pub const SAD: &str = "sad";
    /// `"angry"` — tense, forceful.
    pub const ANGRY: &str = "angry";
    /// `"calm"` — measured, even.
    pub const CALM: &str = "calm";
    /// `"excited"` — uptempo, higher pitch.
    pub const EXCITED: &str = "excited";
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-modality batches
// ─────────────────────────────────────────────────────────────────────────────

/// One STT or A2F request handed to the engine.
///
/// Not `Serialize`/`Deserialize` because [`AudioInput::Stream`] holds an
/// `mpsc::Receiver`. To persist a request, construct from
/// [`AudioPayload`] and serialize that.
///
/// # Examples
///
/// ```
/// use atomr_infer_core::{
///     AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams,
///     AudioPayload, TranscribeOptions,
/// };
/// use bytes::Bytes;
///
/// let batch = AudioBatch {
///     request_id: "req-1".into(),
///     model: "whisper-1".into(),
///     input: AudioInput::Static(AudioPayload::Bytes {
///         data: Bytes::from_static(&[0u8; 0]),
///         params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
///     }),
///     stream: false,
///     options: AudioOptions::Transcribe(TranscribeOptions::default()),
///     estimated_units: 5,
/// };
/// assert_eq!(batch.request_id, "req-1");
/// ```
#[derive(Debug)]
pub struct AudioBatch {
    /// Identifier of the originating request. Surfaces back on every
    /// output chunk for correlation.
    pub request_id: String,
    /// Model name (provider-specific).
    pub model: String,
    /// Input audio. See [`AudioInput`].
    pub input: AudioInput,
    /// True if the caller wants chunk-by-chunk streaming.
    pub stream: bool,
    /// Per-modality options.
    pub options: AudioOptions,
    /// Estimated work in modality-specific units — audio-seconds for
    /// STT, frames for A2F. Used by admission control.
    pub estimated_units: u32,
}

/// One TTS request handed to the engine.
///
/// # Examples
///
/// ```
/// use atomr_infer_core::{SpeechBatch, SynthOptions, VoiceRef};
///
/// let batch = SpeechBatch {
///     request_id: "req-2".into(),
///     model: "tts-1".into(),
///     text: "hello world".into(),
///     voice: VoiceRef::Named("alloy".into()),
///     options: SynthOptions::default(),
///     stream: true,
///     emit_alignment: false,
///     estimated_characters: 11,
/// };
/// assert_eq!(batch.estimated_characters, 11);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechBatch {
    /// Identifier of the originating request.
    pub request_id: String,
    /// Model name (provider-specific).
    pub model: String,
    /// Text to synthesize.
    pub text: String,
    /// Voice selection.
    pub voice: VoiceRef,
    /// Synthesis options.
    #[serde(default)]
    pub options: SynthOptions,
    /// True if the caller wants chunk-by-chunk streaming.
    #[serde(default)]
    pub stream: bool,
    /// True if the caller wants per-word / per-viseme alignment frames
    /// alongside the audio.
    #[serde(default)]
    pub emit_alignment: bool,
    /// Estimated character count of `text`. Used by admission control.
    pub estimated_characters: u32,
}

/// One bidirectional realtime session (OpenAI Realtime, Gemini Live).
///
/// Not `Serialize`/`Deserialize` — holds mpsc channels owned by the
/// caller and the runtime adapter for the lifetime of the session.
pub struct RealtimeBatch {
    /// Identifier of the originating session.
    pub request_id: String,
    /// Model name (provider-specific).
    pub model: String,
    /// Voice selection for synthesized turns.
    pub voice: VoiceRef,
    /// Default synthesis options. Per-turn options may override.
    pub options: SynthOptions,
    /// Caller → runtime: input frames, text turns, control messages.
    pub inbound: mpsc::Receiver<RealtimeIn>,
    /// Runtime → caller: synthesized audio, transcripts, alignment.
    pub outbound: mpsc::Sender<RealtimeOut>,
}

impl RealtimeBatch {
    /// Convenience constructor accepting `Into<String>` for the
    /// identifiers. Mirrors the struct-literal form but easier on
    /// the eyes at call sites.
    pub fn new(
        request_id: impl Into<String>,
        model: impl Into<String>,
        voice: VoiceRef,
        options: SynthOptions,
        inbound: mpsc::Receiver<RealtimeIn>,
        outbound: mpsc::Sender<RealtimeOut>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            model: model.into(),
            voice,
            options,
            inbound,
            outbound,
        }
    }
}

impl std::fmt::Debug for RealtimeBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealtimeBatch")
            .field("request_id", &self.request_id)
            .field("model", &self.model)
            .field("voice", &self.voice)
            .field("options", &self.options)
            .field("inbound", &"<mpsc::Receiver<RealtimeIn>>")
            .field("outbound", &"<mpsc::Sender<RealtimeOut>>")
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-modality output chunks
// ─────────────────────────────────────────────────────────────────────────────

/// One streamed STT chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptChunk {
    pub request_id: String,
    /// True for the final chunk of a turn / utterance.
    pub is_final: bool,
    /// Transcript text delta (for streaming) or full text (for unary
    /// or final chunks).
    pub text: String,
    /// Start timestamp in milliseconds from the start of the audio
    /// input.
    pub ts_start_ms: u32,
    /// End timestamp in milliseconds from the start of the audio
    /// input.
    pub ts_end_ms: u32,
    /// Diarized speaker identifier. `None` when diarization is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
    /// Per-word timestamps when [`TranscribeOptions::word_timestamps`]
    /// is true. Per FR-STT-001 §7.3 these arrive inline rather than on
    /// a sibling stream.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordTiming>,
    /// Cost / accounting attribution for this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<CostEstimate>,
}

/// One streamed TTS chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechChunk {
    pub request_id: String,
    /// True for the terminal chunk of the synthesis.
    pub is_final: bool,
    /// PCM (or compressed, per `params.format`) audio bytes for this
    /// chunk.
    #[serde(with = "bytes_b64")]
    pub audio_pcm_chunk: Bytes,
    /// Wire format of `audio_pcm_chunk`.
    pub params: AudioParams,
    /// Optional alignment for the audio in this chunk (when the caller
    /// requested it via [`SpeechBatch::emit_alignment`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<AlignmentDelta>,
    /// Cost / accounting attribution for this chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<CostEstimate>,
}

/// One streamed A2F blendshape frame.
///
/// `weights` is in **ARKit canonical order** — 52 floats indexed by
/// Apple's [`BlendShapeLocation`] enum
/// (<https://developer.apple.com/documentation/arkit/arblendshapelocation>).
/// Provider adapters normalize from the A2F-native ordering into ARKit
/// canonical order at the wire boundary.
///
/// [`BlendShapeLocation`]: https://developer.apple.com/documentation/arkit/arblendshapelocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendshapeChunk {
    pub request_id: String,
    /// True for the terminal chunk of the stream.
    pub is_final: bool,
    /// Timestamp in milliseconds from the start of the input audio.
    pub timestamp_ms: u32,
    /// Blendshape weights in `[0, 1]`, ARKit canonical order.
    #[serde(with = "weights_52")]
    pub weights: [f32; 52],
}

// ─────────────────────────────────────────────────────────────────────────────
// Alignment / timing primitives
// ─────────────────────────────────────────────────────────────────────────────

/// A single word's timing — shared by STT transcripts and TTS alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTiming {
    /// The word itself.
    pub text: String,
    /// Start timestamp in milliseconds from the start of the audio.
    pub ts_start_ms: u32,
    /// End timestamp in milliseconds from the start of the audio.
    pub ts_end_ms: u32,
    /// Recognition confidence in `[0, 1]`, where present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl WordTiming {
    /// Invariant: `ts_start_ms <= ts_end_ms` and any `confidence` is
    /// inside `[0, 1]`.
    pub fn is_valid(&self) -> bool {
        self.ts_start_ms <= self.ts_end_ms && self.confidence.map_or(true, |c| (0.0..=1.0).contains(&c))
    }
}

/// One alignment delta — accompanies a [`SpeechChunk`] when the caller
/// asks for alignment, or arrives as a [`RealtimeOut::Alignment`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlignmentDelta {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<WordTiming>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visemes: Vec<Viseme>,
}

/// One viseme (mouth-shape) emission. Per FR-TTS-001 §8.2 we carry the
/// provider's raw viseme id rather than normalizing to a single canon
/// — providers ship different viseme tables and premature normalization
/// would lock us to one before we've seen two.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Viseme {
    /// Provider-specific viseme identifier.
    pub id: u8,
    pub ts_start_ms: u32,
    pub ts_end_ms: u32,
    /// Activation weight in `[0, 1]`.
    pub weight: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Voice selection
// ─────────────────────────────────────────────────────────────────────────────

/// How a TTS voice is selected.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum VoiceRef {
    /// Provider's catalog name (e.g. OpenAI `"alloy"`, ElevenLabs voice
    /// name).
    Named(String),
    /// Provider-specific opaque id (e.g. ElevenLabs voice id, Piper
    /// voice path).
    Id(String),
    /// Voice cloning from a reference audio sample. Provider must
    /// support cloning (XTTS, ElevenLabs).
    ClonedFrom(AudioPayload),
}

// ─────────────────────────────────────────────────────────────────────────────
// Realtime session messages
// ─────────────────────────────────────────────────────────────────────────────

/// One inbound message to a [`RealtimeBatch`] session.
#[derive(Debug)]
#[non_exhaustive]
pub enum RealtimeIn {
    /// Push a raw audio frame into the session.
    AudioFrame { pcm: Bytes, params: AudioParams },
    /// Inject a text turn (the model responds with audio).
    Text(String),
    /// Commit the current input turn — the provider should start
    /// responding.
    Commit,
    /// Cancel any in-flight response and clear input buffers.
    Interrupt,
    /// Tear down the session.
    Close,
}

/// One outbound message from a [`RealtimeBatch`] session.
#[derive(Debug)]
#[non_exhaustive]
pub enum RealtimeOut {
    /// One synthesized audio frame.
    AudioFrame { pcm: Bytes, params: AudioParams },
    /// One transcript delta (input or response, depending on `role`).
    Transcript {
        role: TranscriptRole,
        text: String,
        is_final: bool,
    },
    /// Alignment frames for the most recent audio output.
    Alignment(AlignmentDelta),
    /// Terminal error closing the session.
    Error(InferenceError),
    /// Session ended cleanly.
    Done,
}

/// Whether a [`RealtimeOut::Transcript`] is the user's input or the
/// model's response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TranscriptRole {
    User,
    Assistant,
}

// ─────────────────────────────────────────────────────────────────────────────
// serde adapters
// ─────────────────────────────────────────────────────────────────────────────

mod bytes_b64 {
    use bytes::Bytes;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::base64_compat as b64;

    pub fn serialize<S: Serializer>(b: &Bytes, s: S) -> Result<S::Ok, S::Error> {
        b64::encode(b).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Bytes, D::Error> {
        let s = String::deserialize(d)?;
        b64::decode(&s).map(Bytes::from).map_err(serde::de::Error::custom)
    }
}

/// Minimal RFC 4648 §4 base64 encoder/decoder, used for serializing
/// `Bytes` fields. Self-contained to avoid adding a dependency to
/// `inference-core` purely for serde audio support.
mod base64_compat {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(input: &[u8]) -> String {
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        let mut chunks = input.chunks_exact(3);
        for chunk in chunks.by_ref() {
            let n = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | (chunk[2] as u32);
            out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
            out.push(TABLE[(n & 0x3f) as usize] as char);
        }
        let rem = chunks.remainder();
        match rem.len() {
            1 => {
                let n = (rem[0] as u32) << 16;
                out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
                out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
                out.push('=');
                out.push('=');
            }
            2 => {
                let n = ((rem[0] as u32) << 16) | ((rem[1] as u32) << 8);
                out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
                out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
                out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
                out.push('=');
            }
            _ => {}
        }
        out
    }

    pub fn decode(input: &str) -> Result<Vec<u8>, &'static str> {
        let trimmed = input.trim_end_matches('=');
        let mut out = Vec::with_capacity(trimmed.len() * 3 / 4);
        let mut buf: u32 = 0;
        let mut bits: u32 = 0;
        for c in trimmed.bytes() {
            let v = match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                _ => return Err("invalid base64 character"),
            };
            buf = (buf << 6) | v as u32;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((buf >> bits) as u8);
                buf &= (1u32 << bits) - 1;
            }
        }
        Ok(out)
    }
}

mod weights_52 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(arr: &[f32; 52], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(arr.iter())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 52], D::Error> {
        let v = Vec::<f32>::deserialize(d)?;
        if v.len() != 52 {
            return Err(serde::de::Error::custom(format!(
                "expected 52 blendshape weights, got {}",
                v.len()
            )));
        }
        let mut out = [0.0_f32; 52];
        out.copy_from_slice(&v);
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn audio_format_classification() {
        assert!(AudioFormat::Pcm16Le.is_pcm());
        assert!(AudioFormat::PcmF32Le.is_pcm());
        assert!(!AudioFormat::OggOpus.is_pcm());
        assert!(AudioFormat::Mp3.requires_decode());
        assert!(!AudioFormat::Wav.requires_decode());
        assert!(!AudioFormat::Pcm16Le.requires_decode());
    }

    #[test]
    fn audio_params_validation() {
        assert!(AudioParams::new(16_000, 1, AudioFormat::Pcm16Le).is_valid());
        assert!(AudioParams::new(8_000, 2, AudioFormat::PcmF32Le).is_valid());
        assert!(AudioParams::new(96_000, 8, AudioFormat::Pcm24Le).is_valid());
        assert!(!AudioParams::new(7_999, 1, AudioFormat::Pcm16Le).is_valid());
        assert!(!AudioParams::new(96_001, 1, AudioFormat::Pcm16Le).is_valid());
        assert!(!AudioParams::new(16_000, 0, AudioFormat::Pcm16Le).is_valid());
        assert!(!AudioParams::new(16_000, 9, AudioFormat::Pcm16Le).is_valid());
    }

    #[test]
    fn audio_payload_borrows_params() {
        let params = AudioParams::new(16_000, 1, AudioFormat::Pcm16Le);
        let p = AudioPayload::Bytes {
            data: Bytes::from_static(b"\x00\x00"),
            params,
        };
        assert_eq!(p.params(), &params);
    }

    #[test]
    fn audio_input_borrows_params() {
        let params = AudioParams::new(24_000, 1, AudioFormat::Pcm16Le);
        let (_tx, rx) = mpsc::channel::<Bytes>(1);
        let input = AudioInput::Stream { params, rx };
        assert_eq!(input.params(), &params);

        let static_input = AudioInput::Static(AudioPayload::Bytes {
            data: Bytes::new(),
            params,
        });
        assert_eq!(static_input.params(), &params);
    }

    #[test]
    fn audio_payload_serde_round_trip() {
        let payload = AudioPayload::Bytes {
            data: Bytes::from_static(&[1, 2, 3, 4, 5]),
            params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let back: AudioPayload = serde_json::from_str(&json).unwrap();
        match back {
            AudioPayload::Bytes { data, params } => {
                assert_eq!(&*data, &[1, 2, 3, 4, 5]);
                assert_eq!(params, AudioParams::new(16_000, 1, AudioFormat::Pcm16Le));
            }
            _ => panic!("variant changed across round-trip"),
        }
    }

    #[test]
    fn audio_payload_url_serde_round_trip() {
        let payload = AudioPayload::Url {
            url: "https://example.com/audio.wav".parse().unwrap(),
            params: AudioParams::new(48_000, 2, AudioFormat::Wav),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let back: AudioPayload = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, AudioPayload::Url { .. }));
    }

    #[test]
    fn speech_batch_serde_round_trip() {
        let batch = SpeechBatch {
            request_id: "r".into(),
            model: "tts-1".into(),
            text: "hi".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions {
                speed: Some(1.1),
                emotion: Some(emotion_presets::HAPPY.into()),
                ..Default::default()
            },
            stream: true,
            emit_alignment: false,
            estimated_characters: 2,
        };
        let json = serde_json::to_string(&batch).unwrap();
        let back: SpeechBatch = serde_json::from_str(&json).unwrap();
        assert_eq!(back.text, "hi");
        assert_eq!(back.estimated_characters, 2);
        assert_eq!(back.options.speed, Some(1.1));
        assert_eq!(back.options.emotion.as_deref(), Some("happy"));
    }

    #[test]
    fn transcribe_options_serde_round_trip() {
        let opts = TranscribeOptions {
            language: Some("en".into()),
            interim_results: true,
            word_timestamps: true,
            diarize: true,
            prompt: Some("hint".into()),
            temperature: Some(0.0),
        };
        let json = serde_json::to_string(&opts).unwrap();
        let back: TranscribeOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.language.as_deref(), Some("en"));
        assert!(back.interim_results);
        assert!(back.diarize);
    }

    #[test]
    fn synth_options_serde_round_trip() {
        let opts = SynthOptions {
            speed: Some(0.9),
            pitch_semitones: Some(-1.0),
            sample_rate_hz: Some(24_000),
            format: Some(AudioFormat::PcmF32Le),
            emotion: Some("calm".into()),
            language: Some("en".into()),
        };
        let json = serde_json::to_string(&opts).unwrap();
        let back: SynthOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.format, Some(AudioFormat::PcmF32Le));
    }

    #[test]
    fn a2f_options_serde_round_trip() {
        let opts = A2FOptions {
            fps: Some(60),
            emotion: Some("excited".into()),
            smoothing_ms: Some(30),
            strength: Some(0.8),
        };
        let json = serde_json::to_string(&opts).unwrap();
        let back: A2FOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.fps, Some(60));
        assert_eq!(back.strength, Some(0.8));
    }

    #[test]
    fn audio_options_tagged_serde() {
        let opts = AudioOptions::Transcribe(TranscribeOptions {
            language: Some("ja".into()),
            ..Default::default()
        });
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"modality\":\"transcribe\""));
        let back: AudioOptions = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, AudioOptions::Transcribe(_)));
    }

    #[test]
    fn voice_ref_serde_round_trip() {
        let v = VoiceRef::Id("voice-xyz".into());
        let json = serde_json::to_string(&v).unwrap();
        let back: VoiceRef = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, VoiceRef::Id(s) if s == "voice-xyz"));
    }

    #[test]
    fn transcript_chunk_serde_round_trip_with_words() {
        let chunk = TranscriptChunk {
            request_id: "r".into(),
            is_final: true,
            text: "hello world".into(),
            ts_start_ms: 0,
            ts_end_ms: 1_000,
            speaker_id: Some("S1".into()),
            words: vec![
                WordTiming {
                    text: "hello".into(),
                    ts_start_ms: 0,
                    ts_end_ms: 500,
                    confidence: Some(0.95),
                },
                WordTiming {
                    text: "world".into(),
                    ts_start_ms: 500,
                    ts_end_ms: 1_000,
                    confidence: Some(0.88),
                },
            ],
            usage: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: TranscriptChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.words.len(), 2);
        assert_eq!(back.words[1].text, "world");
    }

    #[test]
    fn blendshape_chunk_serde_round_trip() {
        let mut weights = [0.0_f32; 52];
        weights[10] = 0.5;
        weights[51] = 1.0;
        let chunk = BlendshapeChunk {
            request_id: "r".into(),
            is_final: false,
            timestamp_ms: 33,
            weights,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        let back: BlendshapeChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(back.weights[10], 0.5);
        assert_eq!(back.weights[51], 1.0);
        assert_eq!(back.weights[0], 0.0);
    }

    #[test]
    fn blendshape_chunk_serde_rejects_wrong_length() {
        let bad = "{\"request_id\":\"r\",\"is_final\":false,\"timestamp_ms\":0,\"weights\":[0.0,0.0]}";
        let err = serde_json::from_str::<BlendshapeChunk>(bad).unwrap_err();
        assert!(err.to_string().contains("52"));
    }

    #[test]
    fn word_timing_invariants() {
        let good = WordTiming {
            text: "x".into(),
            ts_start_ms: 0,
            ts_end_ms: 100,
            confidence: Some(0.5),
        };
        assert!(good.is_valid());

        let bad_order = WordTiming {
            text: "x".into(),
            ts_start_ms: 200,
            ts_end_ms: 100,
            confidence: None,
        };
        assert!(!bad_order.is_valid());

        let bad_conf = WordTiming {
            text: "x".into(),
            ts_start_ms: 0,
            ts_end_ms: 100,
            confidence: Some(1.5),
        };
        assert!(!bad_conf.is_valid());
    }

    #[test]
    fn alignment_delta_round_trip() {
        let d = AlignmentDelta {
            words: vec![WordTiming {
                text: "hi".into(),
                ts_start_ms: 0,
                ts_end_ms: 200,
                confidence: None,
            }],
            visemes: vec![Viseme {
                id: 3,
                ts_start_ms: 0,
                ts_end_ms: 100,
                weight: 0.7,
            }],
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: AlignmentDelta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.words.len(), 1);
        assert_eq!(back.visemes[0].id, 3);
    }

    #[test]
    fn realtime_batch_constructs() {
        let (_tx_in, rx_in) = mpsc::channel::<RealtimeIn>(4);
        let (tx_out, _rx_out) = mpsc::channel::<RealtimeOut>(4);
        let batch = RealtimeBatch {
            request_id: "rt".into(),
            model: "gpt-4o-realtime".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions::default(),
            inbound: rx_in,
            outbound: tx_out,
        };
        assert_eq!(batch.request_id, "rt");
        // Debug impl doesn't panic on channel fields:
        let _ = format!("{batch:?}");
    }

    #[test]
    fn realtime_messages_construct() {
        let _in_frame = RealtimeIn::AudioFrame {
            pcm: Bytes::from_static(&[0; 4]),
            params: AudioParams::new(24_000, 1, AudioFormat::Pcm16Le),
        };
        let _in_text = RealtimeIn::Text("hi".into());
        let _in_close = RealtimeIn::Close;
        let _out_done = RealtimeOut::Done;
    }

    #[test]
    fn base64_round_trip() {
        let cases: &[&[u8]] = &[b"", b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"];
        for &c in cases {
            let s = base64_compat::encode(c);
            let back = base64_compat::decode(&s).unwrap();
            assert_eq!(back, c, "round trip failed for {c:?} (encoded as {s:?})");
        }
    }

    proptest! {
        #[test]
        fn prop_audio_params_validity(
            rate in 0u32..200_000,
            channels in 0u8..16,
        ) {
            let p = AudioParams::new(rate, channels, AudioFormat::Pcm16Le);
            let expected = (8_000..=96_000).contains(&rate) && (1..=8).contains(&channels);
            prop_assert_eq!(p.is_valid(), expected);
        }

        #[test]
        fn prop_word_timing_validity(
            start in 0u32..1_000_000,
            end in 0u32..1_000_000,
            conf_raw in -1.0_f32..2.0,
            conf_some in any::<bool>(),
        ) {
            let w = WordTiming {
                text: "x".into(),
                ts_start_ms: start,
                ts_end_ms: end,
                confidence: conf_some.then_some(conf_raw),
            };
            let order_ok = start <= end;
            let conf_ok = !conf_some || (0.0..=1.0).contains(&conf_raw);
            prop_assert_eq!(w.is_valid(), order_ok && conf_ok);
        }

        #[test]
        fn prop_blendshape_chunk_round_trip(seed in any::<u64>()) {
            let mut weights = [0.0_f32; 52];
            // Deterministic-from-seed values in [0,1].
            for (i, w) in weights.iter_mut().enumerate() {
                let mix = seed.wrapping_mul((i as u64) + 1);
                *w = ((mix % 1024) as f32) / 1024.0;
            }
            let chunk = BlendshapeChunk {
                request_id: "r".into(),
                is_final: false,
                timestamp_ms: (seed % 100_000) as u32,
                weights,
            };
            let json = serde_json::to_string(&chunk)?;
            let back: BlendshapeChunk = serde_json::from_str(&json)?;
            prop_assert_eq!(back.weights, weights);
            prop_assert_eq!(back.timestamp_ms, chunk.timestamp_ms);
        }

        #[test]
        fn prop_audio_payload_bytes_round_trip(data in proptest::collection::vec(any::<u8>(), 0..256)) {
            let payload = AudioPayload::Bytes {
                data: Bytes::from(data.clone()),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            };
            let json = serde_json::to_string(&payload)?;
            let back: AudioPayload = serde_json::from_str(&json)?;
            match back {
                AudioPayload::Bytes { data: d, .. } => prop_assert_eq!(&*d, &data[..]),
                _ => prop_assert!(false, "variant changed"),
            }
        }
    }
}
