//! `audio` — PyO3 wrappers for the audio modality types.
//!
//! Wraps the `atomr_infer_core::audio::*` surface introduced in M1.
//! These types let Python callers describe TTS / STT / Audio2Face /
//! Realtime requests and inspect the output chunks. The dispatcher
//! plumbing (`PyCluster::execute_audio`, etc.) lands in the next parity
//! wave — for now these types are constructible and inspectable but
//! cannot yet be handed to `PyCluster::execute`.
//!
//! Coverage:
//! - `PyAudioFormat` / `PyAudioParams` — wire format + sample-rate / channels
//! - `PyAudioPayload` (Bytes / Path / Url variants) — serializable inputs
//! - `PySpeechBatch` — TTS batch
//! - `PyAudioBatch` — STT + A2F batch
//! - `PyA2FOptions` / `PyTranscribeOptions` / `PySynthOptions` — option blobs
//! - `PyVoiceRef` — voice selection (Named / Id / ClonedFrom)
//! - `PyTranscriptChunk` / `PySpeechChunk` / `PyBlendshapeChunk` — outputs
//! - `PyWordTiming` / `PyAlignmentDelta` / `PyViseme`
//!
//! All types use the same `name = "...", module = "atomr_infer._native.audio"`
//! pattern as the existing `core` module.

use std::path::PathBuf;

use bytes::Bytes;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use atomr_infer_core::audio::{
    A2FOptions, AlignmentDelta, AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload,
    BlendshapeChunk, SpeechBatch, SpeechChunk, SynthOptions, TranscribeOptions, TranscriptChunk, Viseme,
    VoiceRef, WordTiming,
};

// ---------------------------------------------------------------------------
// AudioFormat
// ---------------------------------------------------------------------------

/// Wire audio format. Wraps `atomr_infer_core::audio::AudioFormat`.
#[pyclass(name = "AudioFormat", module = "atomr_infer._native.audio", eq)]
#[derive(Clone, PartialEq, Eq)]
pub struct PyAudioFormat {
    pub(crate) inner: AudioFormat,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyAudioFormat {
    #[classattr]
    fn PCM16_LE() -> Self {
        Self {
            inner: AudioFormat::Pcm16Le,
        }
    }
    #[classattr]
    fn PCM24_LE() -> Self {
        Self {
            inner: AudioFormat::Pcm24Le,
        }
    }
    #[classattr]
    fn PCM_F32_LE() -> Self {
        Self {
            inner: AudioFormat::PcmF32Le,
        }
    }
    #[classattr]
    fn OGG_OPUS() -> Self {
        Self {
            inner: AudioFormat::OggOpus,
        }
    }
    #[classattr]
    fn MP3() -> Self {
        Self {
            inner: AudioFormat::Mp3,
        }
    }
    #[classattr]
    fn FLAC() -> Self {
        Self {
            inner: AudioFormat::Flac,
        }
    }
    #[classattr]
    fn WAV() -> Self {
        Self {
            inner: AudioFormat::Wav,
        }
    }

    fn __repr__(&self) -> String {
        format!("AudioFormat({:?})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// AudioParams
// ---------------------------------------------------------------------------

#[pyclass(name = "AudioParams", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PyAudioParams {
    pub(crate) inner: AudioParams,
}

#[pymethods]
impl PyAudioParams {
    #[new]
    fn new(sample_rate_hz: u32, channels: u8, format: PyAudioFormat) -> Self {
        Self {
            inner: AudioParams::new(sample_rate_hz, channels, format.inner),
        }
    }

    #[getter]
    fn sample_rate_hz(&self) -> u32 {
        self.inner.sample_rate_hz
    }
    #[getter]
    fn channels(&self) -> u8 {
        self.inner.channels
    }
    #[getter]
    fn format(&self) -> PyAudioFormat {
        PyAudioFormat {
            inner: self.inner.format.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioParams(sample_rate_hz={}, channels={}, format={:?})",
            self.inner.sample_rate_hz, self.inner.channels, self.inner.format
        )
    }
}

// ---------------------------------------------------------------------------
// AudioPayload
// ---------------------------------------------------------------------------

#[pyclass(name = "AudioPayload", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PyAudioPayload {
    pub(crate) inner: AudioPayload,
}

#[pymethods]
impl PyAudioPayload {
    /// Build from in-memory bytes.
    #[staticmethod]
    fn from_bytes(data: &Bound<'_, PyBytes>, params: PyAudioParams) -> Self {
        Self {
            inner: AudioPayload::Bytes {
                data: Bytes::copy_from_slice(data.as_bytes()),
                params: params.inner,
            },
        }
    }

    /// Build from a local file path.
    #[staticmethod]
    fn from_path(path: PathBuf, params: PyAudioParams) -> Self {
        Self {
            inner: AudioPayload::Path {
                path,
                params: params.inner,
            },
        }
    }

    /// Build from a URL.
    #[staticmethod]
    fn from_url(url: &str, params: PyAudioParams) -> PyResult<Self> {
        let parsed = url::Url::parse(url).map_err(|e| PyValueError::new_err(format!("invalid url: {e}")))?;
        Ok(Self {
            inner: AudioPayload::Url {
                url: parsed,
                params: params.inner,
            },
        })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            AudioPayload::Bytes { data, .. } => format!("AudioPayload(Bytes len={})", data.len()),
            AudioPayload::Path { path, .. } => format!("AudioPayload(Path {:?})", path),
            AudioPayload::Url { url, .. } => format!("AudioPayload(Url {})", url),
            _ => "AudioPayload(<other>)".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// VoiceRef
// ---------------------------------------------------------------------------

#[pyclass(name = "VoiceRef", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PyVoiceRef {
    pub(crate) inner: VoiceRef,
}

#[pymethods]
impl PyVoiceRef {
    /// Named voice (e.g. "alloy", "shimmer").
    #[staticmethod]
    fn named(name: &str) -> Self {
        Self {
            inner: VoiceRef::Named(name.to_owned()),
        }
    }

    /// Provider-specific voice ID (e.g. ElevenLabs 21-char ID).
    #[staticmethod]
    fn id(value: &str) -> Self {
        Self {
            inner: VoiceRef::Id(value.to_owned()),
        }
    }

    /// Voice cloned from a reference audio sample.
    #[staticmethod]
    fn cloned_from(payload: PyAudioPayload) -> Self {
        Self {
            inner: VoiceRef::ClonedFrom(payload.inner),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            VoiceRef::Named(n) => format!("VoiceRef::Named({n:?})"),
            VoiceRef::Id(i) => format!("VoiceRef::Id({i:?})"),
            VoiceRef::ClonedFrom(_) => "VoiceRef::ClonedFrom(<payload>)".to_owned(),
            _ => "VoiceRef(<other>)".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// TranscribeOptions / SynthOptions / A2FOptions
// ---------------------------------------------------------------------------

#[pyclass(name = "TranscribeOptions", module = "atomr_infer._native.audio")]
#[derive(Clone, Default)]
pub struct PyTranscribeOptions {
    pub(crate) inner: TranscribeOptions,
}

#[pymethods]
impl PyTranscribeOptions {
    #[new]
    #[pyo3(signature = (language=None, prompt=None, interim_results=false, word_timestamps=false, diarize=false))]
    fn new(
        language: Option<String>,
        prompt: Option<String>,
        interim_results: bool,
        word_timestamps: bool,
        diarize: bool,
    ) -> Self {
        Self {
            inner: TranscribeOptions {
                language,
                prompt,
                interim_results,
                word_timestamps,
                diarize,
                ..Default::default()
            },
        }
    }
}

#[pyclass(name = "SynthOptions", module = "atomr_infer._native.audio")]
#[derive(Clone, Default)]
pub struct PySynthOptions {
    pub(crate) inner: SynthOptions,
}

#[pymethods]
impl PySynthOptions {
    #[new]
    #[pyo3(signature = (format=None, sample_rate_hz=None, speed=None, emotion=None))]
    fn new(
        format: Option<PyAudioFormat>,
        sample_rate_hz: Option<u32>,
        speed: Option<f32>,
        emotion: Option<String>,
    ) -> Self {
        Self {
            inner: SynthOptions {
                format: format.map(|f| f.inner),
                sample_rate_hz,
                speed,
                emotion,
                ..Default::default()
            },
        }
    }
}

#[pyclass(name = "A2FOptions", module = "atomr_infer._native.audio")]
#[derive(Clone, Default)]
pub struct PyA2FOptions {
    pub(crate) inner: A2FOptions,
}

#[pymethods]
impl PyA2FOptions {
    #[new]
    #[pyo3(signature = (fps=None, emotion=None))]
    fn new(fps: Option<u32>, emotion: Option<String>) -> Self {
        Self {
            inner: A2FOptions {
                fps,
                emotion,
                ..Default::default()
            },
        }
    }
}

// ---------------------------------------------------------------------------
// SpeechBatch (TTS)
// ---------------------------------------------------------------------------

#[pyclass(name = "SpeechBatch", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PySpeechBatch {
    pub(crate) inner: SpeechBatch,
}

#[pymethods]
impl PySpeechBatch {
    #[new]
    #[pyo3(signature = (request_id, model, text, voice, options=None, stream=false, emit_alignment=false))]
    fn new(
        request_id: String,
        model: String,
        text: String,
        voice: PyVoiceRef,
        options: Option<PySynthOptions>,
        stream: bool,
        emit_alignment: bool,
    ) -> Self {
        let estimated_characters = text.chars().count() as u32;
        Self {
            inner: SpeechBatch {
                request_id,
                model,
                text,
                voice: voice.inner,
                options: options.map(|o| o.inner).unwrap_or_default(),
                stream,
                emit_alignment,
                estimated_characters,
            },
        }
    }

    #[getter]
    fn request_id(&self) -> &str {
        &self.inner.request_id
    }
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }
    #[getter]
    fn stream(&self) -> bool {
        self.inner.stream
    }
    #[getter]
    fn emit_alignment(&self) -> bool {
        self.inner.emit_alignment
    }
    #[getter]
    fn estimated_characters(&self) -> u32 {
        self.inner.estimated_characters
    }

    fn __repr__(&self) -> String {
        format!(
            "SpeechBatch(request_id={:?}, model={:?}, chars={})",
            self.inner.request_id, self.inner.model, self.inner.estimated_characters
        )
    }
}

// ---------------------------------------------------------------------------
// AudioBatch (STT + A2F)
// ---------------------------------------------------------------------------

#[pyclass(name = "AudioBatch", module = "atomr_infer._native.audio")]
pub struct PyAudioBatch {
    pub(crate) inner: AudioBatch,
}

#[pymethods]
impl PyAudioBatch {
    /// Build a transcription (STT) batch from a serializable payload.
    #[staticmethod]
    #[pyo3(signature = (request_id, model, payload, options=None, stream=false, estimated_units=0))]
    fn transcribe(
        request_id: String,
        model: String,
        payload: PyAudioPayload,
        options: Option<PyTranscribeOptions>,
        stream: bool,
        estimated_units: u32,
    ) -> Self {
        Self {
            inner: AudioBatch {
                request_id,
                model,
                input: AudioInput::Static(payload.inner),
                stream,
                options: AudioOptions::Transcribe(options.map(|o| o.inner).unwrap_or_default()),
                estimated_units,
            },
        }
    }

    /// Build an Audio2Face batch from a serializable payload.
    #[staticmethod]
    #[pyo3(signature = (request_id, model, payload, options=None, stream=false, estimated_units=0))]
    fn audio2face(
        request_id: String,
        model: String,
        payload: PyAudioPayload,
        options: Option<PyA2FOptions>,
        stream: bool,
        estimated_units: u32,
    ) -> Self {
        Self {
            inner: AudioBatch {
                request_id,
                model,
                input: AudioInput::Static(payload.inner),
                stream,
                options: AudioOptions::Audio2Face(options.map(|o| o.inner).unwrap_or_default()),
                estimated_units,
            },
        }
    }

    #[getter]
    fn request_id(&self) -> &str {
        &self.inner.request_id
    }
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }
    #[getter]
    fn stream(&self) -> bool {
        self.inner.stream
    }
    #[getter]
    fn estimated_units(&self) -> u32 {
        self.inner.estimated_units
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioBatch(request_id={:?}, model={:?})",
            self.inner.request_id, self.inner.model
        )
    }
}

// ---------------------------------------------------------------------------
// Output chunks
// ---------------------------------------------------------------------------

#[pyclass(name = "WordTiming", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PyWordTiming {
    pub(crate) inner: WordTiming,
}

#[pymethods]
impl PyWordTiming {
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }
    #[getter]
    fn ts_start_ms(&self) -> u32 {
        self.inner.ts_start_ms
    }
    #[getter]
    fn ts_end_ms(&self) -> u32 {
        self.inner.ts_end_ms
    }
    #[getter]
    fn confidence(&self) -> Option<f32> {
        self.inner.confidence
    }
}

impl From<WordTiming> for PyWordTiming {
    fn from(inner: WordTiming) -> Self {
        Self { inner }
    }
}

#[pyclass(name = "Viseme", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PyViseme {
    pub(crate) inner: Viseme,
}

#[pymethods]
impl PyViseme {
    #[getter]
    fn id(&self) -> u8 {
        self.inner.id
    }
    #[getter]
    fn ts_start_ms(&self) -> u32 {
        self.inner.ts_start_ms
    }
    #[getter]
    fn ts_end_ms(&self) -> u32 {
        self.inner.ts_end_ms
    }
    #[getter]
    fn weight(&self) -> f32 {
        self.inner.weight
    }
}

#[pyclass(name = "AlignmentDelta", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PyAlignmentDelta {
    pub(crate) inner: AlignmentDelta,
}

#[pymethods]
impl PyAlignmentDelta {
    #[getter]
    fn words(&self) -> Vec<PyWordTiming> {
        self.inner.words.iter().cloned().map(PyWordTiming::from).collect()
    }
    #[getter]
    fn visemes(&self) -> Vec<PyViseme> {
        self.inner
            .visemes
            .iter()
            .cloned()
            .map(|v| PyViseme { inner: v })
            .collect()
    }
}

#[pyclass(name = "TranscriptChunk", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PyTranscriptChunk {
    pub(crate) inner: TranscriptChunk,
}

#[pymethods]
impl PyTranscriptChunk {
    #[getter]
    fn request_id(&self) -> &str {
        &self.inner.request_id
    }
    #[getter]
    fn is_final(&self) -> bool {
        self.inner.is_final
    }
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }
    #[getter]
    fn ts_start_ms(&self) -> u32 {
        self.inner.ts_start_ms
    }
    #[getter]
    fn ts_end_ms(&self) -> u32 {
        self.inner.ts_end_ms
    }
    #[getter]
    fn speaker_id(&self) -> Option<&str> {
        self.inner.speaker_id.as_deref()
    }
    #[getter]
    fn words(&self) -> Vec<PyWordTiming> {
        self.inner.words.iter().cloned().map(PyWordTiming::from).collect()
    }
}

impl From<TranscriptChunk> for PyTranscriptChunk {
    fn from(inner: TranscriptChunk) -> Self {
        Self { inner }
    }
}

#[pyclass(name = "SpeechChunk", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PySpeechChunk {
    pub(crate) inner: SpeechChunk,
}

#[pymethods]
impl PySpeechChunk {
    #[getter]
    fn request_id(&self) -> &str {
        &self.inner.request_id
    }
    #[getter]
    fn is_final(&self) -> bool {
        self.inner.is_final
    }
    #[getter]
    fn audio_pcm_chunk<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.audio_pcm_chunk)
    }
    #[getter]
    fn params(&self) -> PyAudioParams {
        PyAudioParams {
            inner: self.inner.params.clone(),
        }
    }
    #[getter]
    fn alignment(&self) -> Option<PyAlignmentDelta> {
        self.inner
            .alignment
            .clone()
            .map(|a| PyAlignmentDelta { inner: a })
    }
}

impl From<SpeechChunk> for PySpeechChunk {
    fn from(inner: SpeechChunk) -> Self {
        Self { inner }
    }
}

#[pyclass(name = "BlendshapeChunk", module = "atomr_infer._native.audio")]
#[derive(Clone)]
pub struct PyBlendshapeChunk {
    pub(crate) inner: BlendshapeChunk,
}

#[pymethods]
impl PyBlendshapeChunk {
    #[getter]
    fn request_id(&self) -> &str {
        &self.inner.request_id
    }
    #[getter]
    fn is_final(&self) -> bool {
        self.inner.is_final
    }
    #[getter]
    fn timestamp_ms(&self) -> u32 {
        self.inner.timestamp_ms
    }
    /// 52 ARKit-canonical blendshape weights in `[0.0, 1.0]`.
    #[getter]
    fn weights(&self) -> Vec<f32> {
        self.inner.weights.to_vec()
    }
}

impl From<BlendshapeChunk> for PyBlendshapeChunk {
    fn from(inner: BlendshapeChunk) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// Submodule registration
// ---------------------------------------------------------------------------

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new_bound(py, "audio")?;
    sub.add_class::<PyAudioFormat>()?;
    sub.add_class::<PyAudioParams>()?;
    sub.add_class::<PyAudioPayload>()?;
    sub.add_class::<PyVoiceRef>()?;
    sub.add_class::<PyTranscribeOptions>()?;
    sub.add_class::<PySynthOptions>()?;
    sub.add_class::<PyA2FOptions>()?;
    sub.add_class::<PySpeechBatch>()?;
    sub.add_class::<PyAudioBatch>()?;
    sub.add_class::<PyWordTiming>()?;
    sub.add_class::<PyViseme>()?;
    sub.add_class::<PyAlignmentDelta>()?;
    sub.add_class::<PyTranscriptChunk>()?;
    sub.add_class::<PySpeechChunk>()?;
    sub.add_class::<PyBlendshapeChunk>()?;
    m.add_submodule(&sub)?;
    Ok(())
}
