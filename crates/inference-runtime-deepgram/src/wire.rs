//! Wire types for the Deepgram WSS streaming protocol.
//!
//! Deepgram's listen-stream endpoint accepts raw audio frames (binary
//! WS messages) on the uplink and emits JSON transcript envelopes on
//! the downlink. The runner sends a single `CloseStream` text frame
//! to flush + terminate cleanly.
//!
//! Source: <https://developers.deepgram.com/reference/streaming>.

#![cfg(feature = "stt-deepgram")]

use serde::{Deserialize, Serialize};

/// Inbound JSON envelope.
///
/// Deepgram emits two envelope types on the downlink:
///
/// - `Results` — a transcript chunk with one or more alternatives,
///   `is_final` indicating whether the segment is interim or finalised,
///   and a `speech_final` flag set on the last chunk of an utterance.
/// - `Metadata` — emitted on session open and end; carries the request
///   id and the negotiated model. We ignore this for streaming output
///   but parse it so unknown fields don't bail.
///
/// The two share the `type` discriminator field.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub(crate) enum InboundEnvelope {
    Results(ResultsEnvelope),
    Metadata(MetadataEnvelope),
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResultsEnvelope {
    /// The current segment's start time in seconds.
    #[serde(default)]
    pub start: f32,
    /// The current segment's duration in seconds.
    #[serde(default)]
    pub duration: f32,
    /// Whether this transcript reflects a finalised segment (no more
    /// edits) — false for interim chunks.
    #[serde(default)]
    pub is_final: bool,
    /// Whether this is the final chunk of an utterance (Deepgram's
    /// VAD endpoint detection). Only set when `endpointing != "false"`.
    #[serde(default)]
    pub speech_final: bool,
    pub channel: ResultsChannel,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ResultsChannel {
    /// Always at least one alternative; we pick the first one for the
    /// emitted transcript text. Higher-ranked alternatives could be
    /// surfaced in a future enhancement.
    pub alternatives: Vec<TranscriptAlternative>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) struct TranscriptAlternative {
    #[serde(default)]
    pub transcript: String,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub words: Vec<DeepgramWord>,
}

/// One word in the alternative's transcript with its time-range and
/// (optional) speaker label.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DeepgramWord {
    #[serde(default)]
    pub word: String,
    /// The token Deepgram chose for output (may differ from `word`
    /// when `smart_format=true` upper-cases or punctuates).
    #[serde(default)]
    pub punctuated_word: Option<String>,
    pub start: f32,
    pub end: f32,
    #[serde(default)]
    pub confidence: Option<f32>,
    /// Speaker label as a 0-indexed integer when diarization is on,
    /// absent otherwise.
    #[serde(default)]
    pub speaker: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) struct MetadataEnvelope {
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub model_info: Option<serde_json::Value>,
}

/// Terminal text frame the runner sends to ask Deepgram to flush the
/// final transcript and close the stream.
#[derive(Debug, Serialize)]
pub(crate) struct CloseStream {
    #[serde(rename = "type")]
    pub type_: &'static str,
}

impl CloseStream {
    pub fn new() -> Self {
        Self { type_: "CloseStream" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_envelope_decodes_minimal_interim() {
        let raw = r#"{"type":"Results","is_final":false,"start":0.0,"duration":0.5,
            "channel":{"alternatives":[{"transcript":"hello","confidence":0.9,"words":[]}]}}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        match parsed {
            InboundEnvelope::Results(r) => {
                assert!(!r.is_final);
                assert!(!r.speech_final);
                assert_eq!(r.channel.alternatives[0].transcript, "hello");
                assert_eq!(r.channel.alternatives[0].confidence, Some(0.9));
                assert!(r.channel.alternatives[0].words.is_empty());
            }
            other => panic!("expected Results, got {other:?}"),
        }
    }

    #[test]
    fn results_envelope_decodes_words_with_speaker() {
        let raw = r#"{"type":"Results","is_final":true,"speech_final":true,"start":0.0,"duration":1.0,
            "channel":{"alternatives":[{"transcript":"hi there","words":[
                {"word":"hi","start":0.0,"end":0.3,"speaker":0,"confidence":0.97},
                {"word":"there","punctuated_word":"there.","start":0.4,"end":0.9,"speaker":1}
            ]}]}}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        match parsed {
            InboundEnvelope::Results(r) => {
                assert!(r.is_final);
                assert!(r.speech_final);
                let words = &r.channel.alternatives[0].words;
                assert_eq!(words.len(), 2);
                assert_eq!(words[0].speaker, Some(0));
                assert_eq!(words[1].punctuated_word.as_deref(), Some("there."));
            }
            other => panic!("expected Results, got {other:?}"),
        }
    }

    #[test]
    fn metadata_envelope_decodes() {
        let raw = r#"{"type":"Metadata","request_id":"abc"}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        match parsed {
            InboundEnvelope::Metadata(m) => assert_eq!(m.request_id.as_deref(), Some("abc")),
            other => panic!("expected Metadata, got {other:?}"),
        }
    }

    #[test]
    fn unknown_type_falls_through() {
        let raw = r#"{"type":"SpeechStarted"}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        assert!(matches!(parsed, InboundEnvelope::Other));
    }

    #[test]
    fn close_stream_serializes() {
        let s = serde_json::to_string(&CloseStream::new()).unwrap();
        assert_eq!(s, r#"{"type":"CloseStream"}"#);
    }
}
