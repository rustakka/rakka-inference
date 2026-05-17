//! Wire types for the AssemblyAI Universal-Streaming v3 protocol.
//!
//! AssemblyAI's streaming endpoint accepts raw audio frames (binary
//! WS messages) on the uplink and emits JSON envelopes on the
//! downlink. The runner sends a single `Terminate` text frame to flush
//! + terminate cleanly.
//!
//! Source: <https://www.assemblyai.com/docs/speech-to-text/universal-streaming>.

#![cfg(feature = "stt-assemblyai")]

use serde::{Deserialize, Serialize};

/// Inbound JSON envelope.
///
/// AssemblyAI v3 emits three envelope types on the downlink:
///
/// - `Begin` — session opened, carries the session id.
/// - `Turn` — a per-turn transcript update; the runner emits one
///   `TranscriptChunk` per envelope. `end_of_turn = true` marks the
///   turn-final chunk.
/// - `Termination` — session about to close (server-initiated or after
///   our `Terminate` flush). We ignore this for streaming output but
///   parse it so unknown fields don't bail.
///
/// The three share the `type` discriminator field.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
pub(crate) enum InboundEnvelope {
    Begin(BeginEnvelope),
    Turn(TurnEnvelope),
    Termination(TerminationEnvelope),
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) struct BeginEnvelope {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub expires_at: Option<u64>,
}

/// A turn update from the v3 streaming protocol.
///
/// `transcript` carries the rolling text of the turn; `words` carries
/// per-token timing (the per-token `word_is_final` flag separates the
/// stable prefix from the unstable suffix). `end_of_turn = true`
/// signals the final update for this turn — there is exactly one of
/// these per spoken turn, in contrast to Deepgram's segment-final /
/// utterance-final distinction.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) struct TurnEnvelope {
    /// Monotonically increasing per-session turn index. Starts at 0.
    #[serde(default)]
    pub turn_order: u32,
    /// True when this turn's text has been Punctuated & Formatted by
    /// the provider (requires `format_turns=true` on connect).
    #[serde(default)]
    pub turn_is_formatted: bool,
    /// True for the turn-final update.
    #[serde(default)]
    pub end_of_turn: bool,
    /// Confidence in the turn boundary detection. Not surfaced.
    #[serde(default)]
    pub end_of_turn_confidence: Option<f32>,
    /// Rolling transcript text — full text-so-far for this turn.
    #[serde(default)]
    pub transcript: String,
    /// Per-token timings.
    #[serde(default)]
    pub words: Vec<AssemblyWord>,
}

/// One word in the turn's transcript with its time-range and stability
/// flag.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) struct AssemblyWord {
    #[serde(default)]
    pub text: String,
    /// Start in milliseconds (v3 uses ms directly, unlike v2's seconds).
    #[serde(default)]
    pub start: u32,
    /// End in milliseconds.
    #[serde(default)]
    pub end: u32,
    #[serde(default)]
    pub confidence: Option<f32>,
    /// True when the token has stabilised (no further edits expected).
    #[serde(default)]
    pub word_is_final: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub(crate) struct TerminationEnvelope {
    #[serde(default)]
    pub audio_duration_seconds: Option<f32>,
    #[serde(default)]
    pub session_duration_seconds: Option<f32>,
}

/// Terminal text frame the runner sends to ask AssemblyAI to flush the
/// final turn and close the stream.
#[derive(Debug, Serialize)]
pub(crate) struct Terminate {
    #[serde(rename = "type")]
    pub type_: &'static str,
}

impl Terminate {
    pub fn new() -> Self {
        Self { type_: "Terminate" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn begin_envelope_decodes() {
        let raw = r#"{"type":"Begin","id":"abc","expires_at":1700000000}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        match parsed {
            InboundEnvelope::Begin(b) => {
                assert_eq!(b.id.as_deref(), Some("abc"));
                assert_eq!(b.expires_at, Some(1700000000));
            }
            other => panic!("expected Begin, got {other:?}"),
        }
    }

    #[test]
    fn turn_envelope_decodes_partial() {
        let raw = r#"{"type":"Turn","turn_order":0,"end_of_turn":false,
            "transcript":"hello",
            "words":[{"text":"hello","start":0,"end":300,"confidence":0.9,"word_is_final":false}]}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        match parsed {
            InboundEnvelope::Turn(t) => {
                assert!(!t.end_of_turn);
                assert_eq!(t.transcript, "hello");
                assert_eq!(t.words.len(), 1);
                assert_eq!(t.words[0].end, 300);
                assert!(!t.words[0].word_is_final);
            }
            other => panic!("expected Turn, got {other:?}"),
        }
    }

    #[test]
    fn turn_envelope_decodes_final_formatted() {
        let raw = r#"{"type":"Turn","turn_order":1,"turn_is_formatted":true,
            "end_of_turn":true,"end_of_turn_confidence":0.95,
            "transcript":"Hello world.",
            "words":[
                {"text":"Hello","start":0,"end":300,"word_is_final":true},
                {"text":"world","start":400,"end":900,"word_is_final":true}
            ]}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        match parsed {
            InboundEnvelope::Turn(t) => {
                assert!(t.end_of_turn);
                assert!(t.turn_is_formatted);
                assert_eq!(t.transcript, "Hello world.");
                assert_eq!(t.words.len(), 2);
                assert!(t.words[1].word_is_final);
            }
            other => panic!("expected Turn, got {other:?}"),
        }
    }

    #[test]
    fn termination_envelope_decodes() {
        let raw = r#"{"type":"Termination","audio_duration_seconds":1.5,
            "session_duration_seconds":3.2}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        match parsed {
            InboundEnvelope::Termination(t) => {
                assert_eq!(t.audio_duration_seconds, Some(1.5));
                assert_eq!(t.session_duration_seconds, Some(3.2));
            }
            other => panic!("expected Termination, got {other:?}"),
        }
    }

    #[test]
    fn unknown_type_falls_through() {
        let raw = r#"{"type":"SessionInformation"}"#;
        let parsed: InboundEnvelope = serde_json::from_str(raw).unwrap();
        assert!(matches!(parsed, InboundEnvelope::Other));
    }

    #[test]
    fn terminate_serializes() {
        let s = serde_json::to_string(&Terminate::new()).unwrap();
        assert_eq!(s, r#"{"type":"Terminate"}"#);
    }
}
