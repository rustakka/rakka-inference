//! OpenAI Realtime wire types — outbound events (client → server) and
//! inbound events (server → client).
//!
//! Reference: <https://platform.openai.com/docs/guides/realtime>

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Outbound (client → server)
// ---------------------------------------------------------------------------

/// `session.update` — sent immediately after connect to configure voice,
/// modalities, and audio formats.
#[derive(Debug, Serialize)]
pub(crate) struct SessionUpdate<'a> {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub session: SessionConfig<'a>,
}

impl<'a> SessionUpdate<'a> {
    pub fn new(session: SessionConfig<'a>) -> Self {
        Self {
            type_: "session.update",
            session,
        }
    }
}

/// Session configuration payload embedded in [`SessionUpdate`].
#[derive(Debug, Serialize)]
pub(crate) struct SessionConfig<'a> {
    pub voice: &'a str,
    pub modalities: &'a [&'a str],
    pub input_audio_format: &'a str,
    pub output_audio_format: &'a str,
    pub input_audio_transcription: InputAudioTranscription,
}

/// Request input audio transcription in the session.
#[derive(Debug, Serialize)]
pub(crate) struct InputAudioTranscription {
    pub model: &'static str,
}

impl Default for InputAudioTranscription {
    fn default() -> Self {
        Self { model: "whisper-1" }
    }
}

/// `input_audio_buffer.append` — stream raw base64-encoded PCM audio.
#[derive(Debug, Serialize)]
pub(crate) struct InputAudioAppend<'a> {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub audio: &'a str,
}

impl<'a> InputAudioAppend<'a> {
    pub fn new(audio: &'a str) -> Self {
        Self {
            type_: "input_audio_buffer.append",
            audio,
        }
    }
}

/// `input_audio_buffer.commit` — flush the audio buffer.
#[derive(Debug, Serialize)]
pub(crate) struct InputAudioCommit {
    #[serde(rename = "type")]
    pub type_: &'static str,
}

impl InputAudioCommit {
    pub fn new() -> Self {
        Self {
            type_: "input_audio_buffer.commit",
        }
    }
}

/// `response.cancel` — interrupt the current response.
#[derive(Debug, Serialize)]
pub(crate) struct ResponseCancel {
    #[serde(rename = "type")]
    pub type_: &'static str,
}

impl ResponseCancel {
    pub fn new() -> Self {
        Self {
            type_: "response.cancel",
        }
    }
}

/// `conversation.item.create` — add a user text message.
#[derive(Debug, Serialize)]
pub(crate) struct ConversationItemCreate<'a> {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub item: ConversationItem<'a>,
}

impl<'a> ConversationItemCreate<'a> {
    pub fn user_text(text: &'a str) -> Self {
        Self {
            type_: "conversation.item.create",
            item: ConversationItem {
                type_: "message",
                role: "user",
                content: vec![ContentItem {
                    type_: "input_text",
                    text,
                }],
            },
        }
    }
}

/// A conversation item (message).
#[derive(Debug, Serialize)]
pub(crate) struct ConversationItem<'a> {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub role: &'static str,
    pub content: Vec<ContentItem<'a>>,
}

/// A single content item inside a conversation item.
#[derive(Debug, Serialize)]
pub(crate) struct ContentItem<'a> {
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub text: &'a str,
}

/// `response.create` — ask the model to start generating a response.
#[derive(Debug, Serialize)]
pub(crate) struct ResponseCreate {
    #[serde(rename = "type")]
    pub type_: &'static str,
}

impl ResponseCreate {
    pub fn new() -> Self {
        Self {
            type_: "response.create",
        }
    }
}

// ---------------------------------------------------------------------------
// Inbound (server → client)
// ---------------------------------------------------------------------------

/// The full set of inbound event types from the OpenAI Realtime API.
///
/// Fields that don't fit a specific variant are captured by `Other`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum InboundEvent {
    /// Audio PCM delta (base64-encoded) from the model.
    #[serde(rename = "response.audio.delta")]
    ResponseAudioDelta(AudioDelta),

    /// Transcript delta for the model's audio output.
    #[serde(rename = "response.audio_transcript.delta")]
    ResponseAudioTranscriptDelta(TranscriptDelta),

    /// Final transcript for the model's audio output.
    #[serde(rename = "response.audio_transcript.done")]
    ResponseAudioTranscriptDone(TranscriptDone),

    /// Final transcript for the user's input audio.
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    InputAudioTranscriptionCompleted(InputTranscriptCompleted),

    /// The model has finished generating a response.
    #[serde(rename = "response.done")]
    ResponseDone,

    /// An error occurred on the server side.
    #[serde(rename = "error")]
    Error(ErrorEnvelope),

    /// All other event types (session.created, session.updated, etc.).
    #[serde(other)]
    Other,
}

/// Payload of `response.audio.delta`.
#[derive(Debug, Deserialize)]
pub(crate) struct AudioDelta {
    /// Base64-encoded PCM16 LE audio bytes.
    #[serde(default)]
    pub delta: String,
}

/// Payload of `response.audio_transcript.delta`.
#[derive(Debug, Deserialize)]
pub(crate) struct TranscriptDelta {
    #[serde(default)]
    pub delta: String,
}

/// Payload of `response.audio_transcript.done`.
#[derive(Debug, Deserialize)]
pub(crate) struct TranscriptDone {
    #[serde(default)]
    pub transcript: String,
}

/// Payload of `conversation.item.input_audio_transcription.completed`.
#[derive(Debug, Deserialize)]
pub(crate) struct InputTranscriptCompleted {
    #[serde(default)]
    pub transcript: String,
}

/// Payload of `error`.
#[derive(Debug, Deserialize)]
pub(crate) struct ErrorEnvelope {
    pub error: ErrorBody,
}

/// Error body inside [`ErrorEnvelope`].
#[derive(Debug, Deserialize)]
pub(crate) struct ErrorBody {
    #[serde(default)]
    pub message: String,
    #[allow(dead_code)]
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_update_serialises() {
        let modalities = &["audio", "text"];
        let su = SessionUpdate::new(SessionConfig {
            voice: "alloy",
            modalities,
            input_audio_format: "pcm16",
            output_audio_format: "pcm16",
            input_audio_transcription: InputAudioTranscription::default(),
        });
        let s = serde_json::to_string(&su).unwrap();
        assert!(s.contains(r#""type":"session.update""#));
        assert!(s.contains("alloy"));
        assert!(s.contains("pcm16"));
    }

    #[test]
    fn audio_append_serialises() {
        let ev = InputAudioAppend::new("AAAA");
        let s = serde_json::to_string(&ev).unwrap();
        assert!(s.contains(r#""type":"input_audio_buffer.append""#));
        assert!(s.contains("AAAA"));
    }

    #[test]
    fn conversation_item_create_serialises() {
        let ev = ConversationItemCreate::user_text("hello");
        let s = serde_json::to_string(&ev).unwrap();
        assert!(s.contains(r#""type":"conversation.item.create""#));
        assert!(s.contains("hello"));
        assert!(s.contains("input_text"));
    }

    #[test]
    fn response_create_serialises() {
        let s = serde_json::to_string(&ResponseCreate::new()).unwrap();
        assert_eq!(s, r#"{"type":"response.create"}"#);
    }

    #[test]
    fn commit_serialises() {
        let s = serde_json::to_string(&InputAudioCommit::new()).unwrap();
        assert_eq!(s, r#"{"type":"input_audio_buffer.commit"}"#);
    }

    #[test]
    fn cancel_serialises() {
        let s = serde_json::to_string(&ResponseCancel::new()).unwrap();
        assert_eq!(s, r#"{"type":"response.cancel"}"#);
    }

    #[test]
    fn inbound_audio_delta_deserialises() {
        let json = r#"{"type":"response.audio.delta","delta":"AAAA"}"#;
        let ev: InboundEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(ev, InboundEvent::ResponseAudioDelta(_)));
    }

    #[test]
    fn inbound_transcript_delta_deserialises() {
        let json = r#"{"type":"response.audio_transcript.delta","delta":"hello"}"#;
        let ev: InboundEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(ev, InboundEvent::ResponseAudioTranscriptDelta(_)));
    }

    #[test]
    fn inbound_transcript_done_deserialises() {
        let json = r#"{"type":"response.audio_transcript.done","transcript":"hello world"}"#;
        let ev: InboundEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(ev, InboundEvent::ResponseAudioTranscriptDone(_)));
    }

    #[test]
    fn inbound_input_transcript_done_deserialises() {
        let json = r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"hi"}"#;
        let ev: InboundEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(ev, InboundEvent::InputAudioTranscriptionCompleted(_)));
    }

    #[test]
    fn inbound_response_done_deserialises() {
        let json = r#"{"type":"response.done"}"#;
        let ev: InboundEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(ev, InboundEvent::ResponseDone));
    }

    #[test]
    fn inbound_error_deserialises() {
        let json = r#"{"type":"error","error":{"message":"oops","type":"server_error"}}"#;
        let ev: InboundEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(ev, InboundEvent::Error(_)));
    }

    #[test]
    fn inbound_unknown_is_other() {
        let json = r#"{"type":"session.created","session":{}}"#;
        let ev: InboundEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(ev, InboundEvent::Other));
    }
}
