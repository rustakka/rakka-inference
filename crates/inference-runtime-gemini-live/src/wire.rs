//! Wire types for the Gemini Live BidiGenerateContent protocol.
//!
//! Gemini Live uses bidirectional JSON text frames over a WebSocket
//! connection. The initial frame must be a `BidiGenerateContentSetup`
//! message; subsequent frames carry client content turns or realtime
//! audio input. The server emits `BidiGenerateContentServerContent`
//! with model-generated audio and text, or `BidiGenerateContentToolCall`.
//!
//! Reference: <https://ai.google.dev/api/multimodal-live>

#![cfg(feature = "tts-gemini-live")]

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Outbound (client → server)
// ─────────────────────────────────────────────────────────────────────────────

/// Initial frame: `BidiGenerateContentSetup`.
///
/// Must be the first frame sent after connecting. The server responds
/// with `{"setupComplete":{}}` when ready to receive content.
#[derive(Debug, Serialize)]
pub(crate) struct Setup<'a> {
    pub setup: SetupConfig<'a>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SetupConfig<'a> {
    pub model: &'a str,
    pub generation_config: GenerationConfig<'a>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerationConfig<'a> {
    pub response_modalities: &'a [&'a str],
}

/// Text turn: `BidiGenerateContentClientContent`.
///
/// Sent when the caller sends `RealtimeIn::Text(s)`. The `turnComplete`
/// flag tells the model this turn is done and it should respond.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ClientContent {
    pub client_content: ClientContentInner,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ClientContentInner {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turns: Option<Vec<ContentTurn>>,
    pub turn_complete: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct ContentTurn {
    pub role: &'static str,
    pub parts: Vec<ContentPart>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ContentPart {
    pub text: String,
}

/// Audio input: `BidiGenerateContentRealtimeInput`.
///
/// Sent when the caller sends `RealtimeIn::AudioFrame`. Carries one PCM
/// chunk as a base64-encoded `mediaChunk`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RealtimeInput {
    pub realtime_input: RealtimeInputInner,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RealtimeInputInner {
    pub media_chunks: Vec<MediaChunk>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MediaChunk {
    pub mime_type: String,
    pub data: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Inbound (server → client)
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level inbound envelope from Gemini Live.
///
/// The server can send:
/// - `{"setupComplete":{}}` — initial handshake complete.
/// - `{"serverContent":{...}}` — model turn with audio / text / turn flags.
/// - `{"toolCall":{...}}` — tool call (ignored in this runner).
/// - Any other envelope → `Other` (silently ignored).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub(crate) enum Inbound {
    #[serde(rename = "setupComplete")]
    SetupComplete(serde_json::Value),
    #[serde(rename = "serverContent")]
    ServerContent(ServerContentEnvelope),
    #[serde(rename = "toolCall")]
    ToolCall(serde_json::Value),
    #[serde(other)]
    Other,
}

/// Contents of a `serverContent` envelope.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ServerContentEnvelope {
    /// The model's turn — may contain audio and/or text parts.
    #[serde(default)]
    pub model_turn: Option<ModelTurn>,
    /// True when the model has finished its response turn.
    #[serde(default)]
    pub turn_complete: Option<bool>,
    /// True when the model was interrupted (e.g. by a `Commit`).
    /// Surfaced for completeness; the runner currently treats it as a no-op.
    #[serde(default)]
    #[allow(dead_code)]
    pub interrupted: Option<bool>,
}

/// One model turn in a `serverContent` response.
#[derive(Debug, Deserialize)]
pub(crate) struct ModelTurn {
    #[serde(default)]
    pub parts: Vec<ModelPart>,
}

/// One part in a model turn — either inline audio data or text.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModelPart {
    /// Inline audio data (base64 PCM).
    #[serde(default)]
    pub inline_data: Option<InlineData>,
    /// Text transcript from the model.
    #[serde(default)]
    pub text: Option<String>,
}

/// Inline data blob (audio PCM in base64).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InlineData {
    /// MIME type, e.g. `"audio/pcm;rate=24000"`.
    /// Used for validation; always `audio/pcm;rate=<hz>` in the current API.
    #[allow(dead_code)]
    pub mime_type: String,
    /// Base64-encoded PCM bytes.
    pub data: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_serializes_with_camel_case() {
        let msg = Setup {
            setup: SetupConfig {
                model: "models/gemini-2.0-flash-exp",
                generation_config: GenerationConfig {
                    response_modalities: &["AUDIO"],
                },
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"setup\""), "json: {json}");
        assert!(json.contains("\"generationConfig\""), "json: {json}");
        assert!(json.contains("\"responseModalities\""), "json: {json}");
        assert!(json.contains("\"AUDIO\""), "json: {json}");
    }

    #[test]
    fn client_content_text_turn_serializes() {
        let msg = ClientContent {
            client_content: ClientContentInner {
                turns: Some(vec![ContentTurn {
                    role: "user",
                    parts: vec![ContentPart { text: "hello".into() }],
                }]),
                turn_complete: true,
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"clientContent\""), "json: {json}");
        assert!(json.contains("\"turnComplete\":true"), "json: {json}");
        assert!(json.contains("\"hello\""), "json: {json}");
    }

    #[test]
    fn commit_serializes_as_turn_complete_only() {
        let msg = ClientContent {
            client_content: ClientContentInner {
                turns: None,
                turn_complete: true,
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("\"turns\""), "should not have turns: {json}");
        assert!(json.contains("\"turnComplete\":true"), "json: {json}");
    }

    #[test]
    fn realtime_input_serializes() {
        let msg = RealtimeInput {
            realtime_input: RealtimeInputInner {
                media_chunks: vec![MediaChunk {
                    mime_type: "audio/pcm;rate=16000".into(),
                    data: "AQID".into(),
                }],
            },
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"realtimeInput\""), "json: {json}");
        assert!(json.contains("\"mediaChunks\""), "json: {json}");
        assert!(json.contains("\"mimeType\""), "json: {json}");
    }

    #[test]
    fn setup_complete_deserializes() {
        let raw = r#"{"setupComplete":{}}"#;
        let parsed: Inbound = serde_json::from_str(raw).unwrap();
        assert!(matches!(parsed, Inbound::SetupComplete(_)));
    }

    #[test]
    fn server_content_with_audio_deserializes() {
        let raw = r#"{
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "AQID"}}
                    ]
                }
            }
        }"#;
        let parsed: Inbound = serde_json::from_str(raw).unwrap();
        match parsed {
            Inbound::ServerContent(sc) => {
                let turn = sc.model_turn.unwrap();
                assert_eq!(turn.parts.len(), 1);
                let p = &turn.parts[0];
                assert!(p.inline_data.is_some());
                assert_eq!(p.inline_data.as_ref().unwrap().mime_type, "audio/pcm;rate=24000");
                assert_eq!(p.inline_data.as_ref().unwrap().data, "AQID");
            }
            other => panic!("expected ServerContent, got {other:?}"),
        }
    }

    #[test]
    fn server_content_with_text_deserializes() {
        let raw = r#"{"serverContent":{"modelTurn":{"parts":[{"text":"hello there"}]}}}"#;
        let parsed: Inbound = serde_json::from_str(raw).unwrap();
        match parsed {
            Inbound::ServerContent(sc) => {
                let turn = sc.model_turn.unwrap();
                assert_eq!(turn.parts[0].text.as_deref(), Some("hello there"));
            }
            other => panic!("expected ServerContent, got {other:?}"),
        }
    }

    #[test]
    fn server_content_turn_complete_deserializes() {
        let raw = r#"{"serverContent":{"turnComplete":true}}"#;
        let parsed: Inbound = serde_json::from_str(raw).unwrap();
        match parsed {
            Inbound::ServerContent(sc) => {
                assert_eq!(sc.turn_complete, Some(true));
                assert!(sc.model_turn.is_none());
            }
            other => panic!("expected ServerContent, got {other:?}"),
        }
    }

    #[test]
    fn unknown_envelope_gracefully_fails() {
        // The Gemini Live API discriminates by top-level key. Unknown
        // envelopes (e.g. future API additions) fail deserialization and
        // the runner's downlink task ignores them via the Err(_) arm.
        let raw = r#"{"someUnknownField":{}}"#;
        let result: Result<Inbound, _> = serde_json::from_str(raw);
        // Either parses as Other (unit) or fails — both are handled in
        // the downlink task.
        match result {
            Ok(Inbound::Other) => {} // unit variant matched
            Err(_) => {}             // parse failure → downlink ignores it
            Ok(other) => panic!("unexpected variant: {other:?}"),
        }
    }
}
