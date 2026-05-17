//! Wire types for the ElevenLabs TTS API.
//!
//! Two distinct shapes:
//!
//! - **HTTPS one-shot**: `POST /v1/text-to-speech/{voice_id}` —
//!   JSON body in, audio bytes out (default `mpeg`, override via
//!   `?output_format=…`).
//! - **WS streaming**: `WSS /v1/text-to-speech/{voice_id}/stream-input` —
//!   sequence of JSON text frames; the runner kicks the session off
//!   with an `InitMessage` (voice settings, generation config) then
//!   sends one `TextMessage { text }` per chunk and a final flush
//!   message. Inbound frames carry base64 audio + optional alignment.
//!
//! Source: <https://elevenlabs.io/docs/api-reference>.

#![cfg(feature = "tts-elevenlabs")]

use serde::{Deserialize, Serialize};

/// HTTPS request body for `POST /v1/text-to-speech/{voice_id}`.
#[derive(Debug, Serialize)]
pub(crate) struct SpeechRequest<'a> {
    pub model_id: &'a str,
    pub text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_settings: Option<VoiceSettings>,
}

/// Subset of the `voice_settings` blob ElevenLabs accepts.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct VoiceSettings {
    /// `0.0..=1.0`. Lower → more expressive, higher → more consistent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<f32>,
    /// `0.0..=1.0`. Higher → closer to the cloned voice's timbre.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f32>,
    /// `0.0..=1.0`. Provider-specific; we forward it from
    /// `SynthOptions::pitch_semitones` mapped to a 0..1 range.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f32>,
    /// `true` → enable the speaker boost post-processor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
}

// ─────────────────────────────────────────────────────────────────────────────
// WS streaming protocol
// ─────────────────────────────────────────────────────────────────────────────

/// First message the runner pushes onto the WS once the session opens.
/// Carries the per-session config the API needs to start synthesising.
#[derive(Debug, Serialize)]
pub(crate) struct WsInitMessage<'a> {
    /// First text chunk — ElevenLabs requires the init frame to carry
    /// at least one space; the runner concatenates the caller's first
    /// text chunk here.
    pub text: &'a str,
    pub model_id: &'a str,
    pub xi_api_key: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_settings: Option<VoiceSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<WsGenerationConfig>,
}

#[derive(Debug, Serialize)]
pub(crate) struct WsGenerationConfig {
    /// Length in milliseconds of audio chunks the server buffers
    /// before emitting an alignment frame. Defaults to ElevenLabs'
    /// internal heuristic when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_length_schedule: Option<Vec<u32>>,
}

/// Subsequent text frame on the WS stream. The runner sends one of
/// these per caller text chunk; pushing an empty `text` flushes the
/// remaining audio without closing the session.
#[derive(Debug, Serialize)]
pub(crate) struct WsTextMessage<'a> {
    pub text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub try_trigger_generation: Option<bool>,
}

/// Inbound JSON envelope received on the WS stream.
///
/// `audio` is base64-encoded raw PCM (`mp3`/`pcm` according to the
/// `output_format` query parameter). `alignment` is per-character
/// timing in milliseconds. Either field may be missing — ElevenLabs
/// emits empty pings between speech.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WsInboundFrame {
    #[serde(default)]
    pub audio: Option<String>,
    #[serde(default)]
    pub alignment: Option<WsAlignment>,
    /// When `true`, the server has flushed the last chunk for the
    /// current input. The runner emits the corresponding
    /// `SpeechChunk` with `is_final = true`.
    #[serde(default)]
    pub is_final: Option<bool>,
}

/// Per-character timing carried alongside an audio frame.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct WsAlignment {
    pub chars: Vec<String>,
    pub char_start_times_ms: Vec<u32>,
    pub char_durations_ms: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_request_serializes_compactly() {
        let body = SpeechRequest {
            model_id: "eleven_turbo_v2_5",
            text: "hello",
            voice_settings: Some(VoiceSettings {
                stability: Some(0.5),
                similarity_boost: Some(0.75),
                style: None,
                use_speaker_boost: Some(true),
            }),
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"model_id\":\"eleven_turbo_v2_5\""));
        assert!(json.contains("\"text\":\"hello\""));
        assert!(json.contains("\"stability\":0.5"));
        assert!(json.contains("\"similarity_boost\":0.75"));
        assert!(!json.contains("\"style\""));
        assert!(json.contains("\"use_speaker_boost\":true"));
    }

    #[test]
    fn ws_init_message_includes_credentials() {
        let body = WsInitMessage {
            text: " ",
            model_id: "eleven_multilingual_v2",
            xi_api_key: "sk-fake",
            voice_settings: None,
            generation_config: None,
        };
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"xi_api_key\":\"sk-fake\""));
        assert!(json.contains("\"model_id\":\"eleven_multilingual_v2\""));
    }

    #[test]
    fn ws_inbound_frame_decodes_optional_fields() {
        let with_audio = r#"{"audio":"aGVsbG8=","is_final":false}"#;
        let frame: WsInboundFrame = serde_json::from_str(with_audio).unwrap();
        assert_eq!(frame.audio.as_deref(), Some("aGVsbG8="));
        assert_eq!(frame.is_final, Some(false));
        assert!(frame.alignment.is_none());

        let with_align = r#"{"audio":null,"alignment":{"chars":["h","i"],"char_start_times_ms":[0,50],"char_durations_ms":[50,50]}}"#;
        let frame: WsInboundFrame = serde_json::from_str(with_align).unwrap();
        assert!(frame.audio.is_none());
        let a = frame.alignment.unwrap();
        assert_eq!(a.chars, vec!["h".to_string(), "i".to_string()]);
        assert_eq!(a.char_start_times_ms, vec![0, 50]);

        let empty_ping = r#"{}"#;
        let frame: WsInboundFrame = serde_json::from_str(empty_ping).unwrap();
        assert!(frame.audio.is_none());
        assert!(frame.alignment.is_none());
        assert!(frame.is_final.is_none());
    }
}
