//! Wire types for `POST /v1/audio/speech`.
//!
//! Only the request body needs serialisation — the response body is
//! raw audio bytes (the format is `response_format`, defaulting to
//! `"pcm"` for streaming-friendly framing).

#[cfg(feature = "tts-openai")]
use serde::Serialize;

#[cfg(feature = "tts-openai")]
use atomr_infer_core::audio::{AudioFormat, SpeechBatch, VoiceRef};

/// `POST /v1/audio/speech` body. See
/// <https://platform.openai.com/docs/api-reference/audio/createSpeech>.
#[cfg(feature = "tts-openai")]
#[derive(Debug, Serialize)]
pub(crate) struct SpeechRequest<'a> {
    pub model: &'a str,
    pub input: &'a str,
    pub voice: &'a str,
    pub response_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<&'a str>,
}

#[cfg(feature = "tts-openai")]
impl<'a> SpeechRequest<'a> {
    pub(crate) fn from_batch(batch: &'a SpeechBatch, response_format: &'static str) -> Self {
        Self {
            model: batch.model.as_str(),
            input: batch.text.as_str(),
            voice: voice_name(&batch.voice),
            response_format,
            speed: batch.options.speed,
            instructions: batch.options.emotion.as_deref(),
        }
    }
}

/// OpenAI's `voice` field is a string — map [`VoiceRef`] to the
/// closest single-string representation it understands.
#[cfg(feature = "tts-openai")]
pub(crate) fn voice_name(voice: &VoiceRef) -> &str {
    match voice {
        VoiceRef::Named(s) | VoiceRef::Id(s) => s.as_str(),
        // Cloned voices are not supported by the public OpenAI TTS API;
        // fall back to the conservative default. Callers wanting
        // voice cloning should choose ElevenLabs / XTTS.
        VoiceRef::ClonedFrom(_) => "alloy",
        _ => "alloy",
    }
}

/// Translate a caller-requested [`AudioFormat`] into the matching
/// OpenAI `response_format` string. Returns `None` when the requested
/// format is one OpenAI does not accept.
#[cfg(feature = "tts-openai")]
pub(crate) fn response_format_str(format: AudioFormat) -> Option<&'static str> {
    match format {
        AudioFormat::Pcm16Le | AudioFormat::PcmF32Le | AudioFormat::Pcm24Le => Some("pcm"),
        AudioFormat::Mp3 => Some("mp3"),
        AudioFormat::OggOpus => Some("opus"),
        AudioFormat::Flac => Some("flac"),
        AudioFormat::Wav => Some("wav"),
        _ => None,
    }
}

#[cfg(all(test, feature = "tts-openai"))]
mod tests {
    use super::*;
    use atomr_infer_core::audio::{SpeechBatch, SynthOptions, VoiceRef};

    fn sample_batch() -> SpeechBatch {
        SpeechBatch {
            request_id: "r".into(),
            model: "tts-1".into(),
            text: "hello".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions {
                speed: Some(1.1),
                ..Default::default()
            },
            stream: true,
            emit_alignment: false,
            estimated_characters: 5,
        }
    }

    #[test]
    fn request_serialises_minimal_fields() {
        let batch = sample_batch();
        let body = SpeechRequest::from_batch(&batch, "pcm");
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"model\":\"tts-1\""));
        assert!(json.contains("\"input\":\"hello\""));
        assert!(json.contains("\"voice\":\"alloy\""));
        assert!(json.contains("\"response_format\":\"pcm\""));
        assert!(json.contains("\"speed\":1.1"));
        // emotion=None → instructions omitted.
        assert!(!json.contains("instructions"));
    }

    #[test]
    fn cloned_voice_falls_back_to_alloy() {
        let voice = VoiceRef::ClonedFrom(atomr_infer_core::audio::AudioPayload::Bytes {
            data: bytes::Bytes::new(),
            params: atomr_infer_core::audio::AudioParams::new(
                16_000,
                1,
                atomr_infer_core::audio::AudioFormat::Pcm16Le,
            ),
        });
        assert_eq!(voice_name(&voice), "alloy");
    }

    #[test]
    fn format_mapping_covers_all_variants() {
        assert_eq!(response_format_str(AudioFormat::Pcm16Le), Some("pcm"));
        assert_eq!(response_format_str(AudioFormat::Mp3), Some("mp3"));
        assert_eq!(response_format_str(AudioFormat::Wav), Some("wav"));
        assert_eq!(response_format_str(AudioFormat::OggOpus), Some("opus"));
        assert_eq!(response_format_str(AudioFormat::Flac), Some("flac"));
    }
}
