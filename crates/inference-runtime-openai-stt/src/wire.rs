//! Wire types for `POST /v1/audio/transcriptions`.
//!
//! The request side is multipart/form-data, encoded by `reqwest`'s
//! `multipart` helper. The response side has two shapes:
//!
//! - `response_format=json` — `{"text": "..."}`
//! - `response_format=verbose_json` — `{"text": "...", "segments":
//!   [{start, end, text, ...}], "words": [{word, start, end, ...}]}`

#[cfg(feature = "stt-openai")]
use serde::Deserialize;

#[cfg(feature = "stt-openai")]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PlainResponse {
    pub text: String,
}

#[cfg(feature = "stt-openai")]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VerboseResponse {
    pub text: String,
    #[serde(default)]
    pub segments: Vec<VerboseSegment>,
    #[serde(default)]
    pub words: Vec<VerboseWord>,
}

#[cfg(feature = "stt-openai")]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VerboseSegment {
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    pub text: String,
    /// Average log-probability of the tokens; not all responses
    /// include it. Captured for future cost/quality estimation.
    #[allow(dead_code)]
    #[serde(default)]
    pub avg_logprob: Option<f32>,
}

#[cfg(feature = "stt-openai")]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VerboseWord {
    pub word: String,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
}

#[cfg(all(test, feature = "stt-openai"))]
mod tests {
    use super::*;

    #[test]
    fn plain_response_parses() {
        let body = r#"{"text":"hello world"}"#;
        let r: PlainResponse = serde_json::from_str(body).unwrap();
        assert_eq!(r.text, "hello world");
    }

    #[test]
    fn verbose_response_with_segments_and_words() {
        let body = r#"{
            "text": "hello world",
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "hello", "avg_logprob": -0.1},
                {"start": 0.5, "end": 1.0, "text": " world"}
            ],
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0}
            ]
        }"#;
        let r: VerboseResponse = serde_json::from_str(body).unwrap();
        assert_eq!(r.text, "hello world");
        assert_eq!(r.segments.len(), 2);
        assert_eq!(r.segments[1].text, " world");
        assert_eq!(r.words.len(), 2);
        assert_eq!(r.words[0].end, 0.5);
    }

    #[test]
    fn verbose_response_without_words() {
        let body = r#"{"text":"hi","segments":[{"start":0.0,"end":0.2,"text":"hi"}]}"#;
        let r: VerboseResponse = serde_json::from_str(body).unwrap();
        assert!(r.words.is_empty());
        assert_eq!(r.segments[0].end, 0.2);
    }
}
