//! # inference-runtime-gemini-live
//!
//! Gemini Live bidirectional realtime speech runtime for `atomr-infer`.
//! Implements [`atomr_infer_core::runner::RealtimeRunner`] against the
//! `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent`
//! endpoint via the shared `atomr_infer_runtime_ws_core` transport
//! (available only when the `tts-gemini-live` feature is on).
//!
//! ## Build profiles
//!
//! | Build                                                                            | Result                                                  |
//! |----------------------------------------------------------------------------------|-------------------------------------------------------|
//! | `cargo build -p atomr-infer-runtime-gemini-live`                                 | Stub — `open_session` returns `Internal("tts-gemini-live feature disabled at build time")`. |
//! | `cargo build -p atomr-infer-runtime-gemini-live --features tts-gemini-live`      | Real path — bidirectional WSS session, PCM audio I/O, setup handshake. |
//!
//! ## Session lifecycle
//!
//! On `open_session`:
//! 1. Connect to the Gemini Live endpoint with the API key embedded in the
//!    URL query string (`?key=<api_key>`).
//! 2. Send a `BidiGenerateContentSetup` message configuring the model and
//!    requesting audio response modality.
//! 3. Wait for the `setupComplete` response before forwarding any
//!    [`atomr_infer_core::audio::RealtimeIn`] frames from the caller.
//! 4. Uplink task: translate [`atomr_infer_core::audio::RealtimeIn`] variants into Gemini Live
//!    JSON messages and send as text WS frames.
//! 5. Downlink task: decode Gemini Live JSON messages into
//!    [`atomr_infer_core::audio::RealtimeOut`] variants and send to the
//!    caller's outbound channel.
//!
//! ## Auth
//!
//! Unlike OpenAI Realtime (which uses an `Authorization: Bearer` header),
//! Gemini Live embeds the API key as a `?key=<api_key>` URL query
//! parameter on the initial WebSocket upgrade request. No auth headers
//! are required.
//!
//! ## PCM audio format
//!
//! Audio received from the model arrives as `audio/pcm;rate=24000` inline
//! data in `modelTurn` parts. The runner decodes the base64 payload and
//! emits [`atomr_infer_core::audio::RealtimeOut::AudioFrame`] with
//! [`atomr_infer_core::audio::AudioParams`] `{24_000, 1, Pcm16Le}`.
//!
//! Audio sent to the model must be [`atomr_infer_core::audio::AudioFormat::Pcm16Le`];
//! other formats surface [`atomr_infer_core::error::InferenceError::UnsupportedAudioFormat`].
//!
//! ## Voice selection
//!
//! Gemini Live encodes voice selection at the model level rather than as
//! a separate API parameter. [`atomr_infer_core::audio::VoiceRef::Named`]
//! and [`atomr_infer_core::audio::VoiceRef::Id`] are accepted as hints;
//! [`atomr_infer_core::audio::VoiceRef::ClonedFrom`] surfaces
//! [`atomr_infer_core::error::InferenceError::BadRequest`].
//!
//! ## Source
//!
//! `FR-TTS-001` (realtime section). Reference: <https://ai.google.dev/api/multimodal-live>.
//! See [`docs/audio-modalities.md`](../../docs/audio-modalities.md).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod cost;
pub mod error;
mod runner;
#[cfg(feature = "tts-gemini-live")]
mod wire;

pub use config::{GeminiLiveApiKey, GeminiLiveConfig};
pub use cost::{per_million_tokens_usd, per_minute_usd};
pub use error::GeminiLiveError;
pub use runner::GeminiLiveRunner;

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runner::RealtimeRunner;
    use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

    #[cfg(not(feature = "tts-gemini-live"))]
    fn runner() -> GeminiLiveRunner {
        GeminiLiveRunner::new_stub()
    }

    #[cfg(feature = "tts-gemini-live")]
    fn runner() -> GeminiLiveRunner {
        use arc_swap::ArcSwap;
        use atomr_infer_remote_core::http::build_client;
        use atomr_infer_remote_core::session::SessionSnapshot;
        use secrecy::SecretString;
        use std::sync::Arc;

        let cfg = GeminiLiveConfig::defaults_for_gemini_live(GeminiLiveApiKey::Env {
            name: "GEMINI_API_KEY".into(),
        });
        let client = build_client(&Default::default(), "test/0").expect("build client");
        let snap = Arc::new(ArcSwap::from_pointee(SessionSnapshot {
            client,
            credential: SecretString::from("fake-api-key".to_string()),
        }));
        GeminiLiveRunner::new(cfg, snap).expect("construct runner")
    }

    #[test]
    fn runner_reports_runtime_kind_and_transport() {
        let r = runner();
        assert_eq!(r.runtime_kind(), RuntimeKind::RealtimeSpeech);
        assert_eq!(
            r.transport_kind(),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::Gemini,
            }
        );
    }

    #[cfg(not(feature = "tts-gemini-live"))]
    #[tokio::test]
    async fn open_session_without_feature_returns_internal_error() {
        use atomr_infer_core::audio::{RealtimeBatch, SynthOptions, VoiceRef};
        use atomr_infer_core::error::InferenceError;
        use tokio::sync::mpsc;

        let mut r = runner();
        let (_tx_in, rx_in) = mpsc::channel(4);
        let (tx_out, _rx_out) = mpsc::channel(4);
        let batch = RealtimeBatch {
            request_id: "t".into(),
            model: "gemini-2.0-flash-exp".into(),
            voice: VoiceRef::Named("default".into()),
            options: SynthOptions::default(),
            inbound: rx_in,
            outbound: tx_out,
        };
        match r.open_session(batch).await {
            Err(InferenceError::Internal(msg)) => {
                assert!(msg.contains("tts-gemini-live feature disabled"), "{msg}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }
}
