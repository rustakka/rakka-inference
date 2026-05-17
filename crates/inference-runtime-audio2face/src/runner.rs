//! `Audio2FaceRunner` — [`A2FRunner`] implementation for NVIDIA Audio2Face-3D.
//!
//! # Build-time gates
//!
//! * **`audio2face` feature off**: every public method returns
//!   `Audio2FaceError::FeatureDisabled` immediately.
//! * **`audio2face` feature on, wrong arch**: returns
//!   `Audio2FaceError::UnsupportedArch` immediately at runtime.
//! * **`audio2face` feature on, linux x86_64**: the real path executes.
//!
//! # Transport (current milestone)
//!
//! The real path uses an **in-memory blendshape generator** that produces
//! deterministic 30 fps frames driven by the input audio's frame count.
//! This validates the full trait-implementation + ARKit-ordering-normalization
//! story without requiring a live NVIDIA A2F-3D server.
//!
//! The gRPC transport (`tonic::Channel` → `PushAudioStream` streaming RPC)
//! is scaffolded in this file with `// TODO(grpc-transport)` markers. A
//! follow-up PR will replace the generator with the real network path once
//! CI has access to an A2F endpoint.

use async_trait::async_trait;

use atomr_infer_core::audio::AudioBatch;
use atomr_infer_core::error::InferenceResult;
use atomr_infer_core::runner::{A2FRunHandle, A2FRunner, SessionRebuildCause};
use atomr_infer_core::runtime::{ProviderKind, RuntimeKind, TransportKind};

use crate::config::Audio2FaceConfig;
use crate::error::Audio2FaceError;

// Imports used only by the feature-on path — suppress dead-code lint.
#[cfg(feature = "audio2face")]
use futures::stream::{self, BoxStream, StreamExt};

#[cfg(feature = "audio2face")]
use atomr_infer_core::audio::{AudioInput, AudioOptions, BlendshapeChunk};

#[cfg(feature = "audio2face")]
use crate::arkit::a2f_to_arkit;

/// Runner for NVIDIA Omniverse Audio2Face-3D.
///
/// Construct via [`Audio2FaceRunner::new`]. See the crate-level docs
/// for gate behaviour and transport status.
pub struct Audio2FaceRunner {
    // Used by the feature-on path for endpoint logging and config.
    #[allow(dead_code)]
    config: Audio2FaceConfig,
    // TODO(grpc-transport): replace with `tonic::transport::Channel` once
    // the real gRPC path is wired. The channel is established lazily on the
    // first `execute_audio2face` call so construction stays cheap.
}

impl Audio2FaceRunner {
    /// Create a new runner from the given config.
    ///
    /// Construction always succeeds (the gRPC channel is established lazily).
    pub fn new(config: Audio2FaceConfig) -> Self {
        Self { config }
    }
}

// ---------------------------------------------------------------------------
// Feature-off stub
// ---------------------------------------------------------------------------

#[cfg(not(feature = "audio2face"))]
#[async_trait]
impl A2FRunner for Audio2FaceRunner {
    async fn execute_audio2face(&mut self, _batch: AudioBatch) -> InferenceResult<A2FRunHandle> {
        Err(Audio2FaceError::FeatureDisabled.into())
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Err(Audio2FaceError::FeatureDisabled.into())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Audio2Face
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork {
            provider: ProviderKind::NvidiaA2F,
        }
    }
}

// ---------------------------------------------------------------------------
// Feature-on path
// ---------------------------------------------------------------------------

#[cfg(feature = "audio2face")]
#[async_trait]
impl A2FRunner for Audio2FaceRunner {
    #[tracing::instrument(
        skip(self, batch),
        fields(request_id = %batch.request_id, model = %batch.model)
    )]
    async fn execute_audio2face(&mut self, batch: AudioBatch) -> InferenceResult<A2FRunHandle> {
        // Arch gate — enforced at runtime so wrong-arch builds still link.
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            return Err(Audio2FaceError::UnsupportedArch.into());
        }

        // Options gate — must be Audio2Face variant.
        let a2f_opts = match &batch.options {
            AudioOptions::Audio2Face(o) => o.clone(),
            _ => {
                return Err(Audio2FaceError::BadRequest(
                    "expected AudioOptions::Audio2Face; got Transcribe variant".into(),
                )
                .into());
            }
        };

        let fps = a2f_opts.fps.unwrap_or(30).max(1);
        let frame_duration_ms = 1_000 / fps.max(1);

        // Derive frame count from input.
        let frame_count = match &batch.input {
            AudioInput::Static(payload) => {
                // Use estimated_units (frame count) if the batch provides it,
                // otherwise default to 1 second at the configured fps.
                if batch.estimated_units > 0 {
                    batch.estimated_units as usize
                } else {
                    // Fall back: estimate from payload size for PCM16LE mono
                    // at 16 kHz. Each sample = 2 bytes, each frame = 1/fps s.
                    match payload {
                        atomr_infer_core::audio::AudioPayload::Bytes { data, params } => {
                            let samples_per_frame = params.sample_rate_hz / fps.max(1);
                            let bytes_per_sample = 2u32; // PCM16LE
                            let total_samples =
                                data.len() as u32 / (bytes_per_sample * params.channels as u32).max(1);
                            (total_samples / samples_per_frame.max(1)).max(1) as usize
                        }
                        _ => fps as usize, // 1 second default for path/url
                    }
                }
            }
            AudioInput::Stream { .. } => {
                // TODO(grpc-transport): for streaming input, forward PCM chunks
                // to A2F via PushAudioStream and yield BlendshapeOutput frames
                // as they arrive. For now, emit 1 second of frames.
                tracing::warn!(
                    "AudioInput::Stream with audio2face is not yet wired to gRPC; \
                     emitting 1-second stub output. TODO(grpc-transport)"
                );
                fps as usize
            }
        };

        let request_id = batch.request_id.clone();
        let frame_count = frame_count.max(1);

        // TODO(grpc-transport): replace the generator below with a real
        // tonic::Channel call to the A2F gRPC service:
        //
        //   let channel = tonic::transport::Channel::from_shared(uri)?
        //       .connect_timeout(self.config.connect_timeout)
        //       .connect()
        //       .await?;
        //   // ... stream PushAudioRequest frames, collect BlendshapeOutput
        //
        // For now: emit a deterministic sine-driven in-memory stream.
        tracing::warn!(
            request_id = %request_id,
            endpoint = %self.config.endpoint,
            "audio2face gRPC transport not yet wired; using in-memory generator. \
             TODO(grpc-transport)"
        );

        let stream = build_generator_stream(request_id, frame_count, frame_duration_ms);

        Ok(A2FRunHandle::streaming(stream))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        // TODO(grpc-transport): drop and re-connect the tonic::Channel.
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::Audio2Face
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::RemoteNetwork {
            provider: ProviderKind::NvidiaA2F,
        }
    }
}

/// Build a deterministic blendshape stream from a frame count + duration.
///
/// Weights are generated using sine waves on a handful of indices to
/// produce visible, non-trivial motion. The pattern is deterministic so
/// tests can assert exact values.
#[cfg(feature = "audio2face")]
fn build_generator_stream(
    request_id: String,
    frame_count: usize,
    frame_duration_ms: u32,
) -> BoxStream<'static, InferenceResult<BlendshapeChunk>> {
    let frames: Vec<InferenceResult<BlendshapeChunk>> = (0..frame_count)
        .map(|i| {
            let t = i as f32 / frame_count.max(1) as f32;
            let timestamp_ms: u32 = (i as u32).saturating_mul(frame_duration_ms);

            // Build raw weights — vary a few mouth/jaw indices via sine.
            let mut raw = [0.0f32; 52];
            // jawOpen (index 17)
            raw[17] = (t * std::f32::consts::TAU * 2.0).sin().abs() * 0.6;
            // mouthSmileLeft (23), mouthSmileRight (24)
            raw[23] = (t * std::f32::consts::TAU * 1.5).sin().abs() * 0.4;
            raw[24] = raw[23];
            // eyeBlinkLeft (0), eyeBlinkRight (7) — blink every ~2 s
            let blink = if (t * 4.0).fract() > 0.9 { 0.8 } else { 0.0 };
            raw[0] = blink;
            raw[7] = blink;

            // Apply A2F→ARKit normalisation (currently passthrough).
            let weights = a2f_to_arkit(&raw);

            Ok(BlendshapeChunk {
                request_id: request_id.clone(),
                is_final: i == frame_count - 1,
                timestamp_ms,
                weights,
            })
        })
        .collect();

    stream::iter(frames).boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::audio::{
        A2FOptions, AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload,
    };
    #[allow(unused_imports)]
    use atomr_infer_core::error::InferenceError;

    #[allow(dead_code)]
    fn make_batch(request_id: &str, frames: u32) -> AudioBatch {
        AudioBatch {
            request_id: request_id.into(),
            model: "audio2face-3d".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: bytes::Bytes::from_static(b"\x00\x00"),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: false,
            options: AudioOptions::Audio2Face(A2FOptions {
                fps: Some(30),
                ..Default::default()
            }),
            estimated_units: frames,
        }
    }

    #[test]
    fn runner_new_succeeds() {
        let cfg = crate::config::Audio2FaceConfig::defaults_for_a2f();
        let _ = Audio2FaceRunner::new(cfg);
    }

    #[test]
    fn runtime_kind_is_audio2face() {
        let r = Audio2FaceRunner::new(crate::config::Audio2FaceConfig::defaults_for_a2f());
        assert_eq!(r.runtime_kind(), RuntimeKind::Audio2Face);
    }

    #[test]
    fn transport_kind_is_nvidia_a2f() {
        let r = Audio2FaceRunner::new(crate::config::Audio2FaceConfig::defaults_for_a2f());
        assert!(matches!(
            r.transport_kind(),
            TransportKind::RemoteNetwork {
                provider: ProviderKind::NvidiaA2F
            }
        ));
    }

    #[cfg(not(feature = "audio2face"))]
    #[tokio::test]
    async fn feature_disabled_returns_error() {
        let mut r = Audio2FaceRunner::new(crate::config::Audio2FaceConfig::defaults_for_a2f());
        let batch = make_batch("req-1", 30);
        let result = r.execute_audio2face(batch).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, InferenceError::Internal(_)));
    }
}
