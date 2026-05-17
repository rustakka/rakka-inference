//! Hand-written protobuf/gRPC wire types for the Audio2Face-3D service.
//!
//! These types mirror the NVIDIA Omniverse Audio2Face-3D gRPC API without
//! requiring `protoc` at build time. The proto contract is approximately:
//!
//! ```protobuf
//! service A2FController {
//!     rpc PushAudioStream(stream PushAudioRequest) returns (stream BlendshapeOutput);
//! }
//! message PushAudioRequest {
//!     bytes audio_chunk = 1;
//!     int32 sample_rate = 2;
//!     int32 channels    = 3;
//! }
//! message BlendshapeOutput {
//!     int64 time_code      = 1;  // microseconds from start
//!     repeated float blend_shapes = 2;  // length 52, A2F-native order
//! }
//! ```
//!
//! For this milestone the in-memory generator (see `runner.rs`) produces
//! [`BlendshapeOutput`] values directly without going over the network.
//! The gRPC plumbing is scaffolded as TODO for when a live A2F server is
//! available.

use prost::Message;

// ---------------------------------------------------------------------------
// Request message
// ---------------------------------------------------------------------------

/// Audio data sent to the A2F gRPC endpoint.
#[derive(Clone, PartialEq, Message)]
pub struct PushAudioRequest {
    /// Raw PCM audio bytes (PCM16LE by default).
    #[prost(bytes = "bytes", tag = "1")]
    pub audio_chunk: bytes::Bytes,

    /// Sampling rate of `audio_chunk` in Hz (e.g. 16000).
    #[prost(int32, tag = "2")]
    pub sample_rate: i32,

    /// Number of audio channels (1 = mono).
    #[prost(int32, tag = "3")]
    pub channels: i32,
}

// ---------------------------------------------------------------------------
// Response message
// ---------------------------------------------------------------------------

/// One frame of blendshape weights returned by the A2F gRPC endpoint.
///
/// `blend_shapes` carries 52 weights in A2F-native ordering. Pass them
/// through [`crate::arkit::a2f_to_arkit`] before surfacing to callers.
#[derive(Clone, PartialEq, Message)]
pub struct BlendshapeOutput {
    /// Frame timestamp in microseconds from the start of the audio stream.
    #[prost(int64, tag = "1")]
    pub time_code: i64,

    /// Blendshape weights in A2F-native ordering. Expect 52 elements on a
    /// well-behaved server; fewer indicates a partial/error frame.
    #[prost(float, repeated, tag = "2")]
    pub blend_shapes: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_audio_request_prost_encode_decode_roundtrip() {
        let req = PushAudioRequest {
            audio_chunk: bytes::Bytes::from_static(&[0u8, 1, 2, 3]),
            sample_rate: 16_000,
            channels: 1,
        };
        let mut buf = Vec::new();
        req.encode(&mut buf).expect("encode should not fail");
        let decoded = PushAudioRequest::decode(buf.as_slice()).expect("decode should not fail");
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.audio_chunk, bytes::Bytes::from_static(&[0, 1, 2, 3]));
    }

    #[test]
    fn blendshape_output_prost_encode_decode_roundtrip() {
        let weights: Vec<f32> = (0..52).map(|i| i as f32 * 0.01).collect();
        let out = BlendshapeOutput {
            time_code: 33_333,
            blend_shapes: weights.clone(),
        };
        let mut buf = Vec::new();
        out.encode(&mut buf).expect("encode should not fail");
        let decoded = BlendshapeOutput::decode(buf.as_slice()).expect("decode should not fail");
        assert_eq!(decoded.time_code, 33_333);
        assert_eq!(decoded.blend_shapes.len(), 52);
        for (a, b) in decoded.blend_shapes.iter().zip(weights.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn empty_blendshape_output() {
        let out = BlendshapeOutput {
            time_code: 0,
            blend_shapes: vec![],
        };
        let mut buf = Vec::new();
        out.encode(&mut buf).unwrap();
        let decoded = BlendshapeOutput::decode(buf.as_slice()).unwrap();
        assert!(decoded.blend_shapes.is_empty());
    }
}
