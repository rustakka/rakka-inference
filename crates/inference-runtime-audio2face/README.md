# atomr-infer-runtime-audio2face

NVIDIA Omniverse Audio2Face-3D runtime for [`atomr-infer`].

Implements [`A2FRunner`][atomr_infer_core::A2FRunner] ingesting 16 kHz mono
PCM audio and emitting a stream of [`BlendshapeChunk`][atomr_infer_core::BlendshapeChunk]s
with 52 ARKit-canonical blendshape weights each.

## Feature flag

```toml
atomr-infer-runtime-audio2face = { workspace = true, features = ["audio2face"] }
```

Without `--features audio2face` all APIs compile and link but
`execute_audio2face` returns an error at runtime.

## Architecture gate

Audio2Face-3D requires **Linux x86_64**. On other platforms the feature-on
path still compiles but returns `InferenceError::Internal("audio2face requires
Linux x86_64")` at runtime. This design keeps the crate in the workspace dep
graph on all platforms while the unsupported path fails gracefully.

## ARKit canonical blendshape ordering

`BlendshapeChunk::weights` carries 52 values in Apple's ARKit canonical order
([`ARBlendShapeLocation`][arkit]):

| Index | Name | Index | Name |
|-------|------|-------|------|
| 0 | `eyeBlinkLeft` | 26 | `mouthFrownRight` |
| 1 | `eyeLookDownLeft` | 27 | `mouthDimpleLeft` |
| 2 | `eyeLookInLeft` | 28 | `mouthDimpleRight` |
| 3 | `eyeLookOutLeft` | 29 | `mouthStretchLeft` |
| 4 | `eyeLookUpLeft` | 30 | `mouthStretchRight` |
| 5 | `eyeSquintLeft` | 31 | `mouthRollLower` |
| 6 | `eyeWideLeft` | 32 | `mouthRollUpper` |
| 7 | `eyeBlinkRight` | 33 | `mouthShrugLower` |
| 8 | `eyeLookDownRight` | 34 | `mouthShrugUpper` |
| 9 | `eyeLookInRight` | 35 | `mouthPressLeft` |
| 10 | `eyeLookOutRight` | 36 | `mouthPressRight` |
| 11 | `eyeLookUpRight` | 37 | `mouthLowerDownLeft` |
| 12 | `eyeSquintRight` | 38 | `mouthLowerDownRight` |
| 13 | `eyeWideRight` | 39 | `mouthUpperUpLeft` |
| 14 | `jawForward` | 40 | `mouthUpperUpRight` |
| 15 | `jawLeft` | 41 | `browDownLeft` |
| 16 | `jawRight` | 42 | `browDownRight` |
| 17 | `jawOpen` | 43 | `browInnerUp` |
| 18 | `mouthClose` | 44 | `browOuterUpLeft` |
| 19 | `mouthFunnel` | 45 | `browOuterUpRight` |
| 20 | `mouthPucker` | 46 | `cheekPuff` |
| 21 | `mouthLeft` | 47 | `cheekSquintLeft` |
| 22 | `mouthRight` | 48 | `cheekSquintRight` |
| 23 | `mouthSmileLeft` | 49 | `noseSneerLeft` |
| 24 | `mouthSmileRight` | 50 | `noseSneerRight` |
| 25 | `mouthFrownLeft` | 51 | `tongueOut` |

The A2F→ARKit index permutation is currently a **passthrough** —
`a2f[i]` maps to `arkit[i]` unchanged. The real NVIDIA mapping table
requires the Omniverse Audio2Face-3D documentation. See
`src/arkit.rs:a2f_to_arkit` and the `TODO(a2f-normalization)` marker.

## gRPC protocol

The NVIDIA Audio2Face-3D service exposes approximately:

```protobuf
service A2FController {
    rpc PushAudioStream(stream PushAudioRequest) returns (stream BlendshapeOutput);
}
message PushAudioRequest {
    bytes audio_chunk = 1;  // raw PCM16LE
    int32 sample_rate = 2;
    int32 channels    = 3;
}
message BlendshapeOutput {
    int64 time_code         = 1;  // microseconds
    repeated float blend_shapes = 2;  // length 52, A2F-native order
}
```

Wire types are hand-written using `prost` — no `protoc` or `tonic-build`
build dependency required.

**gRPC transport status:** the current milestone uses an in-memory generator
instead of a live gRPC connection. The transport is scaffolded with
`TODO(grpc-transport)` markers in `src/runner.rs`.

## Quick start

```rust,no_run
use atomr_infer_core::audio::{
    A2FOptions, AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload,
};
use atomr_infer_core::runner::A2FRunner;
use atomr_infer_runtime_audio2face::{Audio2FaceConfig, Audio2FaceRunner};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let cfg = Audio2FaceConfig::defaults_for_a2f()
        .with_endpoint("localhost:50051".into());

    let mut runner = Audio2FaceRunner::new(cfg);

    let batch = AudioBatch {
        request_id: "req-001".into(),
        model: "audio2face-3d".into(),
        input: AudioInput::Static(AudioPayload::Bytes {
            data: bytes::Bytes::new(), // replace with real PCM bytes
            params: AudioParams::a2f_default(),
        }),
        stream: true,
        options: AudioOptions::Audio2Face(A2FOptions {
            target_fps: 30,
            emotion_preset: Some("neutral".into()),
        }),
        estimated_units: 30,
    };

    let handle = runner.execute_audio2face(batch).await.unwrap();
    let mut stream = handle.into_stream();

    while let Some(Ok(chunk)) = stream.next().await {
        println!("t={}ms weights={:?}", chunk.timestamp_ms, &chunk.weights[..5]);
        if chunk.is_final { break; }
    }
}
```

[arkit]: https://developer.apple.com/documentation/arkit/arblendshapelocation
[`atomr-infer`]: https://docs.rs/atomr-infer
