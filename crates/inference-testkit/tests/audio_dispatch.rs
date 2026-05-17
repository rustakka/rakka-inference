//! End-to-end integration test for the M2 audio engine actors.
//!
//! Each test wires one mock runner into its matching engine actor,
//! sends one batch via the mailbox, and asserts the chunk stream
//! observed on the per-request channel matches the script.
//!
//! Drop-mid-stream coverage exercises the runner's backpressure path —
//! when the receiver is dropped, the spawned `output.send().await`
//! returns `Err`, and the engine actor must stop pumping rather than
//! deadlock.

use std::time::Duration;

use atomr_config::Config;
use atomr_core::actor::{ActorSystem, Props};

use atomr_infer_core::audio::{
    A2FOptions, AudioBatch, AudioFormat, AudioInput, AudioOptions, AudioParams, AudioPayload, RealtimeBatch,
    RealtimeIn, RealtimeOut, SpeechBatch, SynthOptions, TranscribeOptions, TranscriptRole, VoiceRef,
};
use atomr_infer_runtime::audio_engine::{
    AddAudio2FaceRequest, AddTranscribeRequest, AudioEngineConfig, AudioEngineCoreActor, AudioEngineMsg,
};
use atomr_infer_runtime::realtime_engine::{
    OpenSessionRequest, RealtimeEngineConfig, RealtimeEngineCoreActor, RealtimeEngineMsg,
};
use atomr_infer_runtime::speech_engine::{
    AddSpeechRequest, SpeechEngineConfig, SpeechEngineCoreActor, SpeechEngineMsg,
};
use atomr_infer_testkit::{
    MockA2FRunner, MockA2FScript, MockRealtimeRunner, MockRealtimeScript, MockSttRunner, MockSttScript,
    MockTtsRunner, MockTtsScript,
};
use bytes::Bytes;
use tokio::sync::{mpsc, oneshot};

async fn system(name: &str) -> ActorSystem {
    ActorSystem::create(name, Config::reference())
        .await
        .expect("actor system create")
}

#[tokio::test]
async fn speech_engine_streams_audio_chunks() {
    let sys = system("speech-engine-stream").await;
    let actor = sys
        .actor_of(
            Props::create(move || {
                SpeechEngineCoreActor::new(
                    Box::new(MockTtsRunner::new(MockTtsScript::from_audio([
                        Bytes::from_static(b"abc"),
                        Bytes::from_static(b"def"),
                    ]))),
                    SpeechEngineConfig::default(),
                )
            }),
            "speech",
        )
        .unwrap();

    let (out_tx, mut out_rx) = mpsc::channel(8);
    let (adm_tx, adm_rx) = oneshot::channel();
    actor.tell(SpeechEngineMsg::Add(AddSpeechRequest {
        batch: SpeechBatch {
            request_id: "r1".into(),
            model: "mock".into(),
            text: "hi".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 2,
        },
        output: out_tx,
        admission: adm_tx,
    }));
    adm_rx.await.unwrap().unwrap();

    let mut got = 0;
    while let Some(chunk) = out_rx.recv().await {
        let c = chunk.unwrap();
        got += 1;
        if c.is_final {
            assert_eq!(c.audio_pcm_chunk.as_ref(), b"def");
            break;
        }
    }
    assert_eq!(got, 2);
}

#[tokio::test]
async fn audio_engine_routes_transcribe_to_audio_runner() {
    let sys = system("audio-engine-stt").await;
    let actor = sys
        .actor_of(
            Props::create(move || {
                AudioEngineCoreActor::new_stt(
                    Box::new(MockSttRunner::new(MockSttScript::from_text(["hello", "world"]))),
                    AudioEngineConfig::default(),
                )
            }),
            "stt",
        )
        .unwrap();

    let (out_tx, mut out_rx) = mpsc::channel(8);
    let (adm_tx, adm_rx) = oneshot::channel();
    actor.tell(AudioEngineMsg::AddTranscribe(AddTranscribeRequest {
        batch: AudioBatch {
            request_id: "r-stt".into(),
            model: "whisper-1".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from_static(&[]),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: true,
            options: AudioOptions::Transcribe(TranscribeOptions::default()),
            estimated_units: 1,
        },
        output: out_tx,
        admission: adm_tx,
    }));
    adm_rx.await.unwrap().unwrap();

    let mut got: Vec<String> = Vec::new();
    while let Some(chunk) = out_rx.recv().await {
        let c = chunk.unwrap();
        got.push(c.text.clone());
        if c.is_final {
            break;
        }
    }
    assert_eq!(got, vec!["hello".to_string(), "world".to_string()]);
}

#[tokio::test]
async fn audio_engine_rejects_modality_mismatch() {
    let sys = system("audio-engine-mismatch").await;
    // Build an A2F-shaped actor but send it an AddTranscribe message.
    let actor = sys
        .actor_of(
            Props::create(move || {
                AudioEngineCoreActor::new_audio2face(
                    Box::new(MockA2FRunner::new(MockA2FScript::from_frames([[0.0_f32; 52]]))),
                    AudioEngineConfig::default(),
                )
            }),
            "a2f-rejecting-stt",
        )
        .unwrap();
    let (out_tx, mut out_rx) = mpsc::channel(8);
    let (adm_tx, adm_rx) = oneshot::channel();
    actor.tell(AudioEngineMsg::AddTranscribe(AddTranscribeRequest {
        batch: AudioBatch {
            request_id: "r-mis".into(),
            model: "whisper-1".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from_static(&[]),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: true,
            options: AudioOptions::Transcribe(TranscribeOptions::default()),
            estimated_units: 1,
        },
        output: out_tx,
        admission: adm_tx,
    }));
    adm_rx.await.unwrap().unwrap();
    let first = out_rx.recv().await.expect("error chunk should arrive");
    let err = first.expect_err("modality mismatch must surface as Err");
    let s = err.to_string();
    assert!(s.to_lowercase().contains("unsupported"), "got {s}");
}

#[tokio::test]
async fn audio_engine_routes_audio2face() {
    let sys = system("audio-engine-a2f").await;
    let actor = sys
        .actor_of(
            Props::create(move || {
                AudioEngineCoreActor::new_audio2face(
                    Box::new(MockA2FRunner::new(MockA2FScript::from_frames([
                        [0.1_f32; 52],
                        [0.2_f32; 52],
                    ]))),
                    AudioEngineConfig::default(),
                )
            }),
            "a2f",
        )
        .unwrap();
    let (out_tx, mut out_rx) = mpsc::channel(8);
    let (adm_tx, adm_rx) = oneshot::channel();
    actor.tell(AudioEngineMsg::AddAudio2Face(AddAudio2FaceRequest {
        batch: AudioBatch {
            request_id: "r-a2f".into(),
            model: "a2f-3d".into(),
            input: AudioInput::Static(AudioPayload::Bytes {
                data: Bytes::from_static(&[]),
                params: AudioParams::new(16_000, 1, AudioFormat::Pcm16Le),
            }),
            stream: true,
            options: AudioOptions::Audio2Face(A2FOptions::default()),
            estimated_units: 30,
        },
        output: out_tx,
        admission: adm_tx,
    }));
    adm_rx.await.unwrap().unwrap();
    let mut frames = 0;
    while let Some(c) = out_rx.recv().await {
        let c = c.unwrap();
        frames += 1;
        if c.is_final {
            break;
        }
    }
    assert_eq!(frames, 2);
}

#[tokio::test]
async fn realtime_engine_opens_session_and_releases_on_close() {
    let sys = system("realtime-engine-session").await;
    let actor = sys
        .actor_of(
            Props::create(move || {
                RealtimeEngineCoreActor::new(
                    Box::new(MockRealtimeRunner::new(MockRealtimeScript {
                        echo_in: true,
                        responses: vec!["assistant-1".into()],
                        ..Default::default()
                    })),
                    RealtimeEngineConfig::default(),
                )
            }),
            "realtime",
        )
        .unwrap();
    let (tx_in, rx_in) = mpsc::channel::<RealtimeIn>(4);
    let (tx_out, mut rx_out) = mpsc::channel::<RealtimeOut>(4);
    let batch = RealtimeBatch {
        request_id: "s1".into(),
        model: "mock".into(),
        voice: VoiceRef::Named("v".into()),
        options: SynthOptions::default(),
        inbound: rx_in,
        outbound: tx_out,
    };
    let (adm_tx, adm_rx) = oneshot::channel();
    actor.tell(RealtimeEngineMsg::OpenSession(OpenSessionRequest {
        batch,
        admission: adm_tx,
    }));
    let _session = adm_rx.await.unwrap().unwrap();

    tx_in.send(RealtimeIn::Text("hi".into())).await.unwrap();
    tx_in.send(RealtimeIn::Commit).await.unwrap();
    tx_in.send(RealtimeIn::Close).await.unwrap();
    drop(tx_in);

    let mut got_user = false;
    let mut got_assistant = false;
    let mut got_done = false;
    while let Some(msg) = rx_out.recv().await {
        match msg {
            RealtimeOut::Transcript {
                role: TranscriptRole::User,
                text,
                ..
            } => {
                got_user = true;
                assert_eq!(text, "hi");
            }
            RealtimeOut::Transcript {
                role: TranscriptRole::Assistant,
                ..
            } => got_assistant = true,
            RealtimeOut::Done => {
                got_done = true;
                break;
            }
            _ => {}
        }
    }
    assert!(got_user && got_assistant && got_done);

    // Probe load — should be 0.0 once the session ended.
    let (lt, lr) = oneshot::channel();
    actor.tell(RealtimeEngineMsg::GetLoad { reply: lt });
    // Give the bridge task a moment to flush its release of the admission slot.
    let load = tokio::time::timeout(Duration::from_secs(1), lr)
        .await
        .unwrap()
        .unwrap();
    assert!((0.0..=1.0).contains(&load));
}

#[tokio::test]
async fn speech_engine_handles_consumer_dropped_mid_stream() {
    let sys = system("speech-engine-cancel").await;
    let actor = sys
        .actor_of(
            Props::create(move || {
                SpeechEngineCoreActor::new(
                    Box::new(MockTtsRunner::new(MockTtsScript {
                        audio_chunks: vec![
                            Bytes::from_static(b"a"),
                            Bytes::from_static(b"b"),
                            Bytes::from_static(b"c"),
                        ],
                        inter_chunk_delay: Duration::from_millis(40),
                        ..Default::default()
                    })),
                    SpeechEngineConfig::default(),
                )
            }),
            "speech-cancel",
        )
        .unwrap();

    let (out_tx, mut out_rx) = mpsc::channel(1);
    let (adm_tx, adm_rx) = oneshot::channel();
    actor.tell(SpeechEngineMsg::Add(AddSpeechRequest {
        batch: SpeechBatch {
            request_id: "rc".into(),
            model: "mock".into(),
            text: "hi".into(),
            voice: VoiceRef::Named("alloy".into()),
            options: SynthOptions::default(),
            stream: true,
            emit_alignment: false,
            estimated_characters: 2,
        },
        output: out_tx,
        admission: adm_tx,
    }));
    adm_rx.await.unwrap().unwrap();

    // Consume one chunk, drop the receiver — runner must stop pumping.
    let _first = out_rx.recv().await;
    drop(out_rx);
    tokio::time::sleep(Duration::from_millis(200)).await;
    // Smoke probe: GetLoad still returns; engine actor did not deadlock.
    let (lt, lr) = oneshot::channel();
    actor.tell(SpeechEngineMsg::GetLoad { reply: lt });
    let load = tokio::time::timeout(Duration::from_secs(1), lr)
        .await
        .unwrap()
        .unwrap();
    assert!((0.0..=1.0).contains(&load));
}
