//! `AudioEngineCoreActor` — per-replica orchestrator for STT
//! (`FR-STT-001`) and Audio2Face (`FR-A2F-001`).
//!
//! Holds **either** a `Box<dyn AudioRunner>` (STT) **or** a
//! `Box<dyn A2FRunner>` (A2F) — both ingest [`AudioBatch`] but emit
//! different chunk types, so the actor has two message variants
//! ([`AudioEngineMsg::AddTranscribe`], [`AudioEngineMsg::AddAudio2Face`])
//! and an internal `RunnerKind` discriminator. Mixing the two in a
//! single actor would be an error; constructors enforce one or the
//! other.
//!
//! Admission counts **audio-seconds** for STT and **frames** for A2F.

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};

use atomr_infer_core::audio::{AudioBatch, BlendshapeChunk, TranscriptChunk};
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::{A2FRunner, AudioRunner};
use atomr_infer_core::runtime::RuntimeKind;

#[derive(Clone)]
pub struct AudioEngineConfig {
    /// Maximum concurrent in-flight audio requests.
    pub max_concurrent: u32,
    /// Mailbox depth for pending engine messages.
    pub queue_capacity: usize,
}

impl Default for AudioEngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 16,
            queue_capacity: 256,
        }
    }
}

enum RunnerCell {
    Stt(Arc<AsyncMutex<Box<dyn AudioRunner>>>),
    A2F(Arc<AsyncMutex<Box<dyn A2FRunner>>>),
}

pub struct AddTranscribeRequest {
    pub batch: AudioBatch,
    pub output: mpsc::Sender<Result<TranscriptChunk, InferenceError>>,
    pub admission: oneshot::Sender<Result<(), InferenceError>>,
}

pub struct AddAudio2FaceRequest {
    pub batch: AudioBatch,
    pub output: mpsc::Sender<Result<BlendshapeChunk, InferenceError>>,
    pub admission: oneshot::Sender<Result<(), InferenceError>>,
}

#[allow(clippy::large_enum_variant)]
pub enum AudioEngineMsg {
    AddTranscribe(AddTranscribeRequest),
    AddAudio2Face(AddAudio2FaceRequest),
    GetLoad { reply: oneshot::Sender<f64> },
}

pub struct AudioEngineCoreActor {
    runner: RunnerCell,
    config: AudioEngineConfig,
    in_flight: Arc<Mutex<u32>>,
}

impl AudioEngineCoreActor {
    /// Build an STT-shaped engine actor.
    pub fn new_stt(runner: Box<dyn AudioRunner>, config: AudioEngineConfig) -> Self {
        Self {
            runner: RunnerCell::Stt(Arc::new(AsyncMutex::new(runner))),
            config,
            in_flight: Arc::new(Mutex::new(0)),
        }
    }

    /// Build an A2F-shaped engine actor.
    pub fn new_audio2face(runner: Box<dyn A2FRunner>, config: AudioEngineConfig) -> Self {
        Self {
            runner: RunnerCell::A2F(Arc::new(AsyncMutex::new(runner))),
            config,
            in_flight: Arc::new(Mutex::new(0)),
        }
    }

    fn try_admit(&self) -> Result<(), InferenceError> {
        let mut g = self.in_flight.lock();
        if *g >= self.config.max_concurrent {
            return Err(InferenceError::Backpressure("audio engine at capacity".into()));
        }
        *g += 1;
        Ok(())
    }

    fn reject_modality_mismatch<T>(
        method: &str,
        runtime: RuntimeKind,
        output: mpsc::Sender<Result<T, InferenceError>>,
        admission: oneshot::Sender<Result<(), InferenceError>>,
    ) where
        T: Send + 'static,
    {
        let err = InferenceError::Unsupported {
            method: method.into(),
            runtime,
        };
        let _ = admission.send(Ok(()));
        tokio::spawn(async move {
            let _ = output.send(Err(err)).await;
        });
    }
}

#[async_trait]
impl Actor for AudioEngineCoreActor {
    type Msg = AudioEngineMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            AudioEngineMsg::AddTranscribe(req) => match &self.runner {
                RunnerCell::A2F(_) => {
                    Self::reject_modality_mismatch(
                        "execute_audio",
                        RuntimeKind::Audio2Face,
                        req.output,
                        req.admission,
                    );
                }
                RunnerCell::Stt(runner) => match self.try_admit() {
                    Err(e) => {
                        let _ = req.admission.send(Err(e));
                    }
                    Ok(()) => {
                        let _ = req.admission.send(Ok(()));
                        let runner = runner.clone();
                        let in_flight = self.in_flight.clone();
                        let output = req.output;
                        let batch = req.batch;
                        tokio::spawn(async move {
                            let mut g = runner.lock().await;
                            match g.execute_audio(batch).await {
                                Ok(handle) => {
                                    let mut s = handle.into_stream();
                                    while let Some(chunk) = s.next().await {
                                        if output.send(chunk).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = output.send(Err(e)).await;
                                }
                            }
                            drop(g);
                            let mut g = in_flight.lock();
                            *g = g.saturating_sub(1);
                        });
                    }
                },
            },
            AudioEngineMsg::AddAudio2Face(req) => match &self.runner {
                RunnerCell::Stt(_) => {
                    Self::reject_modality_mismatch(
                        "execute_audio2face",
                        RuntimeKind::SpeechToText,
                        req.output,
                        req.admission,
                    );
                }
                RunnerCell::A2F(runner) => match self.try_admit() {
                    Err(e) => {
                        let _ = req.admission.send(Err(e));
                    }
                    Ok(()) => {
                        let _ = req.admission.send(Ok(()));
                        let runner = runner.clone();
                        let in_flight = self.in_flight.clone();
                        let output = req.output;
                        let batch = req.batch;
                        tokio::spawn(async move {
                            let mut g = runner.lock().await;
                            match g.execute_audio2face(batch).await {
                                Ok(handle) => {
                                    let mut s = handle.into_stream();
                                    while let Some(chunk) = s.next().await {
                                        if output.send(chunk).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = output.send(Err(e)).await;
                                }
                            }
                            drop(g);
                            let mut g = in_flight.lock();
                            *g = g.saturating_sub(1);
                        });
                    }
                },
            },
            AudioEngineMsg::GetLoad { reply } => {
                let load = *self.in_flight.lock() as f64 / self.config.max_concurrent as f64;
                let _ = reply.send(load);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_sane() {
        let c = AudioEngineConfig::default();
        assert!(c.max_concurrent > 0);
        assert!(c.queue_capacity > 0);
    }
}
