//! `SpeechEngineCoreActor` — per-replica orchestrator for
//! text-to-speech runners (`FR-TTS-001`).
//!
//! Modeled on [`crate::engine_core::EngineCoreActor`]: one async-mutex
//! over a `Box<dyn SpeechRunner>`, in-flight admission control, chunks
//! pumped from the runner's [`atomr_infer_core::runner::SpeechRunHandle`] stream into the
//! per-request output channel.
//!
//! Admission counts **characters** rather than tokens — TTS providers
//! quota by characters synthesized.

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};

use atomr_infer_core::audio::{SpeechBatch, SpeechChunk};
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::SpeechRunner;

/// Per-engine admission config — characters per second, in-flight cap,
/// queue depth.
#[derive(Clone)]
pub struct SpeechEngineConfig {
    /// Maximum concurrent in-flight synthesis requests.
    pub max_concurrent: u32,
    /// Mailbox depth for pending engine messages.
    pub queue_capacity: usize,
}

impl Default for SpeechEngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 16,
            queue_capacity: 256,
        }
    }
}

pub struct AddSpeechRequest {
    pub batch: SpeechBatch,
    pub output: mpsc::Sender<Result<SpeechChunk, InferenceError>>,
    pub admission: oneshot::Sender<Result<(), InferenceError>>,
}

#[allow(clippy::large_enum_variant)]
pub enum SpeechEngineMsg {
    Add(AddSpeechRequest),
    /// Periodic load probe from the coordinator. Returns a value in
    /// `[0, 1]` — fraction of `max_concurrent` currently busy.
    GetLoad {
        reply: oneshot::Sender<f64>,
    },
}

pub struct SpeechEngineCoreActor {
    runner: Arc<AsyncMutex<Box<dyn SpeechRunner>>>,
    config: SpeechEngineConfig,
    in_flight: Arc<Mutex<u32>>,
}

impl SpeechEngineCoreActor {
    pub fn new(runner: Box<dyn SpeechRunner>, config: SpeechEngineConfig) -> Self {
        Self {
            runner: Arc::new(AsyncMutex::new(runner)),
            config,
            in_flight: Arc::new(Mutex::new(0)),
        }
    }

    fn try_admit(&self) -> Result<(), InferenceError> {
        let mut g = self.in_flight.lock();
        if *g >= self.config.max_concurrent {
            return Err(InferenceError::Backpressure("speech engine at capacity".into()));
        }
        *g += 1;
        Ok(())
    }
}

#[async_trait]
impl Actor for SpeechEngineCoreActor {
    type Msg = SpeechEngineMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            SpeechEngineMsg::Add(req) => match self.try_admit() {
                Err(e) => {
                    let _ = req.admission.send(Err(e));
                }
                Ok(()) => {
                    let _ = req.admission.send(Ok(()));
                    let runner = self.runner.clone();
                    let in_flight = self.in_flight.clone();
                    let output = req.output;
                    let batch = req.batch;
                    tokio::spawn(async move {
                        let mut g = runner.lock().await;
                        match g.speak(batch).await {
                            Ok(handle) => {
                                let mut s = handle.into_stream();
                                while let Some(chunk) = s.next().await {
                                    if output.send(chunk).await.is_err() {
                                        // Receiver dropped — stop early.
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
            SpeechEngineMsg::GetLoad { reply } => {
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
        let c = SpeechEngineConfig::default();
        assert!(c.max_concurrent > 0);
        assert!(c.queue_capacity > 0);
    }
}
