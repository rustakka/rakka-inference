//! `EngineCoreActor` — local-GPU per-replica orchestrator. Doc §4, §5.1.
//!
//! Wraps a `Box<dyn ModelRunner>` whose `transport_kind() ==
//! LocalGpu`. The continuous-batch scheduler and KV-cache manager are
//! per-runtime *modules* (vLLM has them; TensorRT/ORT batch by
//! stacking inputs); this actor just owns the runner, dispatches
//! `ExecuteBatch` requests through it, and pumps the resulting chunk
//! stream into the per-request output channel.
//!
//! `RemoteEngineCoreActor` (in `inference-remote-core`) is the
//! network-shaped sibling.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;
use rakka_core::actor::{Actor, Context};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};

use inference_core::batch::ExecuteBatch;
use inference_core::error::InferenceError;
use inference_core::runner::ModelRunner;
use inference_core::tokens::TokenChunk;

#[derive(Clone)]
pub struct LocalEngineConfig {
    pub max_concurrent: u32,
    pub queue_capacity: usize,
}

impl Default for LocalEngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 32,
            queue_capacity: 1024,
        }
    }
}

pub struct AddRequest {
    pub batch: ExecuteBatch,
    pub output: mpsc::Sender<Result<TokenChunk, InferenceError>>,
    pub admission: oneshot::Sender<Result<(), InferenceError>>,
}

pub enum EngineCoreMsg {
    Add(AddRequest),
    /// Request a load-score snapshot. Used by `DpCoordinatorActor`'s
    /// periodic poll.
    GetLoad {
        reply: oneshot::Sender<f64>,
    },
}

pub struct EngineCoreActor {
    /// Async mutex because `ModelRunner::execute` is held across an
    /// await; a `parking_lot::Mutex` guard would not be `Send` over
    /// the await boundary.
    runner: Arc<AsyncMutex<Box<dyn ModelRunner>>>,
    config: LocalEngineConfig,
    in_flight: Arc<Mutex<u32>>,
}

impl EngineCoreActor {
    pub fn new(runner: Box<dyn ModelRunner>, config: LocalEngineConfig) -> Self {
        Self {
            runner: Arc::new(AsyncMutex::new(runner)),
            config,
            in_flight: Arc::new(Mutex::new(0)),
        }
    }

    fn try_admit(&self) -> Result<(), InferenceError> {
        let mut g = self.in_flight.lock();
        if *g >= self.config.max_concurrent {
            return Err(InferenceError::Backpressure("engine at capacity".into()));
        }
        *g += 1;
        Ok(())
    }

    fn release(&self) {
        let mut g = self.in_flight.lock();
        *g = g.saturating_sub(1);
    }
}

#[async_trait]
impl Actor for EngineCoreActor {
    type Msg = EngineCoreMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            EngineCoreMsg::Add(req) => match self.try_admit() {
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
                        // Hold the async mutex across the execute()
                        // await — single runner owns the GPU context
                        // exclusively for the duration of a batched
                        // step. For runtimes that batch across
                        // requests (vLLM), `execute` returns quickly
                        // after enqueueing onto the engine's internal
                        // step loop.
                        let mut g = runner.lock().await;
                        match g.execute(batch).await {
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
                        let mut g = in_flight.lock();
                        *g = g.saturating_sub(1);
                    });
                    self.release();
                }
            },
            EngineCoreMsg::GetLoad { reply } => {
                let load = *self.in_flight.lock() as f64 / self.config.max_concurrent as f64;
                let _ = reply.send(load);
            }
        }
    }
}
