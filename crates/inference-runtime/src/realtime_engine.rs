//! `RealtimeEngineCoreActor` — per-replica orchestrator for
//! bidirectional realtime speech sessions (`FR-TTS-001`, realtime
//! section).
//!
//! Structurally different from the other engine actors: each
//! [`RealtimeBatch`] becomes a long-lived session, not a one-shot
//! batch. The actor holds a `Box<dyn RealtimeRunner>` whose
//! `open_session` spawns its own session adapter task; this actor
//! supervises that task's lifetime via an `outbound` relay so the
//! per-replica admission slot is released when either side of the
//! session drops.
//!
//! Admission unit is **concurrent sessions** rather than tokens /
//! characters / audio-seconds.

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};

use atomr_infer_core::audio::{RealtimeBatch, RealtimeOut};
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::{RealtimeRunner, RealtimeSession};

/// Per-engine admission config.
#[derive(Clone)]
pub struct RealtimeEngineConfig {
    /// Maximum number of concurrent live sessions.
    pub max_concurrent: u32,
    /// Mailbox depth for pending engine messages.
    pub queue_capacity: usize,
    /// Capacity of the internal outbound relay channel that bridges
    /// the runner's adapter task to the caller's outbound receiver.
    /// Bounded to apply backpressure on slow consumers.
    pub outbound_relay_capacity: usize,
}

impl Default for RealtimeEngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 8,
            queue_capacity: 64,
            outbound_relay_capacity: 64,
        }
    }
}

pub struct OpenSessionRequest {
    pub batch: RealtimeBatch,
    /// Returns the session handle on success, or admission/error on
    /// failure. The caller uses the session for cancel + telemetry.
    pub admission: oneshot::Sender<Result<RealtimeSession, InferenceError>>,
}

#[allow(clippy::large_enum_variant)]
pub enum RealtimeEngineMsg {
    OpenSession(OpenSessionRequest),
    /// Periodic load probe — fraction of `max_concurrent` currently
    /// occupied by live sessions.
    GetLoad {
        reply: oneshot::Sender<f64>,
    },
}

pub struct RealtimeEngineCoreActor {
    runner: Arc<AsyncMutex<Box<dyn RealtimeRunner>>>,
    config: RealtimeEngineConfig,
    in_flight: Arc<Mutex<u32>>,
}

impl RealtimeEngineCoreActor {
    pub fn new(runner: Box<dyn RealtimeRunner>, config: RealtimeEngineConfig) -> Self {
        Self {
            runner: Arc::new(AsyncMutex::new(runner)),
            config,
            in_flight: Arc::new(Mutex::new(0)),
        }
    }

    fn try_admit(&self) -> Result<(), InferenceError> {
        let mut g = self.in_flight.lock();
        if *g >= self.config.max_concurrent {
            return Err(InferenceError::Backpressure("realtime engine at capacity".into()));
        }
        *g += 1;
        Ok(())
    }
}

#[async_trait]
impl Actor for RealtimeEngineCoreActor {
    type Msg = RealtimeEngineMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            RealtimeEngineMsg::OpenSession(req) => match self.try_admit() {
                Err(e) => {
                    let _ = req.admission.send(Err(e));
                }
                Ok(()) => {
                    let runner = self.runner.clone();
                    let in_flight = self.in_flight.clone();
                    let relay_cap = self.config.outbound_relay_capacity;

                    // Intercept the caller's outbound sender so we can
                    // observe session end (either the runner closes its
                    // adapter or the caller drops the receiver) and
                    // release the admission slot accordingly.
                    let OpenSessionRequest { mut batch, admission } = req;
                    let caller_outbound = std::mem::replace(
                        &mut batch.outbound,
                        // Placeholder; immediately replaced below.
                        mpsc::channel(1).0,
                    );
                    let (relay_tx, mut relay_rx) = mpsc::channel::<RealtimeOut>(relay_cap);
                    batch.outbound = relay_tx;

                    tokio::spawn(async move {
                        let res = {
                            let mut g = runner.lock().await;
                            g.open_session(batch).await
                        };
                        match res {
                            Ok(session) => {
                                let _ = admission.send(Ok(session));
                                while let Some(out_msg) = relay_rx.recv().await {
                                    if caller_outbound.send(out_msg).await.is_err() {
                                        // Caller dropped its outbound
                                        // receiver. Drain remaining
                                        // adapter output by dropping
                                        // relay_rx — the runner's
                                        // adapter will see its outbound
                                        // close and shut down.
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = admission.send(Err(e));
                            }
                        }
                        let mut g = in_flight.lock();
                        *g = g.saturating_sub(1);
                    });
                }
            },
            RealtimeEngineMsg::GetLoad { reply } => {
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
        let c = RealtimeEngineConfig::default();
        assert!(c.max_concurrent > 0);
        assert!(c.queue_capacity > 0);
        assert!(c.outbound_relay_capacity > 0);
    }
}
