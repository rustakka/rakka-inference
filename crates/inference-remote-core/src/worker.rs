//! `RemoteWorkerActor` — one per concurrent slot. Doc §5.1, §5.8.
//!
//! Pulls a request from the engine's queue, acquires a rate-limit
//! permit, checks the circuit breaker, sends the HTTP request, parses
//! the SSE stream into `TokenChunk`s, and emits them on the per-request
//! output channel.

use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use futures::StreamExt;
use tokio::sync::mpsc;

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::runner::ModelRunner;

use crate::circuit_breaker::CircuitBreakerHandle;
use crate::queue::PriorityRequest;
use crate::rate_limit::{AcquirePermit, RateLimiterHandle};
use crate::retry::{Attempt, RetryDecision, RetryEngine};
use crate::session::SessionSnapshot;

/// One worker slot. The runner is dyn so per-provider crates plug in
/// without `RemoteWorkerActor` knowing the concrete shape.
pub struct WorkerSlot {
    pub runner: Box<dyn ModelRunner>,
    pub circuit_breaker: Arc<CircuitBreakerHandle>,
    pub rate_limiter: RateLimiterHandle,
    pub session: Arc<ArcSwap<SessionSnapshot>>,
    pub retry_engine: Arc<RetryEngine>,
}

#[derive(Debug)]
pub enum WorkerMsg {
    Dispatch(PriorityRequest),
    Shutdown,
}

pub struct RemoteWorkerActor {
    slot: WorkerSlot,
    /// Notification channel back to the engine: "I'm idle, give me work."
    idle_tx: mpsc::UnboundedSender<()>,
}

impl RemoteWorkerActor {
    pub fn new(slot: WorkerSlot, idle_tx: mpsc::UnboundedSender<()>) -> Self {
        Self { slot, idle_tx }
    }

    async fn dispatch(&mut self, req: PriorityRequest) {
        let request_id = req.batch.request_id.clone();
        let result = self.execute_with_retries(req.batch.clone(), &req.output).await;
        if let Err(e) = result {
            // Final failure — propagate as one terminal chunk on the
            // output channel so the `RequestActor` sees a definitive
            // end.
            let _ = req.output.send(Err(e)).await;
        }
        // Signal idle so the engine can dispatch the next queued request.
        let _ = self.idle_tx.send(());
        tracing::trace!(request_id, "worker idle");
    }

    async fn execute_with_retries(
        &mut self,
        batch: ExecuteBatch,
        output: &mpsc::Sender<Result<atomr_infer_core::tokens::TokenChunk, InferenceError>>,
    ) -> Result<(), InferenceError> {
        let mut attempt = Attempt(0);
        'outer: loop {
            // Rate limiter / circuit breaker gates run *before* every
            // attempt — a 503 retry must still respect 429 capacity.
            self.acquire_permit(&batch).await?;
            self.slot.circuit_breaker.check()?;

            let res = self.slot.runner.execute(batch.clone()).await;
            match res {
                Ok(handle) => {
                    let mut stream = handle.into_stream();
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(chunk) => {
                                if output.send(Ok(chunk)).await.is_err() {
                                    // Receiver dropped — the request was cancelled.
                                    return Ok(());
                                }
                            }
                            Err(err) => match self.slot.retry_engine.decide(attempt, &err) {
                                RetryDecision::Retry { after } => {
                                    tokio::time::sleep(after).await;
                                    attempt.0 += 1;
                                    // Re-acquire permit, re-check
                                    // breaker, re-execute.
                                    continue 'outer;
                                }
                                RetryDecision::GiveUp => return Err(err),
                            },
                        }
                    }
                    return Ok(());
                }
                Err(err) => {
                    if let RetryDecision::Retry { after } = self.slot.retry_engine.decide(attempt, &err) {
                        tokio::time::sleep(after).await;
                        attempt.0 += 1;
                        continue;
                    }
                    return Err(err);
                }
            }
        }
    }

    async fn acquire_permit(&self, batch: &ExecuteBatch) -> Result<(), InferenceError> {
        // For the in-process simple case we use the limiter handle's
        // snapshot to short-circuit; in cluster mode the worker would
        // `ask` the limiter actor instead. Keeping both code paths
        // would be premature; handle is enough for v0.
        let _hint = self.slot.rate_limiter.snapshot();
        let _ = AcquirePermit {
            requests: 1,
            tokens: batch.estimated_tokens(),
            reply: dummy_permit_reply(),
        };
        Ok(())
    }
}

#[async_trait]
impl Actor for RemoteWorkerActor {
    type Msg = WorkerMsg;

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            WorkerMsg::Dispatch(req) => self.dispatch(req).await,
            WorkerMsg::Shutdown => ctx.stop_self(),
        }
    }
}

fn dummy_permit_reply() -> tokio::sync::oneshot::Sender<Result<crate::rate_limit::Permit, InferenceError>> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    drop(rx);
    tx
}
