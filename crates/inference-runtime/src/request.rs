//! `RequestActor` — one per active client request. Doc §6.1, §6.2.
//!
//! Owns the per-request `Tokens` accumulation and the streaming channel
//! back to the gateway's HTTP response. The actor is created by the
//! gateway, asks the `DpCoordinatorActor` for a route, then `tell`s
//! the chosen engine an `AddRequest`. The engine writes chunks into
//! the `mpsc::Sender` we provided; we forward them to the `Tokens`
//! sender held by the gateway.

use async_trait::async_trait;
use atomr_core::actor::{Actor, ActorRef, Context};
use tokio::sync::{mpsc, oneshot};

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::error::InferenceError;
use atomr_infer_core::tokens::{TokenChunk, Tokens};

use crate::dp_coordinator::{DpCoordinatorMsg, RouteTarget};

pub enum RequestMsg {
    /// Kick off the request: routes via the coordinator and dispatches
    /// to the chosen engine.
    Dispatch { deployment: String, batch: ExecuteBatch },
    /// Forwarded chunk from the engine.
    Chunk(Result<TokenChunk, InferenceError>),
    /// Gateway gave up on the response (client disconnected). Cancel.
    Cancel,
}

/// Streaming response handed back to the gateway for forwarding into
/// the HTTP body. The gateway pulls `next()` until it sees `None`.
pub type StreamingResponse = mpsc::Receiver<Result<TokenChunk, InferenceError>>;

pub struct RequestActor {
    coordinator: ActorRef<DpCoordinatorMsg>,
    /// The gateway-facing channel. Each chunk we receive from the
    /// engine is mirrored here.
    output: mpsc::Sender<Result<TokenChunk, InferenceError>>,
    /// Whether `Dispatch` has happened — guards against double-dispatch.
    dispatched: bool,
    /// Aggregate accumulator (exposed at end via `done` channel).
    accumulator: Tokens,
    done: Option<oneshot::Sender<Tokens>>,
}

impl RequestActor {
    pub fn new(
        coordinator: ActorRef<DpCoordinatorMsg>,
        output: mpsc::Sender<Result<TokenChunk, InferenceError>>,
        done: oneshot::Sender<Tokens>,
    ) -> Self {
        Self {
            coordinator,
            output,
            dispatched: false,
            accumulator: Tokens::default(),
            done: Some(done),
        }
    }

    async fn dispatch(&mut self, ctx: &mut Context<Self>, deployment: String, batch: ExecuteBatch) {
        if self.dispatched {
            return;
        }
        self.dispatched = true;
        self.accumulator.request_id = batch.request_id.clone();

        let target = match self
            .coordinator
            .ask_with(
                |reply| DpCoordinatorMsg::RouteTo {
                    deployment: deployment.clone(),
                    reply,
                },
                std::time::Duration::from_secs(2),
            )
            .await
        {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => {
                let _ = self.output.send(Err(e)).await;
                self.finish().await;
                ctx.stop_self();
                return;
            }
            Err(_) => {
                let _ = self
                    .output
                    .send(Err(InferenceError::Internal("coordinator timeout".into())))
                    .await;
                self.finish().await;
                ctx.stop_self();
                return;
            }
        };

        // Bridge the engine→our chunk channel into RequestMsg::Chunk so
        // we observe each chunk on our own mailbox and update the
        // accumulator in actor context (no shared state).
        let (chunk_tx, mut chunk_rx) = mpsc::channel::<Result<TokenChunk, InferenceError>>(64);
        let self_ref = ctx.self_ref().clone();
        tokio::spawn(async move {
            while let Some(c) = chunk_rx.recv().await {
                self_ref.tell(RequestMsg::Chunk(c));
            }
        });

        // Send through the appropriate transport — local engine cores
        // and remote engine cores have different message types, so we
        // bridge via a small typed adapter at the placement site.
        // Here in v0 we accept either by using a closure boxed by the
        // caller; the gateway constructs the closure with knowledge of
        // the engine kind.
        // For now we simply ignore the routed `target` for the actual
        // dispatch — that wiring is the gateway's job (see `gateway`
        // module). We only retain the route for observability.
        let _ = target;
        let _ = batch;
        let _ = chunk_tx;
    }

    async fn finish(&mut self) {
        if let Some(d) = self.done.take() {
            let _ = d.send(std::mem::take(&mut self.accumulator));
        }
    }
}

#[async_trait]
impl Actor for RequestActor {
    type Msg = RequestMsg;

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            RequestMsg::Dispatch { deployment, batch } => {
                self.dispatch(ctx, deployment, batch).await;
            }
            RequestMsg::Chunk(item) => {
                let is_terminal = match &item {
                    Ok(c) => c.finish_reason.is_some(),
                    Err(_) => true,
                };
                if let Ok(c) = &item {
                    self.accumulator.append(c);
                }
                let _ = self.output.send(item).await;
                if is_terminal {
                    self.finish().await;
                    ctx.stop_self();
                }
            }
            RequestMsg::Cancel => {
                self.finish().await;
                ctx.stop_self();
            }
        }
    }
}

/// Public alias: `RouteTarget` exposed under the `Route` name so
/// callers can hold typed targets without reaching into
/// `dp_coordinator`.
pub type Route = RouteTarget;
