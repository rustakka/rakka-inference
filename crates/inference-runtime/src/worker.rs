//! Local-GPU worker — two-tier supervision adapter (doc §4, §5.3).
//!
//! `WorkerActor` is the **stable** parent: addressable, supervised by
//! the engine-core, never restarts. Its child `ContextActor` is
//! **restartable** and owns the runtime-specific resources (CUDA
//! context, weights, etc). When the runner reports
//! `CudaContextPoisoned` the parent panics with the
//! [`rakka_accel::cuda::error::CONTEXT_POISONED_TAG`] marker so that
//! [`rakka_accel::cuda::error::device_supervisor_strategy`] routes the
//! failure to `Directive::Restart`.
//!
//! The supervision *policy* (3 retries / 60s, decider, marker tags) is
//! re-used verbatim from rakka-accel's `error` module — that's the
//! upstream substrate for the doc's §5.11 two-tier pattern. The
//! *body* this crate adds is the runtime-polymorphic
//! `Box<dyn ModelRunner>` slot, which is inference-specific.
//!
//! Per-runtime crates supply the runner via the `WorkerSlot` factory.
//! Remote runtimes go through `inference-remote-core::RemoteWorkerActor`
//! instead.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use rakka_core::supervision::SupervisorStrategy;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::{mpsc, oneshot};

use inference_core::batch::ExecuteBatch;
use inference_core::error::InferenceError;
use inference_core::runner::{ModelRunner, SessionRebuildCause};
use inference_core::tokens::TokenChunk;

/// What the parent hands to its child on construction. The runner
/// owns the GPU context indirectly (via `cudarc::driver::CudaContext`,
/// `rakka_accel::cuda::device::DeviceState`, or whatever the backend uses);
/// when the parent decides to rebuild, it constructs a fresh
/// `WorkerSlot` and the child cell starts anew.
pub struct WorkerSlot {
    pub runner: Box<dyn ModelRunner>,
}

pub enum WorkerMsg {
    Execute(ExecuteBatch, mpsc::Sender<Result<TokenChunk, InferenceError>>),
    /// Forwarded from the runner when a sticky CUDA error is detected.
    /// Triggers a child restart.
    ContextPoisoned(String),
    /// Operator-triggered rebuild.
    RebuildSession {
        cause: SessionRebuildCause,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },
}

pub enum ContextMsg {
    Execute(ExecuteBatch, mpsc::Sender<Result<TokenChunk, InferenceError>>),
    Rebuild {
        cause: SessionRebuildCause,
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },
}

pub struct WorkerActor {
    /// Slot factory — invoked once on initial child spawn and once per
    /// rebuild. Per-runtime crates supply this.
    slot_factory: Box<dyn Fn() -> WorkerSlot + Send + Sync>,
    child: Option<ActorRef<ContextMsg>>,
    parent_to_child_seq: u64,
}

impl WorkerActor {
    pub fn new<F>(slot_factory: F) -> Self
    where
        F: Fn() -> WorkerSlot + Send + Sync + 'static,
    {
        Self { slot_factory: Box::new(slot_factory), child: None, parent_to_child_seq: 0 }
    }

    fn spawn_child(&mut self, ctx: &mut Context<Self>) {
        // Factory is called once per spawn. ContextActor itself isn't
        // restarted by rakka's supervisor — we tear it down and spawn
        // a fresh one with a new slot when context poisoning happens.
        self.parent_to_child_seq += 1;
        let name = format!("ctx-{}", self.parent_to_child_seq);
        let cell = Mutex::new(Some((self.slot_factory)()));
        let props = Props::create(move || {
            let s = cell
                .lock()
                .take()
                .expect("worker context factory invoked twice");
            ContextActor::new(s)
        });
        match ctx.spawn(props, &name) {
            Ok(addr) => self.child = Some(addr),
            Err(e) => tracing::error!(?e, "spawn ContextActor failed"),
        }
    }
}

#[async_trait]
impl Actor for WorkerActor {
    type Msg = WorkerMsg;

    async fn pre_start(&mut self, ctx: &mut Context<Self>) {
        self.spawn_child(ctx);
    }

    fn supervisor_strategy(&self) -> SupervisorStrategy {
        // With the `local-gpu` feature on, defer to the upstream
        // supervisor strategy (3 retries / 60s window, decider over
        // `ContextPoisoned` / `OutOfMemory` / `Unrecoverable` markers).
        // Without the feature, fall back to a hand-rolled policy that
        // restarts on the same string-tag panic-message — this keeps
        // the workspace buildable for `remote-only` consumers that
        // don't pull rakka-accel but still happen to mount a local
        // ModelRunner (e.g. inference-testkit's MockRunner in tests).
        #[cfg(feature = "local-gpu")]
        {
            // The CUDA backend is re-exported at `rakka_accel::cuda`
            // when the `cuda` feature is on. We carry that feature
            // forward via our own `local-gpu` feature.
            rakka_accel::cuda::error::device_supervisor_strategy()
        }
        #[cfg(not(feature = "local-gpu"))]
        {
            use rakka_core::supervision::{Directive, OneForOneStrategy};
            OneForOneStrategy::new()
                .with_max_retries(3)
                .with_within(std::time::Duration::from_secs(60))
                .with_decider(|err| {
                    // Mirror rakka_accel::cuda::error::decider's tag set.
                    if err.contains("ContextPoisoned") {
                        Directive::Restart
                    } else if err.contains("OutOfMemory") {
                        Directive::Resume
                    } else if err.contains("Unrecoverable") {
                        Directive::Stop
                    } else {
                        Directive::Escalate
                    }
                })
                .into()
        }
    }

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            WorkerMsg::Execute(batch, output) => {
                let Some(child) = self.child.as_ref() else { return };
                child.tell(ContextMsg::Execute(batch, output));
            }
            WorkerMsg::ContextPoisoned(reason) => {
                tracing::warn!(reason, "context poisoned — rebuilding child");
                if let Some(child) = self.child.take() {
                    child.stop();
                }
                self.spawn_child(ctx);
            }
            WorkerMsg::RebuildSession { cause, reply } => {
                let Some(child) = self.child.as_ref() else {
                    let _ = reply.send(Err(InferenceError::Internal("no child".into())));
                    return;
                };
                child.tell(ContextMsg::Rebuild { cause, reply });
            }
        }
    }
}

// ---------------------------------------------------------------------------

/// `ContextActor` — restartable child holding the CUDA context (or the
/// remote-network analogue). Distinct from
/// `rakka_accel::cuda::device::ContextActor`: that one specialises to CUDA
/// memory / streams; this one holds the polymorphic
/// `Box<dyn ModelRunner>` so the same supervision shape covers
/// remote-network runners too.
pub struct ContextActor {
    runner: Arc<AsyncMutex<Box<dyn ModelRunner>>>,
}

impl ContextActor {
    pub fn new(slot: WorkerSlot) -> Self {
        Self { runner: Arc::new(AsyncMutex::new(slot.runner)) }
    }
}

#[async_trait]
impl Actor for ContextActor {
    type Msg = ContextMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            ContextMsg::Execute(batch, output) => {
                let runner = self.runner.clone();
                tokio::spawn(async move {
                    let mut g = runner.lock().await;
                    match g.execute(batch).await {
                        Ok(handle) => {
                            drop(g); // release runner mutex while we drain
                            let mut s = handle.into_stream();
                            while let Some(chunk) = s.next().await {
                                if output.send(chunk).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            // Sticky CUDA errors propagate as panics
                            // tagged with rakka_accel's CONTEXT_POISONED_TAG
                            // so the parent's supervisor strategy can
                            // route them to Restart.
                            if matches!(e, InferenceError::CudaContextPoisoned(_)) {
                                let _ = output.send(Err(e.clone())).await;
                                #[cfg(feature = "local-gpu")]
                                panic!(
                                    "{}: {e}",
                                    rakka_accel::cuda::error::CONTEXT_POISONED_TAG
                                );
                                #[cfg(not(feature = "local-gpu"))]
                                panic!("ContextPoisoned: {e}");
                            }
                            let _ = output.send(Err(e)).await;
                        }
                    }
                });
            }
            ContextMsg::Rebuild { cause, reply } => {
                let runner = self.runner.clone();
                tokio::spawn(async move {
                    let mut g = runner.lock().await;
                    let r = g.rebuild_session(cause).await;
                    let _ = reply.send(r);
                });
            }
        }
    }
}
