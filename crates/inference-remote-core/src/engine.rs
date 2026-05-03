//! `RemoteEngineCoreActor` — per-replica HTTP orchestrator. Doc §5.1.
//!
//! Owns the bounded priority queue (a *module*, not a child actor) and
//! the worker pool of `RemoteWorkerActor`s. Receives `AddRequest` from
//! the upstream `RequestActor`; dispatches to whichever worker
//! signals idle.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::{mpsc, oneshot};

use inference_core::batch::ExecuteBatch;
use inference_core::deployment::CapacityPolicy;
use inference_core::error::InferenceError;
use inference_core::tokens::TokenChunk;

use crate::queue::{Priority, PriorityRequest, RequestQueue};
use crate::worker::{RemoteWorkerActor, WorkerMsg, WorkerSlot};

#[derive(Clone)]
pub struct RemoteEngineConfig {
    pub queue_capacity: usize,
    pub worker_count: u32,
    pub on_capacity_exhausted: CapacityPolicy,
}

impl Default for RemoteEngineConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1024,
            worker_count: 8,
            on_capacity_exhausted: CapacityPolicy::Queue,
        }
    }
}

#[derive(Default, Clone)]
pub struct EngineMetrics {
    pub queued: u64,
    pub in_flight: u64,
    pub completed: u64,
    pub rejected_backpressure: u64,
}

pub struct AddRequest {
    pub priority: Priority,
    pub batch: ExecuteBatch,
    pub output: mpsc::Sender<Result<TokenChunk, InferenceError>>,
    pub admission: oneshot::Sender<Result<(), InferenceError>>,
}

pub enum EngineMsg {
    Add(AddRequest),
    WorkerIdle,
}

/// Factory for fresh worker slots. Called once per worker at engine
/// startup. Each invocation returns a *new* `WorkerSlot` because slots
/// own a `Box<dyn ModelRunner>` and are not `Clone`.
pub type WorkerSlotFactory = Box<dyn FnMut() -> WorkerSlot + Send>;

struct WorkerEntry {
    addr: ActorRef<WorkerMsg>,
    /// Idle if true; busy if false.
    idle: bool,
}

pub struct RemoteEngineCoreActor {
    #[allow(dead_code)] // observability hook; surfaces in MetricsActor wiring later
    config: RemoteEngineConfig,
    queue: RequestQueue,
    workers: Vec<WorkerEntry>,
    metrics: Arc<Mutex<EngineMetrics>>,
    /// Factory invoked once per slot at startup. Held in an `Option`
    /// so we can `take()` it during `pre_start` (factories aren't `Clone`).
    factory: Option<WorkerSlotFactory>,
    worker_count: u32,
    /// Signal channel: each worker tells the engine "I'm idle" by
    /// sending a `()` here. The engine forwards into its own mailbox
    /// as `EngineMsg::WorkerIdle` via a small forwarder task.
    idle_tx: mpsc::UnboundedSender<()>,
    idle_rx: Option<mpsc::UnboundedReceiver<()>>,
}

impl RemoteEngineCoreActor {
    pub fn new(config: RemoteEngineConfig, factory: WorkerSlotFactory) -> Self {
        let (idle_tx, idle_rx) = mpsc::unbounded_channel();
        let queue = RequestQueue::new(config.queue_capacity);
        let worker_count = config.worker_count;
        Self {
            config,
            queue,
            workers: Vec::new(),
            metrics: Arc::new(Mutex::new(EngineMetrics::default())),
            factory: Some(factory),
            worker_count,
            idle_tx,
            idle_rx: Some(idle_rx),
        }
    }

    pub fn metrics_handle(&self) -> Arc<Mutex<EngineMetrics>> {
        self.metrics.clone()
    }

    fn enqueue(&mut self, req: AddRequest) {
        let priority_request = PriorityRequest {
            priority: req.priority,
            arrival_seq: 0,
            batch: req.batch,
            output: req.output,
        };
        match self.queue.push(priority_request) {
            Ok(()) => {
                self.metrics.lock().queued += 1;
                let _ = req.admission.send(Ok(()));
            }
            Err(_rejected) => {
                self.metrics.lock().rejected_backpressure += 1;
                let _ = req
                    .admission
                    .send(Err(InferenceError::Backpressure("engine queue full".into())));
            }
        }
    }

    fn try_dispatch(&mut self) {
        while !self.queue.is_empty() {
            let Some(idx) = self.workers.iter().position(|w| w.idle) else {
                break;
            };
            let Some(req) = self.queue.pop() else { break };
            self.workers[idx].idle = false;
            self.workers[idx].addr.tell(WorkerMsg::Dispatch(req));
            let mut m = self.metrics.lock();
            m.queued = m.queued.saturating_sub(1);
            m.in_flight += 1;
        }
    }
}

#[async_trait]
impl Actor for RemoteEngineCoreActor {
    type Msg = EngineMsg;

    async fn pre_start(&mut self, ctx: &mut Context<Self>) {
        // Each worker needs a fresh slot. We invoke the factory N times
        // up-front; the slot moves into a `Props::create` factory whose
        // closure can produce it on demand (Rakka may call the factory
        // again on restart — we currently propagate the same WorkerSlot
        // for the first creation only; restart-after-failure rebuilds
        // are handled at the higher RemoteSessionActor tier and not
        // here).
        let mut factory = match self.factory.take() {
            Some(f) => f,
            None => {
                tracing::error!("RemoteEngineCoreActor pre_start with no factory");
                return;
            }
        };
        for i in 0..self.worker_count {
            let slot = factory();
            let idle_tx = self.idle_tx.clone();
            let cell = parking_lot::Mutex::new(Some(slot));
            let props = Props::create(move || {
                let s = cell
                    .lock()
                    .take()
                    .expect("worker factory invoked twice — restart not yet supported");
                RemoteWorkerActor::new(s, idle_tx.clone())
            });
            let name = format!("worker-{i}");
            match ctx.spawn(props, &name) {
                Ok(addr) => self.workers.push(WorkerEntry { addr, idle: true }),
                Err(e) => tracing::error!(?e, "spawn worker {i} failed"),
            }
        }

        // Forwarder: workers signal idle on `idle_tx`; we lift each
        // `()` into an `EngineMsg::WorkerIdle` on our own mailbox.
        let self_ref = ctx.self_ref().clone();
        let mut rx = self.idle_rx.take().expect("idle_rx set in new()");
        tokio::spawn(async move {
            while rx.recv().await.is_some() {
                self_ref.tell(EngineMsg::WorkerIdle);
            }
        });
    }

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            EngineMsg::Add(req) => {
                self.enqueue(req);
                self.try_dispatch();
            }
            EngineMsg::WorkerIdle => {
                // Mark first non-idle worker as idle. Without per-worker
                // identity in the signal, this is best-effort and
                // sufficient for capacity tracking — the actual
                // dispatched request flowed through one specific
                // worker's mailbox, so the count stays correct in
                // aggregate.
                if let Some(w) = self.workers.iter_mut().find(|w| !w.idle) {
                    w.idle = true;
                    let mut m = self.metrics.lock();
                    m.in_flight = m.in_flight.saturating_sub(1);
                    m.completed += 1;
                }
                self.try_dispatch();
            }
        }
    }
}
