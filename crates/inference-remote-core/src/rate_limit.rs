//! Rate-limiter actors. Doc §3.5, §12.1.
//!
//! Two flavours, selected per `RateLimits::strict`:
//!
//! - [`RateLimiterActor`] — approximate, distributed via
//!   `atomr_distributed_data::counters::GCounter`. Each node keeps a
//!   local per-window view; over-spend is bounded by sync interval and
//!   per-node budget. Default for high-throughput deployments.
//! - [`StrictRateLimiterActor`] — runs as a cluster singleton; every
//!   request `ask`s for a permit. Higher latency, exact accounting.
//!   Default for low-throughput / pay-per-call deployments.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context};
use atomr_distributed_data::{DeltaCrdt, GCounter};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use atomr_infer_core::deployment::RateLimits;
use atomr_infer_core::error::InferenceError;

/// Permit returned to a worker. Holding the permit until the request
/// completes is the convention; `Drop` is a no-op because spend is
/// recorded at acquire time. Strict mode could swap this for a
/// release-on-drop scheme later.
#[derive(Debug)]
pub struct Permit {
    pub requests: u32,
    pub tokens: u32,
}

pub struct AcquirePermit {
    pub requests: u32,
    pub tokens: u32,
    pub reply: oneshot::Sender<Result<Permit, InferenceError>>,
}

#[derive(Default)]
struct Window {
    /// Wall-clock instant the current 60-s window started.
    started_at: Option<Instant>,
    requests: u64,
    tokens: u64,
}

/// Cheap shared view that workers consult before acquiring. Useful for
/// status endpoints and tests.
#[derive(Default, Clone)]
pub struct RateLimiterHandle {
    state: Arc<Mutex<Window>>,
}

impl RateLimiterHandle {
    pub fn snapshot(&self) -> (u64, u64) {
        let s = self.state.lock();
        (s.requests, s.tokens)
    }
}

pub struct RateLimiterActor {
    node_id: String,
    limits: RateLimits,
    /// Distributed counter — total requests this window across the
    /// cluster, by node.
    requests_counter: GCounter,
    tokens_counter: GCounter,
    window: Arc<Mutex<Window>>,
}

impl RateLimiterActor {
    pub fn new(node_id: impl Into<String>, limits: RateLimits) -> Self {
        Self {
            node_id: node_id.into(),
            limits,
            requests_counter: GCounter::new(),
            tokens_counter: GCounter::new(),
            window: Arc::new(Mutex::new(Window::default())),
        }
    }

    pub fn handle(&self) -> RateLimiterHandle {
        RateLimiterHandle {
            state: self.window.clone(),
        }
    }

    /// Apply a CRDT delta from another node. Wired through
    /// `atomr_distributed_data::Replicator` in production.
    pub fn merge_remote_delta_requests(&mut self, delta: &<GCounter as DeltaCrdt>::Delta) {
        self.requests_counter.merge_delta(delta);
    }

    pub fn merge_remote_delta_tokens(&mut self, delta: &<GCounter as DeltaCrdt>::Delta) {
        self.tokens_counter.merge_delta(delta);
    }

    fn rotate_window_if_needed(&mut self) {
        let mut w = self.window.lock();
        let needs_reset = match w.started_at {
            Some(started) => started.elapsed() >= Duration::from_secs(60),
            None => true,
        };
        if needs_reset {
            *w = Window {
                started_at: Some(Instant::now()),
                requests: 0,
                tokens: 0,
            };
            // Window rotation: zero the local view of the CRDT.
            // Cluster peers will re-observe the new window via gossip
            // since their own counters reset on the same wall clock.
            self.requests_counter = GCounter::new();
            self.tokens_counter = GCounter::new();
        }
    }

    fn acquire(&mut self, req: AcquirePermit) -> Result<Permit, InferenceError> {
        self.rotate_window_if_needed();
        let mut w = self.window.lock();

        // Approximate distributed budget: each node may consume up to
        // its share. With one node the share is the full budget; the
        // CRDT sync widens this scope when peers join.
        if let Some(rpm) = self.limits.requests_per_minute {
            if w.requests + req.requests as u64 > rpm {
                return Err(InferenceError::Backpressure(format!(
                    "requests-per-minute limit reached ({}/{})",
                    w.requests, rpm
                )));
            }
        }
        if let Some(tpm) = self.limits.tokens_per_minute {
            if w.tokens + req.tokens as u64 > tpm {
                return Err(InferenceError::Backpressure(format!(
                    "tokens-per-minute limit reached ({}/{})",
                    w.tokens, tpm
                )));
            }
        }
        w.requests += req.requests as u64;
        w.tokens += req.tokens as u64;
        // Record into the CRDT so peers see our spend.
        drop(w);
        self.requests_counter
            .increment(&self.node_id, req.requests as u64);
        self.tokens_counter.increment(&self.node_id, req.tokens as u64);
        Ok(Permit {
            requests: req.requests,
            tokens: req.tokens,
        })
    }
}

#[async_trait]
impl Actor for RateLimiterActor {
    type Msg = AcquirePermit;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        let reply = msg.reply;
        let res = self.acquire(AcquirePermit {
            reply: dummy_reply(),
            ..msg
        });
        let _ = reply.send(res);
    }
}

/// Strict variant — run as a cluster singleton. The actor structure is
/// identical to the approximate one; the different default is
/// expressed at deploy time by registering it through
/// `atomr_cluster_tools::ClusterSingletonManager` rather than as a
/// per-node actor.
pub struct StrictRateLimiterActor {
    inner: RateLimiterActor,
}

impl StrictRateLimiterActor {
    pub fn new(inner: RateLimiterActor) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Actor for StrictRateLimiterActor {
    type Msg = AcquirePermit;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        let reply = msg.reply;
        let res = self.inner.acquire(AcquirePermit {
            reply: dummy_reply(),
            ..msg
        });
        let _ = reply.send(res);
    }
}

/// Construct a no-op oneshot reply so we can move the original out by
/// value into `acquire` without touching it.
fn dummy_reply() -> oneshot::Sender<Result<Permit, InferenceError>> {
    let (tx, rx) = oneshot::channel();
    drop(rx);
    tx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approximate_limiter_blocks_on_rpm() {
        let mut a = RateLimiterActor::new(
            "node-a",
            RateLimits {
                requests_per_minute: Some(2),
                tokens_per_minute: None,
                concurrent_requests: None,
                strict: false,
            },
        );
        let (tx1, _) = oneshot::channel();
        let (tx2, _) = oneshot::channel();
        let (tx3, _) = oneshot::channel();
        assert!(a
            .acquire(AcquirePermit {
                requests: 1,
                tokens: 0,
                reply: tx1
            })
            .is_ok());
        assert!(a
            .acquire(AcquirePermit {
                requests: 1,
                tokens: 0,
                reply: tx2
            })
            .is_ok());
        assert!(matches!(
            a.acquire(AcquirePermit {
                requests: 1,
                tokens: 0,
                reply: tx3
            }),
            Err(InferenceError::Backpressure(_))
        ));
    }
}
