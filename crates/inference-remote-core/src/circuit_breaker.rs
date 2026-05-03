//! Circuit-breaker actor (doc §3.5, §12.2). One per `(provider,
//! endpoint)`.
//!
//! Originally intended to wrap `rakka_core::pattern::CircuitBreaker`,
//! but that primitive's `CircuitBreakerError<E>` is not publicly
//! re-exported, which makes composing it as an intermediary error
//! type unworkable. The state machine itself is small (closed → open
//! after N failures → half-open after `open_duration` → closed on
//! probe success), so we implement it here directly with atomics.
//!
//! Provides:
//! - typed messages so other actors can `tell` / `ask`
//! - observability events on transitions
//! - operator escape hatch (`ForceOpen { duration }`) for incident
//!   response (doc §11.5)

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;
use rakka_core::actor::{Actor, Context};
use tokio::sync::oneshot;

use inference_core::error::InferenceError;
use inference_core::runtime::{CircuitBreakerConfig, ProviderKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Cheap synchronous handle that `RemoteWorkerActor`s share without
/// going through the actor mailbox on the hot path.
pub struct CircuitBreakerHandle {
    provider: ProviderKind,
    config: CircuitBreakerConfig,
    failures: AtomicU32,
    /// Wall-clock time the breaker opened, in `Instant`-derived nanos
    /// since process start. `0` means "not open".
    opened_at_ns: AtomicU64,
    /// Per-process anchor so `opened_at_ns` is meaningful as elapsed
    /// duration without incurring `SystemTime` overhead in the hot path.
    epoch: Instant,
    /// Operator-set forced-open expiry (`ForceOpen` message).
    forced_open_until: Mutex<Option<Instant>>,
}

impl CircuitBreakerHandle {
    pub fn new(provider: ProviderKind, config: CircuitBreakerConfig) -> Arc<Self> {
        Arc::new(Self {
            provider,
            config,
            failures: AtomicU32::new(0),
            opened_at_ns: AtomicU64::new(0),
            epoch: Instant::now(),
            forced_open_until: Mutex::new(None),
        })
    }

    pub fn provider(&self) -> &ProviderKind {
        &self.provider
    }

    pub fn state(&self) -> CircuitState {
        if let Some(until) = *self.forced_open_until.lock() {
            if Instant::now() < until {
                return CircuitState::Open;
            }
        }
        let opened = self.opened_at_ns.load(Ordering::Acquire);
        if opened == 0 {
            return CircuitState::Closed;
        }
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        if now_ns.saturating_sub(opened) >= self.config.open_duration.as_nanos() as u64 {
            CircuitState::HalfOpen
        } else {
            CircuitState::Open
        }
    }

    /// Quick gate before sending a request. Returns `Err(CircuitOpen)`
    /// when open. Half-open lets one probe through.
    pub fn check(&self) -> Result<(), InferenceError> {
        match self.state() {
            CircuitState::Closed | CircuitState::HalfOpen => Ok(()),
            CircuitState::Open => {
                let opened_ns = self.opened_at_ns.load(Ordering::Acquire);
                let opened_at_unix_ms = chrono::Utc::now().timestamp_millis().saturating_sub(
                    (self.epoch.elapsed().as_nanos() as u64).saturating_sub(opened_ns) as i64 / 1_000_000,
                ) as u64;
                let retry_at_unix_ms =
                    opened_at_unix_ms.saturating_add(self.config.open_duration.as_millis() as u64);
                Err(InferenceError::CircuitOpen {
                    provider: self.provider.clone(),
                    opened_at_unix_ms,
                    retry_at_unix_ms,
                })
            }
        }
    }

    pub fn record_success(&self) {
        self.failures.store(0, Ordering::Release);
        self.opened_at_ns.store(0, Ordering::Release);
    }

    /// Increment the failure counter; flip to Open if we cross the
    /// threshold. Idempotent for already-open breakers.
    pub fn record_failure(&self) {
        let n = self.failures.fetch_add(1, Ordering::AcqRel) + 1;
        if n >= self.config.failure_threshold && self.opened_at_ns.load(Ordering::Acquire) == 0 {
            tracing::warn!(provider = ?self.provider, failures = n, "circuit opened");
            self.opened_at_ns
                .store(self.epoch.elapsed().as_nanos() as u64, Ordering::Release);
        }
    }

    /// Wrap an async block. Successes / failures contribute to the
    /// state machine; non-circuit errors (content filter, 4xx) flow
    /// through without affecting the breaker.
    pub async fn run<F, Fut, T>(&self, f: F) -> Result<T, InferenceError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, InferenceError>>,
    {
        self.check()?;
        match f().await {
            Ok(v) => {
                self.record_success();
                Ok(v)
            }
            Err(e) => {
                if e.counts_as_circuit_failure() {
                    self.record_failure();
                }
                Err(e)
            }
        }
    }

    pub(crate) fn force_open(&self, duration: Duration) {
        *self.forced_open_until.lock() = Some(Instant::now() + duration);
    }
}

// --- Actor wrapper ---------------------------------------------------------

#[derive(Debug)]
pub enum CircuitBreakerMsg {
    Check {
        reply: oneshot::Sender<Result<(), InferenceError>>,
    },
    GetState {
        reply: oneshot::Sender<CircuitState>,
    },
    ForceOpen {
        duration: Duration,
    },
}

pub struct CircuitBreakerActor {
    handle: Arc<CircuitBreakerHandle>,
}

impl CircuitBreakerActor {
    pub fn new(handle: Arc<CircuitBreakerHandle>) -> Self {
        Self { handle }
    }

    pub fn handle(&self) -> Arc<CircuitBreakerHandle> {
        self.handle.clone()
    }
}

#[async_trait]
impl Actor for CircuitBreakerActor {
    type Msg = CircuitBreakerMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: Self::Msg) {
        match msg {
            CircuitBreakerMsg::Check { reply } => {
                let _ = reply.send(self.handle.check());
            }
            CircuitBreakerMsg::GetState { reply } => {
                let _ = reply.send(self.handle.state());
            }
            CircuitBreakerMsg::ForceOpen { duration } => {
                tracing::warn!(provider = ?self.handle.provider(), ?duration, "circuit forced open");
                self.handle.force_open(duration);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn opens_after_threshold_and_check_returns_circuit_open() {
        let h = CircuitBreakerHandle::new(
            ProviderKind::OpenAi,
            CircuitBreakerConfig {
                failure_threshold: 2,
                open_duration: Duration::from_millis(30),
                half_open_max_probes: 1,
            },
        );
        for _ in 0..2 {
            let _ = h
                .run(|| async {
                    Err::<(), _>(InferenceError::ServerError {
                        status: 503,
                        body: None,
                    })
                })
                .await;
        }
        assert_eq!(h.state(), CircuitState::Open);
        assert!(matches!(h.check(), Err(InferenceError::CircuitOpen { .. })));
    }

    #[tokio::test]
    async fn half_open_after_duration_then_closes_on_success() {
        let h = CircuitBreakerHandle::new(
            ProviderKind::Anthropic,
            CircuitBreakerConfig {
                failure_threshold: 1,
                open_duration: Duration::from_millis(10),
                half_open_max_probes: 1,
            },
        );
        let _ = h
            .run(|| async {
                Err::<(), _>(InferenceError::ServerError {
                    status: 500,
                    body: None,
                })
            })
            .await;
        assert_eq!(h.state(), CircuitState::Open);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(h.state(), CircuitState::HalfOpen);
        let r: Result<(), _> = h.run(|| async { Ok::<(), InferenceError>(()) }).await;
        assert!(r.is_ok());
        assert_eq!(h.state(), CircuitState::Closed);
    }
}
