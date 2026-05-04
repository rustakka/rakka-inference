//! Per-request retry decision logic. Doc §3.5 (Backoff on 429), §12.3.
//!
//! `RetryEngine` is intentionally a value, not an actor — the retry
//! loop runs inside one `RemoteWorkerActor::execute` call and a
//! mailbox hop per attempt is gratuitous overhead.

use std::time::Duration;

use atomr_infer_core::deployment::RetryPolicy;
use atomr_infer_core::error::InferenceError;

use crate::backoff::{compute_backoff, BackoffPolicy};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Attempt(pub u32);

impl Attempt {
    pub fn zero() -> Self {
        Self(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    Retry { after: Duration },
    GiveUp,
}

pub struct RetryEngine {
    policy: RetryPolicy,
    backoff: BackoffPolicy,
    idempotent: bool,
}

impl RetryEngine {
    pub fn new(policy: RetryPolicy, idempotent: bool) -> Self {
        let backoff = BackoffPolicy::from(&policy);
        Self {
            policy,
            backoff,
            idempotent,
        }
    }

    /// Decide whether to retry after a failed attempt. `attempt` is the
    /// 0-indexed attempt that just failed (so `0` means we've made one
    /// attempt and are deciding whether to make a second).
    pub fn decide(&self, attempt: Attempt, err: &InferenceError) -> RetryDecision {
        if !self.idempotent {
            return RetryDecision::GiveUp;
        }
        if attempt.0 >= self.policy.max_retries {
            return RetryDecision::GiveUp;
        }
        if !err.is_retryable() {
            return RetryDecision::GiveUp;
        }
        // 429 with server-provided `Retry-After` overrides the policy.
        if let InferenceError::RateLimited {
            retry_after: Some(server_ra),
            ..
        } = err
        {
            if self.policy.respect_retry_after {
                return RetryDecision::Retry { after: *server_ra };
            }
        }
        RetryDecision::Retry {
            after: compute_backoff(&self.backoff, attempt.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runtime::{JitterKind, ProviderKind};

    fn policy() -> RetryPolicy {
        RetryPolicy {
            max_retries: 3,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            jitter: JitterKind::None,
            respect_retry_after: true,
        }
    }

    #[test]
    fn retries_on_429_until_max() {
        let e = RetryEngine::new(policy(), true);
        let err = InferenceError::RateLimited {
            provider: ProviderKind::OpenAi,
            retry_after: None,
        };
        assert!(matches!(e.decide(Attempt(0), &err), RetryDecision::Retry { .. }));
        assert!(matches!(e.decide(Attempt(2), &err), RetryDecision::Retry { .. }));
        assert!(matches!(e.decide(Attempt(3), &err), RetryDecision::GiveUp));
    }

    #[test]
    fn no_retry_on_content_filter() {
        let e = RetryEngine::new(policy(), true);
        let err = InferenceError::ContentFiltered {
            reason: "harmful".into(),
        };
        assert!(matches!(e.decide(Attempt(0), &err), RetryDecision::GiveUp));
    }

    #[test]
    fn no_retry_when_not_idempotent() {
        let e = RetryEngine::new(policy(), false);
        let err = InferenceError::ServerError {
            status: 503,
            body: None,
        };
        assert!(matches!(e.decide(Attempt(0), &err), RetryDecision::GiveUp));
    }

    #[test]
    fn server_retry_after_wins_when_respected() {
        let e = RetryEngine::new(policy(), true);
        let err = InferenceError::RateLimited {
            provider: ProviderKind::OpenAi,
            retry_after: Some(Duration::from_secs(5)),
        };
        match e.decide(Attempt(0), &err) {
            RetryDecision::Retry { after } => assert_eq!(after, Duration::from_secs(5)),
            other => panic!("expected retry, got {other:?}"),
        }
    }
}
