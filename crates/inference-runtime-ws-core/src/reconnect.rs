//! Exponential-backoff reconnect engine.
//!
//! Thin wrapper around [`atomr_infer_remote_core::BackoffPolicy`]
//! that adds a "max attempts" ceiling and a notion of "first
//! attempt is free" — provider WS clients try once with no delay
//! before entering the reconnect loop.
//!
//! The state machine is intentionally non-async; callers drive
//! `next_delay()` from their own retry loop and `sleep` themselves.
//! Tests do not need to fake time.

use std::time::Duration;

use atomr_infer_remote_core::backoff::{compute_backoff, BackoffPolicy};

/// Holds the next-attempt counter and the policy.
#[derive(Debug, Clone)]
pub struct ReconnectEngine {
    policy: BackoffPolicy,
    max_attempts: u32,
    attempt: u32,
}

impl ReconnectEngine {
    /// `max_attempts == 0` means "retry forever".
    pub fn new(policy: BackoffPolicy, max_attempts: u32) -> Self {
        Self {
            policy,
            max_attempts,
            attempt: 0,
        }
    }

    /// Call after a successful connect. Resets the counter so the
    /// next dropped connection starts the backoff from scratch.
    pub fn reset(&mut self) {
        self.attempt = 0;
    }

    /// Returns the delay to wait before the next attempt, or `None`
    /// if the configured `max_attempts` cap has been reached.
    ///
    /// The first call returns `Some(Duration::ZERO)` so the initial
    /// attempt is free. Subsequent calls advance the counter and
    /// consult [`compute_backoff`].
    pub fn next_delay(&mut self) -> Option<Duration> {
        if self.attempt == 0 {
            self.attempt = 1;
            return Some(Duration::ZERO);
        }
        if self.max_attempts != 0 && self.attempt >= self.max_attempts {
            return None;
        }
        let d = compute_backoff(&self.policy, self.attempt.saturating_sub(1));
        self.attempt = self.attempt.saturating_add(1);
        Some(d)
    }

    /// Number of attempts so far (1 == first attempt has fired).
    pub fn attempts(&self) -> u32 {
        self.attempt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::runtime::JitterKind;

    fn policy() -> BackoffPolicy {
        BackoffPolicy {
            initial: Duration::from_millis(100),
            max: Duration::from_secs(2),
            multiplier: 2.0,
            jitter: JitterKind::None,
        }
    }

    #[test]
    fn first_attempt_is_free() {
        let mut r = ReconnectEngine::new(policy(), 0);
        assert_eq!(r.next_delay(), Some(Duration::ZERO));
        assert_eq!(r.attempts(), 1);
    }

    #[test]
    fn subsequent_attempts_grow() {
        let mut r = ReconnectEngine::new(policy(), 0);
        assert_eq!(r.next_delay(), Some(Duration::ZERO));
        assert_eq!(r.next_delay(), Some(Duration::from_millis(100)));
        assert_eq!(r.next_delay(), Some(Duration::from_millis(200)));
        assert_eq!(r.next_delay(), Some(Duration::from_millis(400)));
    }

    #[test]
    fn caps_at_max_attempts() {
        let mut r = ReconnectEngine::new(policy(), 3);
        assert_eq!(r.next_delay(), Some(Duration::ZERO));
        assert!(r.next_delay().is_some());
        assert!(r.next_delay().is_some());
        assert_eq!(r.next_delay(), None);
    }

    #[test]
    fn reset_restarts_the_counter() {
        let mut r = ReconnectEngine::new(policy(), 0);
        let _ = r.next_delay();
        let _ = r.next_delay();
        r.reset();
        assert_eq!(r.next_delay(), Some(Duration::ZERO));
    }
}
