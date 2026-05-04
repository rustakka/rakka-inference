//! Exponential backoff + jitter, mirroring rakka's
//! `pattern::backoff::BackoffOptions` shape but specialised for the
//! per-request retry loop inside `RemoteWorkerActor`.
//!
//! For supervisor-level "actor restarted N times" backoff we still use
//! `atomr_core::pattern::backoff::BackoffOptions` directly. This module
//! is the per-call analogue.

use std::time::Duration;

use atomr_infer_core::deployment::RetryPolicy;
use atomr_infer_core::runtime::JitterKind;

#[derive(Debug, Clone)]
pub struct BackoffPolicy {
    pub initial: Duration,
    pub max: Duration,
    pub multiplier: f64,
    pub jitter: JitterKind,
}

impl From<&RetryPolicy> for BackoffPolicy {
    fn from(p: &RetryPolicy) -> Self {
        Self {
            initial: p.initial_backoff,
            max: p.max_backoff,
            multiplier: p.backoff_multiplier,
            jitter: p.jitter,
        }
    }
}

/// Compute the next backoff. `attempt` is 0-indexed.
pub fn compute_backoff(policy: &BackoffPolicy, attempt: u32) -> Duration {
    let base_ms = policy.initial.as_millis() as f64 * policy.multiplier.powi(attempt as i32);
    let capped = base_ms.min(policy.max.as_millis() as f64);
    let with_jitter = match policy.jitter {
        JitterKind::None => capped,
        JitterKind::Equal => capped * 0.5 + capped * pseudo_random_01(attempt) * 0.5,
        JitterKind::Full => capped * pseudo_random_01(attempt),
    };
    Duration::from_millis(with_jitter.max(0.0) as u64)
}

/// Deterministic pseudo-randomness — tests don't want real entropy.
/// Same idiom rakka uses in `pattern::backoff`.
fn pseudo_random_01(seed: u32) -> f64 {
    ((seed.wrapping_mul(2_654_435_761)) % 10_000) as f64 / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_caps() {
        let p = BackoffPolicy {
            initial: Duration::from_millis(100),
            max: Duration::from_millis(2_000),
            multiplier: 2.0,
            jitter: JitterKind::None,
        };
        assert_eq!(compute_backoff(&p, 0), Duration::from_millis(100));
        assert_eq!(compute_backoff(&p, 1), Duration::from_millis(200));
        assert_eq!(compute_backoff(&p, 2), Duration::from_millis(400));
        // Capped:
        assert_eq!(compute_backoff(&p, 10), Duration::from_millis(2_000));
    }
}
