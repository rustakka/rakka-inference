//! Ping/pong keepalive tracker.
//!
//! Lives next to the receive loop. Each time an inbound frame
//! arrives the receiver calls [`Keepalive::observe`]; periodically
//! [`Keepalive::tick`] is called and decides whether to emit a ping
//! and/or whether the link has gone idle past
//! `idle_timeout`.
//!
//! Time is passed in explicitly so tests don't need a fake clock.

use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct KeepaliveConfig {
    /// How often to consider sending a ping after the last observed
    /// inbound frame. Zero disables ping emission.
    pub ping_interval: Duration,
    /// How long the link may stay quiet before [`Keepalive::tick`]
    /// reports `KeepaliveAction::Dead`. Zero disables liveness.
    pub idle_timeout: Duration,
}

impl Default for KeepaliveConfig {
    fn default() -> Self {
        Self {
            ping_interval: Duration::from_secs(15),
            idle_timeout: Duration::from_secs(60),
        }
    }
}

/// What the caller should do after [`Keepalive::tick`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeepaliveAction {
    /// Nothing to do; the link is healthy.
    Idle,
    /// Send a ping frame to coax a pong from the peer.
    SendPing,
    /// Link has been silent past `idle_timeout`; tear it down.
    Dead,
}

#[derive(Debug, Clone)]
pub struct Keepalive {
    cfg: KeepaliveConfig,
    last_seen: Instant,
    last_ping: Option<Instant>,
}

impl Keepalive {
    pub fn new(cfg: KeepaliveConfig, now: Instant) -> Self {
        Self {
            cfg,
            last_seen: now,
            last_ping: None,
        }
    }

    /// Note that an inbound frame just arrived.
    pub fn observe(&mut self, now: Instant) {
        self.last_seen = now;
        self.last_ping = None;
    }

    /// Evaluate the link health at `now` and return the suggested
    /// action. Callers typically call this from a `tokio::time::interval`.
    pub fn tick(&mut self, now: Instant) -> KeepaliveAction {
        let since_seen = now.saturating_duration_since(self.last_seen);
        if !self.cfg.idle_timeout.is_zero() && since_seen >= self.cfg.idle_timeout {
            return KeepaliveAction::Dead;
        }
        if self.cfg.ping_interval.is_zero() {
            return KeepaliveAction::Idle;
        }
        let need_ping = match self.last_ping {
            None => since_seen >= self.cfg.ping_interval,
            Some(prev) => now.saturating_duration_since(prev) >= self.cfg.ping_interval,
        };
        if need_ping {
            self.last_ping = Some(now);
            return KeepaliveAction::SendPing;
        }
        KeepaliveAction::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ping: u64, idle: u64) -> KeepaliveConfig {
        KeepaliveConfig {
            ping_interval: Duration::from_secs(ping),
            idle_timeout: Duration::from_secs(idle),
        }
    }

    #[test]
    fn idle_when_just_seen_a_frame() {
        let t0 = Instant::now();
        let mut k = Keepalive::new(cfg(5, 30), t0);
        k.observe(t0);
        assert_eq!(k.tick(t0 + Duration::from_secs(1)), KeepaliveAction::Idle);
    }

    #[test]
    fn ping_after_interval_then_idle_until_next_interval() {
        let t0 = Instant::now();
        let mut k = Keepalive::new(cfg(5, 30), t0);
        assert_eq!(k.tick(t0 + Duration::from_secs(5)), KeepaliveAction::SendPing);
        assert_eq!(k.tick(t0 + Duration::from_secs(6)), KeepaliveAction::Idle);
        assert_eq!(k.tick(t0 + Duration::from_secs(10)), KeepaliveAction::SendPing);
    }

    #[test]
    fn observing_a_frame_clears_pending_ping() {
        let t0 = Instant::now();
        let mut k = Keepalive::new(cfg(5, 30), t0);
        let _ = k.tick(t0 + Duration::from_secs(5)); // sends ping
        k.observe(t0 + Duration::from_secs(6));
        assert_eq!(k.tick(t0 + Duration::from_secs(7)), KeepaliveAction::Idle);
    }

    #[test]
    fn idle_timeout_marks_link_dead() {
        let t0 = Instant::now();
        let mut k = Keepalive::new(cfg(5, 30), t0);
        assert_eq!(k.tick(t0 + Duration::from_secs(30)), KeepaliveAction::Dead);
    }

    #[test]
    fn ping_interval_zero_disables_pings() {
        let t0 = Instant::now();
        let mut k = Keepalive::new(cfg(0, 30), t0);
        assert_eq!(k.tick(t0 + Duration::from_secs(20)), KeepaliveAction::Idle);
        assert_eq!(k.tick(t0 + Duration::from_secs(30)), KeepaliveAction::Dead);
    }
}
