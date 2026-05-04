//! Bounded priority queue for `RemoteEngineCoreActor`. Per doc §5.2 the
//! queue is a *module*, not an actor — every per-message hop would add
//! mailbox latency for no architectural payoff.

use std::collections::BinaryHeap;

use atomr_infer_core::batch::ExecuteBatch;
use atomr_infer_core::tokens::TokenChunk;

/// Priority class — higher values served first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Background,
    Normal,
    High,
    Critical,
}

#[derive(Debug)]
pub struct PriorityRequest {
    pub priority: Priority,
    /// Stable arrival sequence — breaks ties at the same priority so
    /// FIFO order is preserved.
    pub arrival_seq: u64,
    pub batch: ExecuteBatch,
    /// Channel the worker writes streamed chunks into. The
    /// `RequestActor` owns the receiver.
    pub output: tokio::sync::mpsc::Sender<Result<TokenChunk, atomr_infer_core::error::InferenceError>>,
}

// BinaryHeap is a max-heap. We want highest priority first, but for
// equal priorities we want the *lowest* sequence number first (oldest).
// Custom Ord flips arrival_seq.
impl PartialEq for PriorityRequest {
    fn eq(&self, o: &Self) -> bool {
        self.priority == o.priority && self.arrival_seq == o.arrival_seq
    }
}
impl Eq for PriorityRequest {}
impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for PriorityRequest {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&o.priority)
            .then_with(|| o.arrival_seq.cmp(&self.arrival_seq))
    }
}

pub struct RequestQueue {
    inner: BinaryHeap<PriorityRequest>,
    capacity: usize,
    next_seq: u64,
}

impl RequestQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: BinaryHeap::new(),
            capacity,
            next_seq: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn is_full(&self) -> bool {
        self.inner.len() >= self.capacity
    }

    /// Enqueue. Caller must already have classified the priority.
    /// Returns `Err` carrying the request if full. The request is
    /// boxed so the `Result` doesn't carry a fat `Err` variant on the
    /// happy path (queue mailboxes pass this around frequently).
    #[allow(clippy::result_large_err)] // PriorityRequest is intentionally big; boxing it everywhere costs more.
    pub fn push(&mut self, mut req: PriorityRequest) -> Result<(), PriorityRequest> {
        if self.is_full() {
            return Err(req);
        }
        req.arrival_seq = self.next_seq;
        self.next_seq += 1;
        self.inner.push(req);
        Ok(())
    }

    pub fn pop(&mut self) -> Option<PriorityRequest> {
        self.inner.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(priority: Priority) -> PriorityRequest {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        PriorityRequest {
            priority,
            arrival_seq: 0,
            batch: ExecuteBatch {
                request_id: "r".into(),
                model: "m".into(),
                messages: vec![],
                sampling: Default::default(),
                stream: false,
                estimated_tokens: 1,
            },
            output: tx,
        }
    }

    #[test]
    fn priority_first_then_fifo() {
        let mut q = RequestQueue::new(8);
        q.push(req(Priority::Normal)).unwrap();
        q.push(req(Priority::Normal)).unwrap();
        q.push(req(Priority::High)).unwrap();
        let first = q.pop().unwrap();
        assert_eq!(first.priority, Priority::High);
        let second = q.pop().unwrap();
        assert_eq!(second.priority, Priority::Normal);
        assert_eq!(second.arrival_seq, 0);
    }

    #[test]
    fn full_queue_rejects() {
        let mut q = RequestQueue::new(1);
        assert!(q.push(req(Priority::Normal)).is_ok());
        assert!(q.push(req(Priority::Normal)).is_err());
    }
}
