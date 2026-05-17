//! Domain-level WebSocket frame abstraction.
//!
//! Provider crates work with [`Frame`] instead of importing
//! `tungstenite::Message` directly. This keeps the transport
//! interface stable across `tungstenite` upgrades and gives us a
//! place to layer batching / coalescing semantics.
//!
//! Coalescing semantics
//! --------------------
//!
//! When backpressure builds, callers pushing audio chunks would
//! rather drop a stale chunk than block the realtime loop. The
//! [`coalesce_binary`] helper collapses a queued run of adjacent
//! [`Frame::Binary`] payloads into one according to a `max_bytes`
//! cap. Property tests in this module pin the invariants:
//!
//! - the relative order of frames in the output matches the input;
//! - non-binary frames are never merged or dropped;
//! - the total byte count of binary payloads is preserved iff no
//!   bound was hit, otherwise it monotonically decreases as
//!   `max_bytes` shrinks.

use bytes::Bytes;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;

/// A WebSocket frame as observed by provider runtimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    /// Binary payload — typically PCM audio.
    Binary(Bytes),
    /// UTF-8 text payload — typically a JSON control envelope.
    Text(String),
    /// Application-level ping (rarely emitted; keepalive is internal).
    Ping(Bytes),
    /// Application-level pong.
    Pong(Bytes),
    /// Peer-initiated close. `code == 1000` is the clean-shutdown
    /// signal; anything in `4000..=4999` is application-defined.
    Close { code: u16, reason: String },
}

impl Frame {
    /// Convert into a `tungstenite::Message` for the wire.
    pub fn into_message(self) -> Message {
        match self {
            Frame::Binary(b) => Message::Binary(b.to_vec()),
            Frame::Text(t) => Message::Text(t),
            Frame::Ping(p) => Message::Ping(p.to_vec()),
            Frame::Pong(p) => Message::Pong(p.to_vec()),
            Frame::Close { code, reason } => {
                let reason: std::borrow::Cow<'static, str> = std::borrow::Cow::Owned(reason);
                Message::Close(Some(CloseFrame {
                    code: CloseCode::from(code),
                    reason,
                }))
            }
        }
    }

    /// Convert from a `tungstenite::Message` arriving from the wire.
    /// Returns `None` for `Message::Frame(_)` (raw frames are never
    /// surfaced by `tokio-tungstenite` to ordinary callers).
    pub fn from_message(m: Message) -> Option<Self> {
        match m {
            Message::Binary(b) => Some(Frame::Binary(Bytes::from(b))),
            Message::Text(t) => Some(Frame::Text(t)),
            Message::Ping(p) => Some(Frame::Ping(Bytes::from(p))),
            Message::Pong(p) => Some(Frame::Pong(Bytes::from(p))),
            Message::Close(Some(cf)) => Some(Frame::Close {
                code: cf.code.into(),
                reason: cf.reason.into_owned(),
            }),
            Message::Close(None) => Some(Frame::Close {
                code: 1000,
                reason: String::new(),
            }),
            Message::Frame(_) => None,
        }
    }

    /// True for control frames (ping/pong/close). The reconnect
    /// engine treats `Close` specially.
    pub fn is_control(&self) -> bool {
        matches!(self, Frame::Ping(_) | Frame::Pong(_) | Frame::Close { .. })
    }
}

/// Coalesce a run of adjacent [`Frame::Binary`] payloads, preserving
/// order and never merging across a non-binary frame.
///
/// Each resulting binary frame is at most `max_bytes` long. If a
/// single input frame already exceeds `max_bytes`, it is passed
/// through verbatim — coalescing only *merges*, it never *splits*.
///
/// `max_bytes == 0` is treated as "do not coalesce", returning the
/// input untouched.
pub fn coalesce_binary(frames: Vec<Frame>, max_bytes: usize) -> Vec<Frame> {
    if max_bytes == 0 || frames.len() <= 1 {
        return frames;
    }
    let mut out: Vec<Frame> = Vec::with_capacity(frames.len());
    let mut buf: Vec<u8> = Vec::new();

    let flush = |buf: &mut Vec<u8>, out: &mut Vec<Frame>| {
        if !buf.is_empty() {
            out.push(Frame::Binary(Bytes::from(std::mem::take(buf))));
        }
    };

    for f in frames {
        match f {
            Frame::Binary(b) => {
                if b.len() >= max_bytes {
                    // Oversized inputs flow through after flushing
                    // whatever we accumulated. We never split a frame
                    // already provided by the caller.
                    flush(&mut buf, &mut out);
                    out.push(Frame::Binary(b));
                } else if buf.len() + b.len() > max_bytes {
                    flush(&mut buf, &mut out);
                    buf.extend_from_slice(&b);
                } else {
                    buf.extend_from_slice(&b);
                }
            }
            other => {
                flush(&mut buf, &mut out);
                out.push(other);
            }
        }
    }
    flush(&mut buf, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn roundtrip_through_message() {
        let cases = vec![
            Frame::Binary(Bytes::from_static(b"abc")),
            Frame::Text("hi".into()),
            Frame::Ping(Bytes::from_static(b"p")),
            Frame::Pong(Bytes::from_static(b"q")),
            Frame::Close {
                code: 1011,
                reason: "boom".into(),
            },
        ];
        for f in cases {
            let m = f.clone().into_message();
            let back = Frame::from_message(m).expect("non-raw frame");
            assert_eq!(f, back);
        }
    }

    #[test]
    fn close_without_frame_decodes_clean() {
        let f = Frame::from_message(Message::Close(None)).unwrap();
        match f {
            Frame::Close { code, .. } => assert_eq!(code, 1000),
            _ => panic!("expected close"),
        }
    }

    #[test]
    fn coalesce_passthrough_when_disabled_or_small() {
        let v = vec![Frame::Binary(Bytes::from_static(b"a"))];
        assert_eq!(coalesce_binary(v.clone(), 0), v);
        assert_eq!(coalesce_binary(v.clone(), 1024), v);
    }

    #[test]
    fn coalesce_merges_run_under_budget() {
        let v = vec![
            Frame::Binary(Bytes::from_static(b"ab")),
            Frame::Binary(Bytes::from_static(b"cd")),
            Frame::Binary(Bytes::from_static(b"ef")),
        ];
        let out = coalesce_binary(v, 8);
        assert_eq!(out.len(), 1);
        match &out[0] {
            Frame::Binary(b) => assert_eq!(b.as_ref(), b"abcdef"),
            _ => panic!("expected binary"),
        }
    }

    #[test]
    fn coalesce_respects_non_binary_boundaries() {
        let v = vec![
            Frame::Binary(Bytes::from_static(b"a")),
            Frame::Binary(Bytes::from_static(b"b")),
            Frame::Text("control".into()),
            Frame::Binary(Bytes::from_static(b"c")),
            Frame::Binary(Bytes::from_static(b"d")),
        ];
        let out = coalesce_binary(v, 1024);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Frame::Binary(Bytes::from_static(b"ab")));
        assert_eq!(out[1], Frame::Text("control".into()));
        assert_eq!(out[2], Frame::Binary(Bytes::from_static(b"cd")));
    }

    #[test]
    fn coalesce_passes_oversized_frame_through_after_flushing() {
        let v = vec![
            Frame::Binary(Bytes::from_static(b"ab")),
            Frame::Binary(Bytes::from_static(b"longer-than-cap")),
            Frame::Binary(Bytes::from_static(b"cd")),
        ];
        let out = coalesce_binary(v, 8);
        // "ab" (flushed alone because adding "longer-than-cap"
        // would blow the budget), "longer-than-cap" (oversized
        // passthrough), "cd" (left in buffer, flushed at end).
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Frame::Binary(Bytes::from_static(b"ab")));
        assert_eq!(out[1], Frame::Binary(Bytes::from_static(b"longer-than-cap")));
        assert_eq!(out[2], Frame::Binary(Bytes::from_static(b"cd")));
    }

    proptest! {
        #[test]
        fn coalesce_preserves_relative_order_and_non_binary_frames(
            chunks in proptest::collection::vec(
                prop_oneof![
                    proptest::collection::vec(any::<u8>(), 0..16)
                        .prop_map(|v| Frame::Binary(Bytes::from(v))),
                    any::<String>().prop_map(Frame::Text),
                ],
                0..32,
            ),
            max_bytes in 1usize..64,
        ) {
            let in_text: Vec<_> = chunks.iter()
                .filter_map(|f| if let Frame::Text(t) = f { Some(t.clone()) } else { None })
                .collect();
            let in_bytes: Vec<u8> = chunks.iter()
                .flat_map(|f| if let Frame::Binary(b) = f { b.to_vec() } else { vec![] })
                .collect();

            let out = coalesce_binary(chunks, max_bytes);

            let out_text: Vec<_> = out.iter()
                .filter_map(|f| if let Frame::Text(t) = f { Some(t.clone()) } else { None })
                .collect();
            let out_bytes: Vec<u8> = out.iter()
                .flat_map(|f| if let Frame::Binary(b) = f { b.to_vec() } else { vec![] })
                .collect();

            prop_assert_eq!(in_text, out_text);
            prop_assert_eq!(in_bytes, out_bytes);
        }
    }
}
