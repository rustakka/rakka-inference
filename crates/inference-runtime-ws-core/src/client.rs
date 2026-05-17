//! `WsClient` — TLS-aware WebSocket client used by audio provider
//! runtimes. Splits the connection into independent
//! [`WsSender`] / [`WsReceiver`] halves so the provider can pump
//! audio uplink and transcript downlink on separate tasks.
//!
//! The connection lifecycle is deliberately small:
//!
//! 1. [`WsClient::connect`] → returns the split halves on success.
//! 2. [`WsSender::send`] → push a [`Frame`] to the wire.
//! 3. [`WsReceiver::next`] → await the next inbound [`Frame`].
//! 4. On send/receive error, the provider consults its
//!    [`crate::ReconnectEngine`] and calls [`WsClient::connect`] again.
//!
//! Keepalive lives one layer up: the provider drives
//! [`crate::Keepalive`] from a `tokio::time::interval` and pushes
//! `Frame::Ping(...)` when it sees `KeepaliveAction::SendPing`.

use std::time::Duration;

use futures::stream::SplitSink;
use futures::stream::SplitStream;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use url::Url;

use crate::error::{WsError, WsResult};
use crate::frame::Frame;

type ClientStream = WebSocketStream<MaybeTlsStream<TcpStream>>;
type ClientSink = SplitSink<ClientStream, tokio_tungstenite::tungstenite::Message>;
type ClientSrc = SplitStream<ClientStream>;

/// Outbound half of a connected WebSocket.
pub struct WsSender {
    sink: ClientSink,
}

impl WsSender {
    /// Push one frame onto the wire. Returns immediately after the
    /// frame is queued in tungstenite's internal buffer.
    pub async fn send(&mut self, frame: Frame) -> WsResult<()> {
        self.sink.send(frame.into_message()).await.map_err(WsError::from)
    }

    /// Flush all queued frames to the socket.
    pub async fn flush(&mut self) -> WsResult<()> {
        self.sink.flush().await.map_err(WsError::from)
    }

    /// Close the connection gracefully with `code` and `reason`.
    /// After this call the sender is unusable.
    pub async fn close(mut self, code: u16, reason: &str) -> WsResult<()> {
        let frame = Frame::Close {
            code,
            reason: reason.to_owned(),
        };
        self.sink
            .send(frame.into_message())
            .await
            .map_err(WsError::from)?;
        self.sink.close().await.map_err(WsError::from)
    }
}

/// Inbound half of a connected WebSocket.
pub struct WsReceiver {
    stream: ClientSrc,
}

impl WsReceiver {
    /// Await the next [`Frame`]. Returns `Ok(None)` when the remote
    /// has closed the stream cleanly.
    pub async fn next(&mut self) -> WsResult<Option<Frame>> {
        loop {
            match self.stream.next().await {
                Some(Ok(msg)) => match Frame::from_message(msg) {
                    Some(f) => return Ok(Some(f)),
                    // `Message::Frame(_)` is internal-only; loop.
                    None => continue,
                },
                Some(Err(e)) => return Err(WsError::from(e)),
                None => return Ok(None),
            }
        }
    }
}

/// Stateless connector. The connection lifecycle (reconnect,
/// keepalive) is driven by the provider; this type exposes only
/// the call that hands back the two split halves.
pub struct WsClient;

impl WsClient {
    /// Open a WebSocket connection to `url`. Times out after
    /// `connect_timeout`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::time::Duration;
    /// use atomr_infer_runtime_ws_core::{Frame, WsClient};
    ///
    /// # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
    /// let (mut tx, mut rx) =
    ///     WsClient::connect("wss://api.example.com/v1/stream", Duration::from_secs(10))
    ///         .await?;
    /// tx.send(Frame::Text("hello".into())).await?;
    /// if let Some(frame) = rx.next().await? {
    ///     println!("got: {:?}", frame);
    /// }
    /// tx.close(1000, "bye").await?;
    /// # Ok(()) }
    /// ```
    pub async fn connect(url: &str, connect_timeout: Duration) -> WsResult<(WsSender, WsReceiver)> {
        Self::connect_with_headers(url, &[], connect_timeout).await
    }

    /// Open a WebSocket connection to `url` with extra upgrade-request
    /// headers (e.g. `Authorization` for Deepgram / AssemblyAI /
    /// Gemini Live). The host header is set automatically; do not pass
    /// it here.
    ///
    /// `headers` is a slice of `(name, value)` pairs. Header names must
    /// be valid ASCII per RFC 7230; values must contain only visible
    /// ASCII / TAB / SP. Invalid pairs surface as
    /// [`WsError::BadUrl`].
    pub async fn connect_with_headers(
        url: &str,
        headers: &[(&str, &str)],
        connect_timeout: Duration,
    ) -> WsResult<(WsSender, WsReceiver)> {
        let parsed = Url::parse(url).map_err(|e| WsError::BadUrl(e.to_string()))?;
        let mut req = parsed
            .as_str()
            .into_client_request()
            .map_err(|e| WsError::BadUrl(format!("ws request: {e}")))?;
        if !headers.is_empty() {
            let h = req.headers_mut();
            for (name, value) in headers {
                let n = HeaderName::from_bytes(name.as_bytes())
                    .map_err(|e| WsError::BadUrl(format!("bad header name {name}: {e}")))?;
                let v = HeaderValue::from_str(value)
                    .map_err(|e| WsError::BadUrl(format!("bad header value for {name}: {e}")))?;
                h.insert(n, v);
            }
        }
        let fut = connect_async(req);
        let (ws, _resp) = match tokio::time::timeout(connect_timeout, fut).await {
            Ok(r) => r.map_err(WsError::from)?,
            Err(_) => {
                return Err(WsError::ConnectTimeout {
                    seconds: connect_timeout.as_secs(),
                });
            }
        };
        let (sink, stream) = ws.split();
        Ok((WsSender { sink }, WsReceiver { stream }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_testkit::MockWsServer;
    use bytes::Bytes;

    #[tokio::test]
    async fn connect_send_receive_round_trip() {
        let server = MockWsServer::start().await.unwrap();
        let (mut tx, mut rx) = WsClient::connect(server.url(), Duration::from_secs(2))
            .await
            .unwrap();

        // Client → server binary frame
        tx.send(Frame::Binary(Bytes::from_static(b"hello")))
            .await
            .unwrap();
        let inbound = server.expect_binary_frames(1, Duration::from_secs(2)).await;
        assert_eq!(inbound.len(), 1);
        assert_eq!(inbound[0].as_ref(), b"hello");

        // Server → client text frame
        server.send_transcript_chunk("world", true).unwrap();
        let f = rx.next().await.unwrap().expect("frame");
        match f {
            Frame::Text(t) => assert!(t.contains("\"world\""), "{t}"),
            other => panic!("expected text, got {other:?}"),
        }

        let _ = tx.close(1000, "bye").await;
    }

    #[tokio::test]
    async fn connect_timeout_when_url_does_not_accept() {
        // 192.0.2.0/24 is TEST-NET-1; will never accept.
        let res = WsClient::connect("ws://192.0.2.1:9/", Duration::from_millis(150)).await;
        match res {
            Err(WsError::ConnectTimeout { .. }) => {}
            Err(other) => panic!("expected ConnectTimeout, got {other:?}"),
            Ok(_) => panic!("unexpectedly connected to TEST-NET-1"),
        }
    }

    #[tokio::test]
    async fn bad_url_is_terminal() {
        let res = WsClient::connect("not-a-url", Duration::from_secs(1)).await;
        match res {
            Err(e @ WsError::BadUrl(_)) => assert!(!e.is_retryable()),
            Err(other) => panic!("expected BadUrl, got {other:?}"),
            Ok(_) => panic!("unexpectedly parsed 'not-a-url'"),
        }
    }

    #[tokio::test]
    async fn close_propagates_to_remote() {
        let server = MockWsServer::start().await.unwrap();
        let (tx, mut rx) = WsClient::connect(server.url(), Duration::from_secs(2))
            .await
            .unwrap();
        tx.close(1000, "bye").await.unwrap();
        // Receiver should observe close or stream end shortly.
        let observed_end = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                match rx.next().await {
                    Ok(Some(Frame::Close { .. })) | Ok(None) => break true,
                    Ok(Some(_)) => continue,
                    Err(_) => break true,
                }
            }
        })
        .await
        .unwrap_or(false);
        assert!(observed_end, "expected close/end after sender close");
    }
}
