//! `MockWsServer` — in-process WebSocket server used by provider WS
//! client tests (Deepgram, AssemblyAI, ElevenLabs, OpenAI Realtime,
//! Gemini Live).
//!
//! Binds to `127.0.0.1:0`, accepts the first client connection, and
//! exposes scripted send / expect helpers. Each test typically:
//!
//! 1. `MockWsServer::start()` → returns the bound URL.
//! 2. Spawns its provider client against the URL.
//! 3. Drives the server-side conversation via `expect_*` and `send_*`.
//!
//! Helper coverage matches the cross-provider needs documented in M2:
//! observe inbound binary audio frames, push down transcript chunks,
//! close with a custom code.

use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

type ServerStream = WebSocketStream<TcpStream>;

/// Inbound frames captured from the connected client. The producer
/// (server accept loop) pushes; tests drain via `expect_*` helpers.
type InboundQueue = mpsc::UnboundedReceiver<Message>;
type InboundSink = mpsc::UnboundedSender<Message>;

/// Outbound frames the test pushes; the accept loop drains and sends.
type OutboundQueue = mpsc::UnboundedReceiver<Message>;
type OutboundSink = mpsc::UnboundedSender<Message>;

/// Server handle.
pub struct MockWsServer {
    url: String,
    inbound: Arc<Mutex<InboundQueue>>,
    outbound: OutboundSink,
}

impl MockWsServer {
    /// Bind to an ephemeral port and start accepting one connection.
    pub async fn start() -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let url = format!("ws://{}/", addr);
        let (in_tx, in_rx): (InboundSink, InboundQueue) = mpsc::unbounded_channel();
        let (out_tx, out_rx): (OutboundSink, OutboundQueue) = mpsc::unbounded_channel();
        tokio::spawn(accept_loop(listener, in_tx, out_rx));
        Ok(Self {
            url,
            inbound: Arc::new(Mutex::new(in_rx)),
            outbound: out_tx,
        })
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// Push one outbound message to the client. No-op if the client
    /// has disconnected.
    pub fn send(&self, msg: Message) -> Result<(), &'static str> {
        self.outbound.send(msg).map_err(|_| "ws server outbound closed")
    }

    /// Push a JSON text frame.
    pub fn send_json(&self, json: serde_json::Value) -> Result<(), &'static str> {
        self.send(Message::Text(json.to_string()))
    }

    /// Push a raw text frame (no JSON wrapping).
    pub fn send_text(&self, text: &str) -> Result<(), &'static str> {
        self.send(Message::Text(text.to_owned()))
    }

    /// Receive next inbound frame and parse it as JSON, with a deadline.
    /// Returns `None` if the connection closes or the frame is not text.
    pub async fn recv_json(&self, timeout: Duration) -> Option<serde_json::Value> {
        match self.recv(timeout).await? {
            Message::Text(t) => serde_json::from_str(&t).ok(),
            _ => None,
        }
    }

    /// Push a transcript chunk (Deepgram / AssemblyAI-shaped envelope).
    pub fn send_transcript_chunk(&self, text: &str, is_final: bool) -> Result<(), &'static str> {
        self.send_json(serde_json::json!({
            "is_final": is_final,
            "channel": {"alternatives": [{"transcript": text}]},
        }))
    }

    /// Close the connection with a code.
    pub fn close_with(&self, code: u16, reason: &str) -> Result<(), &'static str> {
        let reason: std::borrow::Cow<'static, str> = std::borrow::Cow::Owned(reason.to_owned());
        self.send(Message::Close(Some(CloseFrame {
            code: CloseCode::from(code),
            reason,
        })))
    }

    /// Receive next inbound frame, with a deadline.
    pub async fn recv(&self, timeout: Duration) -> Option<Message> {
        let mut guard = self.inbound.lock().await;
        tokio::time::timeout(timeout, guard.recv()).await.ok().flatten()
    }

    /// Collect up to `n` inbound binary frames (e.g. audio chunks
    /// pushed by the client), with a per-frame deadline.
    pub async fn expect_binary_frames(&self, n: usize, timeout: Duration) -> Vec<bytes::Bytes> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match self.recv(timeout).await {
                Some(Message::Binary(b)) => out.push(bytes::Bytes::from(b)),
                Some(_) => continue,
                None => break,
            }
        }
        out
    }
}

async fn accept_loop(listener: TcpListener, in_tx: InboundSink, mut out_rx: OutboundQueue) {
    let (stream, _) = match listener.accept().await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(?e, "mock ws accept failed");
            return;
        }
    };
    let ws: ServerStream = match tokio_tungstenite::accept_async(stream).await {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!(?e, "mock ws handshake failed");
            return;
        }
    };
    let (mut sink, mut stream) = ws.split();
    let send_task = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            let is_close = matches!(msg, Message::Close(_));
            if sink.send(msg).await.is_err() {
                break;
            }
            if is_close {
                break;
            }
        }
    });
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = stream.next().await {
            match msg {
                Ok(m) => {
                    if in_tx.send(m).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let _ = tokio::join!(send_task, recv_task);
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_tungstenite::connect_async;

    #[tokio::test]
    async fn mock_round_trips_one_frame() {
        let server = MockWsServer::start().await.unwrap();
        let (mut client, _) = connect_async(server.url()).await.unwrap();
        client.send(Message::Binary(b"hello".to_vec())).await.unwrap();
        let frames = server.expect_binary_frames(1, Duration::from_secs(2)).await;
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].as_ref(), b"hello");
        server.send_transcript_chunk("world", true).unwrap();
        let resp = client.next().await.unwrap().unwrap();
        if let Message::Text(t) = resp {
            assert!(t.contains("\"world\""));
        } else {
            panic!("unexpected msg shape");
        }
    }
}
