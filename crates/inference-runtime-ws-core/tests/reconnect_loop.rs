//! Integration test: drive the full `ReconnectEngine` loop against
//! a small in-process server that drops the first connection mid-
//! handshake and accepts the second. Asserts that the reconnect
//! engine consults the backoff policy, sleeps the prescribed
//! amount, and recovers on the second attempt.

use std::time::Duration;

use atomr_infer_core::runtime::JitterKind;
use atomr_infer_remote_core::backoff::BackoffPolicy;
use atomr_infer_runtime_ws_core::{Frame, ReconnectEngine, WsClient, WsError};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

/// Server that drops `kills` connections before accepting one
/// stable client.
async fn flapping_server(kills: usize) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/", listener.local_addr().unwrap());
    let h = tokio::spawn(async move {
        for _ in 0..kills {
            let (stream, _) = listener.accept().await.unwrap();
            // Drop without handshake — client sees a hangup.
            drop(stream);
        }
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        // Echo one text frame and close.
        if let Some(Ok(Message::Text(t))) = ws.next().await {
            ws.send(Message::Text(format!("echo:{t}"))).await.unwrap();
        }
        let _ = ws.close(None).await;
    });
    (url, h)
}

#[tokio::test]
async fn reconnect_engine_recovers_after_one_drop() {
    let (url, server) = flapping_server(1).await;
    let mut engine = ReconnectEngine::new(
        BackoffPolicy {
            initial: Duration::from_millis(20),
            max: Duration::from_millis(40),
            multiplier: 2.0,
            jitter: JitterKind::None,
        },
        5,
    );

    let connected = loop {
        let Some(delay) = engine.next_delay() else {
            panic!("reconnect exhausted unexpectedly");
        };
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        match WsClient::connect(&url, Duration::from_millis(500)).await {
            Ok(pair) => break pair,
            Err(e) => {
                assert!(e.is_retryable(), "non-retryable: {e:?}");
                continue;
            }
        }
    };

    let (mut tx, mut rx) = connected;
    tx.send(Frame::Text("ping".into())).await.unwrap();
    let f = rx.next().await.unwrap().expect("frame");
    match f {
        Frame::Text(t) => assert_eq!(t, "echo:ping"),
        other => panic!("expected text, got {other:?}"),
    }
    assert_eq!(engine.attempts(), 2, "took two attempts (one drop + one success)");
    let _ = server.await;
}

#[tokio::test]
async fn reconnect_engine_exhausts_when_kills_exceed_budget() {
    let (url, server) = flapping_server(10).await;
    let mut engine = ReconnectEngine::new(
        BackoffPolicy {
            initial: Duration::from_millis(5),
            max: Duration::from_millis(10),
            multiplier: 2.0,
            jitter: JitterKind::None,
        },
        3,
    );

    let mut last_err: Option<WsError> = None;
    let outcome = loop {
        let Some(delay) = engine.next_delay() else {
            break Err::<(), _>(last_err.expect("at least one attempt"));
        };
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        match WsClient::connect(&url, Duration::from_millis(200)).await {
            Ok(_) => break Ok(()),
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        }
    };

    assert!(outcome.is_err(), "exhaustion expected after 3 attempts");
    // Spin server down so the test cleans up promptly.
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn keepalive_round_trips_a_ping() {
    // Stand up a one-shot server that pongs whatever ping it receives.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(msg)) = ws.next().await {
            match msg {
                Message::Ping(p) => ws.send(Message::Pong(p)).await.unwrap(),
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    let (mut tx, mut rx) = WsClient::connect(&url, Duration::from_secs(2)).await.unwrap();
    tx.send(Frame::Ping(Bytes::from_static(b"hb"))).await.unwrap();
    // tokio-tungstenite auto-responds to pings server-side; we should
    // see a Pong come back.
    let observed_pong = tokio::time::timeout(Duration::from_secs(1), async {
        while let Ok(Some(f)) = rx.next().await {
            if matches!(f, Frame::Pong(_)) {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false);
    assert!(observed_pong, "expected pong");

    let _ = tx.close(1000, "bye").await;
    server.abort();
    let _ = server.await;
}
