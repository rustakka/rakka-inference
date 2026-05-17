//! `MockRealtimeRunner` — deterministic [`RealtimeRunner`] for tests.
//!
//! Spawns one adapter task per session that:
//! - Forwards every inbound message into a configurable script: if
//!   `echo_in` is true the inbound is mirrored back as a synthesized
//!   transcript; otherwise scripted responses fire on `Commit`.
//! - Closes cleanly when the caller sends [`RealtimeIn::Close`] or
//!   drops the inbound sender.

use async_trait::async_trait;
use futures::future::{AbortHandle, Abortable};

use atomr_infer_core::audio::{RealtimeBatch, RealtimeIn, RealtimeOut, TranscriptRole};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::{RealtimeRunner, RealtimeSession, SessionRebuildCause};
use atomr_infer_core::runtime::{RuntimeKind, TransportKind};

#[derive(Debug, Clone, Default)]
pub struct MockRealtimeScript {
    /// Responses fired in order on each [`RealtimeIn::Commit`]. After
    /// the list is exhausted the runner remains idle until close.
    pub responses: Vec<String>,
    /// When true, every inbound [`RealtimeIn::Text`] is mirrored back as
    /// an assistant transcript before any scripted response.
    pub echo_in: bool,
    /// Force `open_session` to return `err` instead of spawning.
    pub fail_with: Option<InferenceError>,
}

pub struct MockRealtimeRunner {
    script: MockRealtimeScript,
}

impl MockRealtimeRunner {
    pub fn new(script: MockRealtimeScript) -> Self {
        Self { script }
    }
}

#[async_trait]
impl RealtimeRunner for MockRealtimeRunner {
    async fn open_session(&mut self, batch: RealtimeBatch) -> InferenceResult<RealtimeSession> {
        if let Some(err) = self.script.fail_with.clone() {
            return Err(err);
        }
        let script = self.script.clone();
        let request_id = batch.request_id.clone();
        let (abort_handle, abort_reg) = AbortHandle::new_pair();
        let RealtimeBatch {
            mut inbound,
            outbound,
            ..
        } = batch;
        let adapter = async move {
            let mut response_idx = 0;
            while let Some(msg) = inbound.recv().await {
                match msg {
                    RealtimeIn::Text(t) => {
                        if script.echo_in
                            && outbound
                                .send(RealtimeOut::Transcript {
                                    role: TranscriptRole::User,
                                    text: t,
                                    is_final: true,
                                })
                                .await
                                .is_err()
                        {
                            break;
                        }
                    }
                    RealtimeIn::AudioFrame { .. } => continue,
                    RealtimeIn::Commit => {
                        if let Some(resp) = script.responses.get(response_idx).cloned() {
                            response_idx += 1;
                            if outbound
                                .send(RealtimeOut::Transcript {
                                    role: TranscriptRole::Assistant,
                                    text: resp,
                                    is_final: true,
                                })
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                    RealtimeIn::Interrupt => continue,
                    RealtimeIn::Close => break,
                    _ => continue,
                }
            }
            let _ = outbound.send(RealtimeOut::Done).await;
        };
        tokio::spawn(Abortable::new(adapter, abort_reg));
        Ok(RealtimeSession::new(request_id, abort_handle))
    }

    async fn rebuild_session(&mut self, _cause: SessionRebuildCause) -> InferenceResult<()> {
        Ok(())
    }

    fn runtime_kind(&self) -> RuntimeKind {
        RuntimeKind::RealtimeSpeech
    }
    fn transport_kind(&self) -> TransportKind {
        TransportKind::UnknownTransport
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_infer_core::audio::{SynthOptions, VoiceRef};
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn mock_echoes_text_and_closes() {
        let (tx_in, rx_in) = mpsc::channel::<RealtimeIn>(4);
        let (tx_out, mut rx_out) = mpsc::channel::<RealtimeOut>(4);
        let batch = RealtimeBatch {
            request_id: "s1".into(),
            model: "mock".into(),
            voice: VoiceRef::Named("v".into()),
            options: SynthOptions::default(),
            inbound: rx_in,
            outbound: tx_out,
        };
        let mut r = MockRealtimeRunner::new(MockRealtimeScript {
            echo_in: true,
            responses: vec!["resp-1".into()],
            ..Default::default()
        });
        let _sess = r.open_session(batch).await.unwrap();
        tx_in.send(RealtimeIn::Text("hello".into())).await.unwrap();
        tx_in.send(RealtimeIn::Commit).await.unwrap();
        tx_in.send(RealtimeIn::Close).await.unwrap();
        drop(tx_in);

        let mut got_user = false;
        let mut got_assistant = false;
        let mut got_done = false;
        while let Some(msg) = rx_out.recv().await {
            match msg {
                RealtimeOut::Transcript {
                    role: TranscriptRole::User,
                    text,
                    ..
                } => {
                    got_user = true;
                    assert_eq!(text, "hello");
                }
                RealtimeOut::Transcript {
                    role: TranscriptRole::Assistant,
                    text,
                    ..
                } => {
                    got_assistant = true;
                    assert_eq!(text, "resp-1");
                }
                RealtimeOut::Done => {
                    got_done = true;
                }
                _ => {}
            }
        }
        assert!(got_user && got_assistant && got_done);
    }
}
