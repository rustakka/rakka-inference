//! Provider-agnostic SSE chunk parsing. Every provider's stream is
//! framed identically (lines beginning `data: <json>` separated by
//! blank lines, terminated by `data: [DONE]`); only the inner JSON
//! shape differs. Per-provider crates layer concrete types on top.

use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::stream::{BoxStream, Stream, StreamExt};

use inference_core::error::InferenceError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseChunk {
    pub event: Option<String>,
    pub data: String,
}

/// Decode a byte-stream from an HTTP body into a stream of SSE chunks,
/// stopping at the provider-specific terminator (`[DONE]` for OpenAI;
/// the per-provider crate may wrap this with its own end-of-stream
/// recognition).
pub fn decode_sse_stream<S>(stream: S) -> BoxStream<'static, Result<SseChunk, InferenceError>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    stream
        .map(|res| res.map_err(|e| InferenceError::NetworkError(e.to_string())))
        .eventsource()
        .map(|res| {
            res.map(|ev| SseChunk {
                event: Some(ev.event).filter(|s| !s.is_empty()),
                data: ev.data,
            })
            .map_err(|e| InferenceError::NetworkError(format!("sse decode: {e}")))
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn parses_two_chunks() {
        let body = "data: {\"x\":1}\n\ndata: {\"x\":2}\n\ndata: [DONE]\n\n"
            .as_bytes()
            .to_vec();
        let s = stream::iter(vec![Ok::<_, reqwest::Error>(Bytes::from(body))]);
        let mut decoded = decode_sse_stream(s);
        let first = decoded.next().await.unwrap().unwrap();
        assert_eq!(first.data, r#"{"x":1}"#);
        let second = decoded.next().await.unwrap().unwrap();
        assert_eq!(second.data, r#"{"x":2}"#);
        let third = decoded.next().await.unwrap().unwrap();
        assert_eq!(third.data, "[DONE]");
    }
}
