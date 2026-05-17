//! Real whisper.cpp inference. Gated on `stt-whisper` **and** a
//! supported host arch (`x86_64` or `aarch64`). Everything else returns
//! [`WhisperError::FeatureDisabled`] / [`WhisperError::UnsupportedArch`]
//! from [`crate::runner::WhisperRunner`].

#![cfg(all(feature = "stt-whisper", any(target_arch = "x86_64", target_arch = "aarch64")))]

use std::path::Path;
use std::sync::Arc;

use atomr_infer_core::audio::{TranscriptChunk, WordTiming};
use parking_lot::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

use crate::config::WhisperConfig;
use crate::error::WhisperError;

/// Lazily-built whisper.cpp context plus the configuration that built it.
pub struct WhisperSession {
    pub ctx: Arc<WhisperContext>,
    pub state: Mutex<WhisperState>,
}

pub fn build_session(cfg: &WhisperConfig) -> Result<Arc<WhisperSession>, WhisperError> {
    let path = cfg.model_path.as_path();
    if !path.is_file() {
        return Err(WhisperError::ModelNotFound {
            path: path.to_path_buf(),
        });
    }
    let path_str = path.to_string_lossy().into_owned();
    let ctx = WhisperContext::new_with_params(&path_str, WhisperContextParameters::default())
        .map_err(|e| WhisperError::Backend(format!("WhisperContext::new: {e}")))?;
    let ctx = Arc::new(ctx);
    let state = ctx
        .create_state()
        .map_err(|e| WhisperError::Backend(format!("WhisperContext::create_state: {e}")))?;
    Ok(Arc::new(WhisperSession {
        ctx,
        state: Mutex::new(state),
    }))
}

/// Run whisper.cpp on a 16 kHz mono f32 PCM buffer. Returns one
/// [`TranscriptChunk`] per Whisper segment, with `is_final = true` set
/// on the last chunk.
pub fn transcribe(
    session: &WhisperSession,
    samples: &[f32],
    cfg: &WhisperConfig,
    request_id: &str,
) -> Result<Vec<TranscriptChunk>, WhisperError> {
    let mut state = session.state.lock();
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    if let Some(n) = cfg.n_threads {
        params.set_n_threads(n as i32);
    }
    if let Some(lang) = cfg.language.as_deref() {
        params.set_language(Some(lang));
    }
    params.set_translate(cfg.translate);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if cfg.word_timestamps {
        params.set_token_timestamps(true);
    }

    state
        .full(params, samples)
        .map_err(|e| WhisperError::Backend(format!("WhisperState::full: {e}")))?;

    let n_segments = state.full_n_segments();
    if n_segments <= 0 {
        // No detectable speech — emit a single empty terminal chunk so
        // consumers can still drain the stream.
        return Ok(vec![TranscriptChunk {
            request_id: request_id.to_owned(),
            is_final: true,
            text: String::new(),
            ts_start_ms: 0,
            ts_end_ms: 0,
            speaker_id: None,
            words: Vec::new(),
            usage: None,
        }]);
    }

    let mut chunks = Vec::with_capacity(n_segments as usize);
    for i in 0..n_segments {
        let segment = state
            .get_segment(i)
            .ok_or_else(|| WhisperError::Backend(format!("get_segment({i}) returned None")))?;
        let text = segment
            .to_str_lossy()
            .map_err(|e| WhisperError::Backend(format!("segment.to_str_lossy({i}): {e}")))?
            .into_owned();
        // whisper.cpp returns timestamps in centiseconds.
        let ts_start_ms = (segment.start_timestamp().max(0) as u64 * 10) as u32;
        let ts_end_ms = (segment.end_timestamp().max(0) as u64 * 10) as u32;

        let words = if cfg.word_timestamps {
            extract_words(&segment)?
        } else {
            Vec::new()
        };

        chunks.push(TranscriptChunk {
            request_id: request_id.to_owned(),
            is_final: i + 1 == n_segments,
            text,
            ts_start_ms,
            ts_end_ms,
            speaker_id: None,
            words,
            usage: None,
        });
    }
    Ok(chunks)
}

fn extract_words(segment: &whisper_rs::WhisperSegment<'_>) -> Result<Vec<WordTiming>, WhisperError> {
    let n = segment.n_tokens();
    let mut out = Vec::with_capacity(n.max(0) as usize);
    for j in 0..n {
        let token = match segment.get_token(j) {
            Some(t) => t,
            None => continue,
        };
        let text_cow = token
            .to_str_lossy()
            .map_err(|e| WhisperError::Backend(format!("token.to_str_lossy({j}): {e}")))?;
        let text = text_cow.into_owned();
        // Skip whisper.cpp's special tokens (`[_BEG_]`, `[_TT_*]`,
        // `<|endoftext|>`, …) — they're not user-visible words.
        if text.starts_with('[') || text.starts_with('<') {
            continue;
        }
        let data = token.token_data();
        let ts_start_ms = (data.t0.max(0) as u64 * 10) as u32;
        let ts_end_ms = (data.t1.max(0) as u64 * 10) as u32;
        out.push(WordTiming {
            text: text.trim().to_owned(),
            ts_start_ms,
            ts_end_ms,
            confidence: Some(data.p),
        });
    }
    Ok(out)
}

/// Convenience for callers that want to check the configured model
/// exists before constructing a runner (e.g. on deployment).
pub fn assert_model_present(path: &Path) -> Result<(), WhisperError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(WhisperError::ModelNotFound {
            path: path.to_path_buf(),
        })
    }
}

// Silence the unused-import warning if the trait is not consulted.
#[allow(dead_code)]
fn _silence_unused() -> Option<WhisperState> {
    None
}
