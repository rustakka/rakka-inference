//! Chat-style autoregressive generation. Runs entirely inside one
//! `tokio::task::spawn_blocking` because `ort::Session::run` is sync;
//! emits `TokenChunk`s through a bounded mpsc channel so the caller
//! gets a real `Stream`.
//!
//! KV-cache handling assumes the HuggingFace Optimum-ONNX layout for
//! causal LMs (the most common ONNX export shape today):
//!
//! - `input_ids`: [1, seq_len] i64
//! - `attention_mask`: [1, past_len + seq_len] i64
//! - `position_ids`: [1, seq_len] i64
//! - `past_key_values.{i}.{key,value}`: [1, n_kv_heads, past_len, head_dim] f32
//! - output `logits`: [1, seq_len, vocab] f32
//! - output `present.{i}.{key,value}`: same as past_kv with new past_len
//!
//! Models that deviate (different dtype on logits or KV cache, missing
//! attention_mask, non-standard KV shapes) will fail with a typed
//! BadRequest naming the offending field. Operators can drop down to
//! `OrtRunner::infer` which has no shape assumptions.

use std::sync::Arc;

use atomr_infer_core::batch::{ExecuteBatch, SamplingParams};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::tokens::{FinishReason, TokenChunk, TokenUsage};
use futures::stream::BoxStream;
use futures::StreamExt;
use ort::session::SessionOutputs;
use ort::value::{Tensor, TensorElementType};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::OrtConfig;
use crate::error::{internal, map_ort_err};
use crate::sampling::{rng_from, sample_next};
use crate::session::OrtState;
use crate::tokenizer::render_chat;
use crate::topology::{ModelKind, Topology};

pub(crate) async fn run_generation(
    state: Arc<OrtState>,
    cfg: OrtConfig,
    batch: ExecuteBatch,
) -> InferenceResult<BoxStream<'static, InferenceResult<TokenChunk>>> {
    if !matches!(state.topology.kind, ModelKind::TextGenWithKv) {
        return Err(InferenceError::BadRequest {
            message: format!(
                "ort: chat-style execute() requires a causal LM with KV cache; \
                 probed kind = {:?}, inputs = {:?}, outputs = {:?}. \
                 Use OrtRunner::infer for embedding / encoder / vision models.",
                state.topology.kind, state.topology.all_input_names, state.topology.all_output_names
            ),
        });
    }
    let Some(tokenizer) = state.tokenizer.clone() else {
        return Err(InferenceError::BadRequest {
            message: "ort: chat-style execute() requires a tokenizer.json — \
                      set OrtConfig::tokenizer_path or place tokenizer.json next \
                      to the ONNX file (or enable ort-hf-hub + set hf_repo)"
                .into(),
        });
    };

    let prompt = render_chat(&batch.messages)?;
    let encoding = tokenizer
        .encode(prompt.clone(), true)
        .map_err(|e| internal("tokenizer.encode", e))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(InferenceError::BadRequest {
            message: "ort: tokenizer produced empty id sequence".into(),
        });
    }

    let max_new = batch
        .sampling
        .max_tokens
        .unwrap_or(cfg.default_max_new_tokens);
    let eos_id = tokenizer.token_to_id("</s>").or_else(|| {
        tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
    });

    let (kv_heads, head_dim) = infer_kv_dims(&state)?;

    let (tx, rx) = mpsc::channel::<InferenceResult<TokenChunk>>(64);
    let request_id = batch.request_id.clone();
    let sampling = batch.sampling.clone();
    let prompt_text_len = prompt.len();

    tokio::task::spawn_blocking(move || {
        let mut rng = rng_from(sampling.seed);
        let mut session = state.session.lock();
        let topo = state.topology.clone();
        let result = run_loop(
            &mut session,
            &topo,
            &tokenizer,
            &request_id,
            &sampling,
            prompt_ids,
            kv_heads,
            head_dim,
            max_new,
            eos_id,
            prompt_text_len,
            &tx,
            &mut rng,
        );
        if let Err(e) = result {
            let _ = tx.blocking_send(Err(e));
        }
    });

    Ok(ReceiverStream::new(rx).boxed())
}

#[allow(clippy::too_many_arguments)]
fn run_loop(
    session: &mut ort::session::Session,
    topo: &Topology,
    tokenizer: &tokenizers::Tokenizer,
    request_id: &str,
    sampling: &SamplingParams,
    prompt_ids: Vec<u32>,
    kv_heads: usize,
    head_dim: usize,
    max_new: u32,
    eos_id: Option<u32>,
    prompt_text_byte_len: usize,
    tx: &mpsc::Sender<InferenceResult<TokenChunk>>,
    rng: &mut rand::rngs::StdRng,
) -> InferenceResult<()> {
    if topo.logits_dtype != Some(TensorElementType::Float32) {
        return Err(InferenceError::BadRequest {
            message: format!(
                "ort: chat-style execute() only supports f32 logits in this build \
                 (probed: {:?}). Re-export with FP32 logits or use infer().",
                topo.logits_dtype
            ),
        });
    }

    let prompt_len = prompt_ids.len();
    let mut all_ids: Vec<u32> = prompt_ids.clone();
    let mut prev_full_decoded = String::new();
    let mut output_tokens: u32 = 0;
    let mut finish: Option<FinishReason> = None;

    // Per-layer past_kv: starts empty (past_len = 0).
    let mut past_kv: Vec<(Vec<f32>, Vec<f32>)> = (0..topo.kv_layers.len())
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    let mut past_len: usize = 0;

    // Step 0 = prefill (whole prompt). Subsequent steps feed back one token.
    let mut step: usize = 0;
    loop {
        let is_prefill = step == 0;
        let cur_input: Vec<i64> = if is_prefill {
            prompt_ids.iter().map(|&t| t as i64).collect()
        } else {
            vec![*all_ids.last().expect("at least one token") as i64]
        };
        let cur_len = cur_input.len();
        let total_attn_len = past_len + cur_len;

        let input_ids = Tensor::from_array(([1i64, cur_len as i64], cur_input)).map_err(map_ort_err)?;
        let attention_mask = if topo.attention_mask_name.is_some() {
            Some(
                Tensor::from_array((
                    [1i64, total_attn_len as i64],
                    vec![1i64; total_attn_len],
                ))
                .map_err(map_ort_err)?,
            )
        } else {
            None
        };
        let position_ids = if topo.position_ids_name.is_some() {
            let positions: Vec<i64> = (past_len as i64..(past_len + cur_len) as i64).collect();
            Some(Tensor::from_array(([1i64, cur_len as i64], positions)).map_err(map_ort_err)?)
        } else {
            None
        };

        let mut kv_tensors: Vec<(String, Tensor<f32>)> = Vec::with_capacity(2 * topo.kv_layers.len());
        for (layer, (k_buf, v_buf)) in topo.kv_layers.iter().zip(past_kv.iter()) {
            let shape = [1i64, kv_heads as i64, past_len as i64, head_dim as i64];
            kv_tensors.push((
                layer.past_key_input.clone(),
                Tensor::from_array((shape, k_buf.clone())).map_err(map_ort_err)?,
            ));
            kv_tensors.push((
                layer.past_value_input.clone(),
                Tensor::from_array((shape, v_buf.clone())).map_err(map_ort_err)?,
            ));
        }

        let outputs: SessionOutputs<'_> = run_with_inputs(
            session,
            topo,
            input_ids,
            attention_mask,
            position_ids,
            kv_tensors,
        )?;

        // Pull logits → take last position → sample.
        let logits_name = topo
            .logits_name
            .as_deref()
            .expect("logits_name set when kind == TextGenWithKv");
        let logits_value = outputs
            .get(logits_name)
            .ok_or_else(|| InferenceError::Internal(format!("ort: missing output '{logits_name}'")))?;
        let (logits_shape, logits_data) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(map_ort_err)?;
        if logits_shape.len() < 3 {
            return Err(InferenceError::Internal(format!(
                "ort: logits has rank {}, expected 3 ([batch, seq, vocab])",
                logits_shape.len()
            )));
        }
        let vocab = logits_shape[logits_shape.len() - 1] as usize;
        let seq = logits_shape[logits_shape.len() - 2] as usize;
        let last_offset = (seq - 1) * vocab;
        let last_logits = &logits_data[last_offset..last_offset + vocab];
        let next_id = sample_next(last_logits, sampling, rng);

        // Pull present_kv → store for next step.
        let mut new_past: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(topo.kv_layers.len());
        for layer in &topo.kv_layers {
            let (_, k_data) = outputs
                .get(layer.present_key_output.as_str())
                .ok_or_else(|| {
                    InferenceError::Internal(format!(
                        "ort: missing output '{}'",
                        layer.present_key_output
                    ))
                })?
                .try_extract_tensor::<f32>()
                .map_err(map_ort_err)?;
            let (_, v_data) = outputs
                .get(layer.present_value_output.as_str())
                .ok_or_else(|| {
                    InferenceError::Internal(format!(
                        "ort: missing output '{}'",
                        layer.present_value_output
                    ))
                })?
                .try_extract_tensor::<f32>()
                .map_err(map_ort_err)?;
            new_past.push((k_data.to_vec(), v_data.to_vec()));
        }
        drop(outputs);
        past_kv = new_past;
        past_len += cur_len;
        all_ids.push(next_id);
        output_tokens += 1;

        // Streaming decode: re-decode full sequence (skipping the
        // prompt prefix), emit only the new suffix unless it ends in
        // a UTF-8 replacement character.
        let generated = &all_ids[prompt_len..];
        let full_decoded = tokenizer
            .decode(generated, true)
            .map_err(|e| internal("tokenizer.decode", e))?;
        let (delta, advance) = decode_delta(&prev_full_decoded, &full_decoded);
        if !delta.is_empty() && tx
            .blocking_send(Ok(TokenChunk {
                request_id: request_id.to_owned(),
                text_delta: delta,
                tool_call_delta: None,
                usage: None,
                finish_reason: None,
            }))
            .is_err()
        {
            return Ok(()); // downstream dropped
        }
        if advance {
            prev_full_decoded = full_decoded.clone();
        }

        // Stop conditions.
        if Some(next_id) == eos_id {
            finish = Some(FinishReason::Stop);
            break;
        }
        if output_tokens >= max_new {
            finish = Some(FinishReason::Length);
            break;
        }
        if !sampling.stop.is_empty() {
            let _ = prompt_text_byte_len; // suppress unused if stop empty
            for stop in &sampling.stop {
                if full_decoded.contains(stop.as_str()) {
                    finish = Some(FinishReason::Stop);
                    break;
                }
            }
            if finish.is_some() {
                break;
            }
        }

        step += 1;
    }

    // Final flush — emit remaining decoded text and the usage chunk.
    let generated = &all_ids[prompt_len..];
    let full_decoded = tokenizer
        .decode(generated, true)
        .map_err(|e| internal("tokenizer.decode", e))?;
    let tail = if full_decoded.len() > prev_full_decoded.len()
        && full_decoded.starts_with(prev_full_decoded.as_str())
    {
        full_decoded[prev_full_decoded.len()..].to_owned()
    } else {
        String::new()
    };
    let _ = tx.blocking_send(Ok(TokenChunk {
        request_id: request_id.to_owned(),
        text_delta: tail,
        tool_call_delta: None,
        usage: Some(TokenUsage {
            input_tokens: prompt_len as u32,
            output_tokens,
            ..Default::default()
        }),
        finish_reason: finish.or(Some(FinishReason::Stop)),
    }));
    Ok(())
}

fn run_with_inputs<'s>(
    session: &'s mut ort::session::Session,
    topo: &Topology,
    input_ids: Tensor<i64>,
    attention_mask: Option<Tensor<i64>>,
    position_ids: Option<Tensor<i64>>,
    kv_tensors: Vec<(String, Tensor<f32>)>,
) -> InferenceResult<SessionOutputs<'s>> {
    use ort::session::SessionInputValue;

    let input_ids_name = topo
        .input_ids_name
        .clone()
        .unwrap_or_else(|| "input_ids".to_owned());
    let mut entries: Vec<(std::borrow::Cow<'static, str>, SessionInputValue<'_>)> = Vec::new();
    entries.push((std::borrow::Cow::Owned(input_ids_name), input_ids.into()));
    if let (Some(name), Some(t)) = (topo.attention_mask_name.clone(), attention_mask) {
        entries.push((std::borrow::Cow::Owned(name), t.into()));
    }
    if let (Some(name), Some(t)) = (topo.position_ids_name.clone(), position_ids) {
        entries.push((std::borrow::Cow::Owned(name), t.into()));
    }
    for (name, t) in kv_tensors {
        entries.push((std::borrow::Cow::Owned(name), t.into()));
    }

    session.run(entries).map_err(map_ort_err)
}

fn infer_kv_dims(state: &OrtState) -> InferenceResult<(usize, usize)> {
    use ort::value::ValueType;
    let Some(layer) = state.topology.kv_layers.first() else {
        return Err(InferenceError::Internal(
            "ort: kv_layers empty after probe — bug in topology::probe".into(),
        ));
    };
    let input = state
        .session
        .lock()
        .inputs()
        .iter()
        .find(|i| i.name() == layer.past_key_input)
        .map(|i| i.dtype().clone())
        .ok_or_else(|| {
            InferenceError::Internal(format!(
                "ort: declared kv input '{}' not found on session",
                layer.past_key_input
            ))
        })?;
    let ValueType::Tensor { shape, .. } = input else {
        return Err(InferenceError::BadRequest {
            message: format!("ort: kv input '{}' is not a tensor", layer.past_key_input),
        });
    };
    if shape.len() != 4 {
        return Err(InferenceError::BadRequest {
            message: format!(
                "ort: kv input '{}' has rank {}, expected 4 ([batch, kv_heads, past, head_dim])",
                layer.past_key_input,
                shape.len()
            ),
        });
    }
    let kv_heads = shape[1];
    let head_dim = shape[3];
    if kv_heads <= 0 || head_dim <= 0 {
        return Err(InferenceError::BadRequest {
            message: format!(
                "ort: kv input '{}' declares dynamic heads ({kv_heads}) or head_dim ({head_dim}); \
                 export with concrete shape on dim 1 and dim 3",
                layer.past_key_input
            ),
        });
    }
    Ok((kv_heads as usize, head_dim as usize))
}

/// Compute the streaming text delta. Returns `(delta, advance)`:
/// `advance` is false when the new text ends in a UTF-8 replacement
/// char (model emitted a partial multi-byte sequence) — caller should
/// withhold the delta and re-decode after the next token.
fn decode_delta(prev: &str, full: &str) -> (String, bool) {
    if let Some(suffix) = full.strip_prefix(prev) {
        if suffix.ends_with('\u{FFFD}') {
            (String::new(), false)
        } else {
            (suffix.to_owned(), true)
        }
    } else {
        // Retraction (uncommon). Emit the new full minus prev's char-len
        // as best-effort; advance unconditionally so we don't loop.
        (full.to_owned(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_emits_simple_suffix() {
        let (d, a) = decode_delta("hello", "hello world");
        assert_eq!(d, " world");
        assert!(a);
    }

    #[test]
    fn delta_withholds_on_replacement_char() {
        let (d, a) = decode_delta("hi", "hi\u{FFFD}");
        assert_eq!(d, "");
        assert!(!a);
    }

    #[test]
    fn delta_handles_retraction_by_resetting() {
        // Unusual case: new text doesn't extend prev. We just emit the
        // new full text and accept duplication rather than looping.
        let (d, a) = decode_delta("hellö", "hi");
        assert_eq!(d, "hi");
        assert!(a);
    }
}
