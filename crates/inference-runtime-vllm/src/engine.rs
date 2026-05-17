//! `VllmEngine` — PyO3 binding around vLLM's V1 synchronous
//! `LLMEngine` (`add_request` + `step` polling).
//!
//! This module is only compiled when the `vllm` feature is on. It is
//! intentionally narrow: a single struct that owns the GIL-pinned
//! engine handle plus a `generate` method that returns a streaming
//! [`RunHandle`].
//!
//! ## Why sync `LLMEngine` not async
//!
//! vLLM exposes both `AsyncLLMEngine` (asyncio-driven) and
//! `LLMEngine` (sync `add_request` / `step`). Bridging asyncio
//! futures into tokio via pyo3-async-runtimes requires a running
//! Python event loop, which is fragile to set up from Rust. The sync
//! API maps cleanly onto our use case: each `step()` advances all
//! in-flight requests one micro-batch and returns finished /
//! partial outputs we can demux ourselves. We trade asyncio's
//! out-of-the-box concurrency for a clearer GIL boundary.
//!
//! ## GIL placement
//!
//! Each `VllmEngine` instance holds an `Arc<PythonGpuBridge>` so all
//! Python calls run through one OS thread. vLLM internally schedules
//! GPU work; the GIL boundary is only held during `add_request` /
//! `step` / `abort_request` calls, never across the kernel itself.
//!
//! ## Streaming
//!
//! Tokens are forwarded into a `tokio::sync::mpsc` channel. The
//! poller task calls `step()` on a small interval; when the consumer
//! drops the [`RunHandle`], the next chunk-send fails and we call
//! `abort_request(request_id)` so the GPU doesn't keep generating
//! into a closed channel.

#![cfg(feature = "vllm")]

use std::sync::Arc;

use futures::stream::StreamExt;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use atomr_infer_core::batch::{ExecuteBatch, Message, MessageContent, Role};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::RunHandle;
use atomr_infer_core::tokens::{FinishReason, TokenChunk, TokenUsage};

use atomr_infer_python_bridge::PythonGpuBridge;

use crate::VllmConfig;

/// Counter used to mint a unique interpreter id per engine instance
/// so the [`PythonGpuBridge`] thread-pool tokens don't collide across
/// hot-restarts.
static INTERPRETER_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Owned vLLM engine handle. Lives behind an `Arc<VllmEngine>` so
/// multiple in-flight `generate` calls can share it.
pub(crate) struct VllmEngine {
    bridge: Arc<PythonGpuBridge>,
    /// `vllm.AsyncLLMEngine` instance. Held inside a `Py<PyAny>` so
    /// the engine survives across GIL releases.
    engine: Mutex<Py<PyAny>>,
    /// Monotonic request-id counter for `engine.add_request(...)`.
    request_seq: std::sync::atomic::AtomicU64,
}

// `Py<PyAny>` is already `Send + Sync` (PyO3 ensures every access
// reacquires the GIL); `Mutex<Py<PyAny>>` and `Arc<PythonGpuBridge>`
// inherit those bounds, so `VllmEngine` is `Send + Sync` without
// any `unsafe impl` (and `forbid(unsafe_code)` stays in force).

impl VllmEngine {
    /// Boot a fresh `AsyncLLMEngine`. Honours `config.hf_cache_dir`
    /// by setting `HF_HOME` on the Python interpreter before
    /// `import vllm`. This is the multi-instance unification hook —
    /// every atomr-infer process pointed at the same cache shares
    /// the same on-disk weights.
    pub(crate) async fn launch(config: &VllmConfig) -> InferenceResult<Self> {
        // Set HF_HOME / HF_HUB_CACHE *before* the bridge inits Python
        // for the first time. SAFETY: env vars are process-global; we
        // accept the read/write race at startup because Python imports
        // its own copy on first import and we control the lifecycle.
        if let Some(cache_dir) = &config.hf_cache_dir {
            std::env::set_var("HF_HOME", cache_dir);
            std::env::set_var("HF_HUB_CACHE", cache_dir.join("hub"));
        }

        // Initialise the Python interpreter exactly once for the
        // process. PyO3 panics if `Python::with_gil` runs before this
        // when the `auto-initialize` feature is off (which is the
        // default — `auto-initialize` would link libpython at build
        // time and complicate cross-compilation).
        pyo3::prepare_freethreaded_python();

        let interpreter_id = INTERPRETER_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bridge = PythonGpuBridge::new(interpreter_id);

        let model = config.model.clone();
        let dtype = config.dtype.clone();
        let tp = config.tensor_parallel_size;
        let gpu_util = config.gpu_memory_utilization;
        let max_len = config.max_model_len;
        let enforce_eager = config.enforce_eager;
        let enable_prefix_caching = config.enable_prefix_caching;
        let enable_chunked_prefill = config.enable_chunked_prefill;
        let max_num_seqs = config.max_num_seqs;
        let block_size = config.block_size;
        let quantization = config.quantization.clone();
        let limit_mm_per_prompt = config.limit_mm_per_prompt.clone();
        let cpu_offload_gb = config.cpu_offload_gb;

        let engine = bridge.with_python(|py| -> PyResult<Py<PyAny>> {
            let vllm = py.import_bound("vllm").map_err(|e| {
                pyo3::exceptions::PyImportError::new_err(format!(
                    "vllm import failed — install with `pip install vllm`: {e}"
                ))
            })?;
            // Use the sync LLMEngine + EngineArgs (not Async*) so
            // we can drive the engine via add_request/step from Rust
            // without needing a Python asyncio event loop.
            let engine_args = vllm.getattr("EngineArgs")?;
            let llm_engine = vllm.getattr("LLMEngine")?;

            // Build the kwargs dict for AsyncEngineArgs.
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("model", &model)?;
            kwargs.set_item("dtype", &dtype)?;
            kwargs.set_item("tensor_parallel_size", tp)?;
            if let Some(g) = gpu_util {
                kwargs.set_item("gpu_memory_utilization", g)?;
            }
            if let Some(m) = max_len {
                kwargs.set_item("max_model_len", m)?;
            }
            // Perf knobs — pass through only when set so the engine
            // falls back to vLLM defaults on `None`.
            if let Some(eager) = enforce_eager {
                kwargs.set_item("enforce_eager", eager)?;
            }
            if let Some(pc) = enable_prefix_caching {
                kwargs.set_item("enable_prefix_caching", pc)?;
            }
            if let Some(cp) = enable_chunked_prefill {
                kwargs.set_item("enable_chunked_prefill", cp)?;
            }
            if let Some(n) = max_num_seqs {
                kwargs.set_item("max_num_seqs", n)?;
            }
            if let Some(b) = block_size {
                kwargs.set_item("block_size", b)?;
            }
            if let Some(q) = &quantization {
                kwargs.set_item("quantization", q.as_str())?;
            }
            if let Some(limits) = &limit_mm_per_prompt {
                let dict = PyDict::new_bound(py);
                for (k, v) in limits {
                    dict.set_item(k, *v)?;
                }
                kwargs.set_item("limit_mm_per_prompt", dict)?;
            }
            if let Some(off) = cpu_offload_gb {
                kwargs.set_item("cpu_offload_gb", off)?;
            }
            // EngineArgs holds the configuration; LLMEngine wraps it.
            let args = engine_args.call((), Some(&kwargs))?;
            let engine = llm_engine.call_method1("from_engine_args", (args,))?;
            Ok(engine.unbind())
        })?;

        tracing::info!(model = %config.model, "vllm LLMEngine launched");

        Ok(Self {
            bridge,
            engine: Mutex::new(engine),
            request_seq: std::sync::atomic::AtomicU64::new(1),
        })
    }

    /// Render messages through the model's tokenizer chat template.
    /// Falls back to the simple `<|role|>` format if the tokenizer
    /// doesn't expose `apply_chat_template` (older vLLM / non-chat
    /// models).
    fn render_chat(&self, messages: &[Message]) -> InferenceResult<String> {
        // Build the python-side `messages` list of `{"role", "content"}`
        // dicts and call `engine.tokenizer.apply_chat_template(...)`.
        // The tokenizer's template knows the right format for each
        // model (Gemma's `<start_of_turn>` / Llama's `[INST]` / etc.).
        self.bridge
            .with_python(|py| -> PyResult<String> {
                let py_messages = PyList::empty_bound(py);
                for m in messages {
                    let d = PyDict::new_bound(py);
                    let role = match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                        _ => "user",
                    };
                    d.set_item("role", role)?;
                    d.set_item("content", message_text(m))?;
                    py_messages.append(d)?;
                }

                let engine_handle = self.engine.lock();
                let bound = engine_handle.bind(py);
                let tokenizer_method = bound
                    .call_method0("get_tokenizer")
                    .or_else(|_| bound.getattr("tokenizer"))
                    .ok();

                let tokenizer = match tokenizer_method {
                    Some(t) => t,
                    None => {
                        return Ok(simple_render(messages));
                    }
                };

                // get_tokenizer() may return a coroutine on V1; if so,
                // fall back to the simple render rather than block.
                if tokenizer.hasattr("apply_chat_template").unwrap_or(false) {
                    let kwargs = PyDict::new_bound(py);
                    kwargs.set_item("tokenize", false)?;
                    kwargs.set_item("add_generation_prompt", true)?;
                    let result = tokenizer.call_method("apply_chat_template", (py_messages,), Some(&kwargs));
                    if let Ok(rendered) = result {
                        if let Ok(s) = rendered.extract::<String>() {
                            return Ok(s);
                        }
                    }
                }
                Ok(simple_render(messages))
            })
            .map_err(|e| InferenceError::Internal(format!("vllm: render_chat: {e}")))
    }

    /// Translate `ExecuteBatch` into a Python prompt + sampling
    /// params, register the request with the engine, and return a
    /// [`RunHandle`] that streams tokens as the engine generates.
    pub(crate) async fn generate(self: &Arc<Self>, batch: ExecuteBatch) -> InferenceResult<RunHandle> {
        let prompt = self.render_chat(&batch.messages)?;
        let request_id = format!(
            "{}-{}",
            batch.request_id,
            self.request_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let max_tokens = batch.sampling.max_tokens.unwrap_or(512);
        let temperature = batch.sampling.temperature.unwrap_or(1.0);
        let top_p = batch.sampling.top_p.unwrap_or(1.0);
        let stop_tokens: Vec<String> = batch.sampling.stop.clone();
        let req_id_for_chunks = batch.request_id.clone();

        // Build SamplingParams + register the request with the engine
        // up front, so by the time `generate` returns the request is
        // already in flight on the GPU.
        self.bridge.with_python(|py| -> PyResult<()> {
            let vllm = py.import_bound("vllm")?;
            let sampling_params_cls = vllm.getattr("SamplingParams")?;
            let sp_kwargs = PyDict::new_bound(py);
            sp_kwargs.set_item("max_tokens", max_tokens)?;
            sp_kwargs.set_item("temperature", temperature)?;
            sp_kwargs.set_item("top_p", top_p)?;
            if !stop_tokens.is_empty() {
                sp_kwargs.set_item("stop", PyList::new_bound(py, &stop_tokens))?;
            }
            let sampling_params = sampling_params_cls.call((), Some(&sp_kwargs))?;

            // LLMEngine.add_request signature in vLLM 0.20:
            //   add_request(request_id: str, prompt: PromptType,
            //               params: SamplingParams, ...)
            // We pass the prompt as a plain str (vLLM accepts it).
            let engine_handle = self.engine.lock();
            let bound = engine_handle.bind(py);
            bound.call_method1("add_request", (&request_id, &prompt, sampling_params))?;
            Ok(())
        })?;

        // Channel for the poller to forward chunks into.
        let (tx, rx) = tokio::sync::mpsc::channel::<InferenceResult<TokenChunk>>(64);

        let engine = Arc::clone(self);
        let request_id_owned = request_id.clone();
        tokio::spawn(async move {
            let outcome = drive_generation(&engine, &request_id_owned, tx.clone(), &req_id_for_chunks).await;
            if let Err(e) = outcome {
                let _ = tx.send(Err(e)).await;
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx).boxed();
        Ok(RunHandle::streaming(stream))
    }
}

/// Poll `engine.step()` and forward this request's chunks to `tx`.
///
/// Each `step()` call returns `list[RequestOutput]` for *all*
/// in-flight requests; we filter for `request_id`. When the consumer
/// drops the [`RunHandle`], the next `tx.send` fails and we call
/// `abort_request(request_id)` to tell vLLM to stop generating.
async fn drive_generation(
    engine: &Arc<VllmEngine>,
    request_id: &str,
    tx: tokio::sync::mpsc::Sender<InferenceResult<TokenChunk>>,
    chunk_request_id: &str,
) -> InferenceResult<()> {
    let mut last_text_len: usize = 0;
    let mut finished_emitted = false;
    let poll_interval = std::time::Duration::from_millis(5);

    loop {
        if finished_emitted {
            return Ok(());
        }

        // Drive one engine step; collect any chunk for our request.
        let chunk_result = engine.bridge.with_python(
            |py| -> PyResult<Option<(String, bool, Option<String>, u32, u32)>> {
                let engine_handle = engine.engine.lock();
                let bound = engine_handle.bind(py);
                let outputs = bound.call_method0("step")?;
                let outputs_list = outputs.downcast::<PyList>()?;
                for i in 0..outputs_list.len() {
                    let req_output = outputs_list.get_item(i)?;
                    let this_id: String = req_output.getattr("request_id")?.extract()?;
                    if this_id != request_id {
                        continue;
                    }
                    let outs = req_output.getattr("outputs")?;
                    let outs_list = outs.downcast::<PyList>()?;
                    if outs_list.len() == 0 {
                        return Ok(None);
                    }
                    let first = outs_list.get_item(0)?;
                    let text: String = first.getattr("text")?.extract()?;
                    let finish_reason: Option<String> =
                        first.getattr("finish_reason")?.extract().ok().flatten();
                    let finished: bool = req_output.getattr("finished")?.extract()?;
                    let prompt_tokens: u32 = req_output
                        .getattr("prompt_token_ids")
                        .and_then(|v| v.len().map(|l| l as u32))
                        .unwrap_or(0);
                    let output_tokens: u32 = first
                        .getattr("token_ids")
                        .and_then(|v| v.len().map(|l| l as u32))
                        .unwrap_or(0);
                    return Ok(Some((
                        text,
                        finished,
                        finish_reason,
                        prompt_tokens,
                        output_tokens,
                    )));
                }
                Ok(None)
            },
        );

        let chunk = match chunk_result {
            Ok(Some(c)) => c,
            Ok(None) => {
                // No new output for our request this step. If the
                // engine has nothing in flight for any request we'd
                // block forever; check that case so we exit cleanly
                // if vLLM dropped the request without telling us.
                let unfinished = engine.bridge.with_python(|py| -> PyResult<bool> {
                    let engine_handle = engine.engine.lock();
                    let bound = engine_handle.bind(py);
                    let r: bool = bound.call_method0("has_unfinished_requests")?.extract()?;
                    Ok(r)
                })?;
                if !unfinished {
                    return Ok(());
                }
                tokio::time::sleep(poll_interval).await;
                continue;
            }
            Err(e) => return Err(InferenceError::Internal(format!("vllm step: {e}"))),
        };

        let (text_total, finished, finish_reason, prompt_tokens, output_tokens) = chunk;
        let delta = if text_total.len() > last_text_len {
            text_total[last_text_len..].to_string()
        } else {
            String::new()
        };
        last_text_len = text_total.len();

        let usage = if finished {
            Some(TokenUsage {
                input_tokens: prompt_tokens,
                output_tokens,
                ..Default::default()
            })
        } else {
            None
        };
        let fin = if finished {
            Some(map_finish_reason(finish_reason.as_deref()))
        } else {
            None
        };

        let out_chunk = TokenChunk {
            request_id: chunk_request_id.to_string(),
            text_delta: delta,
            tool_call_delta: None,
            usage,
            finish_reason: fin,
        };

        if tx.send(Ok(out_chunk)).await.is_err() {
            // Consumer dropped. Abort the request server-side so the
            // engine doesn't keep generating into a closed channel.
            let _ = engine.bridge.with_python(|py| -> PyResult<()> {
                let engine_handle = engine.engine.lock();
                let bound = engine_handle.bind(py);
                bound.call_method1("abort_request", (request_id,))?;
                Ok(())
            });
            return Ok(());
        }

        if finished {
            finished_emitted = true;
        }
    }
}

/// Extract plain text from a [`Message`] (collapsing
/// `MessageContent::Parts` to text-only). Image / tool parts are
/// dropped — vLLM multimodal goes through a separate code path
/// not yet wired here.
fn message_text(m: &Message) -> String {
    match &m.content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                atomr_infer_core::batch::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Fallback chat formatter used when the tokenizer doesn't expose
/// `apply_chat_template`. The model's actual template is preferred —
/// this only fires on old vLLM or non-chat models.
fn simple_render(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            _ => "user",
        };
        out.push_str(&format!("<|{role}|>\n"));
        out.push_str(&message_text(m));
        out.push('\n');
    }
    out.push_str("<|assistant|>\n");
    out
}

fn map_finish_reason(s: Option<&str>) -> FinishReason {
    match s {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("abort") => FinishReason::Stop,
        _ => FinishReason::Stop,
    }
}
