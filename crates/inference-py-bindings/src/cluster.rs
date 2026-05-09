//! `Cluster` — Python handle that owns a registry of `ModelRunner`s and
//! exposes `deploy`, `execute`, `execute_stream`.
//!
//! Async interop via `pyo3_async_runtimes::tokio::future_into_py`. The
//! registry is sync-locked (cheap ops) but each runner has its own
//! async lock so concurrent executes on different deployments don't
//! contend.
//!
//! Only the testkit `MockRunner` is wired today. Real provider runtimes
//! (OpenAI, Anthropic, Gemini, LiteLLM) need session-actor bootstrap +
//! credential plumbing — landing as the next parity wave; for now they
//! return a clear `BadRequest`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use atomr_infer_core::error::{InferenceError, InferenceResult};
use atomr_infer_core::runner::ModelRunner;
use atomr_infer_core::runtime::{RuntimeConfig, RuntimeKind};
use atomr_infer_core::tokens::Tokens;
use atomr_infer_testkit::{MockRunner, MockScript};
use futures::stream::BoxStream;
use futures_util::StreamExt;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_async_runtimes::tokio::future_into_py;
use serde::Deserialize;
use tokio::sync::Mutex as AsyncMutex;

use crate::core::{PyDeployment, PyExecuteBatch, PyTokenChunk, PyTokens};
use crate::errors;

type RunnerCell = Arc<AsyncMutex<Box<dyn ModelRunner>>>;
type Registry = Arc<StdMutex<HashMap<String, RunnerCell>>>;

#[pyclass(name = "Cluster", module = "atomr_infer._native.cluster")]
pub struct PyCluster {
    endpoint: String,
    runners: Registry,
}

#[derive(Deserialize, Default)]
struct MockKindConfig {
    #[serde(default)]
    chunks: Vec<String>,
    #[serde(default)]
    inter_chunk_delay_ms: u64,
}

fn build_runner(deployment: &PyDeployment) -> InferenceResult<Box<dyn ModelRunner>> {
    let kind = deployment.inner.effective_runtime();
    match &kind {
        RuntimeKind::Custom(s) if s == "mock" => {
            let cfg: MockKindConfig = match deployment.inner.runtime_config.clone() {
                Some(RuntimeConfig::Custom { config, .. }) => serde_json::from_value(config)
                    .map_err(|e| InferenceError::BadRequest {
                        message: format!("invalid mock config: {e}"),
                    })?,
                Some(_) => MockKindConfig::default(),
                None => MockKindConfig::default(),
            };
            let chunks = if cfg.chunks.is_empty() {
                vec!["mock-response".to_string()]
            } else {
                cfg.chunks
            };
            let script = MockScript {
                chunks,
                inter_chunk_delay: Duration::from_millis(cfg.inter_chunk_delay_ms),
                fail_with: None,
            };
            Ok(Box::new(MockRunner::new(script)))
        }
        // Remote provider runtimes need session-actor bootstrap which
        // isn't wired through the Python surface yet (next parity wave).
        RuntimeKind::OpenAi | RuntimeKind::Anthropic | RuntimeKind::Gemini | RuntimeKind::LiteLlm => {
            Err(InferenceError::BadRequest {
                message: format!(
                    "remote runtime {kind:?} not yet wired through Python bindings — \
                     use the Rust API for now (binding will land next parity wave)"
                ),
            })
        }
        // Local-GPU runtimes are explicitly out of scope for the
        // portable Python wheel.
        RuntimeKind::Vllm
        | RuntimeKind::TensorRt
        | RuntimeKind::Ort
        | RuntimeKind::Candle
        | RuntimeKind::Cudarc
        | RuntimeKind::MistralRs => Err(InferenceError::BadRequest {
            message: format!("local-GPU runtime {kind:?} not exposed via Python bindings"),
        }),
        RuntimeKind::Python(_) => Err(InferenceError::BadRequest {
            message: "Python-runtime variant requires the python-bridge crate".to_string(),
        }),
        RuntimeKind::Custom(other) => Err(InferenceError::BadRequest {
            message: format!("unknown custom runtime kind: {other:?}"),
        }),
        _ => Err(InferenceError::BadRequest {
            message: format!("runtime kind {kind:?} is not handled"),
        }),
    }
}

#[pymethods]
impl PyCluster {
    /// Open a handle. The endpoint is currently informational —
    /// in-process registries don't talk to a remote control plane yet.
    #[staticmethod]
    fn connect(endpoint: String) -> Self {
        Self {
            endpoint,
            runners: Arc::new(StdMutex::new(HashMap::new())),
        }
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Register a runner for the deployment. Subsequent
    /// `execute(deployment.name, ...)` calls dispatch to it.
    fn deploy(&self, deployment: &PyDeployment) -> PyResult<()> {
        deployment
            .inner
            .validate()
            .map_err(|e| errors::map_str(format!("validation error: {e}")))?;
        let runner = build_runner(deployment).map_err(errors::map)?;
        let cell: RunnerCell = Arc::new(AsyncMutex::new(runner));
        let mut guard = self
            .runners
            .lock()
            .map_err(|e| errors::map_str(format!("registry poisoned: {e}")))?;
        guard.insert(deployment.inner.name.clone(), cell);
        tracing::info!(name = %deployment.inner.name, "py: deploy ok");
        Ok(())
    }

    /// List registered deployment names.
    fn deployments(&self) -> PyResult<Vec<String>> {
        let guard = self
            .runners
            .lock()
            .map_err(|e| errors::map_str(format!("registry poisoned: {e}")))?;
        Ok(guard.keys().cloned().collect())
    }

    /// Async; resolves to a fully-aggregated `Tokens`. Drains the
    /// runner's chunk stream and accumulates text + usage.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        deployment_name: String,
        batch: &PyExecuteBatch,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cell = self.lookup(&deployment_name)?;
        let batch = batch.inner.clone();
        future_into_py(py, async move {
            let mut runner = cell.lock().await;
            let handle = runner.execute(batch).await.map_err(errors::map)?;
            drop(runner); // release runner lock during drain
            let mut stream = handle.into_stream();
            let mut tokens = Tokens::default();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(errors::map)?;
                if tokens.request_id.is_empty() {
                    tokens.request_id = chunk.request_id.clone();
                }
                tokens.append(&chunk);
            }
            Ok(PyTokens::from(tokens))
        })
    }

    /// Async iterator yielding `TokenChunk` per `__anext__`.
    fn execute_stream(&self, deployment_name: String, batch: &PyExecuteBatch) -> PyResult<PyTokenStream> {
        let cell = self.lookup(&deployment_name)?;
        let batch = batch.inner.clone();
        Ok(PyTokenStream {
            state: Arc::new(AsyncMutex::new(StreamState::Pending {
                cell: Some(cell),
                batch: Some(batch),
            })),
        })
    }

    fn __repr__(&self) -> String {
        format!("Cluster(endpoint={:?})", self.endpoint)
    }
}

impl PyCluster {
    fn lookup(&self, deployment_name: &str) -> PyResult<RunnerCell> {
        let guard = self
            .runners
            .lock()
            .map_err(|e| errors::map_str(format!("registry poisoned: {e}")))?;
        guard
            .get(deployment_name)
            .cloned()
            .ok_or_else(|| errors::map_str(format!("no deployment {deployment_name:?}")))
    }
}

// ---------------------------------------------------------------------------
// PyTokenStream — async iterator over TokenChunk
// ---------------------------------------------------------------------------

enum StreamState {
    /// Haven't called `runner.execute()` yet — first `__anext__` will.
    Pending {
        cell: Option<RunnerCell>,
        batch: Option<atomr_infer_core::batch::ExecuteBatch>,
    },
    Active(BoxStream<'static, InferenceResult<atomr_infer_core::tokens::TokenChunk>>),
    Done,
}

#[pyclass(name = "TokenStream", module = "atomr_infer._native.cluster")]
pub struct PyTokenStream {
    state: Arc<AsyncMutex<StreamState>>,
}

#[pymethods]
impl PyTokenStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = slf.state.clone();
        future_into_py(py, async move {
            // Lazily kick off `runner.execute()` on first call so any
            // dispatch error surfaces as a normal Python exception
            // instead of breaking the async-iterator protocol.
            {
                let mut guard = state.lock().await;
                if let StreamState::Pending { cell, batch } = &mut *guard {
                    let cell = cell.take().expect("pending state has cell");
                    let batch = batch.take().expect("pending state has batch");
                    let mut runner = cell.lock().await;
                    let handle = runner.execute(batch).await.map_err(errors::map)?;
                    drop(runner);
                    *guard = StreamState::Active(handle.into_stream());
                }
            }
            // Now pull the next chunk.
            let mut guard = state.lock().await;
            let stream = match &mut *guard {
                StreamState::Active(s) => s,
                StreamState::Done => return Err(PyStopAsyncIteration::new_err("")),
                StreamState::Pending { .. } => unreachable!("just transitioned out"),
            };
            match stream.next().await {
                Some(Ok(chunk)) => Ok(PyTokenChunk::from(chunk)),
                Some(Err(e)) => {
                    *guard = StreamState::Done;
                    Err(errors::map(e))
                }
                None => {
                    *guard = StreamState::Done;
                    Err(PyStopAsyncIteration::new_err(""))
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Submodule registration
// ---------------------------------------------------------------------------

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new_bound(py, "cluster")?;
    sub.add_class::<PyCluster>()?;
    sub.add_class::<PyTokenStream>()?;
    m.add_submodule(&sub)?;
    Ok(())
}
