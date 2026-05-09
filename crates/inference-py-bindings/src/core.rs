//! Core data-type wrappers — `Deployment`, `ExecuteBatch`, message
//! shape, sampling, token streaming primitives. Layout mirrors upstream
//! `atomr/crates/py-bindings/pycore/src/ext_core_extras.rs` (string-tag
//! enums + classes-as-newtypes-around-Rust-values).

use atomr_infer_core::batch::{ContentPart, ExecuteBatch, Message, MessageContent, Role, SamplingParams};
use atomr_infer_core::cost::CostEstimate;
use atomr_infer_core::deployment::{Deployment, Replica};
use atomr_infer_core::tokens::{FinishReason, TokenChunk, TokenUsage, Tokens};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

#[pyclass(name = "Role", module = "atomr_infer._native.core", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyRole {
    pub(crate) inner: Role,
}

#[pymethods]
impl PyRole {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => return Err(PyValueError::new_err(format!("unknown role: {other:?}"))),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!("Role({:?})", self.name())
    }
}

// ---------------------------------------------------------------------------
// ContentPart
// ---------------------------------------------------------------------------

#[pyclass(name = "ContentPart", module = "atomr_infer._native.core")]
#[derive(Clone)]
pub struct PyContentPart {
    pub(crate) inner: ContentPart,
}

#[pymethods]
impl PyContentPart {
    #[staticmethod]
    fn text(text: String) -> Self {
        Self {
            inner: ContentPart::Text { text },
        }
    }

    #[staticmethod]
    fn image_base64(mime: String, data: String) -> Self {
        Self {
            inner: ContentPart::ImageBase64 { mime, data },
        }
    }

    #[staticmethod]
    fn image_url(url: String) -> Self {
        Self {
            inner: ContentPart::ImageUrl { url },
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            ContentPart::Text { .. } => "text",
            ContentPart::ImageBase64 { .. } => "image_base64",
            ContentPart::ImageUrl { .. } => "image_url",
            _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!("ContentPart({})", self.kind())
    }
}

// ---------------------------------------------------------------------------
// MessageContent
// ---------------------------------------------------------------------------

#[pyclass(name = "MessageContent", module = "atomr_infer._native.core")]
#[derive(Clone)]
pub struct PyMessageContent {
    pub(crate) inner: MessageContent,
}

#[pymethods]
impl PyMessageContent {
    #[staticmethod]
    fn text(text: String) -> Self {
        Self {
            inner: MessageContent::Text(text),
        }
    }

    #[staticmethod]
    fn parts(parts: Vec<PyContentPart>) -> Self {
        Self {
            inner: MessageContent::Parts(parts.into_iter().map(|p| p.inner).collect()),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            MessageContent::Text(_) => "text",
            MessageContent::Parts(_) => "parts",
            _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!("MessageContent({})", self.kind())
    }
}

/// Coerce a Python value into `MessageContent`: accepts `str`, an existing
/// `MessageContent`, or a list of `ContentPart`.
fn coerce_content(any: &Bound<'_, PyAny>) -> PyResult<MessageContent> {
    if let Ok(s) = any.extract::<String>() {
        return Ok(MessageContent::Text(s));
    }
    if let Ok(mc) = any.extract::<PyMessageContent>() {
        return Ok(mc.inner);
    }
    if let Ok(parts) = any.extract::<Vec<PyContentPart>>() {
        return Ok(MessageContent::Parts(parts.into_iter().map(|p| p.inner).collect()));
    }
    Err(PyValueError::new_err(
        "content must be str, MessageContent, or list[ContentPart]",
    ))
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

#[pyclass(name = "Message", module = "atomr_infer._native.core")]
#[derive(Clone)]
pub struct PyMessage {
    pub(crate) inner: Message,
}

#[pymethods]
impl PyMessage {
    #[new]
    fn new(role: &PyRole, content: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: Message {
                role: role.inner,
                content: coerce_content(content)?,
            },
        })
    }

    #[getter]
    fn role(&self) -> PyRole {
        PyRole { inner: self.inner.role }
    }

    #[getter]
    fn content(&self) -> PyMessageContent {
        PyMessageContent {
            inner: self.inner.content.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Message(role={:?})", PyRole { inner: self.inner.role }.name())
    }
}

// ---------------------------------------------------------------------------
// SamplingParams
// ---------------------------------------------------------------------------

#[pyclass(name = "SamplingParams", module = "atomr_infer._native.core")]
#[derive(Clone, Default)]
pub struct PySamplingParams {
    pub(crate) inner: SamplingParams,
}

#[pymethods]
impl PySamplingParams {
    #[new]
    #[pyo3(signature = (
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        presence_penalty=None,
        frequency_penalty=None,
        seed=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        max_tokens: Option<u32>,
        stop: Option<Vec<String>>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            inner: SamplingParams {
                temperature,
                top_p,
                top_k,
                max_tokens,
                stop: stop.unwrap_or_default(),
                presence_penalty,
                frequency_penalty,
                seed,
            },
        }
    }

    #[getter]
    fn temperature(&self) -> Option<f32> {
        self.inner.temperature
    }
    #[getter]
    fn top_p(&self) -> Option<f32> {
        self.inner.top_p
    }
    #[getter]
    fn top_k(&self) -> Option<u32> {
        self.inner.top_k
    }
    #[getter]
    fn max_tokens(&self) -> Option<u32> {
        self.inner.max_tokens
    }
    #[getter]
    fn stop(&self) -> Vec<String> {
        self.inner.stop.clone()
    }
    #[getter]
    fn presence_penalty(&self) -> Option<f32> {
        self.inner.presence_penalty
    }
    #[getter]
    fn frequency_penalty(&self) -> Option<f32> {
        self.inner.frequency_penalty
    }
    #[getter]
    fn seed(&self) -> Option<u64> {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingParams(temperature={:?}, max_tokens={:?})",
            self.inner.temperature, self.inner.max_tokens
        )
    }
}

// ---------------------------------------------------------------------------
// ExecuteBatch
// ---------------------------------------------------------------------------

#[pyclass(name = "ExecuteBatch", module = "atomr_infer._native.core")]
#[derive(Clone)]
pub struct PyExecuteBatch {
    pub(crate) inner: ExecuteBatch,
}

#[pymethods]
impl PyExecuteBatch {
    #[new]
    #[pyo3(signature = (
        request_id,
        model,
        messages,
        sampling=None,
        stream=false,
        estimated_tokens=1,
    ))]
    fn new(
        request_id: String,
        model: String,
        messages: Vec<PyMessage>,
        sampling: Option<PySamplingParams>,
        stream: bool,
        estimated_tokens: u32,
    ) -> Self {
        Self {
            inner: ExecuteBatch {
                request_id,
                model,
                messages: messages.into_iter().map(|m| m.inner).collect(),
                sampling: sampling.map(|s| s.inner).unwrap_or_default(),
                stream,
                estimated_tokens,
            },
        }
    }

    #[getter]
    fn request_id(&self) -> &str {
        &self.inner.request_id
    }
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }
    #[getter]
    fn messages(&self) -> Vec<PyMessage> {
        self.inner
            .messages
            .iter()
            .map(|m| PyMessage { inner: m.clone() })
            .collect()
    }
    #[getter]
    fn sampling(&self) -> PySamplingParams {
        PySamplingParams {
            inner: self.inner.sampling.clone(),
        }
    }
    #[getter]
    fn stream(&self) -> bool {
        self.inner.stream
    }
    #[getter]
    fn estimated_tokens(&self) -> u32 {
        self.inner.estimated_tokens
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecuteBatch(request_id={:?}, model={:?}, messages={})",
            self.inner.request_id,
            self.inner.model,
            self.inner.messages.len()
        )
    }
}

// ---------------------------------------------------------------------------
// FinishReason
// ---------------------------------------------------------------------------

#[pyclass(name = "FinishReason", module = "atomr_infer._native.core", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyFinishReason {
    pub(crate) inner: FinishReason,
}

#[pymethods]
impl PyFinishReason {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            "content_filter" => FinishReason::ContentFilter,
            "error" => FinishReason::Error,
            other => return Err(PyValueError::new_err(format!("unknown finish reason: {other:?}"))),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::Error => "error",
            _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!("FinishReason({:?})", self.name())
    }
}

// ---------------------------------------------------------------------------
// TokenUsage
// ---------------------------------------------------------------------------

#[pyclass(name = "TokenUsage", module = "atomr_infer._native.core")]
#[derive(Clone, Copy, Default)]
pub struct PyTokenUsage {
    pub(crate) inner: TokenUsage,
}

#[pymethods]
impl PyTokenUsage {
    #[new]
    #[pyo3(signature = (input_tokens=0, output_tokens=0, reasoning_tokens=0, cached_tokens=0))]
    fn new(input_tokens: u32, output_tokens: u32, reasoning_tokens: u32, cached_tokens: u32) -> Self {
        Self {
            inner: TokenUsage {
                input_tokens,
                output_tokens,
                reasoning_tokens,
                cached_tokens,
            },
        }
    }

    #[getter]
    fn input_tokens(&self) -> u32 {
        self.inner.input_tokens
    }
    #[getter]
    fn output_tokens(&self) -> u32 {
        self.inner.output_tokens
    }
    #[getter]
    fn reasoning_tokens(&self) -> u32 {
        self.inner.reasoning_tokens
    }
    #[getter]
    fn cached_tokens(&self) -> u32 {
        self.inner.cached_tokens
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenUsage(input={}, output={})",
            self.inner.input_tokens, self.inner.output_tokens
        )
    }
}

// ---------------------------------------------------------------------------
// TokenChunk
// ---------------------------------------------------------------------------

#[pyclass(name = "TokenChunk", module = "atomr_infer._native.core")]
#[derive(Clone)]
pub struct PyTokenChunk {
    pub(crate) inner: TokenChunk,
}

#[pymethods]
impl PyTokenChunk {
    #[getter]
    fn request_id(&self) -> &str {
        &self.inner.request_id
    }
    #[getter]
    fn text_delta(&self) -> &str {
        &self.inner.text_delta
    }
    #[getter]
    fn usage(&self) -> Option<PyTokenUsage> {
        self.inner.usage.map(|u| PyTokenUsage { inner: u })
    }
    #[getter]
    fn finish_reason(&self) -> Option<PyFinishReason> {
        self.inner.finish_reason.map(|r| PyFinishReason { inner: r })
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenChunk(request_id={:?}, text_delta={:?})",
            self.inner.request_id, self.inner.text_delta
        )
    }
}

impl From<TokenChunk> for PyTokenChunk {
    fn from(inner: TokenChunk) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// Tokens (final aggregate)
// ---------------------------------------------------------------------------

#[pyclass(name = "Tokens", module = "atomr_infer._native.core")]
#[derive(Clone, Default)]
pub struct PyTokens {
    pub(crate) inner: Tokens,
}

#[pymethods]
impl PyTokens {
    #[getter]
    fn request_id(&self) -> &str {
        &self.inner.request_id
    }
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }
    #[getter]
    fn usage(&self) -> PyTokenUsage {
        PyTokenUsage {
            inner: self.inner.usage,
        }
    }
    #[getter]
    fn finish_reason(&self) -> Option<PyFinishReason> {
        self.inner.finish_reason.map(|r| PyFinishReason { inner: r })
    }

    fn __repr__(&self) -> String {
        format!(
            "Tokens(request_id={:?}, text={:?}, usage={})",
            self.inner.request_id,
            self.inner.text,
            PyTokenUsage {
                inner: self.inner.usage
            }
            .__repr__()
        )
    }
}

impl From<Tokens> for PyTokens {
    fn from(inner: Tokens) -> Self {
        Self { inner }
    }
}

// ---------------------------------------------------------------------------
// CostEstimate
// ---------------------------------------------------------------------------

#[pyclass(name = "CostEstimate", module = "atomr_infer._native.core")]
#[derive(Clone, Copy, Default)]
pub struct PyCostEstimate {
    pub(crate) inner: CostEstimate,
}

#[pymethods]
impl PyCostEstimate {
    #[new]
    #[pyo3(signature = (usd=0.0, input_tokens=0, output_tokens_max=0))]
    fn new(usd: f64, input_tokens: u32, output_tokens_max: u32) -> Self {
        Self {
            inner: CostEstimate {
                usd,
                input_tokens,
                output_tokens_max,
            },
        }
    }

    #[getter]
    fn usd(&self) -> f64 {
        self.inner.usd
    }
    #[getter]
    fn input_tokens(&self) -> u32 {
        self.inner.input_tokens
    }
    #[getter]
    fn output_tokens_max(&self) -> u32 {
        self.inner.output_tokens_max
    }

    fn __repr__(&self) -> String {
        format!("CostEstimate(usd={:.6})", self.inner.usd)
    }
}

// ---------------------------------------------------------------------------
// Replica
// ---------------------------------------------------------------------------

#[pyclass(name = "Replica", module = "atomr_infer._native.core")]
#[derive(Clone)]
pub struct PyReplica {
    pub(crate) inner: Replica,
}

#[pymethods]
impl PyReplica {
    #[new]
    #[pyo3(signature = (deployment, replica_index=0, node=None, gpu_indices=None))]
    fn new(deployment: String, replica_index: u32, node: Option<String>, gpu_indices: Option<Vec<u32>>) -> Self {
        Self {
            inner: Replica {
                deployment,
                replica_index,
                node,
                gpu_indices: gpu_indices.unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn deployment(&self) -> &str {
        &self.inner.deployment
    }
    #[getter]
    fn replica_index(&self) -> u32 {
        self.inner.replica_index
    }
    #[getter]
    fn node(&self) -> Option<&str> {
        self.inner.node.as_deref()
    }
    #[getter]
    fn gpu_indices(&self) -> Vec<u32> {
        self.inner.gpu_indices.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Replica(deployment={:?}, replica_index={})",
            self.inner.deployment, self.inner.replica_index
        )
    }
}

// ---------------------------------------------------------------------------
// Deployment
// ---------------------------------------------------------------------------

#[pyclass(name = "Deployment", module = "atomr_infer._native.core")]
#[derive(Clone)]
pub struct PyDeployment {
    pub(crate) inner: Deployment,
}

#[pymethods]
impl PyDeployment {
    #[new]
    #[pyo3(signature = (
        name,
        model,
        replicas=1,
        gpus=None,
        runtime=None,
        runtime_config=None,
        serving=None,
        budget=None,
        idempotent=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        model: String,
        replicas: u32,
        gpus: Option<u32>,
        runtime: Option<crate::runtime::PyRuntimeKind>,
        runtime_config: Option<crate::runtime::PyRuntimeConfig>,
        serving: Option<crate::config::PyServing>,
        budget: Option<crate::config::PyBudget>,
        idempotent: bool,
    ) -> Self {
        Self {
            inner: Deployment {
                name,
                model,
                runtime: runtime.map(|r| r.inner),
                runtime_config: runtime_config.map(|c| c.inner),
                gpus,
                replicas,
                serving: serving.map(|s| s.inner).unwrap_or_default(),
                budget: budget.map(|b| b.inner),
                idempotent,
            },
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.name = name;
    }

    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }
    #[setter]
    fn set_model(&mut self, model: String) {
        self.inner.model = model;
    }

    #[getter]
    fn replicas(&self) -> u32 {
        self.inner.replicas
    }
    #[setter]
    fn set_replicas(&mut self, replicas: u32) {
        self.inner.replicas = replicas;
    }

    #[getter]
    fn gpus(&self) -> Option<u32> {
        self.inner.gpus
    }
    #[setter]
    fn set_gpus(&mut self, gpus: Option<u32>) {
        self.inner.gpus = gpus;
    }

    #[getter]
    fn runtime(&self) -> Option<crate::runtime::PyRuntimeKind> {
        self.inner.runtime.clone().map(|r| crate::runtime::PyRuntimeKind { inner: r })
    }
    #[setter]
    fn set_runtime(&mut self, runtime: Option<crate::runtime::PyRuntimeKind>) {
        self.inner.runtime = runtime.map(|r| r.inner);
    }

    #[getter]
    fn runtime_config(&self) -> Option<crate::runtime::PyRuntimeConfig> {
        self.inner
            .runtime_config
            .clone()
            .map(|c| crate::runtime::PyRuntimeConfig { inner: c })
    }
    #[setter]
    fn set_runtime_config(&mut self, runtime_config: Option<crate::runtime::PyRuntimeConfig>) {
        self.inner.runtime_config = runtime_config.map(|c| c.inner);
    }

    #[getter]
    fn serving(&self) -> crate::config::PyServing {
        crate::config::PyServing {
            inner: self.inner.serving.clone(),
        }
    }
    #[setter]
    fn set_serving(&mut self, serving: crate::config::PyServing) {
        self.inner.serving = serving.inner;
    }

    #[getter]
    fn budget(&self) -> Option<crate::config::PyBudget> {
        self.inner.budget.clone().map(|b| crate::config::PyBudget { inner: b })
    }
    #[setter]
    fn set_budget(&mut self, budget: Option<crate::config::PyBudget>) {
        self.inner.budget = budget.map(|b| b.inner);
    }

    #[getter]
    fn idempotent(&self) -> bool {
        self.inner.idempotent
    }
    #[setter]
    fn set_idempotent(&mut self, idempotent: bool) {
        self.inner.idempotent = idempotent;
    }

    /// Run the same structural validation that `inference-core` applies
    /// before deploy. Raises `BadRequest` on failure.
    fn validate(&self) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|e| crate::errors::map_str(format!("validation error: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "Deployment(name={:?}, model={:?}, replicas={})",
            self.inner.name, self.inner.model, self.inner.replicas
        )
    }
}

// ---------------------------------------------------------------------------
// Submodule registration
// ---------------------------------------------------------------------------

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new_bound(py, "core")?;
    sub.add_class::<PyRole>()?;
    sub.add_class::<PyContentPart>()?;
    sub.add_class::<PyMessageContent>()?;
    sub.add_class::<PyMessage>()?;
    sub.add_class::<PySamplingParams>()?;
    sub.add_class::<PyExecuteBatch>()?;
    sub.add_class::<PyFinishReason>()?;
    sub.add_class::<PyTokenUsage>()?;
    sub.add_class::<PyTokenChunk>()?;
    sub.add_class::<PyTokens>()?;
    sub.add_class::<PyCostEstimate>()?;
    sub.add_class::<PyReplica>()?;
    sub.add_class::<PyDeployment>()?;
    m.add_submodule(&sub)?;
    Ok(())
}
