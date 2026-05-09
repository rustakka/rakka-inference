//! Runtime / transport / provider taxonomy + per-runtime config wrappers.
//!
//! `RuntimeKind` and `RuntimeConfig` carry payload variants, so we expose
//! them as classes with static-method constructors per variant (matches
//! the Rust idiom). Pure-payload-free enums (`JitterKind`, `ProviderKind`'s
//! non-custom variants) use the simpler `Class("name")` pattern.

use std::time::Duration;

use atomr_infer_core::runtime::{
    CircuitBreakerConfig, JitterKind, ProviderKind, RuntimeConfig, RuntimeKind, TransportKind,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

// ---------------------------------------------------------------------------
// RuntimeKind
// ---------------------------------------------------------------------------

#[pyclass(name = "RuntimeKind", module = "atomr_infer._native.runtime", eq)]
#[derive(Clone, PartialEq, Eq)]
pub struct PyRuntimeKind {
    pub(crate) inner: RuntimeKind,
}

#[pymethods]
impl PyRuntimeKind {
    #[staticmethod]
    fn vllm() -> Self {
        Self { inner: RuntimeKind::Vllm }
    }
    #[staticmethod]
    fn tensorrt() -> Self {
        Self { inner: RuntimeKind::TensorRt }
    }
    #[staticmethod]
    fn ort() -> Self {
        Self { inner: RuntimeKind::Ort }
    }
    #[staticmethod]
    fn candle() -> Self {
        Self { inner: RuntimeKind::Candle }
    }
    #[staticmethod]
    fn cudarc() -> Self {
        Self { inner: RuntimeKind::Cudarc }
    }
    #[staticmethod]
    fn mistralrs() -> Self {
        Self { inner: RuntimeKind::MistralRs }
    }
    #[staticmethod]
    fn openai() -> Self {
        Self { inner: RuntimeKind::OpenAi }
    }
    #[staticmethod]
    fn anthropic() -> Self {
        Self { inner: RuntimeKind::Anthropic }
    }
    #[staticmethod]
    fn gemini() -> Self {
        Self { inner: RuntimeKind::Gemini }
    }
    #[staticmethod]
    fn litellm() -> Self {
        Self { inner: RuntimeKind::LiteLlm }
    }
    #[staticmethod]
    fn python(name: String) -> Self {
        Self { inner: RuntimeKind::Python(name) }
    }
    #[staticmethod]
    fn custom(name: String) -> Self {
        Self { inner: RuntimeKind::Custom(name) }
    }

    /// Snake-case discriminator string (`"vllm"`, `"openai"`, …). For
    /// `Custom(s)` this is `"custom"`; use `tag` to recover `s`.
    #[getter]
    fn name(&self) -> &'static str {
        match &self.inner {
            RuntimeKind::Vllm => "vllm",
            RuntimeKind::TensorRt => "tensorrt",
            RuntimeKind::Ort => "ort",
            RuntimeKind::Candle => "candle",
            RuntimeKind::Cudarc => "cudarc",
            RuntimeKind::MistralRs => "mistralrs",
            RuntimeKind::Python(_) => "python",
            RuntimeKind::OpenAi => "openai",
            RuntimeKind::Anthropic => "anthropic",
            RuntimeKind::Gemini => "gemini",
            RuntimeKind::LiteLlm => "litellm",
            RuntimeKind::Custom(_) => "custom",
            _ => "unknown",
        }
    }

    /// Inner string for `Python(s)` and `Custom(s)`, else `None`.
    #[getter]
    fn tag(&self) -> Option<String> {
        match &self.inner {
            RuntimeKind::Python(s) | RuntimeKind::Custom(s) => Some(s.clone()),
            _ => None,
        }
    }

    fn is_remote(&self) -> bool {
        self.inner.is_remote()
    }

    fn is_local(&self) -> bool {
        self.inner.is_local()
    }

    fn __repr__(&self) -> String {
        match self.tag() {
            Some(s) => format!("RuntimeKind.{}({:?})", self.name(), s),
            None => format!("RuntimeKind.{}", self.name()),
        }
    }
}

// ---------------------------------------------------------------------------
// ProviderKind
// ---------------------------------------------------------------------------

#[pyclass(name = "ProviderKind", module = "atomr_infer._native.runtime", eq)]
#[derive(Clone, PartialEq, Eq)]
pub struct PyProviderKind {
    pub(crate) inner: ProviderKind,
}

#[pymethods]
impl PyProviderKind {
    #[staticmethod]
    fn openai() -> Self {
        Self { inner: ProviderKind::OpenAi }
    }
    #[staticmethod]
    fn anthropic() -> Self {
        Self { inner: ProviderKind::Anthropic }
    }
    #[staticmethod]
    fn gemini() -> Self {
        Self { inner: ProviderKind::Gemini }
    }
    #[staticmethod]
    fn litellm() -> Self {
        Self { inner: ProviderKind::LiteLlm }
    }
    #[staticmethod]
    fn custom(name: String) -> Self {
        Self { inner: ProviderKind::Custom(name) }
    }

    #[getter]
    fn name(&self) -> &'static str {
        match &self.inner {
            ProviderKind::OpenAi => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Gemini => "gemini",
            ProviderKind::LiteLlm => "litellm",
            ProviderKind::Custom(_) => "custom",
            _ => "unknown",
        }
    }

    #[getter]
    fn tag(&self) -> Option<String> {
        match &self.inner {
            ProviderKind::Custom(s) => Some(s.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.tag() {
            Some(s) => format!("ProviderKind.custom({:?})", s),
            None => format!("ProviderKind.{}", self.name()),
        }
    }
}

// ---------------------------------------------------------------------------
// TransportKind
// ---------------------------------------------------------------------------

#[pyclass(name = "TransportKind", module = "atomr_infer._native.runtime")]
#[derive(Clone)]
pub struct PyTransportKind {
    pub(crate) inner: TransportKind,
}

#[pymethods]
impl PyTransportKind {
    #[staticmethod]
    fn local_gpu() -> Self {
        Self { inner: TransportKind::LocalGpu }
    }
    #[staticmethod]
    fn remote_network(provider: PyProviderKind) -> Self {
        Self {
            inner: TransportKind::RemoteNetwork {
                provider: provider.inner,
            },
        }
    }

    #[getter]
    fn name(&self) -> &'static str {
        match &self.inner {
            TransportKind::LocalGpu => "local_gpu",
            TransportKind::RemoteNetwork { .. } => "remote_network",
            _ => "unknown",
        }
    }

    #[getter]
    fn provider(&self) -> Option<PyProviderKind> {
        match &self.inner {
            TransportKind::RemoteNetwork { provider } => Some(PyProviderKind {
                inner: provider.clone(),
            }),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("TransportKind.{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// RuntimeConfig
// ---------------------------------------------------------------------------

#[pyclass(name = "RuntimeConfig", module = "atomr_infer._native.runtime")]
#[derive(Clone)]
pub struct PyRuntimeConfig {
    pub(crate) inner: RuntimeConfig,
}

fn extract_json(any: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    // Convert via the JSON module — pyo3 doesn't ship a built-in
    // PyAny → serde_json::Value path. Acceptable cost for config blobs
    // that are typically built once at deploy time.
    let py = any.py();
    let json_mod = py.import_bound("json")?;
    let dumped: String = json_mod.call_method1("dumps", (any,))?.extract()?;
    serde_json::from_str(&dumped).map_err(|e| PyValueError::new_err(format!("not JSON-serialisable: {e}")))
}

#[pymethods]
impl PyRuntimeConfig {
    #[staticmethod]
    fn vllm(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::Vllm(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn tensorrt(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::TensorRt(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn ort(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::Ort(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn candle(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::Candle(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn cudarc(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::Cudarc(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn mistralrs(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::MistralRs(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn openai(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::OpenAi(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn anthropic(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::Anthropic(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn gemini(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::Gemini(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn litellm(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::LiteLlm(extract_json(config)?),
        })
    }
    #[staticmethod]
    fn custom(kind: String, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: RuntimeConfig::Custom {
                kind,
                config: extract_json(config)?,
            },
        })
    }

    fn runtime_kind(&self) -> PyRuntimeKind {
        PyRuntimeKind {
            inner: self.inner.runtime_kind(),
        }
    }

    fn transport_kind(&self) -> PyTransportKind {
        PyTransportKind {
            inner: self.inner.transport_kind(),
        }
    }

    fn __repr__(&self) -> String {
        format!("RuntimeConfig.{}", self.runtime_kind().name())
    }
}

// ---------------------------------------------------------------------------
// CircuitBreakerConfig
// ---------------------------------------------------------------------------

#[pyclass(name = "CircuitBreakerConfig", module = "atomr_infer._native.runtime")]
#[derive(Clone)]
pub struct PyCircuitBreakerConfig {
    pub(crate) inner: CircuitBreakerConfig,
}

#[pymethods]
impl PyCircuitBreakerConfig {
    #[new]
    #[pyo3(signature = (failure_threshold=10, open_duration_ms=30_000, half_open_max_probes=1))]
    fn new(failure_threshold: u32, open_duration_ms: u64, half_open_max_probes: u32) -> Self {
        Self {
            inner: CircuitBreakerConfig {
                failure_threshold,
                open_duration: Duration::from_millis(open_duration_ms),
                half_open_max_probes,
            },
        }
    }

    #[getter]
    fn failure_threshold(&self) -> u32 {
        self.inner.failure_threshold
    }
    #[getter]
    fn open_duration_ms(&self) -> u64 {
        self.inner.open_duration.as_millis() as u64
    }
    #[getter]
    fn half_open_max_probes(&self) -> u32 {
        self.inner.half_open_max_probes
    }

    fn __repr__(&self) -> String {
        format!(
            "CircuitBreakerConfig(failure_threshold={}, open_duration_ms={})",
            self.inner.failure_threshold,
            self.inner.open_duration.as_millis()
        )
    }
}

// ---------------------------------------------------------------------------
// JitterKind
// ---------------------------------------------------------------------------

#[pyclass(name = "JitterKind", module = "atomr_infer._native.runtime", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyJitterKind {
    pub(crate) inner: JitterKind,
}

#[pymethods]
impl PyJitterKind {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "none" => JitterKind::None,
            "equal" => JitterKind::Equal,
            "full" => JitterKind::Full,
            other => return Err(PyValueError::new_err(format!("unknown jitter kind: {other:?}"))),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            JitterKind::None => "none",
            JitterKind::Equal => "equal",
            JitterKind::Full => "full",
            _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!("JitterKind({:?})", self.name())
    }
}

// ---------------------------------------------------------------------------
// Submodule registration
// ---------------------------------------------------------------------------

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new_bound(py, "runtime")?;
    sub.add_class::<PyRuntimeKind>()?;
    sub.add_class::<PyProviderKind>()?;
    sub.add_class::<PyTransportKind>()?;
    sub.add_class::<PyRuntimeConfig>()?;
    sub.add_class::<PyCircuitBreakerConfig>()?;
    sub.add_class::<PyJitterKind>()?;
    m.add_submodule(&sub)?;
    Ok(())
}
