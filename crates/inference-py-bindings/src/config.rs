//! Per-deployment config wrappers — `Serving`, `RateLimits`,
//! `RetryPolicy`, `Timeouts`, `Budget`, plus the simple action enums.

use std::time::Duration;

use atomr_infer_core::deployment::{Budget, BudgetAction, CapacityPolicy, RateLimits, RetryPolicy, Serving, Timeouts};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::runtime::PyJitterKind;

// ---------------------------------------------------------------------------
// CapacityPolicy
// ---------------------------------------------------------------------------

#[pyclass(name = "CapacityPolicy", module = "atomr_infer._native.config", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyCapacityPolicy {
    pub(crate) inner: CapacityPolicy,
}

#[pymethods]
impl PyCapacityPolicy {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "reject" => CapacityPolicy::Reject,
            "queue" => CapacityPolicy::Queue,
            "fallback" => CapacityPolicy::Fallback,
            other => return Err(PyValueError::new_err(format!("unknown capacity policy: {other:?}"))),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            CapacityPolicy::Reject => "reject",
            CapacityPolicy::Queue => "queue",
            CapacityPolicy::Fallback => "fallback",
        }
    }

    fn __repr__(&self) -> String {
        format!("CapacityPolicy({:?})", self.name())
    }
}

// ---------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------

#[pyclass(name = "Serving", module = "atomr_infer._native.config")]
#[derive(Clone)]
pub struct PyServing {
    pub(crate) inner: Serving,
}

#[pymethods]
impl PyServing {
    #[new]
    #[pyo3(signature = (max_concurrent=32, on_capacity_exhausted=None))]
    fn new(max_concurrent: u32, on_capacity_exhausted: Option<PyCapacityPolicy>) -> Self {
        Self {
            inner: Serving {
                max_concurrent,
                on_capacity_exhausted: on_capacity_exhausted
                    .map(|p| p.inner)
                    .unwrap_or(CapacityPolicy::Queue),
            },
        }
    }

    #[getter]
    fn max_concurrent(&self) -> u32 {
        self.inner.max_concurrent
    }
    #[getter]
    fn on_capacity_exhausted(&self) -> PyCapacityPolicy {
        PyCapacityPolicy {
            inner: self.inner.on_capacity_exhausted,
        }
    }

    fn __repr__(&self) -> String {
        format!("Serving(max_concurrent={})", self.inner.max_concurrent)
    }
}

// ---------------------------------------------------------------------------
// RateLimits
// ---------------------------------------------------------------------------

#[pyclass(name = "RateLimits", module = "atomr_infer._native.config")]
#[derive(Clone, Default)]
pub struct PyRateLimits {
    pub(crate) inner: RateLimits,
}

#[pymethods]
impl PyRateLimits {
    #[new]
    #[pyo3(signature = (requests_per_minute=None, tokens_per_minute=None, concurrent_requests=None, strict=false))]
    fn new(
        requests_per_minute: Option<u64>,
        tokens_per_minute: Option<u64>,
        concurrent_requests: Option<u32>,
        strict: bool,
    ) -> Self {
        Self {
            inner: RateLimits {
                requests_per_minute,
                tokens_per_minute,
                concurrent_requests,
                strict,
            },
        }
    }

    #[getter]
    fn requests_per_minute(&self) -> Option<u64> {
        self.inner.requests_per_minute
    }
    #[getter]
    fn tokens_per_minute(&self) -> Option<u64> {
        self.inner.tokens_per_minute
    }
    #[getter]
    fn concurrent_requests(&self) -> Option<u32> {
        self.inner.concurrent_requests
    }
    #[getter]
    fn strict(&self) -> bool {
        self.inner.strict
    }

    fn __repr__(&self) -> String {
        format!(
            "RateLimits(rpm={:?}, tpm={:?}, strict={})",
            self.inner.requests_per_minute, self.inner.tokens_per_minute, self.inner.strict
        )
    }
}

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

#[pyclass(name = "RetryPolicy", module = "atomr_infer._native.config")]
#[derive(Clone)]
pub struct PyRetryPolicy {
    pub(crate) inner: RetryPolicy,
}

#[pymethods]
impl PyRetryPolicy {
    #[new]
    #[pyo3(signature = (
        max_retries=3,
        initial_backoff_ms=1_000,
        max_backoff_ms=60_000,
        backoff_multiplier=2.0,
        jitter=None,
        respect_retry_after=true,
    ))]
    fn new(
        max_retries: u32,
        initial_backoff_ms: u64,
        max_backoff_ms: u64,
        backoff_multiplier: f64,
        jitter: Option<PyJitterKind>,
        respect_retry_after: bool,
    ) -> PyResult<Self> {
        let jitter_inner = match jitter {
            Some(j) => j.inner,
            None => atomr_infer_core::runtime::JitterKind::Equal,
        };
        Ok(Self {
            inner: RetryPolicy {
                max_retries,
                initial_backoff: Duration::from_millis(initial_backoff_ms),
                max_backoff: Duration::from_millis(max_backoff_ms),
                backoff_multiplier,
                jitter: jitter_inner,
                respect_retry_after,
            },
        })
    }

    #[getter]
    fn max_retries(&self) -> u32 {
        self.inner.max_retries
    }
    #[getter]
    fn initial_backoff_ms(&self) -> u64 {
        self.inner.initial_backoff.as_millis() as u64
    }
    #[getter]
    fn max_backoff_ms(&self) -> u64 {
        self.inner.max_backoff.as_millis() as u64
    }
    #[getter]
    fn backoff_multiplier(&self) -> f64 {
        self.inner.backoff_multiplier
    }
    #[getter]
    fn jitter(&self) -> PyJitterKind {
        PyJitterKind { inner: self.inner.jitter }
    }
    #[getter]
    fn respect_retry_after(&self) -> bool {
        self.inner.respect_retry_after
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryPolicy(max_retries={}, initial_backoff_ms={})",
            self.inner.max_retries,
            self.inner.initial_backoff.as_millis()
        )
    }
}

// ---------------------------------------------------------------------------
// Timeouts
// ---------------------------------------------------------------------------

#[pyclass(name = "Timeouts", module = "atomr_infer._native.config")]
#[derive(Clone)]
pub struct PyTimeouts {
    pub(crate) inner: Timeouts,
}

#[pymethods]
impl PyTimeouts {
    #[new]
    #[pyo3(signature = (request_timeout_ms=30_000, read_timeout_ms=10_000))]
    fn new(request_timeout_ms: u64, read_timeout_ms: u64) -> Self {
        Self {
            inner: Timeouts {
                request_timeout: Duration::from_millis(request_timeout_ms),
                read_timeout: Duration::from_millis(read_timeout_ms),
            },
        }
    }

    #[getter]
    fn request_timeout_ms(&self) -> u64 {
        self.inner.request_timeout.as_millis() as u64
    }
    #[getter]
    fn read_timeout_ms(&self) -> u64 {
        self.inner.read_timeout.as_millis() as u64
    }

    fn __repr__(&self) -> String {
        format!(
            "Timeouts(request_timeout_ms={}, read_timeout_ms={})",
            self.inner.request_timeout.as_millis(),
            self.inner.read_timeout.as_millis()
        )
    }
}

// ---------------------------------------------------------------------------
// BudgetAction
// ---------------------------------------------------------------------------

#[pyclass(name = "BudgetAction", module = "atomr_infer._native.config", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyBudgetAction {
    pub(crate) inner: BudgetAction,
}

#[pymethods]
impl PyBudgetAction {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name {
            "reject" => BudgetAction::Reject,
            "warn" => BudgetAction::Warn,
            "throttle" => BudgetAction::Throttle,
            other => return Err(PyValueError::new_err(format!("unknown budget action: {other:?}"))),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &'static str {
        match self.inner {
            BudgetAction::Reject => "reject",
            BudgetAction::Warn => "warn",
            BudgetAction::Throttle => "throttle",
        }
    }

    fn __repr__(&self) -> String {
        format!("BudgetAction({:?})", self.name())
    }
}

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

#[pyclass(name = "Budget", module = "atomr_infer._native.config")]
#[derive(Clone)]
pub struct PyBudget {
    pub(crate) inner: Budget,
}

#[pymethods]
impl PyBudget {
    #[new]
    #[pyo3(signature = (max_spend_per_hour_usd=None, max_spend_per_day_usd=None, on_exceeded=None))]
    fn new(
        max_spend_per_hour_usd: Option<f64>,
        max_spend_per_day_usd: Option<f64>,
        on_exceeded: Option<PyBudgetAction>,
    ) -> Self {
        Self {
            inner: Budget {
                max_spend_per_hour_usd,
                max_spend_per_day_usd,
                on_exceeded: on_exceeded.map(|a| a.inner).unwrap_or(BudgetAction::Reject),
            },
        }
    }

    #[getter]
    fn max_spend_per_hour_usd(&self) -> Option<f64> {
        self.inner.max_spend_per_hour_usd
    }
    #[getter]
    fn max_spend_per_day_usd(&self) -> Option<f64> {
        self.inner.max_spend_per_day_usd
    }
    #[getter]
    fn on_exceeded(&self) -> PyBudgetAction {
        PyBudgetAction {
            inner: self.inner.on_exceeded,
        }
    }

    fn __repr__(&self) -> String {
        format!("Budget(on_exceeded={:?})", PyBudgetAction { inner: self.inner.on_exceeded }.name())
    }
}

// ---------------------------------------------------------------------------
// Submodule registration
// ---------------------------------------------------------------------------

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let sub = PyModule::new_bound(py, "config")?;
    sub.add_class::<PyCapacityPolicy>()?;
    sub.add_class::<PyServing>()?;
    sub.add_class::<PyRateLimits>()?;
    sub.add_class::<PyRetryPolicy>()?;
    sub.add_class::<PyTimeouts>()?;
    sub.add_class::<PyBudgetAction>()?;
    sub.add_class::<PyBudget>()?;
    m.add_submodule(&sub)?;
    Ok(())
}
