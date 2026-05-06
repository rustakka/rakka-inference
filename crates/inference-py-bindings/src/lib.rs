//! # inference-py-bindings
//!
//! PyO3 bindings exposing the `Deployment` value object and a thin
//! `Cluster` handle to Python callers. Doc §11.1.
//!
//! Default-features-off the crate compiles to an empty rlib so the
//! workspace builds without a Python venv. With `--features python`
//! it builds a `cdylib` suitable for loading as a Python extension
//! module (`pip install maturin && maturin develop --features python`).
//!
//! When loaded via maturin / wheel, the extension module is exposed
//! as `atomr_infer._native`; the user-facing API lives in the
//! pure-Python `python/atomr_infer/` package which re-exports
//! the native classes.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;

    use atomr_infer_core::deployment::Deployment as RsDeployment;

    /// Python-side `Deployment` — owns a `RsDeployment` value object.
    /// Python callers build it via the keyword-arg constructor below
    /// and pass it to `Cluster.deploy(...)`.
    #[pyclass(name = "Deployment")]
    pub struct PyDeployment {
        inner: RsDeployment,
    }

    #[pymethods]
    impl PyDeployment {
        #[new]
        #[pyo3(signature = (name, model, replicas=1, gpus=None))]
        fn new(name: String, model: String, replicas: u32, gpus: Option<u32>) -> Self {
            Self {
                inner: RsDeployment {
                    name,
                    model,
                    runtime: None,
                    runtime_config: None,
                    gpus,
                    replicas,
                    serving: Default::default(),
                    budget: None,
                    idempotent: true,
                },
            }
        }

        fn name(&self) -> &str {
            &self.inner.name
        }

        fn model(&self) -> &str {
            &self.inner.model
        }
    }

    /// `Cluster.connect(endpoint)` — returns a `Cluster` handle.
    /// In v0 this is a placeholder; real cluster connection wires up
    /// with `atomr-cluster` once the binding surface stabilises.
    #[pyclass(name = "Cluster")]
    pub struct PyCluster {
        endpoint: String,
    }

    #[pymethods]
    impl PyCluster {
        #[staticmethod]
        fn connect(endpoint: String) -> Self {
            Self { endpoint }
        }

        fn endpoint(&self) -> &str {
            &self.endpoint
        }

        fn deploy(&self, deployment: &PyDeployment) -> PyResult<()> {
            // TODO(doc §11.5): submit Apply through the cluster's
            // DeploymentManagerActor singleton over atomr-remote IPC.
            tracing::info!(name = %deployment.inner.name, "py: deploy stub");
            Ok(())
        }
    }

    #[pymodule]
    fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyDeployment>()?;
        m.add_class::<PyCluster>()?;
        Ok(())
    }
}
