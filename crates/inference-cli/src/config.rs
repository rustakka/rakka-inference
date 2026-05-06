//! Project-file (`<config>.toml`) parsing. Doc §11.3.

use std::path::Path;

use serde::{Deserialize, Serialize};

use atomr_infer_core::deployment::Deployment;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFile {
    pub cluster: ClusterConfig,
    #[serde(default, rename = "deployment")]
    pub deployments: Vec<Deployment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub name: String,
    /// HTTP gateway bind address. Optional — defaults to 127.0.0.1:8080.
    #[serde(default = "default_bind")]
    pub bind: std::net::SocketAddr,
    /// atomr cluster endpoint (placeholder; v0 runs single-node).
    #[serde(default)]
    pub endpoint: Option<String>,
}

fn default_bind() -> std::net::SocketAddr {
    "127.0.0.1:8080".parse().expect("static addr")
}

impl ProjectFile {
    pub fn from_path(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let body = std::fs::read_to_string(path.as_ref())?;
        Ok(toml::from_str(&body)?)
    }
}
