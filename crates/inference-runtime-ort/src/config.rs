//! Public configuration types. Compiled in every build profile so the
//! `inference-runtime-ort` stub still parses operator config when the
//! `ort` feature is off.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrtConfig {
    pub onnx_path: PathBuf,

    #[serde(default)]
    pub execution_provider: ExecutionProvider,

    /// CUDA device ordinal — only consulted when
    /// `execution_provider == Cuda`.
    #[serde(default)]
    pub device_id: u32,

    /// Where to find `tokenizer.json`. `None` ⇒ probe the directory
    /// next to `onnx_path`, then `hf_repo` if `ort-hf-hub` is on.
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,

    /// HuggingFace repo id — used as a fallback tokenizer source when
    /// the `ort-hf-hub` feature is enabled.
    #[serde(default)]
    pub hf_repo: Option<String>,

    /// Number of intra-op threads. `None` ⇒ ort default (typically the
    /// physical core count).
    #[serde(default)]
    pub intra_threads: Option<usize>,

    /// Hard ceiling on output tokens for chat-style `execute()`. Used
    /// only when `SamplingParams::max_tokens` is `None`.
    #[serde(default = "default_max_new_tokens")]
    pub default_max_new_tokens: u32,
}

fn default_max_new_tokens() -> u32 {
    256
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProvider {
    #[default]
    Cpu,
    Cuda,
    TensorRt,
    DirectMl,
}
