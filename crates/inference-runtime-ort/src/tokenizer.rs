//! Tokenizer resolution + chat-template rendering.

use std::path::PathBuf;

use atomr_infer_core::batch::{ContentPart, Message, MessageContent, Role};
use atomr_infer_core::error::{InferenceError, InferenceResult};
use tokenizers::Tokenizer;

use crate::config::OrtConfig;
use crate::error::internal;

/// Try to find a `tokenizer.json` for the configured ONNX file.
///
/// Resolution order:
/// 1. `cfg.tokenizer_path` if set.
/// 2. `tokenizer.json` in the same directory as `cfg.onnx_path`.
/// 3. `cfg.hf_repo` via hf-hub (only when `ort-hf-hub` feature is on).
///
/// Returns `Ok(None)` when no tokenizer is found anywhere — the
/// generate loop refuses chat-style execute() in that case, but the
/// low-level `infer()` path still works (it doesn't need text).
pub(crate) fn resolve_tokenizer(cfg: &OrtConfig) -> InferenceResult<Option<Tokenizer>> {
    if let Some(path) = &cfg.tokenizer_path {
        return load(path).map(Some);
    }

    if let Some(parent) = cfg.onnx_path.parent() {
        let sibling = parent.join("tokenizer.json");
        if sibling.is_file() {
            return load(&sibling).map(Some);
        }
    }

    #[cfg(feature = "ort-hf-hub")]
    if let Some(repo) = &cfg.hf_repo {
        return resolve_via_hf_hub(repo).map(Some);
    }

    let _ = cfg;
    Ok(None)
}

fn load(path: &PathBuf) -> InferenceResult<Tokenizer> {
    Tokenizer::from_file(path).map_err(|e| internal(&format!("tokenizer.json {}", path.display()), e))
}

#[cfg(feature = "ort-hf-hub")]
fn resolve_via_hf_hub(repo: &str) -> InferenceResult<Tokenizer> {
    use hf_hub::api::sync::Api;
    let api = Api::new().map_err(|e| internal("hf-hub api", e))?;
    let path = api
        .model(repo.to_owned())
        .get("tokenizer.json")
        .map_err(|e| internal(&format!("hf-hub get {repo}/tokenizer.json"), e))?;
    load(&path)
}

/// Fallback chat-template renderer. We don't try to interpret the
/// tokenizer's Jinja `chat_template` (the `tokenizers` crate doesn't
/// surface it stably across 0.20.x); instead we use a generic
/// role-prefixed format that most ONNX-exported models accept and
/// that the operator can override by writing their own template
/// upstream of `execute`.
pub(crate) fn render_chat(messages: &[Message]) -> InferenceResult<String> {
    if messages.is_empty() {
        return Err(InferenceError::BadRequest {
            message: "ort: empty messages list".into(),
        });
    }
    let mut out = String::new();
    for m in messages {
        let role = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            _ => "user",
        };
        out.push_str("<|");
        out.push_str(role);
        out.push_str("|>\n");
        out.push_str(&message_text(m));
        out.push('\n');
    }
    out.push_str("<|assistant|>\n");
    Ok(out)
}

fn message_text(m: &Message) -> String {
    match &m.content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
