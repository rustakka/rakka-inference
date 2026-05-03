//! Default runtime selection — `Deployment::infer_runtime()` (doc §3.2).
//!
//! Operators always override; this is the ergonomic path for the common
//! case where the model name carries the runtime affinity unambiguously
//! (`gpt-4o*` → openai, `claude-*` → anthropic, etc.).

use crate::runtime::RuntimeKind;

/// Map a model name to its default runtime backend.
pub fn infer_runtime(model: &str) -> RuntimeKind {
    let m = model.to_ascii_lowercase();

    // Remote provider model families take precedence. The patterns are
    // intentionally conservative — operators with custom local
    // fine-tunes named `gpt-*` should set `runtime` explicitly.
    if matches_any(&m, &["gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-3.5", "o1-", "o3-", "chatgpt-"]) {
        return RuntimeKind::OpenAi;
    }
    if m.starts_with("claude-") || m.starts_with("anthropic/") {
        return RuntimeKind::Anthropic;
    }
    if m.starts_with("gemini-") || m.starts_with("google/gemini") {
        return RuntimeKind::Gemini;
    }
    if m.contains("via-litellm") || m.starts_with("litellm/") {
        return RuntimeKind::LiteLlm;
    }

    // Local LLM families: prefer Rust-native runtimes where available.
    if m.contains("mistral") {
        return RuntimeKind::MistralRs;
    }

    // Default for local LLM-shaped names where no Rust-native backend
    // is well established.
    RuntimeKind::Vllm
}

fn matches_any(model: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| model.starts_with(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_defaults() {
        assert_eq!(infer_runtime("gpt-4o"), RuntimeKind::OpenAi);
        assert_eq!(infer_runtime("gpt-4o-mini"), RuntimeKind::OpenAi);
        assert_eq!(infer_runtime("o1-preview"), RuntimeKind::OpenAi);
    }

    #[test]
    fn anthropic_defaults() {
        assert_eq!(infer_runtime("claude-sonnet-4"), RuntimeKind::Anthropic);
        assert_eq!(infer_runtime("anthropic/claude-3-haiku"), RuntimeKind::Anthropic);
    }

    #[test]
    fn gemini_defaults() {
        assert_eq!(infer_runtime("gemini-2.0-pro"), RuntimeKind::Gemini);
        assert_eq!(infer_runtime("google/gemini-1.5-flash"), RuntimeKind::Gemini);
    }

    #[test]
    fn local_fallthrough_for_unknown() {
        assert_eq!(infer_runtime("meta-llama/Llama-3.1-70B-Instruct"), RuntimeKind::Vllm);
    }

    #[test]
    fn mistral_picks_rust_native() {
        assert_eq!(infer_runtime("mistralai/Mistral-7B-Instruct-v0.3"), RuntimeKind::MistralRs);
    }
}
