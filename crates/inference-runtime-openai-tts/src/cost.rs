//! Per-character TTS pricing. Snapshot from
//! <https://openai.com/api/pricing/> (May 2026). Not authoritative —
//! deployments should override via their own pricing source.

/// USD per 1 million input characters for a given model. Returns
/// `0.0` when the model is unknown so the cost estimate degrades
/// gracefully (matches the sibling chat-completions crate convention).
pub fn per_million_chars_usd(model: &str) -> f64 {
    match model {
        "tts-1" => 15.0,
        "tts-1-hd" => 30.0,
        "gpt-4o-mini-tts" => 12.0,
        _ => 0.0,
    }
}

/// Estimate the USD cost of synthesizing `chars` characters with
/// `model`.
pub fn estimate_usd(model: &str, chars: u32) -> f64 {
    per_million_chars_usd(model) * (chars as f64) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_have_positive_rates() {
        assert!(per_million_chars_usd("tts-1") > 0.0);
        assert!(per_million_chars_usd("tts-1-hd") > 0.0);
        assert!(per_million_chars_usd("gpt-4o-mini-tts") > 0.0);
    }

    #[test]
    fn unknown_model_is_zero() {
        assert_eq!(per_million_chars_usd("unknown-model"), 0.0);
        assert_eq!(estimate_usd("unknown-model", 1_000), 0.0);
    }

    #[test]
    fn estimate_scales_linearly() {
        let one_m = estimate_usd("tts-1", 1_000_000);
        let two_m = estimate_usd("tts-1", 2_000_000);
        assert!((two_m - 2.0 * one_m).abs() < 1e-9);
    }
}
