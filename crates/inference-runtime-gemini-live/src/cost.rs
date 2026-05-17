//! Pricing helpers for Gemini Live models.
//!
//! Gemini Live billing is per-second-of-audio rather than per-token, so
//! this module exposes `per_minute_usd` as the primary estimator. A
//! `per_million_tokens_usd` stub is also provided for consistency with
//! other runtime cost modules; it returns 0.0 because audio billing is
//! time-based.
//!
//! Rates are approximate snapshots and should be overridden with an
//! operator-supplied pricing source.

/// USD per audio-minute billed for a given Gemini Live model.
///
/// - `gemini-2.0-flash-exp` → 0.0 (free preview tier as of plan date).
/// - All other models → 0.05 USD/min (placeholder).
///
/// Returns `0.0` for unknown models so estimation degrades safely.
pub fn per_minute_usd(model: &str) -> f64 {
    match model {
        "gemini-2.0-flash-exp" => 0.0,
        _ => 0.05,
    }
}

/// USD per million tokens for a given model.
///
/// Gemini Live billing is time-based (audio-seconds), so this always
/// returns 0.0. Provided for API symmetry with other cost modules.
///
/// Known non-zero token prices (text API, not realtime):
/// - `gemini-2.0-flash` → 0.075 USD per million input tokens.
pub fn per_million_tokens_usd(model: &str) -> f64 {
    match model {
        "gemini-2.0-flash-exp" => 0.0,
        _ => 0.0,
    }
}

/// Estimate USD cost of a Gemini Live session lasting `audio_seconds`
/// seconds.
pub fn estimate_usd(model: &str, audio_seconds: u32) -> f64 {
    per_minute_usd(model) * (audio_seconds as f64) / 60.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_exp_is_free() {
        assert_eq!(per_minute_usd("gemini-2.0-flash-exp"), 0.0);
        assert_eq!(per_million_tokens_usd("gemini-2.0-flash-exp"), 0.0);
        assert_eq!(estimate_usd("gemini-2.0-flash-exp", 3600), 0.0);
    }

    #[test]
    fn unknown_model_gets_placeholder_rate() {
        let rate = per_minute_usd("gemini-2.0-pro");
        assert!((rate - 0.05).abs() < 1e-9, "expected 0.05, got {rate}");
    }

    #[test]
    fn per_million_tokens_always_zero() {
        assert_eq!(per_million_tokens_usd("gemini-2.0-flash"), 0.0);
        assert_eq!(per_million_tokens_usd("unknown-model"), 0.0);
    }

    #[test]
    fn estimate_scales_linearly() {
        let model = "gemini-unknown";
        let one_min = estimate_usd(model, 60);
        let ten_min = estimate_usd(model, 600);
        assert!((ten_min - 10.0 * one_min).abs() < 1e-9);
    }
}
