//! Per-character TTS pricing for ElevenLabs models. Approximate
//! snapshot — operators should override with their own pricing
//! source. Numbers from the public ElevenLabs pricing page,
//! billed-per-credit converted to USD-per-character on the Creator tier
//! (~$0.30 / 1000 characters for the flagship `eleven_multilingual_v2`).

/// USD per character billed for a given model. Returns `0.0` for
/// unknown models so estimation degrades safely.
pub fn per_million_chars_usd(model: &str) -> f64 {
    match model {
        "eleven_multilingual_v2" => 300.0,
        "eleven_turbo_v2_5" => 150.0,
        "eleven_flash_v2_5" => 50.0,
        _ => 0.0,
    }
}

/// Estimate USD cost of synthesising `chars` characters with `model`.
pub fn estimate_usd(model: &str, chars: u32) -> f64 {
    per_million_chars_usd(model) * (chars as f64) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_have_positive_rates() {
        assert!(per_million_chars_usd("eleven_multilingual_v2") > 0.0);
        assert!(per_million_chars_usd("eleven_turbo_v2_5") > 0.0);
        assert!(per_million_chars_usd("eleven_flash_v2_5") > 0.0);
    }

    #[test]
    fn unknown_model_is_zero() {
        assert_eq!(per_million_chars_usd("unknown"), 0.0);
        assert_eq!(estimate_usd("unknown", 1_000_000), 0.0);
    }

    #[test]
    fn estimate_scales_linearly() {
        let one_k = estimate_usd("eleven_multilingual_v2", 1_000);
        let ten_k = estimate_usd("eleven_multilingual_v2", 10_000);
        assert!((ten_k - 10.0 * one_k).abs() < 1e-9);
    }
}
