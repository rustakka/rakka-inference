//! Per-second STT pricing for AssemblyAI Streaming models. Approximate
//! snapshot from the public AssemblyAI pricing page (Universal-Streaming
//! tier). Operators should override with their own pricing source.
//!
//! AssemblyAI bills Streaming-v3 at a per-second rate independent of
//! the specific model variant; we expose `per_hour_usd` for parity
//! with the published pricing surface and reduce internally to a
//! per-second figure.

/// USD per audio-hour billed for a given streaming model. Returns
/// `0.0` for unknown models so estimation degrades safely.
pub fn per_hour_usd(model: &str) -> f64 {
    match model {
        // Universal-Streaming v3 — flat per-hour rate at the time of
        // this snapshot.
        "universal" | "universal-streaming" | "universal-v3" => 0.15,
        _ => 0.0,
    }
}

/// Estimate USD cost of transcribing `audio_seconds` seconds of audio
/// with `model`.
pub fn estimate_usd(model: &str, audio_seconds: u32) -> f64 {
    per_hour_usd(model) * (audio_seconds as f64) / 3600.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_have_positive_rates() {
        assert!(per_hour_usd("universal") > 0.0);
        assert!(per_hour_usd("universal-streaming") > 0.0);
    }

    #[test]
    fn unknown_model_is_zero() {
        assert_eq!(per_hour_usd("unknown"), 0.0);
        assert_eq!(estimate_usd("unknown", 600), 0.0);
    }

    #[test]
    fn estimate_scales_linearly() {
        let one_min = estimate_usd("universal", 60);
        let ten_min = estimate_usd("universal", 600);
        assert!((ten_min - 10.0 * one_min).abs() < 1e-9);
    }
}
