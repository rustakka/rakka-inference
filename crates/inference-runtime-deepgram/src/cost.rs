//! Per-minute STT pricing for Deepgram models. Approximate snapshot —
//! operators should override with their own pricing source. Numbers
//! from the public Deepgram pricing page (Pay-as-you-go tier).

/// USD per audio-minute billed for a given model. Returns `0.0` for
/// unknown models so estimation degrades safely.
pub fn per_minute_usd(model: &str) -> f64 {
    match model {
        "nova-2" | "nova-2-general" => 0.0043,
        "nova-2-phonecall" => 0.0058,
        "nova-2-meeting" => 0.0058,
        "nova" => 0.0125,
        "enhanced" => 0.0145,
        "base" => 0.0125,
        _ => 0.0,
    }
}

/// Estimate USD cost of transcribing `audio_seconds` seconds of audio
/// with `model`.
pub fn estimate_usd(model: &str, audio_seconds: u32) -> f64 {
    per_minute_usd(model) * (audio_seconds as f64) / 60.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_have_positive_rates() {
        assert!(per_minute_usd("nova-2") > 0.0);
        assert!(per_minute_usd("nova") > 0.0);
    }

    #[test]
    fn unknown_model_is_zero() {
        assert_eq!(per_minute_usd("unknown"), 0.0);
        assert_eq!(estimate_usd("unknown", 600), 0.0);
    }

    #[test]
    fn estimate_scales_linearly() {
        let one_min = estimate_usd("nova-2", 60);
        let ten_min = estimate_usd("nova-2", 600);
        assert!((ten_min - 10.0 * one_min).abs() < 1e-9);
    }
}
