//! Per-minute STT pricing for OpenAI transcription models. Rough
//! snapshot — operators should override with their own pricing source.

/// USD per minute of audio billed for a given transcription model.
/// Returns `0.0` when the model is unknown.
pub fn per_minute_usd(model: &str) -> f64 {
    match model {
        "whisper-1" => 0.006,
        "gpt-4o-transcribe" => 0.006,
        "gpt-4o-mini-transcribe" => 0.003,
        _ => 0.0,
    }
}

/// Estimate the USD cost of transcribing `audio_seconds` seconds with
/// `model`.
pub fn estimate_usd(model: &str, audio_seconds: u32) -> f64 {
    per_minute_usd(model) * (audio_seconds as f64) / 60.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_have_positive_rates() {
        assert!(per_minute_usd("whisper-1") > 0.0);
        assert!(per_minute_usd("gpt-4o-transcribe") > 0.0);
        assert!(per_minute_usd("gpt-4o-mini-transcribe") > 0.0);
    }

    #[test]
    fn unknown_model_is_zero() {
        assert_eq!(per_minute_usd("unknown-model"), 0.0);
        assert_eq!(estimate_usd("unknown-model", 60), 0.0);
    }

    #[test]
    fn estimate_scales_linearly() {
        let one_min = estimate_usd("whisper-1", 60);
        let ten_min = estimate_usd("whisper-1", 600);
        assert!((ten_min - 10.0 * one_min).abs() < 1e-9);
    }
}
