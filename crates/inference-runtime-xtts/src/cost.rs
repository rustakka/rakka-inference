//! Cost helpers for XTTS TTS.
//!
//! XTTS is a local runtime — no per-request cloud spend. All cost
//! estimates return 0.0.

/// Per-million-character cost in USD for the given model name.
///
/// XTTS is locally-executed; there is no remote metering.
/// Always returns `0.0`.
///
/// # Examples
///
/// ```
/// use atomr_infer_runtime_xtts::cost;
///
/// assert_eq!(cost::per_million_chars_usd("xtts-v2"), 0.0);
/// ```
pub fn per_million_chars_usd(_model: &str) -> f64 {
    0.0
}

/// Estimate the USD cost for synthesizing `chars` characters with `model`.
///
/// Always `0.0` for this local runtime.
///
/// # Examples
///
/// ```
/// use atomr_infer_runtime_xtts::cost;
///
/// assert_eq!(cost::estimate_usd("xtts-v2", 1_000_000), 0.0);
/// ```
pub fn estimate_usd(model: &str, chars: u64) -> f64 {
    per_million_chars_usd(model) * chars as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xtts_rate_is_zero() {
        assert_eq!(per_million_chars_usd("xtts-v2"), 0.0);
    }

    #[test]
    fn unknown_model_returns_zero() {
        assert_eq!(per_million_chars_usd("nonexistent-model-xyz"), 0.0);
    }

    #[test]
    fn estimate_scales_linearly() {
        assert_eq!(estimate_usd("xtts-v2", 0), 0.0);
        assert_eq!(estimate_usd("xtts-v2", 500_000), 0.0);
        assert_eq!(estimate_usd("xtts-v2", 2_000_000), 0.0);
    }
}
