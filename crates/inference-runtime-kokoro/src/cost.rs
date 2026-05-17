//! Cost helpers for Kokoro TTS.
//!
//! Kokoro is a local runtime — no per-request cloud spend. All cost
//! estimates return 0.0. These helpers exist for API-surface parity
//! with remote TTS runtimes that have per-character pricing.

/// Per-million-character cost in USD for the given model name.
///
/// Kokoro is a locally-executed model; there is no remote metering.
/// Always returns `0.0`.
///
/// # Examples
///
/// ```
/// use atomr_infer_runtime_kokoro::cost;
///
/// assert_eq!(cost::per_million_chars_usd("kokoro-82m"), 0.0);
/// ```
pub fn per_million_chars_usd(_model: &str) -> f64 {
    0.0
}

/// Estimate the USD cost for synthesizing `chars` characters with `model`.
///
/// Since Kokoro is a local runtime, this is always `0.0` regardless of
/// character count. Provided for API-surface parity with remote runtimes.
///
/// # Examples
///
/// ```
/// use atomr_infer_runtime_kokoro::cost;
///
/// assert_eq!(cost::estimate_usd("kokoro-82m", 1_000_000), 0.0);
/// ```
pub fn estimate_usd(model: &str, chars: u64) -> f64 {
    per_million_chars_usd(model) * chars as f64 / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kokoro_rate_is_zero() {
        assert_eq!(per_million_chars_usd("kokoro-82m"), 0.0);
    }

    #[test]
    fn unknown_model_returns_zero() {
        assert_eq!(per_million_chars_usd("nonexistent-model-xyz"), 0.0);
    }

    #[test]
    fn estimate_scales_linearly() {
        // 0.0 * anything = 0.0 for local runtimes
        assert_eq!(estimate_usd("kokoro-82m", 0), 0.0);
        assert_eq!(estimate_usd("kokoro-82m", 500_000), 0.0);
        assert_eq!(estimate_usd("kokoro-82m", 2_000_000), 0.0);
    }
}
