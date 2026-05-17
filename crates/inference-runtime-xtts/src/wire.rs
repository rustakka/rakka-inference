//! Wire-level constants for the XTTS ONNX session.
//!
//! XTTS-v2 outputs at 24 000 Hz mono PCM. The speaker encoder expects a
//! 22 050 Hz reference clip (it internally resamples) — callers may pass
//! any common sample rate for the reference audio.

/// Native output sample rate of XTTS-v2 (24 000 Hz).
pub const XTTS_SAMPLE_RATE_HZ: u32 = 24_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rate_is_24khz() {
        assert_eq!(XTTS_SAMPLE_RATE_HZ, 24_000);
    }
}
