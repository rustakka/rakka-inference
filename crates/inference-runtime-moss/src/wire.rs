//! Wire-level constants for the MOSS TTS runtime.
//!
//! MOSS-TTSD outputs 24 000 Hz mono PCM. The model is Linux-only
//! (see `MossRunner` for the platform gate).

/// Native output sample rate of MOSS-TTSD (24 000 Hz).
pub const MOSS_SAMPLE_RATE_HZ: u32 = 24_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rate_is_24khz() {
        assert_eq!(MOSS_SAMPLE_RATE_HZ, 24_000);
    }
}
