//! Wire-level helpers for the Kokoro ONNX session.
//!
//! Kokoro-82M expects a sequence of phoneme token ids as input and
//! returns a `f32[1, 1, samples]` waveform tensor. This module
//! encapsulates the input/output tensor shapes so `runner.rs` stays
//! focused on the SpeechRunner trait contract.
//!
//! Under the `tts-kokoro` feature gate, these helpers interact with
//! the `ort` crate. Without the feature, the module is a no-op stub.

/// Native sample rate of Kokoro-82M voices (24 000 Hz).
pub const KOKORO_SAMPLE_RATE_HZ: u32 = 24_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rate_is_24khz() {
        assert_eq!(KOKORO_SAMPLE_RATE_HZ, 24_000);
    }
}
