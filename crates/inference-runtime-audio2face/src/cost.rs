//! Cost estimation for Audio2Face-3D.
//!
//! NVIDIA Audio2Face-3D is a self-hosted service — there is no per-call
//! cloud billing. All cost functions return 0.0.

/// Return the estimated USD cost per minute of audio for the given model.
///
/// Audio2Face-3D is self-hosted; compute costs are amortized through
/// infrastructure billing rather than per-call charges. This function
/// always returns `0.0`.
pub fn per_minute_usd(_model: &str) -> f64 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_is_zero_for_all_models() {
        assert_eq!(per_minute_usd("audio2face-3d"), 0.0);
        assert_eq!(per_minute_usd("audio2face-3d-v1"), 0.0);
        assert_eq!(per_minute_usd("unknown-model"), 0.0);
        assert_eq!(per_minute_usd(""), 0.0);
    }
}
