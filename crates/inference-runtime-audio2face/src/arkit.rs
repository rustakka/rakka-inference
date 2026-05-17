//! ARKit canonical blendshape ordering and A2F→ARKit adapter.
//!
//! The 52-element ordering is mandated by Apple's `ARBlendShapeLocation`
//! enum, documented at
//! <https://developer.apple.com/documentation/arkit/arblendshapelocation>.
//!
//! # A2F-native vs ARKit ordering
//!
//! NVIDIA Omniverse Audio2Face-3D uses its own internal blendshape ordering
//! that differs from the ARKit canonical ordering. The authoritative mapping
//! table requires the NVIDIA Omniverse documentation, which is not available
//! offline. The [`a2f_to_arkit`] adapter therefore currently passes values
//! through unchanged (identity mapping).
//!
//! **TODO(a2f-normalization):** replace the passthrough with the real index
//! permutation table once the NVIDIA mapping is confirmed. Track at
//! <https://github.com/rustakka/atomr-infer/issues/TODO>.

/// The 52 ARKit-canonical blendshape names in Apple-mandated order.
///
/// Index 0 is `eyeBlinkLeft`; index 51 is `tongueOut`. This ordering is
/// fixed by `ARBlendShapeLocation` and must not be changed.
pub const ARKIT_BLENDSHAPE_NAMES: [&str; 52] = [
    "eyeBlinkLeft",
    "eyeLookDownLeft",
    "eyeLookInLeft",
    "eyeLookOutLeft",
    "eyeLookUpLeft",
    "eyeSquintLeft",
    "eyeWideLeft",
    "eyeBlinkRight",
    "eyeLookDownRight",
    "eyeLookInRight",
    "eyeLookOutRight",
    "eyeLookUpRight",
    "eyeSquintRight",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
];

/// Convert A2F-native blendshape weights to ARKit-canonical ordering.
///
/// # Current behaviour
///
/// The mapping is currently a **passthrough** — `a2f[i]` maps to
/// `arkit[i]` unchanged. This is correct only if the A2F server is
/// already emitting weights in ARKit order (which some server configs do).
///
/// When the authoritative NVIDIA→ARKit index table becomes available,
/// replace the body with the permutation.
///
/// # TODO
///
/// ```text
/// // TODO(a2f-normalization): apply real index permutation from
/// // NVIDIA Omniverse Audio2Face documentation once confirmed.
/// ```
pub fn a2f_to_arkit(a2f: &[f32; 52]) -> [f32; 52] {
    // TODO(a2f-normalization): apply real index permutation from
    // NVIDIA Omniverse Audio2Face documentation once confirmed.
    tracing::debug!("a2f→arkit ordering currently passthrough; map remains TODO(a2f-normalization)");
    *a2f
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn arkit_names_count() {
        assert_eq!(
            ARKIT_BLENDSHAPE_NAMES.len(),
            52,
            "ARKit requires exactly 52 blendshape names"
        );
    }

    #[test]
    fn arkit_names_unique() {
        let set: HashSet<&str> = ARKIT_BLENDSHAPE_NAMES.iter().copied().collect();
        assert_eq!(set.len(), 52, "ARKit blendshape names must all be unique");
    }

    #[test]
    fn arkit_names_known_entries() {
        // Spot-check a few canonical positions.
        assert_eq!(ARKIT_BLENDSHAPE_NAMES[0], "eyeBlinkLeft");
        assert_eq!(ARKIT_BLENDSHAPE_NAMES[7], "eyeBlinkRight");
        assert_eq!(ARKIT_BLENDSHAPE_NAMES[17], "jawOpen");
        assert_eq!(ARKIT_BLENDSHAPE_NAMES[51], "tongueOut");
    }

    #[test]
    fn passthrough_adapter_preserves_values() {
        let mut input = [0.0f32; 52];
        for (i, v) in input.iter_mut().enumerate() {
            *v = i as f32 * 0.01;
        }
        let output = a2f_to_arkit(&input);
        for i in 0..52 {
            assert!(
                (output[i] - input[i]).abs() < f32::EPSILON,
                "passthrough adapter should preserve value at index {i}"
            );
        }
    }

    #[test]
    fn passthrough_adapter_all_zeros() {
        let input = [0.0f32; 52];
        let output = a2f_to_arkit(&input);
        assert_eq!(output, [0.0f32; 52]);
    }

    #[test]
    fn passthrough_adapter_all_ones() {
        let input = [1.0f32; 52];
        let output = a2f_to_arkit(&input);
        assert_eq!(output, [1.0f32; 52]);
    }
}
