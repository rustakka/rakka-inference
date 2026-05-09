//! Pure-Rust sampling. No `ort` dependency — `f32` logits in, token id
//! out. The caller is responsible for converting FP16 / BF16 logits
//! upstream (we keep this module type-clean).
//!
//! Pipeline:
//!   temperature scale → top-k truncate → top-p (nucleus) truncate →
//!   softmax → multinomial sample
//!
//! Greedy fast-path (`temperature <= 0.0`) bypasses the whole stack.

use atomr_infer_core::batch::SamplingParams;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;

pub(crate) fn sample_next(logits: &[f32], params: &SamplingParams, rng: &mut StdRng) -> u32 {
    debug_assert!(!logits.is_empty(), "sample_next: empty logits");

    let temperature = params.temperature.unwrap_or(1.0);
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let mut scored: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, l)| (i as u32, l / temperature))
        .collect();

    if let Some(k) = params.top_k {
        if (k as usize) < scored.len() && k > 0 {
            scored.select_nth_unstable_by(k as usize - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k as usize);
        }
    }

    let mut probs: Vec<(u32, f32)> = softmax(scored);

    if let Some(p) = params.top_p {
        if (0.0..1.0).contains(&p) {
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut acc = 0.0;
            let mut cutoff = probs.len();
            for (i, (_, prob)) in probs.iter().enumerate() {
                acc += prob;
                if acc >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            // Renormalise after truncation.
            let s: f32 = probs.iter().map(|(_, w)| *w).sum();
            if s > 0.0 {
                for (_, w) in &mut probs {
                    *w /= s;
                }
            }
        }
    }

    let weights: Vec<f32> = probs.iter().map(|(_, w)| *w).collect();
    match WeightedIndex::new(&weights) {
        Ok(dist) => probs[dist.sample(rng)].0,
        Err(_) => probs.first().map(|(t, _)| *t).unwrap_or_else(|| argmax(logits)),
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::NEG_INFINITY),
            |(best_i, best), (i, &v)| if v > best { (i, v) } else { (best_i, best) },
        )
        .0 as u32
}

fn softmax(scored: Vec<(u32, f32)>) -> Vec<(u32, f32)> {
    let max = scored
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<(u32, f32)> = scored
        .into_iter()
        .map(|(t, l)| (t, (l - max).exp()))
        .collect();
    let sum: f32 = exps.iter().map(|(_, e)| *e).sum();
    if sum > 0.0 {
        for (_, e) in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Construct a deterministic RNG, optionally seeded from the request.
pub(crate) fn rng_from(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.1, 0.5, 0.2, 0.05];
        let params = SamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(sample_next(&logits, &params, &mut rng), 1);
    }

    #[test]
    fn negative_temp_treated_as_greedy() {
        let logits = vec![0.0, 1.0, 0.5];
        let params = SamplingParams {
            temperature: Some(-1.0),
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(sample_next(&logits, &params, &mut rng), 1);
    }

    #[test]
    fn top_k_truncates() {
        // With top_k = 1, only the argmax is reachable regardless of temp.
        let logits = vec![0.1, 0.5, 0.2, 0.05];
        let params = SamplingParams {
            temperature: Some(1.0),
            top_k: Some(1),
            ..Default::default()
        };
        for seed in 0..5 {
            let mut rng = StdRng::seed_from_u64(seed);
            assert_eq!(sample_next(&logits, &params, &mut rng), 1);
        }
    }

    #[test]
    fn deterministic_from_seed() {
        let logits = vec![0.1, 0.2, 0.3, 0.4, 0.0];
        let params = SamplingParams {
            temperature: Some(1.0),
            ..Default::default()
        };
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        assert_eq!(
            sample_next(&logits, &params, &mut a),
            sample_next(&logits, &params, &mut b)
        );
    }
}
