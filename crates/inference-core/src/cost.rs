//! Cost-estimation primitives. Used by `inference-pipeline`'s
//! `TieredRouter` and by `MetricsActor` for budget enforcement (doc
//! §9.2, §12.4).

use serde::{Deserialize, Serialize};

use crate::batch::ExecuteBatch;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CostEstimate {
    /// USD that the call is *expected* to consume given input + likely
    /// output sizes. For local runtimes this is a per-token amortized
    /// figure; for remote runtimes it's `tokens × $/Mtok`.
    pub usd: f64,
    /// Tokens-in budget reserved for the call.
    pub input_tokens: u32,
    /// Tokens-out budget reserved.
    pub output_tokens_max: u32,
}

pub trait EstimateCost {
    fn estimate(&self, batch: &ExecuteBatch) -> CostEstimate;
}

/// Build a cost estimate from per-million-token rates.
pub fn from_rates(input_per_mtok_usd: f64, output_per_mtok_usd: f64, batch: &ExecuteBatch) -> CostEstimate {
    let in_t = batch.estimated_tokens();
    // No information here about the split between input and output, so
    // assume the documented heuristic: 80 / 20.
    let in_share = ((in_t as f64) * 0.8) as u32;
    let out_share = in_t - in_share;
    let usd = (in_share as f64) * input_per_mtok_usd / 1_000_000.0
        + (out_share as f64) * output_per_mtok_usd / 1_000_000.0;
    CostEstimate {
        usd,
        input_tokens: in_share,
        output_tokens_max: out_share,
    }
}
