//! Per-token cost estimates for OpenAI Realtime models.
//!
//! Pricing is in USD per **million** tokens as of the snapshot date;
//! operators may override via `inference-cli` flags. The Realtime API
//! uses a different pricing axis than chat completions (audio tokens
//! are more expensive than text tokens) — this table captures only the
//! text token rates for budget tracking, as audio token billing depends
//! on the provider's realtime billing API.

use std::collections::HashMap;

use once_cell::sync::Lazy;

/// Per-model pricing entry.
#[derive(Debug, Clone, Copy)]
pub struct RealtimeModelPrice {
    /// Input text tokens (USD / million).
    pub input_text_per_mtok_usd: f64,
    /// Output text tokens (USD / million).
    pub output_text_per_mtok_usd: f64,
    /// Input audio tokens (USD / million).
    pub input_audio_per_mtok_usd: f64,
    /// Output audio tokens (USD / million).
    pub output_audio_per_mtok_usd: f64,
}

/// Published pricing for OpenAI Realtime models.
pub struct OpenAiRealtimePricing {
    table: HashMap<&'static str, RealtimeModelPrice>,
}

impl OpenAiRealtimePricing {
    /// Return the singleton published pricing table.
    pub fn published() -> &'static Self {
        &PRICING
    }

    /// Look up pricing for `model`.  Returns `None` if the model is not
    /// in the table (treat as unknown cost → 0).
    pub fn get(&self, model: &str) -> Option<RealtimeModelPrice> {
        self.table.get(model).copied()
    }
}

static PRICING: Lazy<OpenAiRealtimePricing> = Lazy::new(|| {
    let mut t = HashMap::new();
    t.insert(
        "gpt-4o-realtime-preview",
        RealtimeModelPrice {
            input_text_per_mtok_usd: 5.00,
            output_text_per_mtok_usd: 20.00,
            input_audio_per_mtok_usd: 100.00,
            output_audio_per_mtok_usd: 200.00,
        },
    );
    t.insert(
        "gpt-4o-realtime-preview-2024-12-17",
        RealtimeModelPrice {
            input_text_per_mtok_usd: 5.00,
            output_text_per_mtok_usd: 20.00,
            input_audio_per_mtok_usd: 100.00,
            output_audio_per_mtok_usd: 200.00,
        },
    );
    t.insert(
        "gpt-4o-mini-realtime-preview",
        RealtimeModelPrice {
            input_text_per_mtok_usd: 0.60,
            output_text_per_mtok_usd: 2.40,
            input_audio_per_mtok_usd: 10.00,
            output_audio_per_mtok_usd: 20.00,
        },
    );
    OpenAiRealtimePricing { table: t }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt4o_realtime_has_price() {
        let p = OpenAiRealtimePricing::published()
            .get("gpt-4o-realtime-preview")
            .unwrap();
        assert!(p.input_audio_per_mtok_usd > 0.0);
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(OpenAiRealtimePricing::published().get("unknown-model").is_none());
    }

    #[test]
    fn mini_cheaper_than_full() {
        let full = OpenAiRealtimePricing::published()
            .get("gpt-4o-realtime-preview")
            .unwrap();
        let mini = OpenAiRealtimePricing::published()
            .get("gpt-4o-mini-realtime-preview")
            .unwrap();
        assert!(mini.output_audio_per_mtok_usd < full.output_audio_per_mtok_usd);
    }
}
