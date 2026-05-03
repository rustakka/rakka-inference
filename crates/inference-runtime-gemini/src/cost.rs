use std::collections::HashMap;

use once_cell::sync::Lazy;

#[derive(Debug, Clone, Copy)]
pub struct ModelPrice {
    pub input_per_mtok_usd: f64,
    pub output_per_mtok_usd: f64,
}

pub struct GeminiPricing {
    table: HashMap<&'static str, ModelPrice>,
}

impl GeminiPricing {
    pub fn published() -> &'static Self {
        &PRICING
    }
    pub fn get(&self, model: &str) -> Option<ModelPrice> {
        self.table.get(model).copied()
    }
}

static PRICING: Lazy<GeminiPricing> = Lazy::new(|| {
    let mut t = HashMap::new();
    t.insert(
        "gemini-2.0-pro",
        ModelPrice {
            input_per_mtok_usd: 1.25,
            output_per_mtok_usd: 5.0,
        },
    );
    t.insert(
        "gemini-1.5-pro",
        ModelPrice {
            input_per_mtok_usd: 1.25,
            output_per_mtok_usd: 5.0,
        },
    );
    t.insert(
        "gemini-1.5-flash",
        ModelPrice {
            input_per_mtok_usd: 0.075,
            output_per_mtok_usd: 0.30,
        },
    );
    t.insert(
        "gemini-2.0-flash",
        ModelPrice {
            input_per_mtok_usd: 0.075,
            output_per_mtok_usd: 0.30,
        },
    );
    GeminiPricing { table: t }
});
