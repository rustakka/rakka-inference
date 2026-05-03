use std::collections::HashMap;

use once_cell::sync::Lazy;

#[derive(Debug, Clone, Copy)]
pub struct ModelPrice {
    pub input_per_mtok_usd: f64,
    pub output_per_mtok_usd: f64,
}

pub struct AnthropicPricing {
    table: HashMap<&'static str, ModelPrice>,
}

impl AnthropicPricing {
    pub fn published() -> &'static Self {
        &PRICING
    }
    pub fn get(&self, model: &str) -> Option<ModelPrice> {
        self.table.get(model).copied()
    }
}

static PRICING: Lazy<AnthropicPricing> = Lazy::new(|| {
    let mut t = HashMap::new();
    // List pricing as of doc snapshot.
    t.insert(
        "claude-opus-4",
        ModelPrice {
            input_per_mtok_usd: 15.0,
            output_per_mtok_usd: 75.0,
        },
    );
    t.insert(
        "claude-sonnet-4",
        ModelPrice {
            input_per_mtok_usd: 3.0,
            output_per_mtok_usd: 15.0,
        },
    );
    t.insert(
        "claude-3-5-sonnet",
        ModelPrice {
            input_per_mtok_usd: 3.0,
            output_per_mtok_usd: 15.0,
        },
    );
    t.insert(
        "claude-3-5-haiku",
        ModelPrice {
            input_per_mtok_usd: 0.80,
            output_per_mtok_usd: 4.0,
        },
    );
    t.insert(
        "claude-3-haiku",
        ModelPrice {
            input_per_mtok_usd: 0.25,
            output_per_mtok_usd: 1.25,
        },
    );
    AnthropicPricing { table: t }
});
