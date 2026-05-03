//! Per-million-token pricing for the OpenAI catalog. Values are public
//! list pricing as of the doc snapshot; operators override via
//! `inference-cli` flags.

use std::collections::HashMap;

use once_cell::sync::Lazy;

#[derive(Debug, Clone, Copy)]
pub struct ModelPrice {
    pub input_per_mtok_usd: f64,
    pub output_per_mtok_usd: f64,
}

pub struct OpenAiPricing {
    table: HashMap<&'static str, ModelPrice>,
}

impl OpenAiPricing {
    pub fn published() -> &'static Self {
        &PRICING
    }

    pub fn get(&self, model: &str) -> Option<ModelPrice> {
        self.table.get(model).copied()
    }
}

static PRICING: Lazy<OpenAiPricing> = Lazy::new(|| {
    let mut t = HashMap::new();
    t.insert("gpt-4o", ModelPrice { input_per_mtok_usd: 2.50, output_per_mtok_usd: 10.00 });
    t.insert("gpt-4o-mini", ModelPrice { input_per_mtok_usd: 0.15, output_per_mtok_usd: 0.60 });
    t.insert("gpt-4-turbo", ModelPrice { input_per_mtok_usd: 10.00, output_per_mtok_usd: 30.00 });
    t.insert("o1-preview", ModelPrice { input_per_mtok_usd: 15.00, output_per_mtok_usd: 60.00 });
    t.insert("o1-mini", ModelPrice { input_per_mtok_usd: 3.00, output_per_mtok_usd: 12.00 });
    OpenAiPricing { table: t }
});
