//! Probe an `ort::Session` to figure out what kind of model it is and
//! what inputs the generate loop needs to feed.
//!
//! ONNX exports vary wildly: `past_key_values.0.key` vs `past.0.key`
//! vs `past_key_values_0`; outputs are `present.0.key` or
//! `present_0_key`. We tolerate the variants by matching names with a
//! regex and pairing layers by captured index.

use ort::session::Session;
use ort::value::TensorElementType;

/// What shape of model `Session` we're driving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelKind {
    /// Causal LM with KV-cache inputs (`past_key_values.{i}.{key,value}`).
    /// Generation loop feeds `present.{i}.{...}` outputs back as
    /// `past_key_values.{i}.{...}` inputs.
    TextGenWithKv,
    /// Causal LM without explicit KV-cache (re-runs the prompt each
    /// step). Slower but supported.
    TextGenStateless,
    /// Encoder-only (BERT/embedding/reranker/Whisper-encoder). Chat
    /// `execute()` returns BadRequest; callers should use `infer()`.
    EncoderOnly,
    /// Couldn't identify — `execute()` will refuse with the probed
    /// shape echoed so the operator can debug.
    Unknown,
}

#[derive(Debug, Clone)]
pub(crate) struct KvLayer {
    pub past_key_input: String,
    pub past_value_input: String,
    pub present_key_output: String,
    pub present_value_output: String,
}

#[derive(Debug, Clone)]
pub(crate) struct Topology {
    pub kind: ModelKind,
    pub input_ids_name: Option<String>,
    pub attention_mask_name: Option<String>,
    pub position_ids_name: Option<String>,
    pub logits_name: Option<String>,
    pub logits_dtype: Option<TensorElementType>,
    pub kv_layers: Vec<KvLayer>,
    /// Echo of all input names — used in error messages when probing
    /// fails so the operator can see what the model actually exposes.
    pub all_input_names: Vec<String>,
    pub all_output_names: Vec<String>,
}

impl Topology {
    pub(crate) fn probe(session: &Session) -> Self {
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        let outputs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();

        let input_ids_name = pick_first(&inputs, &["input_ids", "inputs", "tokens"]);
        let attention_mask_name = pick_first(&inputs, &["attention_mask", "mask"]);
        let position_ids_name = pick_first(&inputs, &["position_ids", "positions"]);
        let logits_name = pick_first(&outputs, &["logits", "output", "last_hidden_state"]);

        let logits_dtype = logits_name
            .as_deref()
            .and_then(|name| session.outputs().iter().find(|o| o.name() == name))
            .and_then(|o| match o.dtype() {
                ort::value::ValueType::Tensor { ty, .. } => Some(*ty),
                _ => None,
            });

        let past_re = regex::Regex::new(r"^(?:past_key_values|past)[._]?(\d+)[._]?(key|value)$").unwrap();
        let present_re =
            regex::Regex::new(r"^(?:present|new_key_values|present_key_values)[._]?(\d+)[._]?(key|value)$")
                .unwrap();

        let kv_layers = pair_kv_layers(&inputs, &outputs, &past_re, &present_re);

        let kind = if !kv_layers.is_empty() {
            ModelKind::TextGenWithKv
        } else if input_ids_name.is_some() && logits_name.as_deref() == Some("logits") {
            ModelKind::TextGenStateless
        } else if input_ids_name.is_some() && logits_name.is_some() {
            ModelKind::EncoderOnly
        } else {
            ModelKind::Unknown
        };

        Self {
            kind,
            input_ids_name,
            attention_mask_name,
            position_ids_name,
            logits_name,
            logits_dtype,
            kv_layers,
            all_input_names: inputs.iter().map(|s| (*s).to_owned()).collect(),
            all_output_names: outputs.iter().map(|s| (*s).to_owned()).collect(),
        }
    }
}

fn pick_first(haystack: &[&str], needles: &[&str]) -> Option<String> {
    for n in needles {
        if let Some(found) = haystack.iter().find(|h| h.eq_ignore_ascii_case(n)) {
            return Some((*found).to_owned());
        }
    }
    None
}

fn pair_kv_layers(
    inputs: &[&str],
    outputs: &[&str],
    past_re: &regex::Regex,
    present_re: &regex::Regex,
) -> Vec<KvLayer> {
    use std::collections::BTreeMap;

    // index → (key_name, value_name) on the input side
    let mut past_by_layer: BTreeMap<u32, (Option<String>, Option<String>)> = BTreeMap::new();
    for name in inputs {
        if let Some(c) = past_re.captures(name) {
            let idx: u32 = c[1].parse().unwrap_or(0);
            let entry = past_by_layer.entry(idx).or_default();
            match &c[2] {
                "key" => entry.0 = Some((*name).to_owned()),
                "value" => entry.1 = Some((*name).to_owned()),
                _ => {}
            }
        }
    }
    let mut present_by_layer: BTreeMap<u32, (Option<String>, Option<String>)> = BTreeMap::new();
    for name in outputs {
        if let Some(c) = present_re.captures(name) {
            let idx: u32 = c[1].parse().unwrap_or(0);
            let entry = present_by_layer.entry(idx).or_default();
            match &c[2] {
                "key" => entry.0 = Some((*name).to_owned()),
                "value" => entry.1 = Some((*name).to_owned()),
                _ => {}
            }
        }
    }

    past_by_layer
        .into_iter()
        .filter_map(|(idx, (pk, pv))| {
            let (sk, sv) = present_by_layer.remove(&idx)?;
            Some(KvLayer {
                past_key_input: pk?,
                past_value_input: pv?,
                present_key_output: sk?,
                present_value_output: sv?,
            })
        })
        .collect()
}
