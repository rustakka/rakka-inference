//! Phoneme → id lookup.
//!
//! Piper's ONNX graph takes a flat sequence of phoneme ids. This
//! module owns the lookup table built from the voice manifest's
//! `phoneme_id_map` and converts an input string into the id vector
//! that the session consumes.
//!
//! M4 scope: char-level lookup (each Unicode grapheme is consulted
//! independently). This is correct when callers feed already-
//! phonemized IPA (e.g. the output of `espeak-ng -q -x --ipa`). Real
//! text → IPA via espeak-ng FFI is a documented follow-up; swap in an
//! espeak adapter that emits IPA before [`PhonemeMap::ids_for_text`].

use std::collections::BTreeMap;

use crate::error::PiperError;

/// Standard Piper boundary phonemes — the IPA `_` (pad/separator) and
/// `^` (begin-of-sentence) / `$` (end-of-sentence). The runner wraps
/// every utterance in BOS/EOS so the model gets the punctuation
/// envelope it was trained against.
pub const BOS: &str = "^";
pub const EOS: &str = "$";
pub const PAD: &str = "_";

/// Built from [`crate::config::PiperVoiceManifest::phoneme_id_map`].
#[derive(Debug, Clone)]
pub struct PhonemeMap {
    inner: BTreeMap<String, Vec<i64>>,
}

impl PhonemeMap {
    pub fn new(inner: BTreeMap<String, Vec<i64>>) -> Self {
        Self { inner }
    }

    /// Convert `text` into a flat `Vec<i64>` of phoneme ids ready for
    /// the ONNX session input.
    ///
    /// The output is wrapped in BOS / EOS and PAD-interleaved (the
    /// shape Piper expects), if the corresponding entries are present
    /// in the voice's id map. Unknown characters yield
    /// [`PiperError::UnknownPhoneme`].
    pub fn ids_for_text(&self, text: &str) -> Result<Vec<i64>, PiperError> {
        let mut out: Vec<i64> = Vec::with_capacity(text.len() * 2 + 4);
        let pad = self.inner.get(PAD).cloned();
        if let Some(bos) = self.inner.get(BOS) {
            out.extend_from_slice(bos);
            if let Some(p) = &pad {
                out.extend_from_slice(p);
            }
        }
        for grapheme in text.chars() {
            let key = grapheme.to_string();
            let ids = self
                .inner
                .get(&key)
                .ok_or_else(|| PiperError::UnknownPhoneme { phoneme: key.clone() })?;
            out.extend_from_slice(ids);
            if let Some(p) = &pad {
                out.extend_from_slice(p);
            }
        }
        if let Some(eos) = self.inner.get(EOS) {
            out.extend_from_slice(eos);
        }
        Ok(out)
    }

    /// Lookup by raw key (test helper).
    pub fn get(&self, key: &str) -> Option<&[i64]> {
        self.inner.get(key).map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> PhonemeMap {
        let mut m = BTreeMap::new();
        m.insert("_".into(), vec![0]);
        m.insert("^".into(), vec![1]);
        m.insert("$".into(), vec![2]);
        m.insert("h".into(), vec![10]);
        m.insert("e".into(), vec![11]);
        m.insert("l".into(), vec![12]);
        m.insert("o".into(), vec![13]);
        PhonemeMap::new(m)
    }

    #[test]
    fn wraps_in_bos_pad_eos_and_interleaves_pad() {
        let ids = map().ids_for_text("he").unwrap();
        // ^ _ h _ e _ $
        assert_eq!(ids, vec![1, 0, 10, 0, 11, 0, 2]);
    }

    #[test]
    fn rejects_unknown_phoneme_cleanly() {
        let err = map().ids_for_text("hex").unwrap_err();
        match err {
            PiperError::UnknownPhoneme { phoneme } => assert_eq!(phoneme, "x"),
            other => panic!("expected UnknownPhoneme, got {other:?}"),
        }
    }

    #[test]
    fn empty_text_yields_only_envelope() {
        let ids = map().ids_for_text("").unwrap();
        assert_eq!(ids, vec![1, 0, 2]);
    }

    #[test]
    fn missing_envelope_keys_are_optional() {
        let mut m = BTreeMap::new();
        m.insert("h".into(), vec![10]);
        m.insert("i".into(), vec![14]);
        let p = PhonemeMap::new(m);
        let ids = p.ids_for_text("hi").unwrap();
        assert_eq!(ids, vec![10, 14]);
    }
}
