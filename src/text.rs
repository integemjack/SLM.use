//! Turning generated tokens into text as they arrive.

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

pub fn load(dir: &std::path::Path) -> Result<Tokenizer> {
    let path = dir.join("tokenizer.json");
    Tokenizer::from_file(&path).map_err(|e| anyhow!("reading {}: {e}", path.display()))
}

pub fn encode(tk: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tk
        .encode(text, false)
        .map_err(|e| anyhow!("{e}"))?
        .get_ids()
        .to_vec())
}

/// One token per character where the vocabulary has one, which is how the reasoning
/// tasks were trained. The tokenizer would merge runs of punctuation into single tokens.
pub fn encode_chars(tk: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let mut ids = Vec::with_capacity(text.len());
    let mut buf = [0; 4];
    for c in text.chars() {
        let c = &*c.encode_utf8(&mut buf);
        match tk.token_to_id(c) {
            Some(id) => ids.push(id),
            None => ids.extend(encode(tk, c)?),
        }
    }
    Ok(ids)
}

/// Holds back tokens until they form whole characters: one Chinese character can span
/// several tokens.
pub struct Detokenizer<'a> {
    tk: &'a Tokenizer,
    pending: Vec<u32>,
}

impl<'a> Detokenizer<'a> {
    pub fn new(tk: &'a Tokenizer) -> Self {
        Self {
            tk,
            pending: Vec::new(),
        }
    }

    /// Returns the text that `id` completes, which may be empty.
    pub fn push(&mut self, id: u32) -> Result<String> {
        self.pending.push(id);
        let text = self
            .tk
            .decode(&self.pending, true)
            .map_err(|e| anyhow!("{e}"))?;
        if text.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        self.pending.clear();
        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `((` merges, and a space has no token spelled as itself.
    const MERGING: &str = r#"{
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "post_processor": null, "decoder": null,
        "pre_tokenizer": {
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false
        },
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
            "ignore_merges": false,
            "vocab": { "(": 0, "((": 1, "a": 2, "Ġ": 3 },
            "merges": [["(", "("]]
        }
    }"#;

    #[test]
    fn chars_keeps_what_the_tokenizer_merges_apart() {
        let tk: Tokenizer = MERGING.parse().unwrap();
        assert_eq!(encode(&tk, "((a").unwrap(), [1, 2]);
        assert_eq!(encode_chars(&tk, "((a").unwrap(), [0, 0, 2]);
        assert_eq!(encode_chars(&tk, "( a").unwrap(), [0, 3, 2]);
    }
}
