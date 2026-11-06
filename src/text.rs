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
