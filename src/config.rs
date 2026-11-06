use std::path::Path;

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use serde::Deserialize;

/// How much the model thinks per token. Each level is a loop count taken from the
/// model's own config, and one loop is about one step of reasoning. `reasoning_effort`
/// values from the OpenAI API that have no level of their own are aliases here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Think {
    #[value(alias = "none", alias = "minimal")]
    Low,
    Medium,
    High,
    #[value(alias = "xhigh")]
    Max,
}

/// `config.json` next to the weights, as written by tools/convert.py.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    #[serde(default = "default_kv_heads")]
    pub global_kv_heads: usize,
    #[serde(default)]
    pub global_rope_base: f32,
    #[serde(default)]
    pub global_logn: u32,
    #[serde(default)]
    pub global_kshift: u32,
    #[serde(default)]
    pub global_emb: u32,
    #[serde(default)]
    pub global_far: u32,
    pub prelude: String,
    pub core: String,
    pub coda: String,
    pub window: usize,
    pub max_loops: usize,
    pub default_loops: usize,
    #[serde(default)]
    pub loop_lora: u32,
    #[serde(default)]
    pub state_loops: u32,
    #[serde(default = "default_damp")]
    pub damp: f32,
    pub rope_base: f32,
    pub norm_eps: f32,
}

fn default_kv_heads() -> usize {
    4
}

fn default_damp() -> f32 {
    1.0
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if cfg.loop_lora != 0 {
            bail!("loop_lora models are not supported");
        }
        if cfg.state_loops != 0 {
            bail!("state_loops models are not supported");
        }
        if cfg.global_rope_base != 0.0 {
            bail!("global_rope_base is not supported");
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Half the default, the default, twice the default and the most the model allows.
    pub fn think_loops(&self, level: Think) -> usize {
        let d = self.default_loops;
        match level {
            Think::Low => (d / 2).max(1),
            Think::Medium => d,
            Think::High => (d * 2).min(self.max_loops),
            Think::Max => self.max_loops,
        }
    }

    pub fn loops(&self, n: Option<usize>) -> Result<usize> {
        if self.core.is_empty() {
            return Ok(0);
        }
        let n = n.unwrap_or(self.default_loops);
        if !(1..=self.max_loops).contains(&n) {
            bail!("loops must be between 1 and {}", self.max_loops);
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(default_loops: usize, max_loops: usize) -> Config {
        let text = std::fs::read_to_string("tests/fixtures/tiny/config.json").unwrap();
        Config {
            default_loops,
            max_loops,
            ..serde_json::from_str(&text).unwrap()
        }
    }

    fn levels(c: &Config) -> [usize; 4] {
        [Think::Low, Think::Medium, Think::High, Think::Max].map(|t| c.think_loops(t))
    }

    #[test]
    fn levels_follow_the_model() {
        assert_eq!(levels(&config(4, 16)), [2, 4, 8, 16]);
        assert_eq!(levels(&config(3, 6)), [1, 3, 6, 6]);
        assert_eq!(levels(&config(1, 2)), [1, 1, 2, 2]);
    }
}
