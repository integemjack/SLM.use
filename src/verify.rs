//! Compares decoding against logits saved from the PyTorch reference.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use safetensors::SafeTensors;

use crate::model::Model;
use crate::ops::Ops;

pub struct Report {
    pub steps: usize,
    pub checked: usize,
    pub max_diff: f32,
    pub same_argmax: bool,
}

fn floats(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).map_err(|e| anyhow!("{name}: {e}"))?;
    Ok(t.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn argmax(l: &[f32]) -> usize {
    l.iter()
        .enumerate()
        .fold(0, |b, (i, x)| if *x > l[b] { i } else { b })
}

/// `expected.safetensors` holds `tokens`, `loops`, `positions` and `logits` (positions × vocab).
pub fn run<O: Ops>(model: &Model<O>, dir: &Path) -> Result<Report> {
    let bytes = std::fs::read(dir.join("expected.safetensors"))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let tokens = floats(&st, "tokens")?;
    let loops = floats(&st, "loops")?[0] as usize;
    let positions: Vec<usize> = floats(&st, "positions")?
        .iter()
        .map(|p| *p as usize)
        .collect();
    let want = floats(&st, "logits")?;
    let v = model.cfg.vocab_size;
    if want.len() != positions.len() * v {
        bail!("expected logits do not match positions × vocab");
    }
    let mut se = model.session(Some(loops))?;
    let (mut max_diff, mut same_argmax, mut next) = (0f32, true, 0);
    for (t, tok) in tokens.iter().enumerate() {
        model.step(&mut se, *tok as u32)?;
        if next < positions.len() && positions[next] == t {
            let got = model.logits(&se)?;
            let exp = &want[next * v..(next + 1) * v];
            for (a, b) in got.iter().zip(exp) {
                max_diff = max_diff.max((a - b).abs());
            }
            same_argmax &= argmax(&got) == argmax(exp);
            next += 1;
        }
    }
    Ok(Report {
        steps: tokens.len(),
        checked: next,
        max_diff,
        same_argmax,
    })
}
