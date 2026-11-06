//! Compares decoding against logits saved from the PyTorch reference.

use std::path::Path;

use anyhow::{anyhow, bail, Result};
use safetensors::SafeTensors;

use crate::model::{Model, Session, CHUNK};
use crate::ops::Ops;

#[derive(Debug)]
pub struct Report {
    pub steps: usize,
    pub checked: usize,
    pub max_diff: f32,
    pub same_argmax: bool,
}

struct Expected {
    tokens: Vec<u32>,
    loops: usize,
    positions: Vec<usize>,
    logits: Vec<f32>,
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
fn expected(dir: &Path, vocab: usize) -> Result<Expected> {
    let bytes = std::fs::read(dir.join("expected.safetensors"))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let e = Expected {
        tokens: floats(&st, "tokens")?.iter().map(|t| *t as u32).collect(),
        loops: floats(&st, "loops")?[0] as usize,
        positions: floats(&st, "positions")?
            .iter()
            .map(|p| *p as usize)
            .collect(),
        logits: floats(&st, "logits")?,
    };
    if e.logits.len() != e.positions.len() * vocab {
        bail!("expected logits do not match positions × vocab");
    }
    Ok(e)
}

struct Tally {
    checked: usize,
    max_diff: f32,
    same_argmax: bool,
}

impl Tally {
    fn check<O: Ops>(&mut self, model: &Model<O>, se: &Session<O>, want: &[f32]) -> Result<()> {
        let got = model.logits(se)?;
        for (a, b) in got.iter().zip(want) {
            self.max_diff = self.max_diff.max((a - b).abs());
        }
        self.same_argmax &= argmax(&got) == argmax(want);
        self.checked += 1;
        Ok(())
    }
}

/// Reads the prompt in chunks up to `split`, one of the saved positions, then decodes
/// the remaining tokens one by one. A `split` of 0 is plain decoding from the start.
pub fn run_split<O: Ops>(
    model: &Model<O>,
    dir: &Path,
    split: usize,
    chunk: usize,
) -> Result<Report> {
    let v = model.cfg.vocab_size;
    let e = expected(dir, v)?;
    let mut se = model.session(Some(e.loops))?;
    let mut tally = Tally {
        checked: 0,
        max_diff: 0.0,
        same_argmax: true,
    };
    if split > 0 {
        model.feed(&mut se, &e.tokens[..split], chunk)?;
    }
    let want = |p: usize| {
        e.positions
            .iter()
            .position(|x| *x == p)
            .map(|i| &e.logits[i * v..(i + 1) * v])
    };
    if split > 0 {
        let exp =
            want(split - 1).ok_or_else(|| anyhow!("no reference at position {}", split - 1))?;
        tally.check(model, &se, exp)?;
    }
    for t in split..e.tokens.len() {
        model.step(&mut se, e.tokens[t])?;
        if let Some(exp) = want(t) {
            tally.check(model, &se, exp)?;
        }
    }
    Ok(Report {
        steps: e.tokens.len(),
        checked: tally.checked,
        max_diff: tally.max_diff,
        same_argmax: tally.same_argmax,
    })
}

/// Checks plain decoding, then reading all but the last saved positions as a prompt.
pub fn run<O: Ops>(model: &Model<O>, dir: &Path) -> Result<(Report, Report)> {
    let e = expected(dir, model.cfg.vocab_size)?;
    let split = e.positions[e.positions.len().saturating_sub(2)] + 1;
    Ok((
        run_split(model, dir, 0, 1)?,
        run_split(model, dir, split, CHUNK)?,
    ))
}
