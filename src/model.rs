//! Token-by-token decoding of the Looped Hybrid Reasoner.
//!
//! ```text
//! token → embedding → prelude → e
//! s_0 = e
//! s_r = core(s_{r-1} + W_e·e + loop_embed[r])    r = 1 .. R
//! → coda → logits
//! ```
//!
//! Each loop keeps its own state for the core blocks. Sliding-window caches are ring
//! buffers: attention does not depend on slot order, because positions are already
//! encoded into the keys.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use safetensors::{Dtype, SafeTensors};

use crate::config::Config;
use crate::ops::{at, Attn, Ops};

struct Weights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("model.safetensors");
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            if view.dtype() != Dtype::F32 {
                bail!("{name}: expected f32 weights");
            }
            let data = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.insert(name, (view.shape().to_vec(), data));
        }
        Ok(Self { tensors })
    }

    fn take(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        self.tensors
            .remove(name)
            .ok_or_else(|| anyhow!("missing weight {name}"))
    }
}

struct Mat<B> {
    w: B,
    rows: usize,
    cols: usize,
}

fn mat<O: Ops>(ops: &O, w: &mut Weights, name: &str) -> Result<Mat<O::Buf>> {
    let (shape, data) = w.take(name)?;
    if shape.len() != 2 {
        bail!("{name}: expected a matrix");
    }
    Ok(Mat {
        w: ops.upload(&data)?,
        rows: shape[0],
        cols: shape[1],
    })
}

fn vector<O: Ops>(ops: &O, w: &mut Weights, name: &str) -> Result<O::Buf> {
    ops.upload(&w.take(name)?.1)
}

enum Mixer<B> {
    Window {
        qkv: Mat<B>,
        o: Mat<B>,
    },
    Linear {
        qkvg: Mat<B>,
        decay: Mat<B>,
        bias: B,
        head_norm: B,
        o: Mat<B>,
    },
    Global {
        q: Mat<B>,
        kv: Mat<B>,
        o: Mat<B>,
        norm_e: Option<B>,
    },
}

struct Block<B> {
    norm1: B,
    norm2: B,
    up: Mat<B>,
    down: Mat<B>,
    mixer: Mixer<B>,
}

impl<B> Block<B> {
    fn load<O: Ops<Buf = B>>(
        ops: &O,
        w: &mut Weights,
        prefix: &str,
        kind: char,
        cfg: &Config,
    ) -> Result<Self> {
        let n = |s: &str| format!("{prefix}.{s}");
        let mixer = match kind {
            'S' => Mixer::Window {
                qkv: mat(ops, w, &n("mixer.qkv.weight"))?,
                o: mat(ops, w, &n("mixer.o.weight"))?,
            },
            'L' => Mixer::Linear {
                qkvg: mat(ops, w, &n("mixer.qkvg.weight"))?,
                decay: mat(ops, w, &n("mixer.decay.weight"))?,
                bias: vector(ops, w, &n("mixer.decay.bias"))?,
                head_norm: vector(ops, w, &n("mixer.head_norm.weight"))?,
                o: mat(ops, w, &n("mixer.o.weight"))?,
            },
            'G' => Mixer::Global {
                q: mat(ops, w, &n("mixer.q.weight"))?,
                kv: mat(ops, w, &n("mixer.kv.weight"))?,
                o: mat(ops, w, &n("mixer.o.weight"))?,
                norm_e: if cfg.global_emb != 0 {
                    Some(vector(ops, w, &n("mixer.norm_e.weight"))?)
                } else {
                    None
                },
            },
            k => bail!("unknown layer type {k}"),
        };
        Ok(Self {
            norm1: vector(ops, w, &n("norm1.weight"))?,
            norm2: vector(ops, w, &n("norm2.weight"))?,
            up: mat(ops, w, &n("mlp.up.weight"))?,
            down: mat(ops, w, &n("mlp.down.weight"))?,
            mixer,
        })
    }
}

enum State<B> {
    Window {
        k: B,
        v: B,
    },
    Linear {
        s: B,
    },
    Global {
        k: B,
        v: B,
        prev: B,
        len: usize,
        cap: usize,
        scores: B,
    },
}

/// Buffers reused by every block, so a step allocates nothing.
struct Scratch<B> {
    n: B,
    mix: B,
    q: B,
    k: B,
    y: B,
    y2: B,
    h: B,
    t: B,
    d: B,
    prev_kv: B,
    up: B,
    mid: B,
    scores: B,
    cos: B,
    sin: B,
}

pub struct Model<O: Ops> {
    ops: O,
    pub cfg: Config,
    embed: Mat<O::Buf>,
    prelude: Vec<Block<O::Buf>>,
    core: Vec<Block<O::Buf>>,
    coda: Vec<Block<O::Buf>>,
    inject: Option<Mat<O::Buf>>,
    loop_embed: Option<Mat<O::Buf>>,
    norm_f: O::Buf,
    inv_freq: O::Buf,
}

/// Decoding position and state for a fixed number of loops.
pub struct Session<O: Ops> {
    pub loops: usize,
    pub pos: usize,
    prelude: Vec<State<O::Buf>>,
    core: Vec<Vec<State<O::Buf>>>,
    coda: Vec<State<O::Buf>>,
    sc: Scratch<O::Buf>,
    x: O::Buf,
    s: O::Buf,
    e0: O::Buf,
    e_proj: O::Buf,
    logits: O::Buf,
}

impl<O: Ops> Model<O> {
    pub fn load(ops: O, dir: &Path) -> Result<Self> {
        let cfg = Config::load(dir)?;
        let mut w = Weights::load(dir)?;
        let blocks = |w: &mut Weights, name: &str, kinds: &str| -> Result<Vec<Block<O::Buf>>> {
            kinds
                .chars()
                .enumerate()
                .map(|(i, k)| Block::load(&ops, w, &format!("{name}.{i}"), k, &cfg))
                .collect()
        };
        let prelude = blocks(&mut w, "prelude", &cfg.prelude)?;
        let core = blocks(&mut w, "core", &cfg.core)?;
        let coda = blocks(&mut w, "coda", &cfg.coda)?;
        let looped = !cfg.core.is_empty();
        let inject = if looped {
            Some(mat(&ops, &mut w, "inject_e.weight")?)
        } else {
            None
        };
        let loop_embed = if looped {
            Some(mat(&ops, &mut w, "loop_embed.weight")?)
        } else {
            None
        };
        let embed = mat(&ops, &mut w, "embed.weight")?;
        let norm_f = vector(&ops, &mut w, "norm_f.weight")?;
        let dh = cfg.head_dim();
        let inv: Vec<f32> = (0..dh / 2)
            .map(|i| 1.0 / cfg.rope_base.powf((2 * i) as f32 / dh as f32))
            .collect();
        let inv_freq = ops.upload(&inv)?;
        Ok(Self {
            embed,
            prelude,
            core,
            coda,
            inject,
            loop_embed,
            norm_f,
            inv_freq,
            cfg,
            ops,
        })
    }

    pub fn device(&self) -> String {
        self.ops.name()
    }

    pub fn params(&self) -> usize {
        let m = |x: &Mat<O::Buf>| x.rows * x.cols;
        let c = &self.cfg;
        let block = |b: &Block<O::Buf>| {
            let mixer = match &b.mixer {
                Mixer::Window { qkv, o } => m(qkv) + m(o),
                Mixer::Linear { qkvg, decay, o, .. } => {
                    m(qkvg) + m(decay) + m(o) + c.n_heads + c.head_dim()
                }
                Mixer::Global { q, kv, o, norm_e } => {
                    m(q) + m(kv) + m(o) + norm_e.as_ref().map_or(0, |_| c.d_model)
                }
            };
            mixer + 2 * c.d_model + m(&b.up) + m(&b.down)
        };
        let blocks: usize = self
            .prelude
            .iter()
            .chain(&self.core)
            .chain(&self.coda)
            .map(block)
            .sum();
        blocks
            + m(&self.embed)
            + c.d_model
            + self.inject.as_ref().map_or(0, m)
            + self.loop_embed.as_ref().map_or(0, m)
    }

    fn new_state(&self, b: &Block<O::Buf>) -> Result<State<O::Buf>> {
        let (c, ops) = (&self.cfg, &self.ops);
        let (h, dh) = (c.n_heads, c.head_dim());
        Ok(match b.mixer {
            Mixer::Window { .. } => State::Window {
                k: ops.zeros_cache(h, c.window, dh)?,
                v: ops.zeros_cache(h, c.window, dh)?,
            },
            Mixer::Linear { .. } => State::Linear {
                s: ops.zeros(h * dh * dh)?,
            },
            Mixer::Global { .. } => {
                let cap = 1 << 13;
                let kh = c.global_kv_heads;
                State::Global {
                    k: ops.zeros_cache(kh, cap, dh)?,
                    v: ops.zeros_cache(kh, cap, dh)?,
                    prev: ops.zeros(c.d_model)?,
                    len: 0,
                    cap,
                    scores: ops.zeros(h * cap)?,
                }
            }
        })
    }

    pub fn session(&self, loops: Option<usize>) -> Result<Session<O>> {
        let loops = self.cfg.loops(loops)?;
        let ops = &self.ops;
        let d = self.cfg.d_model;
        let states = |blocks: &[Block<O::Buf>]| -> Result<Vec<State<O::Buf>>> {
            blocks.iter().map(|b| self.new_state(b)).collect()
        };
        let mut core = Vec::with_capacity(loops);
        for _ in 0..loops {
            core.push(states(&self.core)?);
        }
        let z = |n: usize| ops.zeros(n);
        let half = self.cfg.head_dim() / 2;
        let sc = Scratch {
            n: z(d)?,
            mix: z(4 * d)?,
            q: z(d)?,
            k: z(d)?,
            y: z(d)?,
            y2: z(d)?,
            h: z(d)?,
            t: z(d)?,
            d: z(self.cfg.n_heads)?,
            prev_kv: z(2 * d)?,
            up: z(4 * d)?,
            mid: z(4 * d)?,
            scores: z(self.cfg.n_heads * self.cfg.window)?,
            cos: z(half)?,
            sin: z(half)?,
        };
        Ok(Session {
            loops,
            pos: 0,
            prelude: states(&self.prelude)?,
            core,
            coda: states(&self.coda)?,
            sc,
            x: z(d)?,
            s: z(d)?,
            e0: z(d)?,
            e_proj: z(d)?,
            logits: z(self.cfg.vocab_size)?,
        })
    }

    /// Doubles a global cache, which moves every head's data.
    fn grow(
        &self,
        k: &mut O::Buf,
        v: &mut O::Buf,
        scores: &mut O::Buf,
        cap: &mut usize,
    ) -> Result<()> {
        let (kh, dh) = (self.cfg.global_kv_heads, self.cfg.head_dim());
        let (old, new) = (*cap, *cap * 2);
        for buf in [k, v] {
            let data = self.ops.read(buf, kh * old * dh)?;
            let mut out = vec![0f32; kh * new * dh];
            for h in 0..kh {
                out[h * new * dh..(h * new + old) * dh]
                    .copy_from_slice(&data[h * old * dh..(h + 1) * old * dh]);
            }
            *buf = self.ops.upload_cache(&out, kh, new, dh)?;
        }
        *scores = self.ops.zeros(self.cfg.n_heads * new)?;
        *cap = new;
        Ok(())
    }

    fn block(
        &self,
        b: &Block<O::Buf>,
        st: &mut State<O::Buf>,
        x: &mut O::Buf,
        e0: &O::Buf,
        sc: &mut Scratch<O::Buf>,
        pos: usize,
    ) -> Result<()> {
        let ops = &self.ops;
        let c = &self.cfg;
        let (d, heads, dh, eps) = (c.d_model, c.n_heads, c.head_dim(), c.norm_eps);
        let scale = 1.0 / (dh as f32).sqrt();
        ops.rms(&mut sc.n, at(x, 0), &b.norm1, 1, d, eps)?;
        let mut has_h = true;
        match (&b.mixer, st) {
            (Mixer::Window { qkv, o }, State::Window { k, v }) => {
                ops.linear(&mut sc.mix, &qkv.w, qkv.rows, d, at(&sc.n, 0))?;
                ops.rope(&mut sc.q, at(&sc.mix, 0), &sc.cos, &sc.sin, heads, dh)?;
                ops.rope(&mut sc.k, at(&sc.mix, d), &sc.cos, &sc.sin, heads, dh)?;
                let slot = pos % c.window;
                ops.store(k, at(&sc.k, 0), slot, c.window, heads, dh)?;
                ops.store(v, at(&sc.mix, 2 * d), slot, c.window, heads, dh)?;
                let a = Attn {
                    n: (pos + 1).min(c.window),
                    cap: c.window,
                    heads,
                    key_heads: heads,
                    dh,
                    scale,
                };
                ops.attend(&mut sc.y, &mut sc.scores, at(&sc.q, 0), k, v, a)?;
                ops.linear(&mut sc.h, &o.w, d, d, at(&sc.y, 0))?;
            }
            (
                Mixer::Linear {
                    qkvg,
                    decay,
                    bias,
                    head_norm,
                    o,
                },
                State::Linear { s },
            ) => {
                ops.linear(&mut sc.mix, &qkvg.w, qkvg.rows, d, at(&sc.n, 0))?;
                ops.l2norm(&mut sc.q, at(&sc.mix, 0), heads, dh)?;
                ops.l2norm(&mut sc.k, at(&sc.mix, d), heads, dh)?;
                ops.linear(&mut sc.d, &decay.w, heads, d, at(&sc.n, 0))?;
                ops.gla_update(s, at(&sc.k, 0), at(&sc.mix, 2 * d), &sc.d, bias, heads, dh)?;
                ops.gla_read(&mut sc.y, at(&sc.q, 0), s, heads, dh)?;
                ops.rms(&mut sc.y2, at(&sc.y, 0), head_norm, heads, dh, eps)?;
                ops.silu_mul(&mut sc.y, at(&sc.mix, 3 * d), at(&sc.y2, 0), d)?;
                ops.linear(&mut sc.h, &o.w, d, d, at(&sc.y, 0))?;
            }
            (
                Mixer::Global { q, kv, o, norm_e },
                State::Global {
                    k,
                    v,
                    prev,
                    len,
                    cap,
                    scores,
                },
            ) => {
                if let Some(w) = norm_e {
                    ops.rms(&mut sc.t, at(e0, 0), w, 1, d, eps)?;
                    ops.add(&mut sc.y, at(&sc.n, 0), at(&sc.t, 0), d)?;
                    std::mem::swap(&mut sc.n, &mut sc.y);
                }
                let (kh, kd) = (c.global_kv_heads, c.global_kv_heads * dh);
                ops.linear(&mut sc.q, &q.w, d, d, at(&sc.n, 0))?;
                ops.linear(&mut sc.mix, &kv.w, 2 * kd, d, at(&sc.n, 0))?;
                if c.global_logn != 0 {
                    let f = (pos as f32).ln_1p() / (c.global_logn as f32).ln();
                    ops.scale(&mut sc.y, at(&sc.q, 0), d, f)?;
                    std::mem::swap(&mut sc.q, &mut sc.y);
                }
                if *len == *cap {
                    self.grow(k, v, scores, cap)?;
                }
                if c.global_kshift != 0 {
                    ops.linear(&mut sc.prev_kv, &kv.w, 2 * kd, d, at(prev, 0))?;
                    ops.store(k, at(&sc.prev_kv, 0), *len, *cap, kh, dh)?;
                    ops.copy(prev, at(&sc.n, 0), d)?;
                } else {
                    ops.store(k, at(&sc.mix, 0), *len, *cap, kh, dh)?;
                }
                ops.store(v, at(&sc.mix, kd), *len, *cap, kh, dh)?;
                *len += 1;
                let far = if c.global_far != 0 { c.window } else { 0 };
                if *len > far {
                    let a = Attn {
                        n: *len - far,
                        cap: *cap,
                        heads,
                        key_heads: kh,
                        dh,
                        scale,
                    };
                    ops.attend(&mut sc.y, scores, at(&sc.q, 0), k, v, a)?;
                    ops.linear(&mut sc.h, &o.w, d, d, at(&sc.y, 0))?;
                } else {
                    has_h = false;
                }
            }
            _ => bail!("state does not match its block"),
        }
        if has_h {
            ops.add(&mut sc.t, at(x, 0), at(&sc.h, 0), d)?;
            std::mem::swap(x, &mut sc.t);
        }
        ops.rms(&mut sc.n, at(x, 0), &b.norm2, 1, d, eps)?;
        let m = b.down.cols;
        ops.linear(&mut sc.up, &b.up.w, 2 * m, d, at(&sc.n, 0))?;
        ops.silu_mul(&mut sc.mid, at(&sc.up, 0), at(&sc.up, m), m)?;
        ops.linear(&mut sc.h, &b.down.w, d, m, at(&sc.mid, 0))?;
        ops.add(&mut sc.t, at(x, 0), at(&sc.h, 0), d)?;
        std::mem::swap(x, &mut sc.t);
        Ok(())
    }

    /// Feeds one token and leaves its logits in the session.
    pub fn step(&self, se: &mut Session<O>, token: u32) -> Result<()> {
        let ops = &self.ops;
        let d = self.cfg.d_model;
        let pos = se.pos;
        let half = self.cfg.head_dim() / 2;
        ops.rope_table(
            &mut se.sc.cos,
            &mut se.sc.sin,
            &self.inv_freq,
            pos as f32,
            half,
        )?;
        ops.copy(&mut se.e0, at(&self.embed.w, token as usize * d), d)?;
        ops.copy(&mut se.x, at(&se.e0, 0), d)?;
        for (b, st) in self.prelude.iter().zip(&mut se.prelude) {
            self.block(b, st, &mut se.x, &se.e0, &mut se.sc, pos)?;
        }
        if let (Some(inject), Some(loop_embed)) = (&self.inject, &self.loop_embed) {
            ops.linear(&mut se.e_proj, &inject.w, d, d, at(&se.x, 0))?;
            ops.copy(&mut se.s, at(&se.x, 0), d)?;
            for r in 0..se.loops {
                ops.add(&mut se.sc.t, at(&se.s, 0), at(&se.e_proj, 0), d)?;
                ops.add(&mut se.x, at(&se.sc.t, 0), at(&loop_embed.w, r * d), d)?;
                for (b, st) in self.core.iter().zip(&mut se.core[r]) {
                    self.block(b, st, &mut se.x, &se.e0, &mut se.sc, pos)?;
                }
                if self.cfg.damp == 1.0 {
                    std::mem::swap(&mut se.s, &mut se.x);
                } else {
                    ops.mix(&mut se.sc.t, at(&se.s, 0), at(&se.x, 0), d, self.cfg.damp)?;
                    std::mem::swap(&mut se.s, &mut se.sc.t);
                }
            }
            std::mem::swap(&mut se.x, &mut se.s);
        }
        for (b, st) in self.coda.iter().zip(&mut se.coda) {
            self.block(b, st, &mut se.x, &se.e0, &mut se.sc, pos)?;
        }
        ops.rms(
            &mut se.sc.n,
            at(&se.x, 0),
            &self.norm_f,
            1,
            d,
            self.cfg.norm_eps,
        )?;
        ops.linear(
            &mut se.logits,
            &self.embed.w,
            self.cfg.vocab_size,
            d,
            at(&se.sc.n, 0),
        )?;
        se.pos += 1;
        let mut keep = vec![&se.logits];
        for st in se
            .prelude
            .iter()
            .chain(se.core.iter().flatten())
            .chain(&se.coda)
        {
            match st {
                State::Window { k, v } => keep.extend([k, v]),
                State::Linear { s } => keep.push(s),
                State::Global { k, v, prev, .. } => keep.extend([k, v, prev]),
            }
        }
        ops.flush(&keep)
    }

    pub fn logits(&self, se: &Session<O>) -> Result<Vec<f32>> {
        self.ops.read(&se.logits, self.cfg.vocab_size)
    }

    /// Bytes held by a session's caches, without the global layer's unused capacity.
    pub fn state_bytes(&self, se: &Session<O>) -> usize {
        let c = &self.cfg;
        let dh = c.head_dim();
        let one = |st: &State<O::Buf>| match st {
            State::Window { .. } => 2 * c.n_heads * c.window * dh,
            State::Linear { .. } => c.n_heads * dh * dh,
            State::Global { len, .. } => 2 * c.global_kv_heads * len * dh + c.d_model,
        };
        let n: usize = se
            .prelude
            .iter()
            .chain(se.core.iter().flatten())
            .chain(&se.coda)
            .map(one)
            .sum();
        n * 4
    }
}
