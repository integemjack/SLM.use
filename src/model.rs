//! The Looped Hybrid Reasoner, advancing by any number of tokens at a time.
//!
//! ```text
//! token → embedding → prelude → e
//! s_0 = e
//! s_r = core(s_{r-1} + W_e·e + loop_embed[r])    r = 1 .. R
//! → coda → logits
//! ```
//!
//! Each loop keeps its own state for the core blocks. Decoding is one token at a time
//! and a prompt goes in chunks, through the same code, so the state after a chunk is the
//! state that many single steps would leave. Sliding-window caches are rings with room
//! for a chunk beyond the window: a query near the start of a chunk still needs the
//! oldest tokens after the chunk has been stored. Attention does not depend on slot
//! order, because positions are already encoded into the keys.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use safetensors::{Dtype, SafeTensors};

use crate::config::Config;
use crate::ops::{at, Attn, Ops};

/// Most tokens advanced at once.
pub const CHUNK: usize = 256;
/// Query × slot scores a chunk may hold, which limits chunks once a global layer has
/// a long history.
const MAX_SCORES: usize = 1 << 28;

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
        cap: usize,
        scores: B,
    },
}

/// Buffers reused by every block, so advancing allocates only when a chunk grows.
struct Scratch<B> {
    n: B,
    mix: B,
    q: B,
    k: B,
    v: B,
    g: B,
    y: B,
    y2: B,
    h: B,
    t: B,
    d: B,
    shifted: B,
    prev_kv: B,
    up: B,
    gate: B,
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
    last: O::Buf,
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

    fn cap(&self) -> usize {
        self.cfg.window + CHUNK
    }

    fn new_state(&self, b: &Block<O::Buf>) -> Result<State<O::Buf>> {
        let (c, ops) = (&self.cfg, &self.ops);
        let (h, dh) = (c.n_heads, c.head_dim());
        Ok(match b.mixer {
            Mixer::Window { .. } => State::Window {
                k: ops.zeros_cache(h, self.cap(), dh)?,
                v: ops.zeros_cache(h, self.cap(), dh)?,
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
                    cap,
                    scores: ops.zeros(1)?,
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
            v: z(d)?,
            g: z(d)?,
            y: z(d)?,
            y2: z(d)?,
            h: z(d)?,
            t: z(d)?,
            d: z(self.cfg.n_heads)?,
            shifted: z(d)?,
            prev_kv: z(2 * d)?,
            up: z(4 * d)?,
            gate: z(2 * d)?,
            mid: z(2 * d)?,
            scores: z(1)?,
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
            last: z(d)?,
            logits: z(self.cfg.vocab_size)?,
        })
    }

    /// Doubles a global cache, which moves every head's data.
    fn grow(&self, k: &mut O::Buf, v: &mut O::Buf, cap: &mut usize) -> Result<()> {
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
        m: usize,
    ) -> Result<()> {
        let ops = &self.ops;
        let c = &self.cfg;
        let (d, heads, dh, eps) = (c.d_model, c.n_heads, c.head_dim(), c.norm_eps);
        let scale = 1.0 / (dh as f32).sqrt();
        let n = m * d;
        ops.rms(&mut sc.n, at(x, 0), &b.norm1, m, d, eps)?;
        // One token needs no copy to split a row: its columns are already contiguous.
        macro_rules! col {
            ($y:ident, $x:ident, $stride:expr, $off:expr, $len:expr) => {
                if m == 1 {
                    at(&sc.$x, $off)
                } else {
                    ops.columns(&mut sc.$y, &sc.$x, m, $stride, $off, $len)?;
                    at(&sc.$y, 0)
                }
            };
        }
        let mut has_h = true;
        match (&b.mixer, st) {
            (Mixer::Window { qkv, o }, State::Window { k, v }) => {
                ops.linear(&mut sc.mix, &qkv.w, qkv.rows, d, at(&sc.n, 0), m)?;
                let raw = col!(y, mix, 3 * d, 0, d);
                ops.rope(&mut sc.q, raw, &sc.cos, &sc.sin, m, heads, dh)?;
                let raw = col!(y, mix, 3 * d, d, d);
                ops.rope(&mut sc.k, raw, &sc.cos, &sc.sin, m, heads, dh)?;
                let vals = col!(v, mix, 3 * d, 2 * d, d);
                let cap = self.cap();
                ops.store(k, at(&sc.k, 0), pos, m, cap, heads, dh)?;
                ops.store(v, vals, pos, m, cap, heads, dh)?;
                let a = Attn {
                    pos,
                    m,
                    reach: c.window,
                    gap: 0,
                    cap,
                    heads,
                    key_heads: heads,
                    dh,
                    scale,
                };
                ops.attend(&mut sc.y, &mut sc.scores, at(&sc.q, 0), k, v, a)?;
                ops.linear(&mut sc.h, &o.w, d, d, at(&sc.y, 0), m)?;
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
                ops.linear(&mut sc.mix, &qkvg.w, qkvg.rows, d, at(&sc.n, 0), m)?;
                let raw = col!(y, mix, 4 * d, 0, d);
                ops.l2norm(&mut sc.q, raw, m * heads, dh)?;
                let raw = col!(y, mix, 4 * d, d, d);
                ops.l2norm(&mut sc.k, raw, m * heads, dh)?;
                let vals = col!(v, mix, 4 * d, 2 * d, d);
                let gate = col!(g, mix, 4 * d, 3 * d, d);
                ops.linear(&mut sc.d, &decay.w, heads, d, at(&sc.n, 0), m)?;
                ops.gla(
                    &mut sc.y,
                    s,
                    at(&sc.q, 0),
                    at(&sc.k, 0),
                    vals,
                    &sc.d,
                    bias,
                    m,
                    heads,
                    dh,
                )?;
                ops.rms(&mut sc.y2, at(&sc.y, 0), head_norm, m * heads, dh, eps)?;
                ops.silu_mul(&mut sc.y, gate, at(&sc.y2, 0), n)?;
                ops.linear(&mut sc.h, &o.w, d, d, at(&sc.y, 0), m)?;
            }
            (
                Mixer::Global { q, kv, o, norm_e },
                State::Global {
                    k,
                    v,
                    prev,
                    cap,
                    scores,
                },
            ) => {
                if let Some(w) = norm_e {
                    ops.rms(&mut sc.t, at(e0, 0), w, m, d, eps)?;
                    ops.add(&mut sc.y, at(&sc.n, 0), at(&sc.t, 0), n)?;
                    std::mem::swap(&mut sc.n, &mut sc.y);
                }
                let (kh, kd) = (c.global_kv_heads, c.global_kv_heads * dh);
                ops.linear(&mut sc.q, &q.w, d, d, at(&sc.n, 0), m)?;
                ops.linear(&mut sc.mix, &kv.w, 2 * kd, d, at(&sc.n, 0), m)?;
                if c.global_logn != 0 {
                    let log = (c.global_logn as f32).ln();
                    let f: Vec<f32> = (0..m).map(|i| ((pos + i) as f32).ln_1p() / log).collect();
                    ops.scale_rows(&mut sc.y, at(&sc.q, 0), &f, d)?;
                    std::mem::swap(&mut sc.q, &mut sc.y);
                }
                while pos + m > *cap {
                    self.grow(k, v, cap)?;
                }
                let keys = if c.global_kshift != 0 {
                    ops.shift(&mut sc.shifted, prev, at(&sc.n, 0), m, d)?;
                    ops.linear(&mut sc.prev_kv, &kv.w, 2 * kd, d, at(&sc.shifted, 0), m)?;
                    ops.copy(prev, at(&sc.n, (m - 1) * d), d)?;
                    col!(k, prev_kv, 2 * kd, 0, kd)
                } else {
                    col!(k, mix, 2 * kd, 0, kd)
                };
                let vals = col!(v, mix, 2 * kd, kd, kd);
                ops.store(k, keys, pos, m, *cap, kh, dh)?;
                ops.store(v, vals, pos, m, *cap, kh, dh)?;
                let far = if c.global_far != 0 { c.window } else { 0 };
                if pos + m > far {
                    let a = Attn {
                        pos,
                        m,
                        reach: usize::MAX,
                        gap: far,
                        cap: *cap,
                        heads,
                        key_heads: kh,
                        dh,
                        scale,
                    };
                    ops.attend(&mut sc.y, scores, at(&sc.q, 0), k, v, a)?;
                    ops.linear(&mut sc.h, &o.w, d, d, at(&sc.y, 0), m)?;
                } else {
                    has_h = false;
                }
            }
            _ => bail!("state does not match its block"),
        }
        if has_h {
            ops.add(&mut sc.t, at(x, 0), at(&sc.h, 0), n)?;
            std::mem::swap(x, &mut sc.t);
        }
        ops.rms(&mut sc.n, at(x, 0), &b.norm2, m, d, eps)?;
        let hidden = b.down.cols;
        ops.linear(&mut sc.up, &b.up.w, 2 * hidden, d, at(&sc.n, 0), m)?;
        let gate = col!(gate, up, 2 * hidden, 0, hidden);
        let amp = col!(mid, up, 2 * hidden, hidden, hidden);
        ops.silu_mul(&mut sc.y, gate, amp, m * hidden)?;
        ops.linear(&mut sc.h, &b.down.w, d, hidden, at(&sc.y, 0), m)?;
        ops.add(&mut sc.t, at(x, 0), at(&sc.h, 0), n)?;
        std::mem::swap(x, &mut sc.t);
        Ok(())
    }

    /// Advances by up to `CHUNK` tokens. Logits come from the last one, and only when asked.
    fn advance(&self, se: &mut Session<O>, tokens: &[u32], want_logits: bool) -> Result<()> {
        let ops = &self.ops;
        let (d, m, pos) = (self.cfg.d_model, tokens.len(), se.pos);
        ops.rope_table(
            &mut se.sc.cos,
            &mut se.sc.sin,
            &self.inv_freq,
            pos,
            m,
            self.cfg.head_dim() / 2,
        )?;
        ops.gather(&mut se.e0, &self.embed.w, tokens, d)?;
        ops.copy(&mut se.x, at(&se.e0, 0), m * d)?;
        for (b, st) in self.prelude.iter().zip(&mut se.prelude) {
            self.block(b, st, &mut se.x, &se.e0, &mut se.sc, pos, m)?;
        }
        if let (Some(inject), Some(loop_embed)) = (&self.inject, &self.loop_embed) {
            ops.linear(&mut se.e_proj, &inject.w, d, d, at(&se.x, 0), m)?;
            ops.copy(&mut se.s, at(&se.x, 0), m * d)?;
            for r in 0..se.loops {
                ops.add(&mut se.sc.t, at(&se.s, 0), at(&se.e_proj, 0), m * d)?;
                ops.add_row(&mut se.x, at(&se.sc.t, 0), at(&loop_embed.w, r * d), m, d)?;
                for (b, st) in self.core.iter().zip(&mut se.core[r]) {
                    self.block(b, st, &mut se.x, &se.e0, &mut se.sc, pos, m)?;
                }
                if self.cfg.damp == 1.0 {
                    std::mem::swap(&mut se.s, &mut se.x);
                } else {
                    ops.mix(
                        &mut se.sc.t,
                        at(&se.s, 0),
                        at(&se.x, 0),
                        m * d,
                        self.cfg.damp,
                    )?;
                    std::mem::swap(&mut se.s, &mut se.sc.t);
                }
            }
            std::mem::swap(&mut se.x, &mut se.s);
        }
        for (b, st) in self.coda.iter().zip(&mut se.coda) {
            self.block(b, st, &mut se.x, &se.e0, &mut se.sc, pos, m)?;
        }
        if want_logits {
            ops.copy(&mut se.last, at(&se.x, (m - 1) * d), d)?;
            ops.rms(
                &mut se.sc.n,
                at(&se.last, 0),
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
                1,
            )?;
        }
        se.pos += m;
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

    /// Feeds one token and leaves its logits in the session.
    pub fn step(&self, se: &mut Session<O>, token: u32) -> Result<()> {
        self.advance(se, &[token], true)
    }

    fn chunk_at(&self, pos: usize, limit: usize) -> usize {
        let chunk = limit.min(CHUNK);
        if self.cfg.coda.contains('G') || self.cfg.prelude.contains('G') {
            chunk.min((MAX_SCORES / self.cfg.n_heads / (pos + 1)).max(1))
        } else {
            chunk
        }
    }

    /// Feeds a prompt in chunks of at most `limit` tokens and leaves the logits of the
    /// last token in the session.
    pub fn feed(&self, se: &mut Session<O>, tokens: &[u32], limit: usize) -> Result<()> {
        let mut rest = tokens;
        while !rest.is_empty() {
            let m = self.chunk_at(se.pos, limit).min(rest.len());
            let (now, later) = rest.split_at(m);
            self.advance(se, now, later.is_empty())?;
            rest = later;
        }
        Ok(())
    }

    pub fn logits(&self, se: &Session<O>) -> Result<Vec<f32>> {
        self.ops.read(&se.logits, self.cfg.vocab_size)
    }

    /// Bytes held by a session's caches, without the global layer's unused capacity.
    pub fn state_bytes(&self, se: &Session<O>) -> usize {
        let c = &self.cfg;
        let dh = c.head_dim();
        let one = |st: &State<O::Buf>| match st {
            State::Window { .. } => 2 * c.n_heads * self.cap() * dh,
            State::Linear { .. } => c.n_heads * dh * dh,
            State::Global { .. } => 2 * c.global_kv_heads * se.pos * dh + c.d_model,
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
