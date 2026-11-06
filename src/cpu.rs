//! CPU backend, always available and the reference for the others.

use anyhow::Result;
use multiversion::multiversion;
use rayon::prelude::*;

use crate::ops::{At, Attn, Ops};

#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx512f+avx512vl", "aarch64+neon"))]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx512f+avx512vl", "aarch64+neon"))]
fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (o, v) in y.iter_mut().zip(x) {
        *o += a * v;
    }
}

fn sl<'a>(x: At<'a, Vec<f32>>, n: usize) -> &'a [f32] {
    &x.buf[x.offset..x.offset + n]
}

fn resize(y: &mut Vec<f32>, n: usize) {
    if y.len() != n {
        y.resize(n, 0.0);
    }
}

#[derive(Default)]
pub struct Cpu;

impl Ops for Cpu {
    type Buf = Vec<f32>;

    fn name(&self) -> String {
        format!("cpu ({} threads)", rayon::current_num_threads())
    }

    fn zeros(&self, n: usize) -> Result<Vec<f32>> {
        Ok(vec![0.0; n])
    }

    fn upload(&self, data: &[f32]) -> Result<Vec<f32>> {
        Ok(data.to_vec())
    }

    fn read(&self, buf: &Vec<f32>, n: usize) -> Result<Vec<f32>> {
        Ok(buf[..n].to_vec())
    }

    fn flush(&self, _keep: &[&Vec<f32>]) -> Result<()> {
        Ok(())
    }

    fn gather(&self, y: &mut Vec<f32>, w: &Vec<f32>, ids: &[u32], d: usize) -> Result<()> {
        resize(y, ids.len() * d);
        for (o, id) in y.chunks_exact_mut(d).zip(ids) {
            let at = *id as usize * d;
            o.copy_from_slice(&w[at..at + d]);
        }
        Ok(())
    }

    fn copy(&self, y: &mut Vec<f32>, x: At<Vec<f32>>, n: usize) -> Result<()> {
        resize(y, n);
        y.copy_from_slice(sl(x, n));
        Ok(())
    }

    fn shift(
        &self,
        y: &mut Vec<f32>,
        first: &Vec<f32>,
        x: At<Vec<f32>>,
        m: usize,
        d: usize,
    ) -> Result<()> {
        resize(y, m * d);
        y[..d].copy_from_slice(&first[..d]);
        y[d..].copy_from_slice(sl(x, (m - 1) * d));
        Ok(())
    }

    fn columns(
        &self,
        y: &mut Vec<f32>,
        x: &Vec<f32>,
        m: usize,
        stride: usize,
        off: usize,
        len: usize,
    ) -> Result<()> {
        resize(y, m * len);
        for (i, o) in y.chunks_exact_mut(len).enumerate() {
            o.copy_from_slice(&x[i * stride + off..i * stride + off + len]);
        }
        Ok(())
    }

    fn linear(
        &self,
        y: &mut Vec<f32>,
        w: &Vec<f32>,
        rows: usize,
        cols: usize,
        x: At<Vec<f32>>,
        m: usize,
    ) -> Result<()> {
        resize(y, m * rows);
        let x = sl(x, m * cols);
        y.par_iter_mut()
            .with_min_len(64)
            .enumerate()
            .for_each(|(t, o)| {
                let (i, r) = (t / rows, t % rows);
                *o = dot(&w[r * cols..(r + 1) * cols], &x[i * cols..(i + 1) * cols]);
            });
        Ok(())
    }

    fn add(&self, y: &mut Vec<f32>, a: At<Vec<f32>>, b: At<Vec<f32>>, n: usize) -> Result<()> {
        resize(y, n);
        for ((o, p), q) in y.iter_mut().zip(sl(a, n)).zip(sl(b, n)) {
            *o = p + q;
        }
        Ok(())
    }

    fn add_row(
        &self,
        y: &mut Vec<f32>,
        x: At<Vec<f32>>,
        row: At<Vec<f32>>,
        m: usize,
        d: usize,
    ) -> Result<()> {
        resize(y, m * d);
        let row = sl(row, d);
        for (o, r) in y.chunks_exact_mut(d).zip(sl(x, m * d).chunks_exact(d)) {
            for i in 0..d {
                o[i] = r[i] + row[i];
            }
        }
        Ok(())
    }

    fn scale_rows(&self, y: &mut Vec<f32>, x: At<Vec<f32>>, f: &[f32], d: usize) -> Result<()> {
        resize(y, f.len() * d);
        for ((o, r), s) in y
            .chunks_exact_mut(d)
            .zip(sl(x, f.len() * d).chunks_exact(d))
            .zip(f)
        {
            for i in 0..d {
                o[i] = r[i] * s;
            }
        }
        Ok(())
    }

    fn mix(
        &self,
        y: &mut Vec<f32>,
        a: At<Vec<f32>>,
        b: At<Vec<f32>>,
        n: usize,
        t: f32,
    ) -> Result<()> {
        resize(y, n);
        for ((o, p), q) in y.iter_mut().zip(sl(a, n)).zip(sl(b, n)) {
            *o = p + t * (q - p);
        }
        Ok(())
    }

    fn rms(
        &self,
        y: &mut Vec<f32>,
        x: At<Vec<f32>>,
        w: &Vec<f32>,
        groups: usize,
        size: usize,
        eps: f32,
    ) -> Result<()> {
        resize(y, groups * size);
        for (o, g) in y
            .chunks_exact_mut(size)
            .zip(sl(x, groups * size).chunks_exact(size))
        {
            let r = 1.0 / (dot(g, g) / size as f32 + eps).sqrt();
            for i in 0..size {
                o[i] = g[i] * r * w[i];
            }
        }
        Ok(())
    }

    fn silu_mul(&self, y: &mut Vec<f32>, g: At<Vec<f32>>, u: At<Vec<f32>>, n: usize) -> Result<()> {
        resize(y, n);
        for ((o, v), m) in y.iter_mut().zip(sl(g, n)).zip(sl(u, n)) {
            *o = m * v / (1.0 + (-v).exp());
        }
        Ok(())
    }

    fn rope_table(
        &self,
        cos: &mut Vec<f32>,
        sin: &mut Vec<f32>,
        inv: &Vec<f32>,
        pos: usize,
        m: usize,
        half: usize,
    ) -> Result<()> {
        resize(cos, m * half);
        resize(sin, m * half);
        for i in 0..m {
            let p = (pos + i) as f32;
            for j in 0..half {
                let a = p * inv[j];
                cos[i * half + j] = a.cos();
                sin[i * half + j] = a.sin();
            }
        }
        Ok(())
    }

    fn rope(
        &self,
        y: &mut Vec<f32>,
        x: At<Vec<f32>>,
        cos: &Vec<f32>,
        sin: &Vec<f32>,
        m: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        resize(y, m * heads * dh);
        let half = dh / 2;
        for (t, (o, h)) in y
            .chunks_exact_mut(dh)
            .zip(sl(x, m * heads * dh).chunks_exact(dh))
            .enumerate()
        {
            let row = t / heads;
            let (c, s) = (
                &cos[row * half..(row + 1) * half],
                &sin[row * half..(row + 1) * half],
            );
            for i in 0..half {
                let (a, b) = (h[i], h[half + i]);
                o[i] = a * c[i] - b * s[i];
                o[half + i] = a * s[i] + b * c[i];
            }
        }
        Ok(())
    }

    fn l2norm(&self, y: &mut Vec<f32>, x: At<Vec<f32>>, heads: usize, dh: usize) -> Result<()> {
        resize(y, heads * dh);
        for (o, h) in y
            .chunks_exact_mut(dh)
            .zip(sl(x, heads * dh).chunks_exact(dh))
        {
            let r = 1.0 / dot(h, h).sqrt().max(1e-12);
            for i in 0..dh {
                o[i] = h[i] * r;
            }
        }
        Ok(())
    }

    fn gla(
        &self,
        y: &mut Vec<f32>,
        s: &mut Vec<f32>,
        q: At<Vec<f32>>,
        k: At<Vec<f32>>,
        v: At<Vec<f32>>,
        decay: &Vec<f32>,
        bias: &Vec<f32>,
        m: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        resize(y, m * heads * dh);
        let n = m * heads * dh;
        let (q, k, v) = (sl(q, n), sl(k, n), sl(v, n));
        // Tokens stay in order within a head; heads are independent.
        let parts: Vec<Vec<f32>> = s
            .par_chunks_exact_mut(dh * dh)
            .enumerate()
            .map(|(h, state)| {
                let mut out = vec![0f32; m * dh];
                for (i, o) in out.chunks_exact_mut(dh).enumerate() {
                    let at = (i * heads + h) * dh;
                    let d = 1.0 / (1.0 + (decay[i * heads + h] + bias[h]).exp());
                    let vh = &v[at..at + dh];
                    for (row, ka) in state.chunks_exact_mut(dh).zip(&k[at..at + dh]) {
                        for (e, vb) in row.iter_mut().zip(vh) {
                            *e = d * *e + ka * vb;
                        }
                    }
                    for (qa, row) in q[at..at + dh].iter().zip(state.chunks_exact(dh)) {
                        axpy(o, *qa, row);
                    }
                }
                out
            })
            .collect();
        for (h, part) in parts.iter().enumerate() {
            for (i, o) in part.chunks_exact(dh).enumerate() {
                let at = (i * heads + h) * dh;
                y[at..at + dh].copy_from_slice(o);
            }
        }
        Ok(())
    }

    fn store(
        &self,
        cache: &mut Vec<f32>,
        src: At<Vec<f32>>,
        pos: usize,
        m: usize,
        cap: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let src = sl(src, m * heads * dh);
        for i in 0..m {
            let slot = (pos + i) % cap;
            for h in 0..heads {
                let (to, from) = ((h * cap + slot) * dh, (i * heads + h) * dh);
                cache[to..to + dh].copy_from_slice(&src[from..from + dh]);
            }
        }
        Ok(())
    }

    fn attend(
        &self,
        y: &mut Vec<f32>,
        _scores: &mut Vec<f32>,
        q: At<Vec<f32>>,
        k: &Vec<f32>,
        v: &Vec<f32>,
        a: Attn,
    ) -> Result<()> {
        let Attn {
            heads,
            key_heads,
            dh,
            cap,
            scale,
            m,
            ..
        } = a;
        resize(y, m * heads * dh);
        let group = heads / key_heads;
        let q = sl(q, m * heads * dh);
        y.par_chunks_exact_mut(heads * dh)
            .enumerate()
            .for_each(|(i, row)| {
                let range = a.range(i);
                let mut sc = vec![0f32; range.len()];
                for (h, o) in row.chunks_exact_mut(dh).enumerate() {
                    o.fill(0.0);
                    if range.is_empty() {
                        continue;
                    }
                    let base = (h / group) * cap * dh;
                    let qh = &q[(i * heads + h) * dh..(i * heads + h + 1) * dh];
                    let mut max = f32::NEG_INFINITY;
                    for (s, p) in sc.iter_mut().zip(range.clone()) {
                        let at = base + (p % cap) * dh;
                        *s = dot(qh, &k[at..at + dh]) * scale;
                        max = max.max(*s);
                    }
                    let mut sum = 0.0;
                    for s in sc.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for (s, p) in sc.iter().zip(range.clone()) {
                        let at = base + (p % cap) * dh;
                        axpy(o, s / sum, &v[at..at + dh]);
                    }
                }
            });
        Ok(())
    }
}
