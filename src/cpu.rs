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

    fn copy(&self, y: &mut Vec<f32>, x: At<Vec<f32>>, n: usize) -> Result<()> {
        resize(y, n);
        y.copy_from_slice(sl(x, n));
        Ok(())
    }

    fn linear(
        &self,
        y: &mut Vec<f32>,
        w: &Vec<f32>,
        rows: usize,
        cols: usize,
        x: At<Vec<f32>>,
    ) -> Result<()> {
        resize(y, rows);
        let x = sl(x, cols);
        y.par_iter_mut()
            .with_min_len(64)
            .enumerate()
            .for_each(|(r, o)| {
                *o = dot(&w[r * cols..(r + 1) * cols], x);
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

    fn scale(&self, y: &mut Vec<f32>, x: At<Vec<f32>>, n: usize, f: f32) -> Result<()> {
        resize(y, n);
        for (o, p) in y.iter_mut().zip(sl(x, n)) {
            *o = p * f;
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
        pos: f32,
        half: usize,
    ) -> Result<()> {
        resize(cos, half);
        resize(sin, half);
        for i in 0..half {
            let a = pos * inv[i];
            cos[i] = a.cos();
            sin[i] = a.sin();
        }
        Ok(())
    }

    fn rope(
        &self,
        y: &mut Vec<f32>,
        x: At<Vec<f32>>,
        cos: &Vec<f32>,
        sin: &Vec<f32>,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        resize(y, heads * dh);
        let half = dh / 2;
        for (o, h) in y
            .chunks_exact_mut(dh)
            .zip(sl(x, heads * dh).chunks_exact(dh))
        {
            for i in 0..half {
                let (a, b) = (h[i], h[half + i]);
                o[i] = a * cos[i] - b * sin[i];
                o[half + i] = a * sin[i] + b * cos[i];
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

    fn gla_update(
        &self,
        s: &mut Vec<f32>,
        k: At<Vec<f32>>,
        v: At<Vec<f32>>,
        decay: &Vec<f32>,
        bias: &Vec<f32>,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let (k, v) = (sl(k, heads * dh), sl(v, heads * dh));
        s.par_chunks_exact_mut(dh * dh)
            .enumerate()
            .for_each(|(h, m)| {
                let d = 1.0 / (1.0 + (decay[h] + bias[h]).exp());
                let (kh, vh) = (&k[h * dh..(h + 1) * dh], &v[h * dh..(h + 1) * dh]);
                for (row, ka) in m.chunks_exact_mut(dh).zip(kh) {
                    for (o, vb) in row.iter_mut().zip(vh) {
                        *o = d * *o + ka * vb;
                    }
                }
            });
        Ok(())
    }

    fn gla_read(
        &self,
        y: &mut Vec<f32>,
        q: At<Vec<f32>>,
        s: &Vec<f32>,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        resize(y, heads * dh);
        let q = sl(q, heads * dh);
        y.par_chunks_exact_mut(dh).enumerate().for_each(|(h, o)| {
            o.fill(0.0);
            let m = &s[h * dh * dh..(h + 1) * dh * dh];
            for (qa, row) in q[h * dh..(h + 1) * dh].iter().zip(m.chunks_exact(dh)) {
                axpy(o, *qa, row);
            }
        });
        Ok(())
    }

    fn store(
        &self,
        cache: &mut Vec<f32>,
        src: At<Vec<f32>>,
        slot: usize,
        cap: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let src = sl(src, heads * dh);
        for h in 0..heads {
            let at = (h * cap + slot) * dh;
            cache[at..at + dh].copy_from_slice(&src[h * dh..(h + 1) * dh]);
        }
        Ok(())
    }

    fn attend(
        &self,
        y: &mut Vec<f32>,
        scores: &mut Vec<f32>,
        q: At<Vec<f32>>,
        k: &Vec<f32>,
        v: &Vec<f32>,
        a: Attn,
    ) -> Result<()> {
        let Attn {
            n,
            cap,
            heads,
            key_heads,
            dh,
            scale,
        } = a;
        resize(y, heads * dh);
        resize(scores, heads * n);
        let group = heads / key_heads;
        let q = sl(q, heads * dh);
        y.par_chunks_exact_mut(dh)
            .zip(scores.par_chunks_exact_mut(n))
            .enumerate()
            .for_each(|(h, (o, sc))| {
                let base = (h / group) * cap * dh;
                let qh = &q[h * dh..(h + 1) * dh];
                let mut max = f32::NEG_INFINITY;
                for (j, s) in sc.iter_mut().enumerate() {
                    *s = dot(qh, &k[base + j * dh..base + (j + 1) * dh]) * scale;
                    max = max.max(*s);
                }
                let mut sum = 0.0;
                for s in sc.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                o.fill(0.0);
                for (j, s) in sc.iter().enumerate() {
                    axpy(o, s / sum, &v[base + j * dh..base + (j + 1) * dh]);
                }
            });
        Ok(())
    }
}
