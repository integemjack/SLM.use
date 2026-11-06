//! MLX backend for Apple silicon. Weights and token tensors stay flat, as in the other
//! backends; caches keep their `[head][slot][dh]` shape so slots can be written in place.

use anyhow::Result;
use mlx_rs::ops::indexing::{TryIndexMutOp, TryIndexOp};
use mlx_rs::ops::{concatenate_axis, maximum, sigmoid, softmax_axis};
use mlx_rs::{transforms, Array, Dtype};

use crate::ops::{At, Attn, Ops};

/// Scores outside a query's range. Softmax turns them into zero, in fp16 as well.
const MASKED: f32 = -1e4;

pub struct Mlx {
    dtype: Dtype,
}

impl Mlx {
    pub fn new(fp16: bool) -> Result<Self> {
        Ok(Self {
            dtype: if fp16 { Dtype::Float16 } else { Dtype::Float32 },
        })
    }

    fn scalar(&self, v: f32) -> Result<Array> {
        Ok(Array::from_f32(v).as_dtype(self.dtype)?)
    }
}

fn dims(shape: &[usize]) -> Vec<i32> {
    shape.iter().map(|d| *d as i32).collect()
}

fn part(x: At<Array>, n: usize) -> Result<Array> {
    if x.offset == 0 && x.buf.size() == n {
        return Ok(x.buf.clone());
    }
    let (a, b) = (x.offset as i32, (x.offset + n) as i32);
    Ok(x.buf.try_index(a..b)?)
}

fn flat(x: &Array) -> Result<Array> {
    Ok(x.reshape(&[-1])?)
}

impl Ops for Mlx {
    type Buf = Array;

    fn name(&self) -> String {
        format!("mlx ({:?})", self.dtype)
    }

    fn zeros(&self, n: usize) -> Result<Array> {
        Ok(Array::zeros::<f32>(&dims(&[n]))?.as_dtype(self.dtype)?)
    }

    fn zeros_cache(&self, heads: usize, cap: usize, dh: usize) -> Result<Array> {
        Ok(Array::zeros::<f32>(&dims(&[heads, cap, dh]))?.as_dtype(self.dtype)?)
    }

    fn upload(&self, data: &[f32]) -> Result<Array> {
        Ok(Array::from_slice(data, &[data.len() as i32]).as_dtype(self.dtype)?)
    }

    fn upload_cache(&self, data: &[f32], heads: usize, cap: usize, dh: usize) -> Result<Array> {
        Ok(self.upload(data)?.reshape(&dims(&[heads, cap, dh]))?)
    }

    fn read(&self, buf: &Array, n: usize) -> Result<Vec<f32>> {
        let a = flat(buf)?.as_dtype(Dtype::Float32)?;
        a.eval()?;
        Ok(a.as_slice::<f32>()[..n].to_vec())
    }

    fn flush(&self, keep: &[&Array]) -> Result<()> {
        transforms::eval(keep.iter().copied())?;
        Ok(())
    }

    fn gather(&self, y: &mut Array, w: &Array, ids: &[u32], d: usize) -> Result<()> {
        let list: Vec<i32> = ids.iter().map(|i| *i as i32).collect();
        let list = Array::from_slice(&list, &[list.len() as i32]);
        *y = flat(&w.reshape(&[-1, d as i32])?.take_axis(&list, 0)?)?;
        Ok(())
    }

    fn copy(&self, y: &mut Array, x: At<Array>, n: usize) -> Result<()> {
        *y = part(x, n)?;
        Ok(())
    }

    fn shift(&self, y: &mut Array, first: &Array, x: At<Array>, m: usize, d: usize) -> Result<()> {
        let head = part(
            At {
                buf: first,
                offset: 0,
            },
            d,
        )?;
        *y = if m == 1 {
            head
        } else {
            concatenate_axis(&[head, part(x, (m - 1) * d)?], 0)?
        };
        Ok(())
    }

    fn columns(
        &self,
        y: &mut Array,
        x: &Array,
        m: usize,
        stride: usize,
        off: usize,
        len: usize,
    ) -> Result<()> {
        let (a, b) = (off as i32, (off + len) as i32);
        let rows = part(At { buf: x, offset: 0 }, m * stride)?.reshape(&dims(&[m, stride]))?;
        *y = flat(&rows.try_index((.., a..b))?)?;
        Ok(())
    }

    fn linear(
        &self,
        y: &mut Array,
        w: &Array,
        rows: usize,
        cols: usize,
        x: At<Array>,
        m: usize,
    ) -> Result<()> {
        let x = part(x, m * cols)?.reshape(&dims(&[m, cols]))?;
        *y = flat(&x.matmul(w.reshape(&dims(&[rows, cols]))?.swap_axes(0, 1)?)?)?;
        Ok(())
    }

    fn add(&self, y: &mut Array, a: At<Array>, b: At<Array>, n: usize) -> Result<()> {
        *y = part(a, n)?.add(part(b, n)?)?;
        Ok(())
    }

    fn add_row(
        &self,
        y: &mut Array,
        x: At<Array>,
        row: At<Array>,
        m: usize,
        d: usize,
    ) -> Result<()> {
        let rows = part(x, m * d)?.reshape(&dims(&[m, d]))?;
        *y = flat(&rows.add(part(row, d)?)?)?;
        Ok(())
    }

    fn scale_rows(&self, y: &mut Array, x: At<Array>, f: &[f32], d: usize) -> Result<()> {
        let m = f.len();
        let rows = part(x, m * d)?.reshape(&dims(&[m, d]))?;
        let f = Array::from_slice(f, &[m as i32, 1]).as_dtype(self.dtype)?;
        *y = flat(&rows.multiply(f)?)?;
        Ok(())
    }

    fn mix(&self, y: &mut Array, a: At<Array>, b: At<Array>, n: usize, t: f32) -> Result<()> {
        let (a, b) = (part(a, n)?, part(b, n)?);
        *y = a.add(b.subtract(&a)?.multiply(self.scalar(t)?)?)?;
        Ok(())
    }

    fn rms(
        &self,
        y: &mut Array,
        x: At<Array>,
        w: &Array,
        groups: usize,
        size: usize,
        eps: f32,
    ) -> Result<()> {
        // The square overflows in fp16, so the norm is always computed in fp32.
        let g = part(x, groups * size)?
            .reshape(&dims(&[groups, size]))?
            .as_dtype(Dtype::Float32)?;
        let mean = g.square()?.mean_axis(-1, true)?;
        let r = mean.add(Array::from_f32(eps))?.sqrt()?;
        *y = flat(&g.divide(r)?.as_dtype(self.dtype)?.multiply(w)?)?;
        Ok(())
    }

    fn silu_mul(&self, y: &mut Array, g: At<Array>, u: At<Array>, n: usize) -> Result<()> {
        let g = part(g, n)?;
        *y = g.multiply(sigmoid(&g)?)?.multiply(part(u, n)?)?;
        Ok(())
    }

    fn rope_table(
        &self,
        cos: &mut Array,
        sin: &mut Array,
        inv: &Array,
        pos: usize,
        m: usize,
        half: usize,
    ) -> Result<()> {
        // Long positions need fp32 angles.
        let positions: Vec<f32> = (0..m).map(|i| (pos + i) as f32).collect();
        let p = Array::from_slice(&positions, &[m as i32, 1]);
        let a = p.multiply(inv.as_dtype(Dtype::Float32)?.reshape(&[1, half as i32])?)?;
        *cos = a.cos()?.as_dtype(self.dtype)?;
        *sin = a.sin()?.as_dtype(self.dtype)?;
        Ok(())
    }

    fn rope(
        &self,
        y: &mut Array,
        x: At<Array>,
        cos: &Array,
        sin: &Array,
        m: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let half = (dh / 2) as i32;
        let t = part(x, m * heads * dh)?.reshape(&dims(&[m, heads, dh]))?;
        let (a, b) = (
            t.try_index((.., .., 0..half))?,
            t.try_index((.., .., half..2 * half))?,
        );
        let (c, s) = (
            cos.reshape(&[m as i32, 1, half])?,
            sin.reshape(&[m as i32, 1, half])?,
        );
        let left = a.multiply(&c)?.subtract(b.multiply(&s)?)?;
        let right = a.multiply(&s)?.add(b.multiply(&c)?)?;
        *y = flat(&concatenate_axis(&[left, right], 2)?)?;
        Ok(())
    }

    fn l2norm(&self, y: &mut Array, x: At<Array>, heads: usize, dh: usize) -> Result<()> {
        let m = part(x, heads * dh)?.reshape(&dims(&[heads, dh]))?;
        let norm = m.square()?.sum_axis(-1, true)?.sqrt()?;
        *y = flat(&m.divide(maximum(norm, self.scalar(1e-12)?)?)?)?;
        Ok(())
    }

    fn gla(
        &self,
        y: &mut Array,
        s: &mut Array,
        q: At<Array>,
        k: At<Array>,
        v: At<Array>,
        decay: &Array,
        bias: &Array,
        m: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let shape = dims(&[m, heads, dh]);
        let n = m * heads * dh;
        let (q, k, v) = (part(q, n)?, part(k, n)?, part(v, n)?);
        let (q, k, v) = (q.reshape(&shape)?, k.reshape(&shape)?, v.reshape(&shape)?);
        let raw = part(
            At {
                buf: decay,
                offset: 0,
            },
            m * heads,
        )?
        .reshape(&dims(&[m, heads]))?;
        let d = sigmoid(raw.add(bias)?.negative()?)?;
        let (h, dh32) = (heads as i32, dh as i32);
        let mut state = s.reshape(&[h, dh32, dh32])?;
        let mut out = Vec::with_capacity(m);
        for i in 0..m as i32 {
            let di = d.try_index(i)?.reshape(&[h, 1, 1])?;
            let ki = k.try_index(i)?.reshape(&[h, dh32, 1])?;
            let vi = v.try_index(i)?.reshape(&[h, 1, dh32])?;
            state = state.multiply(di)?.add(ki.multiply(vi)?)?;
            let qi = q.try_index(i)?.reshape(&[h, 1, dh32])?;
            out.push(flat(&qi.matmul(&state)?)?);
        }
        *s = flat(&state)?;
        *y = concatenate_axis(&out, 0)?;
        Ok(())
    }

    fn store(
        &self,
        cache: &mut Array,
        src: At<Array>,
        pos: usize,
        m: usize,
        cap: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let vals = part(src, m * heads * dh)?
            .reshape(&dims(&[m, heads, dh]))?
            .swap_axes(0, 1)?;
        let first = pos % cap;
        let n = m.min(cap - first);
        let (a, b) = (first as i32, (first + n) as i32);
        cache.try_index_mut((.., a..b), vals.try_index((.., 0..n as i32))?)?;
        if n < m {
            cache.try_index_mut(
                (.., 0..(m - n) as i32),
                vals.try_index((.., n as i32..m as i32))?,
            )?;
        }
        Ok(())
    }

    fn attend(
        &self,
        y: &mut Array,
        _scores: &mut Array,
        q: At<Array>,
        k: &Array,
        v: &Array,
        a: Attn,
    ) -> Result<()> {
        let (span, base) = (a.span(), a.base());
        let slots: Vec<i32> = (base..base + span).map(|p| (p % a.cap) as i32).collect();
        let slots = Array::from_slice(&slots, &[span as i32]);
        let (kh, group) = (a.key_heads as i32, (a.heads / a.key_heads) as i32);
        let keys = k
            .take_axis(&slots, 1)?
            .reshape(&[kh, 1, span as i32, a.dh as i32])?;
        let vals = v
            .take_axis(&slots, 1)?
            .reshape(&[kh, 1, span as i32, a.dh as i32])?;
        // [query][head][dh] → [key head][group][query][dh]
        let q = part(q, a.m * a.heads * a.dh)?
            .reshape(&dims(&[a.m, a.heads, a.dh]))?
            .swap_axes(0, 1)?
            .reshape(&[kh, group, a.m as i32, a.dh as i32])?;
        let mut mask = vec![MASKED; a.m * span];
        let mut seen = vec![0f32; a.m];
        for i in 0..a.m {
            let range = a.range(i);
            seen[i] = if range.is_empty() { 0.0 } else { 1.0 };
            for p in range {
                mask[i * span + p - base] = 0.0;
            }
        }
        let mask = Array::from_slice(&mask, &[a.m as i32, span as i32]).as_dtype(self.dtype)?;
        let seen = Array::from_slice(&seen, &[a.m as i32, 1]).as_dtype(self.dtype)?;
        let scores = q
            .matmul(keys.swap_axes(-1, -2)?)?
            .multiply(self.scalar(a.scale)?)?;
        let out = softmax_axis(scores.add(mask)?, -1, None)?
            .matmul(vals)?
            .multiply(seen)?;
        // [key head][group][query][dh] → [query][head][dh]
        let out = out.reshape(&dims(&[a.heads, a.m, a.dh]))?.swap_axes(0, 1)?;
        *y = flat(&out)?;
        Ok(())
    }
}
