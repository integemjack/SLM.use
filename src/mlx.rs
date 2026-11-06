//! MLX backend for Apple silicon. Weights and scratch tensors stay flat, as in the
//! other backends; caches keep their `[head][slot][dim]` shape so one slot can be
//! written in place.

use anyhow::Result;
use mlx_rs::ops::indexing::{TryIndexMutOp, TryIndexOp};
use mlx_rs::ops::{concatenate_axis, maximum, sigmoid, softmax_axis};
use mlx_rs::{transforms, Array, Dtype};

use crate::ops::{At, Attn, Ops};

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

    fn copy(&self, y: &mut Array, x: At<Array>, n: usize) -> Result<()> {
        *y = part(x, n)?;
        Ok(())
    }

    fn linear(
        &self,
        y: &mut Array,
        w: &Array,
        rows: usize,
        cols: usize,
        x: At<Array>,
    ) -> Result<()> {
        *y = w.reshape(&dims(&[rows, cols]))?.matmul(part(x, cols)?)?;
        Ok(())
    }

    fn add(&self, y: &mut Array, a: At<Array>, b: At<Array>, n: usize) -> Result<()> {
        *y = part(a, n)?.add(part(b, n)?)?;
        Ok(())
    }

    fn scale(&self, y: &mut Array, x: At<Array>, n: usize, f: f32) -> Result<()> {
        *y = part(x, n)?.multiply(self.scalar(f)?)?;
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
        pos: f32,
        _half: usize,
    ) -> Result<()> {
        // Long positions need fp32 angles.
        let a = inv
            .as_dtype(Dtype::Float32)?
            .multiply(Array::from_f32(pos))?;
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
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let half = (dh / 2) as i32;
        let m = part(x, heads * dh)?.reshape(&dims(&[heads, dh]))?;
        let (a, b) = (
            m.try_index((.., 0..half))?,
            m.try_index((.., half..2 * half))?,
        );
        let left = a.multiply(cos)?.subtract(b.multiply(sin)?)?;
        let right = a.multiply(sin)?.add(b.multiply(cos)?)?;
        *y = flat(&concatenate_axis(&[left, right], 1)?)?;
        Ok(())
    }

    fn l2norm(&self, y: &mut Array, x: At<Array>, heads: usize, dh: usize) -> Result<()> {
        let m = part(x, heads * dh)?.reshape(&dims(&[heads, dh]))?;
        let norm = m.square()?.sum_axis(-1, true)?.sqrt()?;
        *y = flat(&m.divide(maximum(norm, self.scalar(1e-12)?)?)?)?;
        Ok(())
    }

    fn gla_update(
        &self,
        s: &mut Array,
        k: At<Array>,
        v: At<Array>,
        decay: &Array,
        bias: &Array,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let d = sigmoid(decay.add(bias)?.negative()?)?.reshape(&dims(&[heads, 1, 1]))?;
        let k = part(k, heads * dh)?.reshape(&dims(&[heads, dh, 1]))?;
        let v = part(v, heads * dh)?.reshape(&dims(&[heads, 1, dh]))?;
        *s = flat(
            &s.reshape(&dims(&[heads, dh, dh]))?
                .multiply(d)?
                .add(k.multiply(v)?)?,
        )?;
        Ok(())
    }

    fn gla_read(
        &self,
        y: &mut Array,
        q: At<Array>,
        s: &Array,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let q = part(q, heads * dh)?.reshape(&dims(&[heads, 1, dh]))?;
        *y = flat(&q.matmul(s.reshape(&dims(&[heads, dh, dh]))?)?)?;
        Ok(())
    }

    fn store(
        &self,
        cache: &mut Array,
        src: At<Array>,
        slot: usize,
        _cap: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let val = part(src, heads * dh)?.reshape(&dims(&[heads, 1, dh]))?;
        let s = slot as i32;
        cache.try_index_mut((.., s..s + 1), val)?;
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
        let n = a.n as i32;
        let (k, v) = if a.n == a.cap {
            (k.clone(), v.clone())
        } else {
            (k.try_index((.., 0..n))?, v.try_index((.., 0..n))?)
        };
        let q =
            part(q, a.heads * a.dh)?.reshape(&dims(&[a.key_heads, a.heads / a.key_heads, a.dh]))?;
        let scores = q
            .matmul(k.swap_axes(-1, -2)?)?
            .multiply(self.scalar(a.scale)?)?;
        *y = flat(&softmax_axis(scores, -1, None)?.matmul(v)?)?;
        Ok(())
    }
}
