//! The operations one decode step needs. The model is written once against this
//! trait; CPU, CUDA and MLX each implement it.
//!
//! A step works on a single token, so every tensor is flat `f32` data:
//! - heads are `[head][dim]`, D = heads · dh values;
//! - key and value caches are `[head][slot][dim]`, heads · cap · dh values;
//! - the linear attention state is `[head][a][b]`, heads · dh · dh values;
//! - attention scores are `[head][slot]`;
//! - weights are `[row][col]`, as `nn.Linear` stores them.

use anyhow::Result;

/// A read-only position inside a buffer.
pub struct At<'a, B> {
    pub buf: &'a B,
    pub offset: usize,
}

impl<B> Clone for At<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for At<'_, B> {}

pub fn at<B>(buf: &B, offset: usize) -> At<'_, B> {
    At { buf, offset }
}

/// Size of an attention call.
#[derive(Clone, Copy, Debug)]
pub struct Attn {
    /// Number of valid cached slots.
    pub n: usize,
    /// Slots allocated per head.
    pub cap: usize,
    pub heads: usize,
    /// Heads in the cache. Query heads share them in groups of `heads / key_heads`.
    pub key_heads: usize,
    pub dh: usize,
    pub scale: f32,
}

pub trait Ops {
    type Buf;

    fn name(&self) -> String;
    fn zeros(&self, n: usize) -> Result<Self::Buf>;
    /// A zeroed `[heads][cap][dh]` cache. Backends without in-place writes keep the shape.
    fn zeros_cache(&self, heads: usize, cap: usize, dh: usize) -> Result<Self::Buf> {
        self.zeros(heads * cap * dh)
    }
    fn upload(&self, data: &[f32]) -> Result<Self::Buf>;
    /// A `[heads][cap][dh]` cache holding `data`.
    fn upload_cache(
        &self,
        data: &[f32],
        _heads: usize,
        _cap: usize,
        _dh: usize,
    ) -> Result<Self::Buf> {
        self.upload(data)
    }
    /// Waits for pending work and returns the first `n` values.
    fn read(&self, buf: &Self::Buf, n: usize) -> Result<Vec<f32>>;
    /// Called once per token with every buffer that outlives the step.
    fn flush(&self, keep: &[&Self::Buf]) -> Result<()>;

    fn copy(&self, y: &mut Self::Buf, x: At<Self::Buf>, n: usize) -> Result<()>;
    /// y = w·x, with `w` of shape rows × cols.
    fn linear(
        &self,
        y: &mut Self::Buf,
        w: &Self::Buf,
        rows: usize,
        cols: usize,
        x: At<Self::Buf>,
    ) -> Result<()>;
    fn add(&self, y: &mut Self::Buf, a: At<Self::Buf>, b: At<Self::Buf>, n: usize) -> Result<()>;
    fn scale(&self, y: &mut Self::Buf, x: At<Self::Buf>, n: usize, f: f32) -> Result<()>;
    /// y = a + t·(b − a).
    fn mix(
        &self,
        y: &mut Self::Buf,
        a: At<Self::Buf>,
        b: At<Self::Buf>,
        n: usize,
        t: f32,
    ) -> Result<()>;
    /// RMSNorm over each of `groups` runs of `size` values, sharing the weight `w`.
    fn rms(
        &self,
        y: &mut Self::Buf,
        x: At<Self::Buf>,
        w: &Self::Buf,
        groups: usize,
        size: usize,
        eps: f32,
    ) -> Result<()>;
    /// y = silu(g)·u.
    fn silu_mul(
        &self,
        y: &mut Self::Buf,
        g: At<Self::Buf>,
        u: At<Self::Buf>,
        n: usize,
    ) -> Result<()>;
    fn rope_table(
        &self,
        cos: &mut Self::Buf,
        sin: &mut Self::Buf,
        inv: &Self::Buf,
        pos: f32,
        half: usize,
    ) -> Result<()>;
    fn rope(
        &self,
        y: &mut Self::Buf,
        x: At<Self::Buf>,
        cos: &Self::Buf,
        sin: &Self::Buf,
        heads: usize,
        dh: usize,
    ) -> Result<()>;
    /// L2-normalises each head.
    fn l2norm(&self, y: &mut Self::Buf, x: At<Self::Buf>, heads: usize, dh: usize) -> Result<()>;
    /// S = exp(−softplus(decay + bias))·S + kᵀv for each head.
    #[allow(clippy::too_many_arguments)]
    fn gla_update(
        &self,
        s: &mut Self::Buf,
        k: At<Self::Buf>,
        v: At<Self::Buf>,
        decay: &Self::Buf,
        bias: &Self::Buf,
        heads: usize,
        dh: usize,
    ) -> Result<()>;
    /// y = q·S for each head.
    fn gla_read(
        &self,
        y: &mut Self::Buf,
        q: At<Self::Buf>,
        s: &Self::Buf,
        heads: usize,
        dh: usize,
    ) -> Result<()>;
    /// Writes one `[head][dim]` entry into `slot` of a cache.
    fn store(
        &self,
        cache: &mut Self::Buf,
        src: At<Self::Buf>,
        slot: usize,
        cap: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()>;
    /// Softmax attention of the `[head][dim]` queries over the first `a.n` cached slots.
    fn attend(
        &self,
        y: &mut Self::Buf,
        scores: &mut Self::Buf,
        q: At<Self::Buf>,
        k: &Self::Buf,
        v: &Self::Buf,
        a: Attn,
    ) -> Result<()>;
}
