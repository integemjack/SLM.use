//! The operations the model needs to advance by m tokens at once. The model is written
//! once against this trait; CPU, CUDA and MLX each implement it. With m = 1 this is
//! ordinary decoding, and larger m is how a prompt is read quickly.
//!
//! Everything is flat `f32` data, row by row:
//! - tokens are `[m][d]`, and split into heads `[m][head][dh]`, with d = heads · dh;
//! - key and value caches are `[head][slot][dh]`, a ring of `cap` slots per head;
//! - the linear attention state is `[head][a][b]`, heads · dh · dh values;
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

/// Which cached positions each of m queries sees: the query at position p reads
/// `max(0, p - reach + 1) ..= p - gap`, found in ring slots modulo `cap`.
#[derive(Clone, Copy, Debug)]
pub struct Attn {
    /// Position of the first query.
    pub pos: usize,
    pub m: usize,
    pub reach: usize,
    pub gap: usize,
    pub cap: usize,
    pub heads: usize,
    /// Heads in the cache. Query heads share them in groups of `heads / key_heads`.
    pub key_heads: usize,
    pub dh: usize,
    pub scale: f32,
}

impl Attn {
    /// First position any query sees.
    pub fn base(&self) -> usize {
        self.pos.saturating_sub(self.reach - 1)
    }

    /// Positions from `base` up to the last one any query sees, at least 1.
    pub fn span(&self) -> usize {
        (self.pos + self.m)
            .saturating_sub(self.gap + self.base())
            .max(1)
    }

    /// The range of positions query `i` sees, empty when it sees none yet.
    pub fn range(&self, i: usize) -> std::ops::Range<usize> {
        let p = self.pos + i;
        p.saturating_sub(self.reach - 1)..(p + 1).saturating_sub(self.gap)
    }
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
    /// Called after each advance with every buffer that outlives it.
    fn flush(&self, keep: &[&Self::Buf]) -> Result<()>;

    /// y\[i\] = w\[ids\[i\]\], the embedding lookup.
    fn gather(&self, y: &mut Self::Buf, w: &Self::Buf, ids: &[u32], d: usize) -> Result<()>;
    fn copy(&self, y: &mut Self::Buf, x: At<Self::Buf>, n: usize) -> Result<()>;
    /// y\[0\] = first, y\[i\] = x\[i - 1\].
    fn shift(
        &self,
        y: &mut Self::Buf,
        first: &Self::Buf,
        x: At<Self::Buf>,
        m: usize,
        d: usize,
    ) -> Result<()>;
    /// y\[i\] = x\[i\]\[off..off + len\], with rows of `stride` values.
    #[allow(clippy::too_many_arguments)]
    fn columns(
        &self,
        y: &mut Self::Buf,
        x: &Self::Buf,
        m: usize,
        stride: usize,
        off: usize,
        len: usize,
    ) -> Result<()>;
    /// y\[i\] = w·x\[i\], with `w` of shape rows × cols.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        y: &mut Self::Buf,
        w: &Self::Buf,
        rows: usize,
        cols: usize,
        x: At<Self::Buf>,
        m: usize,
    ) -> Result<()>;
    fn add(&self, y: &mut Self::Buf, a: At<Self::Buf>, b: At<Self::Buf>, n: usize) -> Result<()>;
    /// y\[i\] = x\[i\] + row.
    fn add_row(
        &self,
        y: &mut Self::Buf,
        x: At<Self::Buf>,
        row: At<Self::Buf>,
        m: usize,
        d: usize,
    ) -> Result<()>;
    /// y\[i\] = f\[i\]·x\[i\].
    fn scale_rows(&self, y: &mut Self::Buf, x: At<Self::Buf>, f: &[f32], d: usize) -> Result<()>;
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
    #[allow(clippy::too_many_arguments)]
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
    /// `[m][half]` tables for positions `pos` onward.
    #[allow(clippy::too_many_arguments)]
    fn rope_table(
        &self,
        cos: &mut Self::Buf,
        sin: &mut Self::Buf,
        inv: &Self::Buf,
        pos: usize,
        m: usize,
        half: usize,
    ) -> Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        y: &mut Self::Buf,
        x: At<Self::Buf>,
        cos: &Self::Buf,
        sin: &Self::Buf,
        m: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()>;
    /// L2-normalises each of `heads` runs of `dh` values.
    fn l2norm(&self, y: &mut Self::Buf, x: At<Self::Buf>, heads: usize, dh: usize) -> Result<()>;
    /// For each token in order, S = exp(−softplus(decay + bias))·S + kᵀv, then y = q·S.
    #[allow(clippy::too_many_arguments)]
    fn gla(
        &self,
        y: &mut Self::Buf,
        s: &mut Self::Buf,
        q: At<Self::Buf>,
        k: At<Self::Buf>,
        v: At<Self::Buf>,
        decay: &Self::Buf,
        bias: &Self::Buf,
        m: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()>;
    /// Writes m `[head][dh]` entries into the ring for positions `pos` onward.
    #[allow(clippy::too_many_arguments)]
    fn store(
        &self,
        cache: &mut Self::Buf,
        src: At<Self::Buf>,
        pos: usize,
        m: usize,
        cap: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()>;
    /// Softmax attention. `scores` is scratch; a query that sees nothing yields zeros.
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
