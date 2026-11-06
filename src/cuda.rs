//! CUDA backend. It loads the NVIDIA driver at run time and runs kernels compiled
//! ahead of time to PTX, so it needs no CUDA toolkit, cuBLAS or NVRTC.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::ops::{At, Attn, Ops};

const PTX: &str = include_str!("../kernels/slm.ptx");
const BLOCK: usize = 32;
const RMS_THREADS: u32 = 64;
/// Slices per query head in an attention call.
const ATTEND_SLICES: usize = 256;

pub type Buf = CudaSlice<f32>;

struct Kernels {
    linear: CudaFunction,
    copy: CudaFunction,
    add: CudaFunction,
    scale: CudaFunction,
    mix: CudaFunction,
    rms: CudaFunction,
    silu_mul: CudaFunction,
    rope_table: CudaFunction,
    rope: CudaFunction,
    l2norm: CudaFunction,
    gla_update: CudaFunction,
    gla_read: CudaFunction,
    store: CudaFunction,
    attn_score: CudaFunction,
    attn_exp: CudaFunction,
    attn_sum: CudaFunction,
    attn_join: CudaFunction,
}

pub struct Cuda {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    k: Kernels,
}

fn view<'a>(x: At<'a, Buf>, n: usize) -> CudaView<'a, f32> {
    x.buf.slice(x.offset..x.offset + n)
}

fn cfg(threads: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (threads.div_ceil(BLOCK) as u32, 1, 1),
        block_dim: (BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Cuda {
    /// Returns `Ok(None)` on machines without a usable NVIDIA driver or GPU.
    pub fn probe() -> Result<Option<Self>> {
        // cudarc panics when the driver library is missing, so a machine without one
        // must not reach that message.
        let hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let ctx = panic::catch_unwind(AssertUnwindSafe(|| CudaContext::new(0)));
        panic::set_hook(hook);
        let ctx = match ctx {
            Ok(Ok(ctx)) => ctx,
            _ => return Ok(None),
        };
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(Ptx::from_src(PTX))
            .context("loading kernels")?;
        let f = |n: &str| {
            module
                .load_function(n)
                .with_context(|| format!("kernel {n}"))
        };
        let k = Kernels {
            linear: f("slm_linear")?,
            copy: f("slm_copy")?,
            add: f("slm_add")?,
            scale: f("slm_scale")?,
            mix: f("slm_mix")?,
            rms: f("slm_rms")?,
            silu_mul: f("slm_silu_mul")?,
            rope_table: f("slm_rope_table")?,
            rope: f("slm_rope")?,
            l2norm: f("slm_l2norm")?,
            gla_update: f("slm_gla_update")?,
            gla_read: f("slm_gla_read")?,
            store: f("slm_store")?,
            attn_score: f("slm_attn_score")?,
            attn_exp: f("slm_attn_exp")?,
            attn_sum: f("slm_attn_sum")?,
            attn_join: f("slm_attn_join")?,
        };
        Ok(Some(Self { ctx, stream, k }))
    }

    fn ensure(&self, y: &mut Buf, n: usize) -> Result<()> {
        if y.len() < n {
            *y = self.stream.alloc_zeros::<f32>(n)?;
        }
        Ok(())
    }
}

macro_rules! launch {
    ($self:ident, $f:ident, $threads:expr, $($arg:expr),+ $(,)?) => {
        launch!(@go $self, $f, cfg($threads), $($arg),+)
    };
    (@go $self:ident, $f:ident, $cfg:expr, $($arg:expr),+) => {{
        let mut b = $self.stream.launch_builder(&$self.k.$f);
        $(b.arg($arg);)+
        // SAFETY: argument lists match kernels/slm.cu and every buffer was sized by the caller.
        unsafe { b.launch($cfg) }.map_err(|e| anyhow!("{}: {e}", stringify!($f)))?;
    }};
}

impl Ops for Cuda {
    type Buf = Buf;

    fn name(&self) -> String {
        format!(
            "cuda ({})",
            self.ctx.name().unwrap_or_else(|_| "unknown GPU".into())
        )
    }

    fn zeros(&self, n: usize) -> Result<Buf> {
        Ok(self.stream.alloc_zeros::<f32>(n.max(1))?)
    }

    fn upload(&self, data: &[f32]) -> Result<Buf> {
        Ok(self.stream.clone_htod(data)?)
    }

    fn read(&self, buf: &Buf, n: usize) -> Result<Vec<f32>> {
        let mut out = vec![0f32; n];
        self.stream.memcpy_dtoh(&buf.slice(..n), &mut out)?;
        self.stream.synchronize()?;
        Ok(out)
    }

    fn flush(&self, _keep: &[&Buf]) -> Result<()> {
        Ok(())
    }

    fn copy(&self, y: &mut Buf, x: At<Buf>, n: usize) -> Result<()> {
        self.ensure(y, n)?;
        let n32 = n as i32;
        let x = view(x, n);
        launch!(self, copy, n, &*y, &x, &n32);
        Ok(())
    }

    fn linear(&self, y: &mut Buf, w: &Buf, rows: usize, cols: usize, x: At<Buf>) -> Result<()> {
        self.ensure(y, rows)?;
        let (r, c) = (rows as i32, cols as i32);
        let x = view(x, cols);
        launch!(self, linear, rows, &*y, w, &x, &r, &c);
        Ok(())
    }

    fn add(&self, y: &mut Buf, a: At<Buf>, b: At<Buf>, n: usize) -> Result<()> {
        self.ensure(y, n)?;
        let n32 = n as i32;
        let (a, b) = (view(a, n), view(b, n));
        launch!(self, add, n, &*y, &a, &b, &n32);
        Ok(())
    }

    fn scale(&self, y: &mut Buf, x: At<Buf>, n: usize, f: f32) -> Result<()> {
        self.ensure(y, n)?;
        let n32 = n as i32;
        let x = view(x, n);
        launch!(self, scale, n, &*y, &x, &n32, &f);
        Ok(())
    }

    fn mix(&self, y: &mut Buf, a: At<Buf>, b: At<Buf>, n: usize, t: f32) -> Result<()> {
        self.ensure(y, n)?;
        let n32 = n as i32;
        let (a, b) = (view(a, n), view(b, n));
        launch!(self, mix, n, &*y, &a, &b, &n32, &t);
        Ok(())
    }

    fn rms(
        &self,
        y: &mut Buf,
        x: At<Buf>,
        w: &Buf,
        groups: usize,
        size: usize,
        eps: f32,
    ) -> Result<()> {
        self.ensure(y, groups * size)?;
        let (g, s) = (groups as i32, size as i32);
        let x = view(x, groups * size);
        let threads = RMS_THREADS.min(size as u32);
        let c = LaunchConfig {
            grid_dim: (groups as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };
        launch!(@go self, rms, c, &*y, &x, w, &g, &s, &eps);
        Ok(())
    }

    fn silu_mul(&self, y: &mut Buf, g: At<Buf>, u: At<Buf>, n: usize) -> Result<()> {
        self.ensure(y, n)?;
        let n32 = n as i32;
        let (g, u) = (view(g, n), view(u, n));
        launch!(self, silu_mul, n, &*y, &g, &u, &n32);
        Ok(())
    }

    fn rope_table(
        &self,
        cos: &mut Buf,
        sin: &mut Buf,
        inv: &Buf,
        pos: f32,
        half: usize,
    ) -> Result<()> {
        self.ensure(cos, half)?;
        self.ensure(sin, half)?;
        let h = half as i32;
        launch!(self, rope_table, half, &*cos, &*sin, inv, &pos, &h);
        Ok(())
    }

    fn rope(
        &self,
        y: &mut Buf,
        x: At<Buf>,
        cos: &Buf,
        sin: &Buf,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        self.ensure(y, heads * dh)?;
        let (h, d) = (heads as i32, dh as i32);
        let x = view(x, heads * dh);
        launch!(self, rope, heads * dh / 2, &*y, &x, cos, sin, &h, &d);
        Ok(())
    }

    fn l2norm(&self, y: &mut Buf, x: At<Buf>, heads: usize, dh: usize) -> Result<()> {
        self.ensure(y, heads * dh)?;
        let (h, d) = (heads as i32, dh as i32);
        let x = view(x, heads * dh);
        launch!(self, l2norm, heads, &*y, &x, &h, &d);
        Ok(())
    }

    fn gla_update(
        &self,
        s: &mut Buf,
        k: At<Buf>,
        v: At<Buf>,
        decay: &Buf,
        bias: &Buf,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let (h, d) = (heads as i32, dh as i32);
        let n = heads * dh;
        let (k, v) = (view(k, n), view(v, n));
        launch!(self, gla_update, n * dh, &*s, &k, &v, decay, bias, &h, &d);
        Ok(())
    }

    fn gla_read(&self, y: &mut Buf, q: At<Buf>, s: &Buf, heads: usize, dh: usize) -> Result<()> {
        self.ensure(y, heads * dh)?;
        let (h, d) = (heads as i32, dh as i32);
        let q = view(q, heads * dh);
        launch!(self, gla_read, heads * dh, &*y, &q, s, &h, &d);
        Ok(())
    }

    fn store(
        &self,
        cache: &mut Buf,
        src: At<Buf>,
        slot: usize,
        cap: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let (sl, c, h, d) = (slot as i32, cap as i32, heads as i32, dh as i32);
        let src = view(src, heads * dh);
        launch!(self, store, heads * dh, &*cache, &src, &sl, &c, &h, &d);
        Ok(())
    }

    fn attend(
        &self,
        y: &mut Buf,
        scores: &mut Buf,
        q: At<Buf>,
        k: &Buf,
        v: &Buf,
        a: Attn,
    ) -> Result<()> {
        let p = ATTEND_SLICES.min(a.n);
        let aux_len = a.heads * p * (2 + a.dh);
        let at = a.heads * a.cap;
        self.ensure(y, a.heads * a.dh)?;
        self.ensure(scores, at + a.heads * ATTEND_SLICES * (2 + a.dh))?;
        let (n, cap, h, kh, d, pi) = (
            a.n as i32,
            a.cap as i32,
            a.heads as i32,
            a.key_heads as i32,
            a.dh as i32,
            p as i32,
        );
        let q = view(q, a.heads * a.dh);
        let (sc, aux) = (scores.slice(..at), scores.slice(at..at + aux_len));
        let threads = a.heads * p;
        launch!(self, attn_score, threads, &sc, &aux, &q, k, &n, &cap, &h, &kh, &d, &pi, &a.scale);
        launch!(self, attn_exp, threads, &sc, &aux, &n, &h, &pi);
        launch!(self, attn_sum, threads, &sc, &aux, v, &n, &cap, &h, &kh, &d, &pi);
        launch!(self, attn_join, a.heads * a.dh, &*y, &aux, &h, &d, &pi);
        Ok(())
    }
}
