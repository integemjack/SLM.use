//! CUDA backend. It loads the NVIDIA driver at run time and runs kernels compiled
//! ahead of time to PTX, so it needs no CUDA toolkit, cuBLAS or NVRTC.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::ops::{At, Attn, Ops};

const PTX: &str = include_str!("../kernels/slm.ptx");
const BLOCK: usize = 32;
/// MAX_DH in kernels/slm.cu.
const MAX_DH: usize = 256;
const RMS_THREADS: u32 = 64;
const GLA_THREADS: u32 = 64;
/// Slices per query in an attention call, sharing about this many threads per head.
const ATTEND_THREADS: usize = 256;

pub type Buf = CudaSlice<f32>;

struct Kernels {
    linear: CudaFunction,
    transpose: CudaFunction,
    linear_many: CudaFunction,
    gather: CudaFunction,
    copy: CudaFunction,
    shift: CudaFunction,
    columns: CudaFunction,
    add: CudaFunction,
    add_row: CudaFunction,
    scale: CudaFunction,
    scale_rows: CudaFunction,
    mix: CudaFunction,
    rms: CudaFunction,
    silu_mul: CudaFunction,
    rope_table: CudaFunction,
    rope: CudaFunction,
    l2norm: CudaFunction,
    gla: CudaFunction,
    store: CudaFunction,
    attn_score: CudaFunction,
    attn_exp: CudaFunction,
    attn_sum: CudaFunction,
    attn_part: CudaFunction,
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
            transpose: f("slm_transpose")?,
            linear_many: f("slm_linear_many")?,
            gather: f("slm_gather")?,
            copy: f("slm_copy")?,
            shift: f("slm_shift")?,
            columns: f("slm_columns")?,
            add: f("slm_add")?,
            add_row: f("slm_add_row")?,
            scale: f("slm_scale")?,
            scale_rows: f("slm_scale_rows")?,
            mix: f("slm_mix")?,
            rms: f("slm_rms")?,
            silu_mul: f("slm_silu_mul")?,
            rope_table: f("slm_rope_table")?,
            rope: f("slm_rope")?,
            l2norm: f("slm_l2norm")?,
            gla: f("slm_gla")?,
            store: f("slm_store")?,
            attn_score: f("slm_attn_score")?,
            attn_exp: f("slm_attn_exp")?,
            attn_sum: f("slm_attn_sum")?,
            attn_part: f("slm_attn_part")?,
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

    fn gather(&self, y: &mut Buf, w: &Buf, ids: &[u32], d: usize) -> Result<()> {
        let m = ids.len();
        self.ensure(y, m * d)?;
        if m == 1 {
            return self.copy(
                y,
                At {
                    buf: w,
                    offset: ids[0] as usize * d,
                },
                d,
            );
        }
        let list: Vec<i32> = ids.iter().map(|i| *i as i32).collect();
        let list = self.stream.clone_htod(&list[..])?;
        let (m32, d32) = (m as i32, d as i32);
        launch!(self, gather, m * d, &*y, w, &list, &m32, &d32);
        // The ids must stay allocated until the kernel has read them.
        self.stream.synchronize()?;
        Ok(())
    }

    fn copy(&self, y: &mut Buf, x: At<Buf>, n: usize) -> Result<()> {
        self.ensure(y, n)?;
        let n32 = n as i32;
        let x = view(x, n);
        launch!(self, copy, n, &*y, &x, &n32);
        Ok(())
    }

    fn shift(&self, y: &mut Buf, first: &Buf, x: At<Buf>, m: usize, d: usize) -> Result<()> {
        self.ensure(y, m * d)?;
        let (m32, d32) = (m as i32, d as i32);
        let x = x.buf.slice(x.offset..);
        launch!(self, shift, m * d, &*y, first, &x, &m32, &d32);
        Ok(())
    }

    fn columns(
        &self,
        y: &mut Buf,
        x: &Buf,
        m: usize,
        stride: usize,
        off: usize,
        len: usize,
    ) -> Result<()> {
        self.ensure(y, m * len)?;
        let (a, b, c, e) = (m as i32, stride as i32, off as i32, len as i32);
        launch!(self, columns, m * len, &*y, x, &a, &b, &c, &e);
        Ok(())
    }

    fn linear(
        &self,
        y: &mut Buf,
        w: &Buf,
        rows: usize,
        cols: usize,
        x: At<Buf>,
        m: usize,
    ) -> Result<()> {
        self.ensure(y, m * rows)?;
        let (r, c, m32) = (rows as i32, cols as i32, m as i32);
        let x = view(x, m * cols);
        if m < BLOCK {
            launch!(self, linear, m * rows, &*y, w, &x, &r, &c, &m32);
            return Ok(());
        }
        let xt = self.stream.alloc_zeros::<f32>(m * cols)?;
        launch!(self, transpose, m * cols, &xt, &x, &m32, &c);
        launch!(self, linear_many, m * rows, &*y, w, &xt, &r, &c, &m32);
        // The transposed copy must stay allocated until the kernel has read it.
        self.stream.synchronize()?;
        Ok(())
    }

    fn add(&self, y: &mut Buf, a: At<Buf>, b: At<Buf>, n: usize) -> Result<()> {
        self.ensure(y, n)?;
        let n32 = n as i32;
        let (a, b) = (view(a, n), view(b, n));
        launch!(self, add, n, &*y, &a, &b, &n32);
        Ok(())
    }

    fn add_row(&self, y: &mut Buf, x: At<Buf>, row: At<Buf>, m: usize, d: usize) -> Result<()> {
        self.ensure(y, m * d)?;
        let (m32, d32) = (m as i32, d as i32);
        let (x, row) = (view(x, m * d), view(row, d));
        launch!(self, add_row, m * d, &*y, &x, &row, &m32, &d32);
        Ok(())
    }

    fn scale_rows(&self, y: &mut Buf, x: At<Buf>, f: &[f32], d: usize) -> Result<()> {
        let m = f.len();
        self.ensure(y, m * d)?;
        let (m32, d32) = (m as i32, d as i32);
        let xv = view(x, m * d);
        if m == 1 {
            launch!(self, scale, d, &*y, &xv, &d32, &f[0]);
            return Ok(());
        }
        let factors = self.stream.clone_htod(f)?;
        launch!(self, scale_rows, m * d, &*y, &xv, &factors, &m32, &d32);
        // The factors must stay allocated until the kernel has read them.
        self.stream.synchronize()?;
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
        pos: usize,
        m: usize,
        half: usize,
    ) -> Result<()> {
        self.ensure(cos, m * half)?;
        self.ensure(sin, m * half)?;
        let (h, m32, p) = (half as i32, m as i32, pos as f32);
        launch!(self, rope_table, m * half, &*cos, &*sin, inv, &p, &h, &m32);
        Ok(())
    }

    fn rope(
        &self,
        y: &mut Buf,
        x: At<Buf>,
        cos: &Buf,
        sin: &Buf,
        m: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        self.ensure(y, m * heads * dh)?;
        let (m32, h, d) = (m as i32, heads as i32, dh as i32);
        let x = view(x, m * heads * dh);
        launch!(
            self,
            rope,
            m * heads * dh / 2,
            &*y,
            &x,
            cos,
            sin,
            &m32,
            &h,
            &d
        );
        Ok(())
    }

    fn l2norm(&self, y: &mut Buf, x: At<Buf>, heads: usize, dh: usize) -> Result<()> {
        self.ensure(y, heads * dh)?;
        let (h, d) = (heads as i32, dh as i32);
        let x = view(x, heads * dh);
        launch!(self, l2norm, heads, &*y, &x, &h, &d);
        Ok(())
    }

    fn gla(
        &self,
        y: &mut Buf,
        s: &mut Buf,
        q: At<Buf>,
        k: At<Buf>,
        v: At<Buf>,
        decay: &Buf,
        bias: &Buf,
        m: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        self.ensure(y, m * heads * dh)?;
        let (m32, h, d) = (m as i32, heads as i32, dh as i32);
        let c = LaunchConfig {
            grid_dim: (heads as u32, 1, 1),
            block_dim: (GLA_THREADS.min(dh as u32), 1, 1),
            shared_mem_bytes: 0,
        };
        let n = m * heads * dh;
        let (q, k, v) = (view(q, n), view(k, n), view(v, n));
        launch!(@go self, gla, c, &*y, &*s, &q, &k, &v, decay, bias, &m32, &h, &d);
        Ok(())
    }

    fn store(
        &self,
        cache: &mut Buf,
        src: At<Buf>,
        pos: usize,
        m: usize,
        cap: usize,
        heads: usize,
        dh: usize,
    ) -> Result<()> {
        let (p, m32, c, h, d) = (pos as i32, m as i32, cap as i32, heads as i32, dh as i32);
        let src = view(src, m * heads * dh);
        launch!(
            self,
            store,
            m * heads * dh,
            &*cache,
            &src,
            &p,
            &m32,
            &c,
            &h,
            &d
        );
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
        if a.dh > MAX_DH {
            bail!("head size {} is above the {MAX_DH} the kernels allow", a.dh);
        }
        let span = a.span();
        let pairs = a.heads * a.m;
        let slices = (ATTEND_THREADS / a.m).clamp(1, span);
        let at = pairs * span;
        self.ensure(y, pairs * a.dh)?;
        self.ensure(scores, at + pairs * slices * (2 + a.dh))?;
        let (pos, m, reach, gap) = (
            a.pos as i32,
            a.m as i32,
            a.reach.min(i32::MAX as usize) as i32,
            a.gap as i32,
        );
        let (sp, cap, h, kh, d, p) = (
            span as i32,
            a.cap as i32,
            a.heads as i32,
            a.key_heads as i32,
            a.dh as i32,
            slices as i32,
        );
        let (sc, aux) = (scores.slice(..at), scores.slice(at..));
        let threads = pairs * slices;
        let q = view(q, pairs * a.dh);
        launch!(
            self, attn_score, threads, &sc, &aux, &q, k, &pos, &m, &reach, &gap, &sp, &cap, &h,
            &kh, &d, &p, &a.scale
        );
        launch!(self, attn_exp, threads, &sc, &aux, &pos, &m, &reach, &gap, &sp, &h, &p);
        if a.m < BLOCK {
            launch!(
                self, attn_part, threads, &sc, &aux, v, &pos, &m, &reach, &gap, &sp, &cap, &h, &kh,
                &d, &p
            );
            launch!(self, attn_join, pairs * a.dh, &*y, &aux, &m, &h, &d, &p);
        } else {
            launch!(
                self,
                attn_sum,
                pairs * a.dh,
                &*y,
                &sc,
                &aux,
                v,
                &pos,
                &m,
                &reach,
                &gap,
                &sp,
                &cap,
                &h,
                &kh,
                &d,
                &p
            );
        }
        Ok(())
    }
}
