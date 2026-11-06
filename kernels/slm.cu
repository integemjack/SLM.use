// CUDA kernels for advancing the model by m tokens. Every tensor is a flat float array
// with the layout described in src/ops.rs. Build with tools/build_kernels.sh; the PTX is
// committed so that neither building nor running the tool needs the CUDA toolkit.
#include <math.h>

#define TID (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL extern "C" __global__ void
// Largest head size; src/cuda.rs rejects anything bigger.
#define MAX_DH 256

// Threads next to each other share a weight row, so a block reads it from memory once.
KERNEL slm_linear(float *y, const float *w, const float *x, int rows, int cols, int m) {
    int t = TID;
    if (t >= m * rows) return;
    int r = t / m, i = t % m;
    const float *row = w + (long long)r * cols;
    const float *in = x + (long long)i * cols;
    float s = 0;
    for (int j = 0; j < cols; j++) s += row[j] * in[j];
    y[(long long)i * rows + r] = s;
}

// The same product for many tokens, with x transposed to [col][token]. Neighbouring
// threads then read neighbouring values, which a GPU fetches far faster than scattered ones.
KERNEL slm_transpose(float *y, const float *x, int m, int cols) {
    long long t = TID;
    if (t >= (long long)m * cols) return;
    int i = (int)(t / cols), j = (int)(t % cols);
    y[(long long)j * m + i] = x[t];
}

KERNEL slm_linear_many(float *y, const float *w, const float *xt, int rows, int cols, int m) {
    int t = TID;
    if (t >= m * rows) return;
    int r = t / m, i = t % m;
    const float *row = w + (long long)r * cols;
    const float *in = xt + i;
    float s = 0;
    for (int j = 0; j < cols; j++) s += row[j] * in[(long long)j * m];
    y[(long long)i * rows + r] = s;
}

KERNEL slm_gather(float *y, const float *w, const int *ids, int m, int d) {
    int t = TID;
    if (t >= m * d) return;
    y[t] = w[(long long)ids[t / d] * d + t % d];
}

KERNEL slm_copy(float *y, const float *x, int n) {
    int i = TID;
    if (i < n) y[i] = x[i];
}

// y[i] = x[i - 1], with y[0] = first.
KERNEL slm_shift(float *y, const float *first, const float *x, int m, int d) {
    int t = TID;
    if (t >= m * d) return;
    y[t] = t < d ? first[t] : x[t - d];
}

KERNEL slm_columns(float *y, const float *x, int m, int stride, int off, int len) {
    int t = TID;
    if (t >= m * len) return;
    y[t] = x[(long long)(t / len) * stride + off + t % len];
}

KERNEL slm_add(float *y, const float *a, const float *b, int n) {
    int i = TID;
    if (i < n) y[i] = a[i] + b[i];
}

KERNEL slm_add_row(float *y, const float *x, const float *row, int m, int d) {
    int t = TID;
    if (t < m * d) y[t] = x[t] + row[t % d];
}

KERNEL slm_scale(float *y, const float *x, int n, float f) {
    int i = TID;
    if (i < n) y[i] = x[i] * f;
}

KERNEL slm_scale_rows(float *y, const float *x, const float *f, int m, int d) {
    int t = TID;
    if (t < m * d) y[t] = x[t] * f[t / d];
}

KERNEL slm_mix(float *y, const float *a, const float *b, int n, float t) {
    int i = TID;
    if (i < n) y[i] = a[i] + t * (b[i] - a[i]);
}

// One block per group. Threads sum slices of the group, then share the partial sums.
KERNEL slm_rms(float *y, const float *x, const float *w, int groups, int size, float eps) {
    extern __shared__ float parts[];
    int g = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    if (g >= groups) return;
    const float *in = x + (long long)g * size;
    float *out = y + (long long)g * size;
    int lo = t * size / T, hi = (t + 1) * size / T;
    float s = 0;
    for (int i = lo; i < hi; i++) s += in[i] * in[i];
    parts[t] = s;
    __syncthreads();
    s = 0;
    for (int i = 0; i < T; i++) s += parts[i];
    float r = 1.0f / sqrtf(s / size + eps);
    for (int i = lo; i < hi; i++) out[i] = in[i] * r * w[i];
}

KERNEL slm_silu_mul(float *y, const float *g, const float *u, int n) {
    int i = TID;
    if (i >= n) return;
    float v = g[i];
    y[i] = u[i] * v / (1.0f + expf(-v));
}

KERNEL slm_rope_table(float *c, float *s, const float *inv, float pos, int half, int m) {
    int t = TID;
    if (t >= m * half) return;
    float a = (pos + t / half) * inv[t % half];
    c[t] = cosf(a);
    s[t] = sinf(a);
}

KERNEL slm_rope(float *y, const float *x, const float *c, const float *s, int m, int heads, int dh) {
    int half = dh / 2, t = TID;
    if (t >= m * heads * half) return;
    int row = t / (heads * half), h = (t / half) % heads, i = t % half;
    const float *in = x + (row * heads + h) * dh;
    float *out = y + (row * heads + h) * dh;
    float cs = c[row * half + i], sn = s[row * half + i];
    float a = in[i], b = in[half + i];
    out[i] = a * cs - b * sn;
    out[half + i] = a * sn + b * cs;
}

KERNEL slm_l2norm(float *y, const float *x, int heads, int dh) {
    int h = TID;
    if (h >= heads) return;
    const float *in = x + h * dh;
    float *out = y + h * dh;
    float s = 0;
    for (int i = 0; i < dh; i++) s += in[i] * in[i];
    float r = 1.0f / fmaxf(sqrtf(s), 1e-12f);
    for (int i = 0; i < dh; i++) out[i] = in[i] * r;
}

// Linear attention over m tokens, one block per head. For each token:
//   S = sigmoid(-(decay + bias)) * S + k^T v, where sigmoid(-x) equals exp(-softplus(x))
//   y = q S
// The barriers keep the tokens in order while the threads share out each matrix.
KERNEL slm_gla(float *y, float *S, const float *q, const float *k, const float *v,
               const float *decay, const float *bias, int m, int heads, int dh) {
    int h = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    if (h >= heads) return;
    float *Sh = S + h * dh * dh;
    int lo = t * dh * dh / T, hi = (t + 1) * dh * dh / T;
    int blo = t * dh / T, bhi = (t + 1) * dh / T;
    for (int i = 0; i < m; i++) {
        int at = (i * heads + h) * dh;
        const float *ki = k + at, *vi = v + at, *qi = q + at;
        float *yi = y + at;
        float d = 1.0f / (1.0f + expf(decay[i * heads + h] + bias[h]));
        for (int e = lo; e < hi; e++) Sh[e] = d * Sh[e] + ki[e / dh] * vi[e % dh];
        __syncthreads();
        for (int b = blo; b < bhi; b++) {
            float s = 0;
            for (int a = 0; a < dh; a++) s += qi[a] * Sh[a * dh + b];
            yi[b] = s;
        }
        __syncthreads();
    }
}

// Writes m tokens' [head][dim] entries into a ring of cap slots.
KERNEL slm_store(float *buf, const float *src, int pos, int m, int cap, int heads, int dh) {
    int t = TID;
    if (t >= m * heads * dh) return;
    int i = t / (heads * dh), h = (t / dh) % heads, d = t % dh;
    long long slot = (pos + i) % cap;
    buf[((long long)h * cap + slot) * dh + d] = src[t];
}

// Softmax attention for m queries at positions pos .. pos + m - 1. The query at position
// p sees the cached positions lo = max(0, p - reach + 1) .. hi = p - gap, and those live
// in ring slots modulo cap. It runs in three passes so that one call can use every core
// of the GPU. Each (head, query) pair splits its positions into P slices and each thread
// owns one slice. With Q = heads * m pairs, sc holds [Q][span] scores, aux holds [Q][P]
// maxima and [Q][P] sums. Query heads share a key head in groups of heads / key_heads.
#define SLICE \
    int t = TID; \
    if (t >= heads * m * P) return; \
    int pair = t / P, part = t % P; \
    int h = pair / m, i = pair % m; \
    int base = pos - reach + 1 > 0 ? pos - reach + 1 : 0; \
    int p_q = pos + i, hi = p_q - gap; \
    int lo = p_q - reach + 1 > 0 ? p_q - reach + 1 : 0; \
    int count = hi >= lo ? hi - lo + 1 : 0; \
    int from = lo + part * count / P, to = lo + (part + 1) * count / P; \
    float *p = sc + (long long)pair * span - base;

KERNEL slm_attn_score(float *sc, float *aux, const float *q, const float *k, int pos, int m,
                      int reach, int gap, int span, int cap, int heads, int key_heads, int dh,
                      int P, float scale) {
    SLICE
    const float *qh = q + (long long)(i * heads + h) * dh;
    const float *keys = k + (long long)(h / (heads / key_heads)) * cap * dh;
    float best = -3.4e38f;
    int slot = from % cap;
    for (int j = from; j < to; j++) {
        const float *kj = keys + (long long)slot * dh;
        if (++slot == cap) slot = 0;
        float s = 0;
        for (int d = 0; d < dh; d++) s += qh[d] * kj[d];
        s *= scale;
        p[j] = s;
        best = fmaxf(best, s);
    }
    aux[t] = best;
}

KERNEL slm_attn_exp(float *sc, float *aux, int pos, int m, int reach, int gap, int span,
                    int heads, int P) {
    SLICE
    long long first = (long long)pair * P;
    float best = aux[first];
    for (int e = 1; e < P; e++) best = fmaxf(best, aux[first + e]);
    float z = 0;
    for (int j = from; j < to; j++) {
        p[j] = expf(p[j] - best);
        z += p[j];
    }
    aux[(long long)heads * m * P + t] = z;
}

// The output pass for a few queries: slices over the slots, as the passes above, with
// [Q][P][dh] partial outputs after the sums in aux.
KERNEL slm_attn_part(float *sc, float *aux, const float *v, int pos, int m, int reach, int gap,
                     int span, int cap, int heads, int key_heads, int dh, int P) {
    SLICE
    long long Q = (long long)heads * m;
    const float *sums = aux + Q * P + (long long)pair * P;
    float z = 0;
    for (int e = 0; e < P; e++) z += sums[e];
    float r = z > 0 ? 1.0f / z : 0;
    float *acc = aux + 2 * Q * P + (long long)t * dh;
    float local[MAX_DH];
    for (int d = 0; d < dh; d++) local[d] = 0;
    const float *vals = v + (long long)(h / (heads / key_heads)) * cap * dh;
    int slot = from % cap;
    for (int j = from; j < to; j++) {
        const float *vj = vals + (long long)slot * dh;
        if (++slot == cap) slot = 0;
        float a = p[j] * r;
        for (int d = 0; d < dh; d++) local[d] += a * vj[d];
    }
    for (int d = 0; d < dh; d++) acc[d] = local[d];
}

KERNEL slm_attn_join(float *y, const float *aux, int m, int heads, int dh, int P) {
    int t = TID;
    if (t >= heads * m * dh) return;
    int pair = t / dh, d = t % dh;
    int h = pair / m, i = pair % m;
    long long Q = (long long)heads * m;
    const float *acc = aux + 2 * Q * P + (long long)pair * P * dh + d;
    float s = 0;
    for (int e = 0; e < P; e++) s += acc[(long long)e * dh];
    y[(long long)(i * heads + h) * dh + d] = s;
}

// The output pass for many queries: one thread per (head, query, dim). Neighbouring
// threads read neighbouring values of each cached row, so a block fetches the row once.
KERNEL slm_attn_sum(float *y, const float *sc, const float *aux, const float *v, int pos, int m,
                    int reach, int gap, int span, int cap, int heads, int key_heads, int dh,
                    int P) {
    int t = TID;
    if (t >= heads * m * dh) return;
    int pair = t / dh, d = t % dh;
    int h = pair / m, i = pair % m;
    int base = pos - reach + 1 > 0 ? pos - reach + 1 : 0;
    int p_q = pos + i, hi = p_q - gap;
    int lo = p_q - reach + 1 > 0 ? p_q - reach + 1 : 0;
    const float *p = sc + (long long)pair * span - base;
    const float *sums = aux + (long long)heads * m * P + (long long)pair * P;
    float z = 0;
    for (int e = 0; e < P; e++) z += sums[e];
    const float *vals = v + (long long)(h / (heads / key_heads)) * cap * dh + d;
    int slot = lo % cap;
    float s = 0;
    for (int j = lo; j <= hi; j++) {
        s += p[j] * vals[(long long)slot * dh];
        if (++slot == cap) slot = 0;
    }
    y[(long long)(i * heads + h) * dh + d] = z > 0 ? s / z : 0;
}
