// CUDA kernels for one decode step. Every tensor is a flat float array with the
// layout described in src/ops.rs. Build with tools/build_kernels.sh; the PTX is
// committed so that neither building nor running the tool needs the CUDA toolkit.
#include <math.h>

#define TID (blockIdx.x * blockDim.x + threadIdx.x)
#define KERNEL extern "C" __global__ void

KERNEL slm_linear(float *y, const float *w, const float *x, int rows, int cols) {
    int r = TID;
    if (r >= rows) return;
    const float *row = w + (long long)r * cols;
    float s = 0;
    for (int j = 0; j < cols; j++) s += row[j] * x[j];
    y[r] = s;
}

KERNEL slm_copy(float *y, const float *x, int n) {
    int i = TID;
    if (i < n) y[i] = x[i];
}

KERNEL slm_add(float *y, const float *a, const float *b, int n) {
    int i = TID;
    if (i < n) y[i] = a[i] + b[i];
}

KERNEL slm_scale(float *y, const float *x, int n, float f) {
    int i = TID;
    if (i < n) y[i] = x[i] * f;
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
    const float *in = x + g * size;
    float *out = y + g * size;
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

KERNEL slm_rope_table(float *c, float *s, const float *inv, float pos, int half) {
    int i = TID;
    if (i >= half) return;
    float a = pos * inv[i];
    c[i] = cosf(a);
    s[i] = sinf(a);
}

KERNEL slm_rope(float *y, const float *x, const float *c, const float *s, int heads, int dh) {
    int half = dh / 2, t = TID;
    if (t >= heads * half) return;
    int h = t / half, i = t % half;
    float a = x[h * dh + i], b = x[h * dh + half + i];
    y[h * dh + i] = a * c[i] - b * s[i];
    y[h * dh + half + i] = a * s[i] + b * c[i];
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

// S = sigmoid(-(decay + bias)) * S + k^T v. sigmoid(-x) equals exp(-softplus(x)).
KERNEL slm_gla_update(float *S, const float *k, const float *v, const float *decay,
                      const float *bias, int heads, int dh) {
    int t = TID;
    if (t >= heads * dh * dh) return;
    int h = t / (dh * dh), a = (t / dh) % dh, b = t % dh;
    float d = 1.0f / (1.0f + expf(decay[h] + bias[h]));
    S[t] = d * S[t] + k[h * dh + a] * v[h * dh + b];
}

KERNEL slm_gla_read(float *y, const float *q, const float *S, int heads, int dh) {
    int t = TID;
    if (t >= heads * dh) return;
    int h = t / dh, b = t % dh;
    float s = 0;
    for (int a = 0; a < dh; a++) s += q[h * dh + a] * S[(h * dh + a) * dh + b];
    y[t] = s;
}

KERNEL slm_store(float *buf, const float *src, int slot, int cap, int heads, int dh) {
    int t = TID;
    if (t >= heads * dh) return;
    int h = t / dh, d = t % dh;
    buf[((long long)h * cap + slot) * dh + d] = src[t];
}

// Softmax attention over n cached keys, in four passes so that one attention can use
// every core of the GPU. Each query head splits its slots into P slices and each thread
// owns one slice. sc holds the scores as [head][n]; aux holds [head][P] maxima, then
// [head][P] sums, then [head][P][dh] partial outputs. Query heads share a key head in
// groups of heads / key_heads.
#define SLICE     int t = TID;     if (t >= heads * P) return;     int h = t / P, part = t % P;     long long lo = (long long)part * n / P, hi = (long long)(part + 1) * n / P;     float *p = sc + (long long)h * n;

KERNEL slm_attn_score(float *sc, float *aux, const float *q, const float *k, int n, int cap,
                      int heads, int key_heads, int dh, int P, float scale) {
    SLICE
    const float *qh = q + h * dh;
    const float *keys = k + (long long)(h / (heads / key_heads)) * cap * dh;
    float m = -3.4e38f;
    for (long long j = lo; j < hi; j++) {
        const float *kj = keys + j * dh;
        float s = 0;
        for (int d = 0; d < dh; d++) s += qh[d] * kj[d];
        s *= scale;
        p[j] = s;
        m = fmaxf(m, s);
    }
    aux[t] = m;
}

KERNEL slm_attn_exp(float *sc, float *aux, int n, int heads, int P) {
    SLICE
    float m = aux[h * P];
    for (int i = 1; i < P; i++) m = fmaxf(m, aux[h * P + i]);
    float z = 0;
    for (long long j = lo; j < hi; j++) {
        p[j] = expf(p[j] - m);
        z += p[j];
    }
    aux[heads * P + t] = z;
}

KERNEL slm_attn_sum(float *sc, float *aux, const float *v, int n, int cap, int heads,
                    int key_heads, int dh, int P) {
    SLICE
    const float *sums = aux + heads * P;
    float z = 0;
    for (int i = 0; i < P; i++) z += sums[h * P + i];
    float r = 1.0f / z;
    float *acc = aux + 2 * heads * P + (long long)t * dh;
    for (int d = 0; d < dh; d++) acc[d] = 0;
    const float *vals = v + (long long)(h / (heads / key_heads)) * cap * dh;
    for (long long j = lo; j < hi; j++) {
        const float *vj = vals + j * dh;
        float a = p[j] * r;
        for (int d = 0; d < dh; d++) acc[d] += a * vj[d];
    }
}

KERNEL slm_attn_join(float *y, const float *aux, int heads, int dh, int P) {
    int t = TID;
    if (t >= heads * dh) return;
    int h = t / dh, d = t % dh;
    const float *acc = aux + 2 * heads * P + (long long)h * P * dh + d;
    float s = 0;
    for (int i = 0; i < P; i++) s += acc[i * dh];
    y[t] = s;
}
