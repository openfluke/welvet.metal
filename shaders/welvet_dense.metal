// Welvet Metal — Dense FormatNone forward (soft-double; Apple Metal has no HW double).
// Contract = Go FormatNone MatVec: sum_k float64(W)*float64(x), then float(sum).
// Compile with fastMathEnabled=NO (see library.mm).
#include <metal_stdlib>
using namespace metal;
inline float act_value(float v, int act);
inline float act_prime(float v, int act);
namespace native_gpu {
inline ulong pack_sd(float2 value);
}

kernel void welvet_smoke_saxpy(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                               constant float &a [[buffer(2)]], constant float &b [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  y[gid] = a * x[gid] + b;
}

// float2 soft-double: .x = hi, .y = lo (Dekker / Knuth; requires no fast-math)

inline float2 sd_two_sum(float a, float b) {
  float s = a + b;
  float v = s - a;
  float e = (a - (s - v)) + (b - v);
  return float2(s, e);
}

inline float2 sd_fast_two_sum(float a, float b) {
  // |a| >= |b|
  float s = a + b;
  float e = b - (s - a);
  return float2(s, e);
}

inline float2 sd_add(float2 a, float2 b) {
  float2 s = sd_two_sum(a.x, b.x);
  float t = a.y + b.y;
  float2 v = sd_two_sum(s.x, t + s.y);
  return v;
}

inline float2 sd_mul_float(float2 a, float c) {
  // (ahi + alo) * c
  float p = a.x * c;
  float e = fma(a.x, c, -p); // exact residual of a.x*c when fma is IEEE
  e = e + a.y * c;
  return sd_fast_two_sum(p, e);
}

inline float sd_to_float(float2 a) {
  // Match (float)(double_value): round the soft-double to float32.
  return a.x + a.y;
}

kernel void welvet_dense_matvec_softdouble_batch(device const float2 *W [[buffer(0)]],
                                                 device const float *x [[buffer(1)]],
                                                 device float *y [[buffer(2)]],
                                                 constant uint &in_dim [[buffer(3)]],
                                                 constant uint &out_dim [[buffer(4)]],
                                                 constant uint &batch [[buffer(5)]],
                                                 uint gid [[thread_position_in_grid]]) {
  const uint total = batch * out_dim;
  if (gid >= total)
    return;
  const uint b = gid / out_dim;
  const uint o = gid % out_dim;
  float2 acc = float2(0.0f, 0.0f);
  const uint row = o * in_dim, xoff = b * in_dim;
  float mx = 0, mw = 0;
  for (uint k = 0; k < in_dim; ++k) {
    mx = max(mx, abs(x[xoff + k]));
    mw = max(mw, abs(W[row + k].x));
  }
  const int shift = mx * mw < 1e-20f ? 64 : 0;
  const bool scale_w = mw < mx;
  for (uint k = 0; k < in_dim; ++k) {
    float2 w = W[row + k];
    float xv = x[xoff + k];
    if (scale_w)
      w = float2(ldexp(w.x, shift), ldexp(w.y, shift));
    else
      xv = ldexp(xv, shift);
    acc = sd_add(acc, sd_mul_float(w, xv));
  }
  y[gid] = ldexp(sd_to_float(acc), -shift);
}

kernel void
welvet_dense_bias_act(device float *pre [[buffer(0)]], device float *post [[buffer(1)]],
                      device const float2 *bias [[buffer(2)]], constant uint &out_dim [[buffer(3)]],
                      constant uint &batch [[buffer(4)]], constant int &act [[buffer(5)]],
                      constant uint &bias_len [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  const uint total = batch * out_dim;
  if (gid >= total)
    return;
  const uint o = gid % out_dim;

  float2 v = float2(pre[gid], 0.0f);
  if (o < bias_len) {
    v = sd_add(v, bias[o]);
    pre[gid] = sd_to_float(v);
  }
  float vf = sd_to_float(v);
  float y = act_value(vf, act);
  post[gid] = y;
}

// d_pre = gy * act'(pre)  — pre is post-bias / pre-act (same buffer as forward `pre`).
kernel void
welvet_dense_act_deriv(device const float *gy [[buffer(0)]], device const float *pre [[buffer(1)]],
                       device float *d_pre [[buffer(2)]], constant uint &out_dim [[buffer(3)]],
                       constant uint &batch [[buffer(4)]], constant int &act [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
  const uint total = batch * out_dim;
  if (gid >= total)
    return;
  const float vf = pre[gid];
  float d = act_prime(vf, act);
  d_pre[gid] = gy[gid] * d;
}

// gx = Wᵀ @ gy  (soft-double W). gid over batch * in_dim.
kernel void welvet_dense_matvect_softdouble_batch(device const float2 *W [[buffer(0)]],
                                                  device const float *gy [[buffer(1)]],
                                                  device float *gx [[buffer(2)]],
                                                  constant uint &in_dim [[buffer(3)]],
                                                  constant uint &out_dim [[buffer(4)]],
                                                  constant uint &batch [[buffer(5)]],
                                                  uint gid [[thread_position_in_grid]]) {
  const uint total = batch * in_dim;
  if (gid >= total)
    return;
  const uint b = gid / in_dim;
  const uint i = gid % in_dim;
  float2 acc = float2(0.0f, 0.0f);
  const uint gyoff = b * out_dim;
  float mg = 0, mw = 0;
  for (uint o = 0; o < out_dim; ++o) {
    mg = max(mg, abs(gy[gyoff + o]));
    mw = max(mw, abs(W[o * in_dim + i].x));
  }
  const int shift = mg * mw < 1e-20f ? 64 : 0;
  const bool scale_w = mw < mg;
  for (uint o = 0; o < out_dim; ++o) {
    float2 w = W[o * in_dim + i];
    float gv = gy[gyoff + o];
    if (scale_w)
      w = float2(ldexp(w.x, shift), ldexp(w.y, shift));
    else
      gv = ldexp(gv, shift);
    acc = sd_add(acc, sd_mul_float(w, gv));
  }
  gx[gid] = ldexp(sd_to_float(acc), -shift);
}

// dW[o,i] = sum_b gy[b,o] * x[b,i]  (f32 outer; matches host f64-accum→f32 cast closely enough).
kernel void
welvet_dense_outer_batch(device const float *gy [[buffer(0)]], device const float *x [[buffer(1)]],
                         device float *dW [[buffer(2)]], constant uint &in_dim [[buffer(3)]],
                         constant uint &out_dim [[buffer(4)]], constant uint &batch [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint total = out_dim * in_dim;
  if (gid >= total)
    return;
  const uint o = gid / in_dim;
  const uint i = gid % in_dim;
  float acc = 0.0f;
  for (uint b = 0; b < batch; ++b) {
    const float g = gy[b * out_dim + o];
    const float xv = x[b * in_dim + i];
    if (abs(acc) < 1e-20f && abs(g * xv) < 1e-20f) {
      acc = ldexp(
          sd_to_float(sd_add(float2(ldexp(acc, 64), 0), sd_mul_float(float2(ldexp(g, 64), 0), xv))),
          -64);
    } else
      acc = sd_to_float(sd_add(float2(acc, 0.0f), sd_mul_float(float2(g, 0.0f), xv)));
  }
  dW[gid] = acc;
}

// Training kernels. Float2 is explicit ALU scratch, never native dtype storage.
inline float2 sd_mul(float2 a, float2 b) {
  return sd_add(sd_mul_float(a, b.x), sd_mul_float(a, b.y));
}

inline float2 sd_div_uint(float2 a, uint n) {
  const float2 d = sd_two_sum(float(n & 0xffff0000u), float(n & 0xffffu));
  const float q = a.x / d.x;
  const float2 residual = sd_add(a, -sd_mul_float(d, q));
  return sd_add(float2(q, 0.0f), float2((residual.x + residual.y) / d.x, 0.0f));
}

// Serial reduction preserves a stable element order. A parallel reduction is future work.
kernel void welvet_mse_loss(device const float *pred [[buffer(0)]],
                            device const float *target [[buffer(1)]],
                            device float2 *loss [[buffer(2)]], constant uint &n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid != 0)
    return;
  float2 sum = float2(0.0f);
  for (uint i = 0; i < n; ++i) {
    const float2 d = sd_two_sum(pred[i], -target[i]);
    sum = sd_add(sum, sd_mul(d, d));
  }
  ((device ulong *)loss)[0] = native_gpu::pack_sd(sd_div_uint(sum, n));
}

kernel void welvet_mse_grad(device const float *pred [[buffer(0)]],
                            device const float *target [[buffer(1)]],
                            device float *grad [[buffer(2)]], constant uint &n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= n)
    return;
  const float2 d = sd_two_sum(pred[gid], -target[gid]);
  grad[gid] = sd_to_float(sd_div_uint(sd_mul_float(d, 2.0f), n));
}

kernel void welvet_sgd_f32(device float *weights [[buffer(0)]],
                           device const float *grad [[buffer(1)]],
                           constant float2 &lr [[buffer(2)]], constant uint &n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= n)
    return;
  // Go Float32 ApplySGD rounds the step before subtraction.
  const float step = abs(grad[gid]) < 1e-20f
                         ? ldexp(sd_to_float(sd_mul_float(lr, ldexp(grad[gid], 64))), -64)
                         : sd_to_float(sd_mul_float(lr, grad[gid]));
  weights[gid] = weights[gid] - step;
}

kernel void welvet_sgd_f32_f64_grad(device float *weights [[buffer(0)]],
                                    device const float2 *grad [[buffer(1)]],
                                    constant float2 &lr [[buffer(2)]],
                                    constant uint &n [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid >= n)
    return;
  const float step =
      abs(grad[gid].x) < 1e-20f
          ? ldexp(sd_to_float(sd_mul(lr, float2(ldexp(grad[gid].x, 64), ldexp(grad[gid].y, 64)))),
                  -64)
          : sd_to_float(sd_mul(lr, grad[gid]));
  weights[gid] = weights[gid] - step;
}

kernel void welvet_accumulate_f32(device float *dst [[buffer(0)]],
                                  device const float *src [[buffer(1)]],
                                  constant uint &n [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid < n)
    dst[gid] = dst[gid] + src[gid];
}
