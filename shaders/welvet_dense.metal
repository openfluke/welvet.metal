// Welvet Metal — Dense FormatNone forward (soft-double; Apple Metal has no HW double).
// Contract = Go FormatNone MatVec: sum_k float64(W)*float64(x), then float(sum).
// Compile with fastMathEnabled=NO (see library.mm).
#include <metal_stdlib>
using namespace metal;

kernel void welvet_smoke_saxpy(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant float& a [[buffer(2)]],
    constant float& b [[buffer(3)]],
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

kernel void welvet_dense_matvec_softdouble_batch(
    device const float2* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = batch * out_dim;
  if (gid >= total) return;
  const uint b = gid / out_dim;
  const uint o = gid % out_dim;
  float2 acc = float2(0.0f, 0.0f);
  const uint row = o * in_dim;
  const uint xoff = b * in_dim;
  for (uint k = 0; k < in_dim; ++k) {
    acc = sd_add(acc, sd_mul_float(W[row + k], x[xoff + k]));
  }
  y[gid] = sd_to_float(acc);
}

kernel void welvet_dense_bias_act(
    device float* pre [[buffer(0)]],
    device float* post [[buffer(1)]],
    device const float2* bias [[buffer(2)]],
    constant uint& out_dim [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant int& act [[buffer(5)]],
    constant uint& bias_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = batch * out_dim;
  if (gid >= total) return;
  const uint o = gid % out_dim;

  float2 v = float2(pre[gid], 0.0f);
  if (o < bias_len) {
    v = sd_add(v, bias[o]);
    pre[gid] = sd_to_float(v);
  }
  float vf = sd_to_float(v);
  float y = vf;
  if (act == 0) {
    y = vf < 0.0f ? 0.0f : vf;
  } else if (act == 6) {
    y = vf < 0.0f ? 0.0f : vf * vf;
  } else if (act == 5) {
    y = vf < 0.0f ? 0.01f * vf : vf;
  } else if (act == 1) {
    y = vf / (1.0f + exp(-vf));
  } else if (act == 2) {
    constexpr float k = 0.7978845608028654f;
    y = 0.5f * vf * (1.0f + tanh(k * (vf + 0.044715f * vf * vf * vf)));
  } else if (act == 3) {
    y = tanh(vf);
  } else if (act == 4) {
    y = 1.0f / (1.0f + exp(-vf));
  }
  post[gid] = y;
}

// d_pre = gy * act'(pre)  — pre is post-bias / pre-act (same buffer as forward `pre`).
kernel void welvet_dense_act_deriv(
    device const float* gy [[buffer(0)]],
    device const float* pre [[buffer(1)]],
    device float* d_pre [[buffer(2)]],
    constant uint& out_dim [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant int& act [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = batch * out_dim;
  if (gid >= total) return;
  const float vf = pre[gid];
  float d = 1.0f;
  if (act == 0) { // ReLU
    d = vf > 0.0f ? 1.0f : 0.0f;
  } else if (act == 6) { // ReLU2
    d = vf > 0.0f ? 2.0f * vf : 0.0f;
  } else if (act == 5) { // LeakyReLU
    d = vf > 0.0f ? 1.0f : 0.01f;
  } else if (act == 1) { // Silu
    const float sig = 1.0f / (1.0f + exp(-vf));
    d = sig * (1.0f + vf * (1.0f - sig));
  } else if (act == 2) { // GELU
    constexpr float k = 0.7978845608028654f;
    const float u = k * (vf + 0.044715f * vf * vf * vf);
    const float th = tanh(u);
    const float du = k * (1.0f + 3.0f * 0.044715f * vf * vf);
    d = 0.5f * (1.0f + th) + 0.5f * vf * (1.0f - th * th) * du;
  } else if (act == 3) { // Tanh
    const float t = tanh(vf);
    d = 1.0f - t * t;
  } else if (act == 4) { // Sigmoid
    const float s = 1.0f / (1.0f + exp(-vf));
    d = s * (1.0f - s);
  }
  d_pre[gid] = gy[gid] * d;
}

// gx = Wᵀ @ gy  (soft-double W). gid over batch * in_dim.
kernel void welvet_dense_matvect_softdouble_batch(
    device const float2* W [[buffer(0)]],
    device const float* gy [[buffer(1)]],
    device float* gx [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = batch * in_dim;
  if (gid >= total) return;
  const uint b = gid / in_dim;
  const uint i = gid % in_dim;
  float2 acc = float2(0.0f, 0.0f);
  const uint gyoff = b * out_dim;
  for (uint o = 0; o < out_dim; ++o) {
    acc = sd_add(acc, sd_mul_float(W[o * in_dim + i], gy[gyoff + o]));
  }
  gx[gid] = sd_to_float(acc);
}

// dW[o,i] = sum_b gy[b,o] * x[b,i]  (f32 outer; matches host f64-accum→f32 cast closely enough).
kernel void welvet_dense_outer_batch(
    device const float* gy [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* dW [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = out_dim * in_dim;
  if (gid >= total) return;
  const uint o = gid / in_dim;
  const uint i = gid % in_dim;
  float2 acc = float2(0.0f, 0.0f);
  for (uint b = 0; b < batch; ++b) {
    const float g = gy[b * out_dim + o];
    const float xv = x[b * in_dim + i];
    acc = sd_add(acc, sd_mul_float(float2(g, 0.0f), xv));
  }
  dW[gid] = sd_to_float(acc);
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
kernel void welvet_mse_loss(device const float* pred [[buffer(0)]],
                            device const float* target [[buffer(1)]],
                            device float2* loss [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid != 0) return;
  float2 sum = float2(0.0f);
  for (uint i = 0; i < n; ++i) {
    const float2 d = sd_two_sum(pred[i], -target[i]);
    sum = sd_add(sum, sd_mul(d, d));
  }
  loss[0] = sd_div_uint(sum, n);
}

kernel void welvet_mse_grad(device const float* pred [[buffer(0)]],
                            device const float* target [[buffer(1)]],
                            device float* grad [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  const float2 d = sd_two_sum(pred[gid], -target[gid]);
  grad[gid] = sd_to_float(sd_div_uint(sd_mul_float(d, 2.0f), n));
}

kernel void welvet_sgd_f32(device float* weights [[buffer(0)]],
                           device const float* grad [[buffer(1)]],
                           constant float2& lr [[buffer(2)]],
                           constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  // Go Float32 ApplySGD rounds the step before subtraction.
  const float step = sd_to_float(sd_mul_float(lr, grad[gid]));
  weights[gid] = weights[gid] - step;
}

kernel void welvet_sgd_f32_f64_grad(device float* weights [[buffer(0)]],
                                    device const float2* grad [[buffer(1)]],
                                    constant float2& lr [[buffer(2)]],
                                    constant uint& n [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  const float step = sd_to_float(sd_mul(lr, grad[gid]));
  weights[gid] = weights[gid] - step;
}

kernel void welvet_accumulate_f32(device float* dst [[buffer(0)]],
                                  device const float* src [[buffer(1)]],
                                  constant uint& n [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid < n) dst[gid] = dst[gid] + src[gid];
}
