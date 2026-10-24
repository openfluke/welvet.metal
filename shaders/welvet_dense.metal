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
