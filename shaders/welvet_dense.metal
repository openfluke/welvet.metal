// Welvet Metal shaders — M0 scaffold.
// Dense MatVec / activations / backward kernels land as gates flip in GAPS.md.
#include <metal_stdlib>
using namespace metal;

// Identity saxpy-style smoke kernel: y[i] = a * x[i] + b
kernel void welvet_smoke_saxpy(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant float& a [[buffer(2)]],
    constant float& b [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  y[gid] = a * x[gid] + b;
}

// Placeholder Dense GEMV (row-major W[out][in]): not wired from host yet.
kernel void welvet_dense_matvec_f32(
    device const float* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= out_dim) return;
  float acc = 0.0f;
  const uint row = gid * in_dim;
  for (uint k = 0; k < in_dim; ++k) {
    acc += W[row + k] * x[k];
  }
  y[gid] = acc;
}
