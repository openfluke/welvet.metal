#pragma once

#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::layernorm {

struct Config {
  int dim = 0;
  double eps = 0.0; // <=0 → 1e-5 (matches Go Validate)
};

Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  WeightStore gamma; // 1×Dim FormatNone
  WeightStore beta;  // 1×Dim FormatNone
};

// γ/β decode_row_f64 + host f64 mean/var (Go/cpp exact_bit contract).
Status new_configured(const Config& cfg, DType dt, const float* gamma_init, const float* beta_init,
                      Layer* out);

// Forward: pre = x̂ = (x-μ)/σ; post = γ⊙x̂+β.
// input shape [batch,dim] or [batch,seq,dim].
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// NormalBP: dγ|dβ (grad_w length 2·Dim) and dx; pre must be x̂ from forward.
Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

// Apply [dγ|dβ] via Gamma/Beta.ApplySGD (FormatNone native).
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

} // namespace welvet::layernorm
