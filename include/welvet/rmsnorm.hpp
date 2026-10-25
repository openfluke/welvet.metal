#pragma once

#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::rmsnorm {

struct Config {
  int dim = 0;
  double eps = 0.0; // <=0 → 1e-6 (matches Go Validate)
};

Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  WeightStore gamma; // 1×Dim FormatNone
};

// γ decode_row_f64 + host f64 RMS (Go/cpp exact_bit contract).
// GPU soft-double RMSNorm is a later speed path — reduction order must stay sequential.
Status new_configured(const Config& cfg, DType dt, const float* gamma_init, Layer* out);

// Forward: pre = x̂ = x/rms; post = γ ⊙ x̂.
// input shape [batch,dim] or [batch,seq,dim].
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// NormalBP: dγ and dx; pre must be x̂ from forward (Go/cpp exact_bit).
Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

// Apply dγ via gamma.ApplySGD (FormatNone native).
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

} // namespace welvet::rmsnorm
