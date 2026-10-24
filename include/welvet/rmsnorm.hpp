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

} // namespace welvet::rmsnorm
