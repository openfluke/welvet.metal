#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "welvet/dtype.hpp"
#include "welvet/status.hpp"

namespace welvet {

enum class QuantFormat : int32_t {
  None = 0,
  // gap? all other quant.Format values from welvet/quant
};

// FormatNone store — MatVec accumulates in float64 (Go parity).
struct WeightStore {
  DType dtype = DType::Float32;
  QuantFormat format = QuantFormat::None;
  int rows = 0;
  int cols = 0;
  std::vector<float> f32;       // Float32 FormatNone payload
  std::vector<uint8_t> native;  // other FormatNone dtypes
  float scale = 1.0f;
  std::vector<double> bias;

  static Status make(int rows, int cols, DType dt, const float* init_or_null, WeightStore* out);
  static Status make_f32(int rows, int cols, const float* init_or_null, WeightStore* out) {
    return make(rows, cols, DType::Float32, init_or_null, out);
  }

  Status encode_format_none(DType dt, const float* scratch, int n);
  Status decode_row_f64(int row, double* dst) const;
  // Decode full matrix to float32 (FormatNone).
  Status export_f32(float* dst) const;

  // y[rows] = W @ x[cols], float64 accumulate then cast to float32.
  Status matvec_f32(const float* x, float* y) const;
  // gx[cols] += W^T @ gy[rows] (float64 accumulate; caller zeros gx).
  Status matvect_f32(const float* gy, float* gx) const;

  // w <- w - lr*dW in native FormatNone storage (Go weights.ApplySGD).
  Status apply_sgd(const double* dW, double lr);

  // Bias ← Bias − lr·dB (float64 host bias; no-op if bias empty). Go ApplyBiasSGD.
  Status apply_bias_sgd(const double* dB, int n, double lr);
};

} // namespace welvet
