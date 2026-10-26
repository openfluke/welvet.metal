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

// FormatNone storage/staging. Numerical execution must use Metal dispatch.
struct WeightStore {
  DType dtype = DType::Float32;
  QuantFormat format = QuantFormat::None;
  int rows = 0;
  int cols = 0;
  std::vector<float> f32;      // Float32 FormatNone payload
  std::vector<uint8_t> native; // other FormatNone dtypes
  float scale = 1.0f;
  std::vector<double> bias;

  static Status make(int rows, int cols, DType dt, const float *init_or_null, WeightStore *out);
  static Status make_f32(int rows, int cols, const float *init_or_null, WeightStore *out) {
    return make(rows, cols, DType::Float32, init_or_null, out);
  }

  Status encode_format_none(DType dt, const float *scratch, int n);
  Status decode_row_f64(int row, double *dst) const;
  // Decode full matrix to float32 (FormatNone).
  Status export_f32(float *dst) const;

  // Legacy host entry: returns Unsupported; use Metal Dense dispatch.
  Status matvec_f32(const float *x, float *y) const;
  // Legacy host entry: returns Unsupported; use Metal Dense backward.
  Status matvect_f32(const float *gy, float *gx) const;

  // Metal SGD plus native storage encoding for all FormatNone dtypes.
  Status apply_sgd(const double *dW, double lr);

  // Re-encode a complete store after GPU blending, including fresh native scale.
  Status set_from_f32_gpu(const std::vector<float>&);

  // Metal float64 bias SGD via explicit software precision; empty bias is a no-op.
  Status apply_bias_sgd(const double *dB, int n, double lr);
};

} // namespace welvet
