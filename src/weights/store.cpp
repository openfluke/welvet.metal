#include "welvet/weights.hpp"

#include <vector>

namespace welvet {

Status WeightStore::make(int rows, int cols, DType dt, const float* init_or_null, WeightStore* out) {
  if (!out || rows <= 0 || cols <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "weights: bad shape");
  }
  out->rows = rows;
  out->cols = cols;
  out->bias.clear();
  const int n = rows * cols;
  std::vector<float> scratch(static_cast<std::size_t>(n), 0.0f);
  if (init_or_null) {
    for (int i = 0; i < n; ++i) {
      scratch[static_cast<std::size_t>(i)] = init_or_null[i];
    }
  }
  return out->encode_format_none(dt, scratch.data(), n);
}

Status WeightStore::matvec_f32(const float* x, float* y) const {
  return Status::Err(StatusCode::Unsupported, "weights.matvec_f32: use Metal Dense dispatch; CPU execution is forbidden");
}

Status WeightStore::matvect_f32(const float* gy, float* gx) const {
  return Status::Err(StatusCode::Unsupported, "weights.matvect_f32: use Metal Dense dispatch; CPU execution is forbidden");
}

Status WeightStore::export_f32(float* dst) const {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "weights: nil dst");
  }
  if (format != QuantFormat::None) {
    return Status::Gap("export_f32 packed formats");
  }
  const int n = rows * cols;
  if (dtype == DType::Float32) {
    if (static_cast<int>(f32.size()) < n) {
      return Status::Err(StatusCode::InvalidArgument, "f32 short");
    }
    for (int i = 0; i < n; ++i) {
      dst[i] = f32[static_cast<std::size_t>(i)];
    }
    return Status::Ok();
  }
  std::vector<double> row(static_cast<std::size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    if (auto st = decode_row_f64(r, row.data()); !st) {
      return st;
    }
    for (int c = 0; c < cols; ++c) {
      dst[r * cols + c] = static_cast<float>(row[static_cast<std::size_t>(c)]);
    }
  }
  return Status::Ok();
}

} // namespace welvet
