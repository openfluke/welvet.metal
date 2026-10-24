#include "welvet/weights.hpp"

#include <cstring>

namespace welvet::weights {

Status new_format_none(int32_t rows, int32_t cols, DType dt, const float* init_f32, Store* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null Store");
  if (rows <= 0 || cols <= 0) return Status::Err(StatusCode::InvalidArgument, "bad shape");
  // Metal FormatNone pack/decode for all 34 dtypes is gap? — only host f32 mirror for scaffold.
  if (dt != DType::Float32) {
    return Status::Gap("weights FormatNone pack/decode on Metal for non-f32");
  }
  out->dtype = dt;
  out->rows = rows;
  out->cols = cols;
  out->host_bytes.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
                         sizeof(float));
  if (init_f32) {
    std::memcpy(out->host_bytes.data(), init_f32, out->host_bytes.size());
  } else {
    std::memset(out->host_bytes.data(), 0, out->host_bytes.size());
  }
  out->metal_buffer = nullptr;
  return Status::Ok();
}

Status matvec_f32(const Store&, const float*, int32_t, float*) {
  return Status::Gap("Metal Dense MatVec / welvet_dense_matvec_f32 host wiring");
}

Status release_device(Store* s) {
  if (!s) return Status::Ok();
  s->metal_buffer = nullptr;
  return Status::Ok();
}

} // namespace welvet::weights
