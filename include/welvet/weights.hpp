#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "welvet/dtype.hpp"
#include "welvet/status.hpp"

namespace welvet::weights {

// FormatNone store — Metal-resident path is gap? until Dense MatVec lands.
struct Store {
  DType dtype = DType::Float32;
  int32_t rows = 0;
  int32_t cols = 0;
  std::vector<uint8_t> host_bytes; // CPU mirror; device buffer later
  void* metal_buffer = nullptr;    // id<MTLBuffer> when resident
};

Status new_format_none(int32_t rows, int32_t cols, DType dt, const float* init_f32,
                       Store* out);
Status matvec_f32(const Store& W, const float* x, int32_t batch, float* y);
Status release_device(Store* s);

} // namespace welvet::weights
