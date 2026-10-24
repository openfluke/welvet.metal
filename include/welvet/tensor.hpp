#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace welvet {

// Minimal host tensor — activations for Dense forward (float32 path first).
struct TensorF32 {
  std::vector<int64_t> shape;
  std::vector<float> data;

  static TensorF32 zeros(int64_t a, int64_t b);
  std::size_t numel() const { return data.size(); }
};

} // namespace welvet
