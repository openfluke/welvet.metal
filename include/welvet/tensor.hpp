#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace welvet {

struct TensorF32 {
  std::vector<int32_t> shape;
  std::vector<float> data;

  std::size_t numel() const { return data.size(); }
};

} // namespace welvet
