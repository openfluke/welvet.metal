#include "welvet/tensor.hpp"

namespace welvet {

TensorF32 TensorF32::zeros(int64_t a, int64_t b) {
  TensorF32 t;
  t.shape = {a, b};
  t.data.assign(static_cast<std::size_t>(a * b), 0.0f);
  return t;
}

} // namespace welvet
