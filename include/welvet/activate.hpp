#pragma once

#include <cstdint>

namespace welvet {

// Mirrors welvet/core/layer.go ActivationType values.
enum class Activation : int32_t {
  Linear = -1,
  ReLU = 0,
  Silu = 1,
  GELU = 2,
  Tanh = 3,
  Sigmoid = 4,
  LeakyReLU = 5,
  ReLU2 = 6,
};

// Pointwise activation in float64 space (same as Go core.Activate).
double activate_f64(double x, Activation act);
double activate_deriv_f64(double x, Activation act);

} // namespace welvet
