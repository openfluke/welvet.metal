#pragma once

#include <cstdint>
#include <string_view>

namespace welvet {

// Mirrors welvet/core activation IDs used by Dense / wire tools.
enum class Activation : int32_t {
  Linear = 0,
  ReLU = 1,
  SiLU = 2,
  GELU = 3,
  Tanh = 4,
  Sigmoid = 5,
  LeakyReLU = 6,
  ReLU2 = 7,
};

std::string_view activation_name(Activation a);
double activate(Activation a, double x);
double activate_deriv(Activation a, double x);

} // namespace welvet
