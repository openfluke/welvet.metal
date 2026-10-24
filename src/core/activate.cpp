#include "welvet/activate.hpp"

#include <cmath>

namespace welvet {

double activate_f64(double x, Activation act) {
  switch (act) {
  case Activation::Linear:
    return x;
  case Activation::ReLU:
    return x < 0.0 ? 0.0 : x;
  case Activation::ReLU2:
    return x < 0.0 ? 0.0 : x * x;
  case Activation::LeakyReLU:
    return x < 0.0 ? 0.01 * x : x;
  case Activation::Silu:
    return x / (1.0 + std::exp(-x));
  case Activation::GELU: {
    constexpr double k = 0.7978845608028654; // sqrt(2/pi)
    return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
  }
  case Activation::Tanh:
    return std::tanh(x);
  case Activation::Sigmoid:
    return 1.0 / (1.0 + std::exp(-x));
  }
  return x;
}

double activate_deriv_f64(double x, Activation act) {
  switch (act) {
  case Activation::Linear:
    return 1.0;
  case Activation::ReLU:
    return x > 0.0 ? 1.0 : 0.0;
  case Activation::ReLU2:
    return x > 0.0 ? 2.0 * x : 0.0;
  case Activation::LeakyReLU:
    return x > 0.0 ? 1.0 : 0.01;
  case Activation::Silu: {
    const double sig = 1.0 / (1.0 + std::exp(-x));
    return sig * (1.0 + x * (1.0 - sig));
  }
  case Activation::GELU: {
    constexpr double k = 0.7978845608028654;
    const double u = k * (x + 0.044715 * x * x * x);
    const double th = std::tanh(u);
    const double du = k * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
  }
  case Activation::Tanh: {
    const double t = std::tanh(x);
    return 1.0 - t * t;
  }
  case Activation::Sigmoid: {
    const double s = 1.0 / (1.0 + std::exp(-x));
    return s * (1.0 - s);
  }
  }
  return 1.0;
}

} // namespace welvet
