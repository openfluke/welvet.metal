#include "welvet/activate.hpp"

#include <cmath>

namespace welvet {

std::string_view activation_name(Activation a) {
  switch (a) {
  case Activation::Linear: return "linear";
  case Activation::ReLU: return "relu";
  case Activation::SiLU: return "silu";
  case Activation::GELU: return "gelu";
  case Activation::Tanh: return "tanh";
  case Activation::Sigmoid: return "sigmoid";
  case Activation::LeakyReLU: return "leaky_relu";
  case Activation::ReLU2: return "relu2";
  }
  return "unknown";
}

double activate(Activation a, double x) {
  switch (a) {
  case Activation::Linear: return x;
  case Activation::ReLU: return x > 0.0 ? x : 0.0;
  case Activation::SiLU: return x / (1.0 + std::exp(-x));
  case Activation::GELU: {
    // Approximate GELU used as host reference until Metal kernel lands.
    constexpr double k = 0.7978845608028654; // sqrt(2/pi)
    return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
  }
  case Activation::Tanh: return std::tanh(x);
  case Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-x));
  case Activation::LeakyReLU: return x > 0.0 ? x : 0.01 * x;
  case Activation::ReLU2: {
    const double r = x > 0.0 ? x : 0.0;
    return r * r;
  }
  }
  return x;
}

double activate_deriv(Activation a, double x) {
  switch (a) {
  case Activation::Linear: return 1.0;
  case Activation::ReLU: return x > 0.0 ? 1.0 : 0.0;
  case Activation::SiLU: {
    const double s = 1.0 / (1.0 + std::exp(-x));
    return s + x * s * (1.0 - s);
  }
  case Activation::GELU: {
    // Match host activate() approx; exact Metal/Go parity comes with the kernel gate.
    constexpr double k = 0.7978845608028654;
    const double u = k * (x + 0.044715 * x * x * x);
    const double t = std::tanh(u);
    const double du = k * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du;
  }
  case Activation::Tanh: {
    const double t = std::tanh(x);
    return 1.0 - t * t;
  }
  case Activation::Sigmoid: {
    const double s = 1.0 / (1.0 + std::exp(-x));
    return s * (1.0 - s);
  }
  case Activation::LeakyReLU: return x > 0.0 ? 1.0 : 0.01;
  case Activation::ReLU2: return x > 0.0 ? 2.0 * x : 0.0;
  }
  return 0.0;
}

} // namespace welvet
