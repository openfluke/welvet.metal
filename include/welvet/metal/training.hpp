#pragma once

#include <cstddef>
#include "welvet/status.hpp"

namespace welvet::metal {

// Numerical work is exclusively Metal. Host pointers are staging/export buffers.
Status mse_f32(const float* pred, const float* target, std::size_t n,
               double* loss, float* grad);
Status sgd_f32(float* weights, const float* grad, std::size_t n, double lr);
Status sgd_f32_f64_grad(float* weights, const double* grad, std::size_t n, double lr);
Status accumulate_f32(float* dst, const float* src, std::size_t n);

} // namespace welvet::metal
