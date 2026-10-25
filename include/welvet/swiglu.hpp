#pragma once

#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::swiglu {

struct Config {
  int input_dim = 0;
  int intermediate_dim = 0;
};

Status validate(const Config& cfg);

// Llama-style ~8/3 intermediate, rounded to multiple of 8.
Config default_ffn(int d_model);

struct Layer {
  Config cfg{};
  dense::Layer Gate, Up, Down;
};

// Gate/Up/Down Dense on Metal; SiLU⊙ host f64 matching Go/cpp (exact_bit).
Status new_configured(const Config& cfg, DType dt, const float* gate_init, const float* up_init,
                      const float* down_init, Layer* out);

// Forward: pre = h = SiLU(Gate(x)) ⊙ Up(x); post = Down(h).
// input shape [batch,in] or [batch,seq,in].
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// NormalBP: grad_w = concat(dGate, dUp, dDown); pre must be h from forward.
Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

// Apply concat dW (Gate then Up then Down) via each Dense ApplyGradSGD.
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

// Flatten Gate|Up|Down weight matrices into dst (length GradWSize).
int grad_w_size(const Layer& layer);
Status export_weights_f32(const Layer& layer, float* dst);

} // namespace welvet::swiglu
