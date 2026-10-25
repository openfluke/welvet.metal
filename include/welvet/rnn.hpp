#pragma once

#include "welvet/dense.hpp"
#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::rnn {

struct Config {
  int input_size = 0;
  int hidden_size = 0;
  int seq_len = 0;
};

int weight_count(const Config& cfg);
Status validate(const Config& cfg);

struct Layer {
  Config cfg{};
  dense::Layer IH; // H×In + bias
  dense::Layer HH; // H×H, no bias
};

// packed: [IH (H×In) | HH (H×H) | bias (H)] as f32; bias → f64 on IH.
Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out);

// Forward: pre/post [batch,seq,hid]. h₀=0. h_t = tanh(IH(x_t)+HH(h_{t-1})).
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

int grad_w_size(const Layer& layer);

// NormalBP BPTT: gradW loom [dIH | dHH | dBias]. pre = linear sums before tanh.
Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

// Export loom pack [IH | HH | bias] as f32 (bias cast from f64).
Status export_packed_f32(const Layer& layer, float* dst);

} // namespace welvet::rnn
