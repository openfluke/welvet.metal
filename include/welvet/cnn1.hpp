#pragma once

#include "welvet/dense.hpp"
#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::cnn1 {

struct Config {
  int in_channels = 0;
  int filters = 0;
  int seq_len = 0;
  int kernel = 0;
  int stride = 0; // <=0 → 1
  int padding = 0;
  Activation activation = Activation::ReLU;
};

int output_length(int seq_len, int kernel, int stride, int padding);
int out_len(const Config& cfg);
int patch_dim(const Config& cfg);
Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  dense::Layer proj;
};

Status new_configured(const Config& cfg, DType dt, const float* w_init, Layer* out);

// Forward: im2col → Dense → loom [batch,filters,outLen]. pre/post both loom layout.
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

int grad_w_size(const Layer& layer);

// NormalBP: loom→dense flat → Dense bwd → col2im. pre must be loom from forward.
Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);
Status export_weights_f32(const Layer& layer, float* dst);

} // namespace welvet::cnn1
