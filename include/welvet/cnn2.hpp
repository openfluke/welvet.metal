#pragma once

#include "welvet/dense.hpp"
#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::cnn2 {

struct Config {
  int in_channels = 0;
  int filters = 0;
  int height = 0;
  int width = 0;
  int kernel = 0; // kH = kW
  int stride = 0; // <=0 → 1
  int padding = 0;
  Activation activation = Activation::ReLU;
};

int output_spatial(int spatial, int kernel, int stride, int padding);
int out_h(const Config& cfg);
int out_w(const Config& cfg);
int patch_dim(const Config& cfg);
Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  dense::Layer proj;
};

Status new_configured(const Config& cfg, DType dt, const float* w_init, Layer* out);

// Forward: im2col → Dense → loom [batch,filters,outH,outW].
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::cnn2
