#pragma once

#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::dense {

struct Layer {
  int32_t in = 0;
  int32_t out = 0;
  Activation act = Activation::Linear;
  DType dtype = DType::Float32;
  weights::Store weights;
  std::vector<double> bias;
};

Status new_layer(int32_t in_dim, int32_t out_dim, Activation act, DType dt, const float* W_init,
                 Layer* layer_out);

// Metal Dense forward — gap? until shaders + residency land.
Status forward_f32(const Layer& layer, const TensorF32& x, TensorF32* pre, TensorF32* post);

} // namespace welvet::dense
