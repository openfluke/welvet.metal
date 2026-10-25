#pragma once

#include "welvet/activate.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <vector>

namespace welvet::sequential {

struct Config {
  int dim = 0;
  int seq_len = 1;
  int depth = 2; // number of Dense Dim→Dim children; 0 → 2
};

int weight_count(const Config& cfg);
Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  std::vector<dense::Layer> children;
};

// packed: concat of Depth × [Dim×Dim] child weights (f32 → FormatNone dtype).
Status new_configured(const Config& cfg, DType dt, Activation child_act, const float* packed_or_null,
                      Layer* out);

// Input [batch,dim] or [batch,seq,dim]. pre/post = last child pre/post, same outer shape.
// Chains Metal dense::forward_f32 (soft-double FormatNone×34 exact_bit).
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

int grad_w_size(const Layer& layer);

// NormalBP: recompute child tape; reverse Dense chain; gradW = concat child dWs.
// Top-level pre unused (may be empty).
Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

// Export loom pack Depth×[Dim×Dim] as f32.
Status export_packed_f32(const Layer& layer, float* dst);

} // namespace welvet::sequential
