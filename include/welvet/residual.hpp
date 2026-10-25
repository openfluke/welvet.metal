#pragma once

#include "welvet/activate.hpp"
#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <vector>

namespace welvet::residual {

struct Config {
  int dim = 0;
  int seq_len = 1;
  int depth = 1; // Dense Dim→Dim children in F; 0 → 1
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

// Input [batch,dim] or [batch,seq,dim]. pre = last F child pre; post = F(x)+x.
// F via Metal dense::forward_f32; skip-add via f64 round-trip (Go AsFloat64+FromFloat64).
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

int grad_w_size(const Layer& layer);

// NormalBP: recompute F; reverse Dense chain; gradIn = g_F + g_y (f64 skip-add).
Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

Status export_packed_f32(const Layer& layer, float* dst);

} // namespace welvet::residual
