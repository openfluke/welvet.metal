#pragma once

#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::dense {

// Metal-only Dense. No host MatVec backend. No silent CPU substitute.
struct Layer {
  int input_height = 0;
  int output_height = 0;
  Activation activation = Activation::Linear;
  WeightStore weights;
};

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float* init_or_null,
                 Layer* layer);

// True when FormatNone Dense forward on Metal can match Go's f64 MatVec contract
// for this dtype. Unsupported dtypes hard-error — never a different path.
bool dtype_supported_metal_forward(DType dt);

// FormatNone Dense forward entirely on Metal (GEMV + bias + act). Staging
// decode of FormatNone → double for upload is host prep only, not a compute fallback.
Status forward_f32(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// NormalBP leaf: dPre = gradOut ⊙ act'(pre); dX = Wᵀ·dPre; dW = outer(dPre, x).
// Host f64 MatVecT + outer (Go BackwardCPUTiled / welvet.cpp parity). pre = post-bias.
Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

// Apply dW via weights.ApplySGD (FormatNone native).
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

} // namespace welvet::dense
