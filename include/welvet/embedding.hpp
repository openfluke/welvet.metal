#pragma once

#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::embedding {

struct Config {
  int vocab_size = 0;
  int embedding_dim = 0;
  int seq_len = 0; // layout hint (must be >0)
};

int weight_count(const Config& cfg);
Status validate(const Config& cfg);

struct Layer {
  Config cfg{};
  WeightStore weights; // Vocab × Emb FormatNone
  Activation activation = Activation::Linear;
};

Status new_configured(const Config& cfg, DType dt, Activation act, const float* init_or_null,
                      Layer* out);

// Input rank 1..3: [T] | [B,T] | [B,T,1|E] (last>1 → token from channel 0).
// pre/post: [B,T,E] (or [T,E] for rank-1). post==pre after optional activation.
// Gather via decode_row_f64 (Go/cpp FormatNone×34 exact_bit).
Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// Scatter gradOut into dW [vocab,emb] by token id; gradIn zeros (same shape as input).
// Matches Go: no ActivateDeriv on scatter (Linear path is the signed NormalBP case).
Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

// Apply dW via weights.ApplySGD (FormatNone native).
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

} // namespace welvet::embedding
