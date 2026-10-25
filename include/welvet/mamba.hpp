#pragma once

#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/seqmix.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <vector>

namespace welvet::mamba {

struct Config {
  int d_model = 0;
  int d_state = 0; // validated; unused in minimal forward
  int expand = 2;  // 0 → 2; inner = expand * d_model
  int seq_len = 0; // expected T
};

int inner_dim(const Config& cfg);
int weight_count(const Config& cfg); // InProj + OutProj + ALog + DSkip
Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  dense::Layer in_proj;
  dense::Layer out_proj;
  std::vector<float> a_log;  // [Inner], always f32
  std::vector<float> d_skip; // [Inner], always f32

  static constexpr seqmix::Kind kind() { return seqmix::mamba_kind(); }
};

// packed: InProj[2*Inner×D] | OutProj[D×Inner] | ALog[Inner] | DSkip[Inner] (f32).
Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out);

// Input [batch, seq, d_model]. Dense soft-double In/Out + host f64 selective scan.
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

int grad_w_size(const Layer& layer);

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

Status export_packed_f32(const Layer& layer, float* dst);

} // namespace welvet::mamba
