#pragma once

#include "welvet/dtype.hpp"
#include "welvet/seqmix.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <vector>

namespace welvet::gdn {

struct Config {
  int hidden_size = 0;
  int num_key_heads = 0;
  int num_value_heads = 0;
  int key_head_dim = 0;
  int value_head_dim = 0;
  int conv_kernel = 4; // <1 → 4
  double eps = 1e-6;   // <=0 → 1e-6
};

int key_dim(const Config &cfg);
int value_dim(const Config &cfg);
int conv_dim(const Config &cfg);
int weight_count(const Config &cfg);
Status validate(Config *cfg);

struct Layer {
  Config cfg{};
  // FormatNone f32 projections (row-major rows×cols).
  std::vector<float> in_qkv;      // [convDim × H]
  std::vector<float> in_z;        // [valDim × H]
  std::vector<float> in_b;        // [nv × H]
  std::vector<float> in_a;        // [nv × H]
  std::vector<float> out;         // [H × valDim]
  std::vector<float> conv_weight; // [convDim · k]
  std::vector<float> a_log;       // [nv]
  std::vector<float> dt_bias;     // [nv]
  std::vector<float> norm_gamma;  // [hdV]

  static constexpr seqmix::Kind kind() { return seqmix::gdn_kind(); }
};

// packed: InQKV|InZ|InB|InA|Out|ConvWeight|ALog|DtBias|NormGamma (all f32).
Status new_configured(const Config &cfg, const float *packed_or_null, Layer *out);

// Input [batch, seq, hidden]. pre/post same shape (Linear). Metal f32 matvec (FormatNone).
Status forward(Layer &layer, const TensorF32 &input, TensorF32 *pre, TensorF32 *post);

// Grad loom length = weight_count (InQKV|InZ|InB|InA|Out|Conv|ALog|DtBias|γ).
int grad_w_size(const Layer &layer);

// Truncated-BPTT NormalBP (f32). State/Conv history stop-grad across tokens.
Status backward_f32(Layer &layer, const TensorF32 &grad_out, const TensorF32 &input,
                    const TensorF32 &pre, TensorF32 *grad_in, TensorF32 *grad_w);

Status apply_grad_sgd(Layer *layer, const TensorF32 &dW, double lr);

Status export_packed_f32(const Layer &layer, float *dst);

} // namespace welvet::gdn
