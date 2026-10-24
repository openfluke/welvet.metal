#pragma once

#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>

namespace welvet::mha {

// Mirrors welvet/layers/mha policy enums (values matter for the WVM1 wire).
enum class MaskKind : int32_t {
  Unspecified = 0,
  Causal = 1,
  Bidirectional = 2,
  SlidingWindow = 3,
  PrefixLM = 4,
  Custom = 5,
};

enum class PosKind : int32_t {
  RoPE = 0,
  None = 1,
  ALiBi = 2,
  RoPEALiBi = 3,
};

enum class AttnMode : int32_t {
  Self = 0,
  Cross = 1,
};

enum class SoftmaxKind : int32_t {
  Standard = 0,
  Sigmoid = 1,
};

struct Config {
  int d_model = 0;
  int num_heads = 0;
  int num_kv_heads = 0; // 0 → num_heads
  int head_dim = 0;     // 0 → d_model/num_heads
  int max_seq_len = 0;  // 0 → 512

  MaskKind mask = MaskKind::Causal;
  PosKind pos = PosKind::RoPE;
  AttnMode mode = AttnMode::Self;
  SoftmaxKind softmax = SoftmaxKind::Standard;

  bool causal = true;
  int window = 0;
  bool window_causal = true;
  int prefix_len = 0;
  double rope_theta = 10000.0;
  double scale_override = 0.0;

  int q_dim() const { return num_heads * head_dim; }
  int kv_dim() const { return num_kv_heads * head_dim; }
  int heads_per_kv() const { return num_heads / num_kv_heads; }
  double scale() const;
  bool uses_rope() const { return pos == PosKind::RoPE || pos == PosKind::RoPEALiBi; }
  bool uses_alibi() const { return pos == PosKind::ALiBi || pos == PosKind::RoPEALiBi; }
};

Status validate(Config* cfg);

Config decoder_causal(int d_model, int num_heads, int num_kv_heads);

struct Layer {
  Config cfg{};
  dense::Layer Q, K, V, O;
};

// FormatNone projections; Q/K/V/O Dense runs on Metal (soft-double GEMV).
// Attention ALU is host f64 — same baseline as welvet.cpp / Go (exact_bit).
Status new_configured(const Config& cfg, DType dt, const float* q_init, const float* k_init,
                      const float* v_init, const float* o_init, Layer* out);

// Self-attn forward. input [batch,seq,d] or [seq,d].
// pre = attn [batch,seq,q_dim]; post = O(attn) [batch,seq,d_model].
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

} // namespace welvet::mha
